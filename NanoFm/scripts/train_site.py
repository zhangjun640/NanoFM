import os
import sys
import torch
import numpy as np
import pandas as pd
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, ArgumentTypeError
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_curve, precision_recall_curve, auc

sys.path.append(os.getcwd())
from NanoFM.model.models_new import SiteLevelModel
from NanoFM.utils.pytorchtools import EarlyStopping
from NanoFM.utils.constants import use_cuda, min_cov
from NanoFM.utils.MyDataSet_site import SiteLevelDataset


def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')


def loss_function():
    return torch.nn.BCELoss()


def adjust_features(features):
    kmer = features[:, :, 0:1, :]
    qual = features[:, :, 4:5, :]
    mis = features[:, :, 5:6, :]
    ins = features[:, :, 6:7, :]
    dele = features[:, :, 7:8, :]
    sigs = features[:, :, 8:, :]
    return torch.cat([kmer, qual, mis, ins, dele, sigs], dim=2)


def load_embedding_dict(parquet_path):
    print(f"Loading embeddings from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    emb_dict = {}
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Parsing Parquet"):
        key = f"{row['chr']}_{row['pos']}"
        v5 = row['center_5mer_emb']
        if hasattr(v5, 'tolist'): v5 = v5.tolist()
        emb_dict[key] = np.array(v5, dtype=np.float32)
    return emb_dict


def get_embeddings_for_batch(sample_info, emb_dict):
    e5 = []
    for info in sample_info:
        parts = str(info).split('|')
        key = f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else str(info)
        if key in emb_dict:
            e5.append(emb_dict[key])
        else:
            e5.append(np.zeros((5, 768), dtype=np.float32))

    e5_t = torch.tensor(np.array(e5)).float()
    if use_cuda: e5_t = e5_t.cuda()
    return e5_t


# [Added]: unified metric computation function to completely resolve list-vs-float comparison errors
def get_metrics(y_true, y_pred):
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)

    pred_label = (y_pred_np >= 0.5).astype(int)
    acc = accuracy_score(y_true_np, pred_label)

    try:
        fpr, tpr, _ = roc_curve(y_true_np, y_pred_np)
        roc_val = auc(fpr, tpr)
    except:
        roc_val = 0

    try:
        precision, recall, _ = precision_recall_curve(y_true_np, y_pred_np)
        pr_val = auc(recall, precision)
    except:
        pr_val = 0

    TP = np.sum(np.logical_and(pred_label == 1, y_true_np == 1))
    FN = np.sum(np.logical_and(pred_label == 0, y_true_np == 1))
    sen = TP / float(TP + FN) if (TP + FN) != 0 else 0

    return {'acc': acc, 'roc_auc': roc_val, 'pr_auc': pr_val, 'sensitivity': sen}


def train_epoch(model, train_dl, optimizer, loss_func, emb_dict, clip_grad):
    model.train()
    train_loss_list, all_y_true, all_y_pred = [], [], []

    for batch in tqdm(train_dl, desc="Training"):
        sample_info, features, labels = batch[0], batch[1].float(), batch[2].float()
        if use_cuda: features, labels = features.cuda(), labels.cuda()
        features = adjust_features(features)

        e5 = get_embeddings_for_batch(sample_info, emb_dict)

        optimizer.zero_grad()
        y_pred = model(features, e5).squeeze(-1)
        loss = loss_func(y_pred, labels)
        loss.backward()

        if clip_grad: torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        train_loss_list.append(loss.item())
        all_y_true.extend(labels.detach().cpu().numpy().flatten())
        all_y_pred.extend(y_pred.detach().cpu().numpy().flatten())

    metrics = get_metrics(all_y_true, all_y_pred)
    metrics['avg_loss'] = np.mean(train_loss_list) if train_loss_list else 0
    return metrics


def val_epoch(model, val_dl, loss_func, emb_dict):
    model.eval()
    val_loss_list, all_y_true, all_y_pred = [], [], []

    with torch.no_grad():
        for batch in tqdm(val_dl, desc="Validation"):
            sample_info, features, labels = batch[0], batch[1].float(), batch[2].float()
            if use_cuda: features, labels = features.cuda(), labels.cuda()
            features = adjust_features(features)
            e5 = get_embeddings_for_batch(sample_info, emb_dict)

            y_pred = model(features, e5).squeeze(-1)
            loss = loss_func(y_pred, labels)

            val_loss_list.append(loss.item())
            all_y_true.extend(labels.cpu().numpy().flatten())
            all_y_pred.extend(y_pred.cpu().numpy().flatten())

    metrics = get_metrics(all_y_true, all_y_pred)
    metrics['avg_loss'] = np.mean(val_loss_list) if val_loss_list else 0
    return metrics


def train(args):
    emb_dict = load_embedding_dict(args.embedding_file)
    model = SiteLevelModel(args.model_type, args.dropout_rate, args.hidden_size,
                           args.seq_lens, args.signal_lens, embedding_size=768,
                           use_cross_attention=args.use_cross_attention,
                           num_heads=args.num_heads, dropout_attn=args.dropout_attn)
    if use_cuda: model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_func = loss_function()

    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume)
        try:
            model.load_state_dict(checkpoint['net'])
        except RuntimeError:
            model.module.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    model_dir = os.path.abspath(args.save_dir)
    if not os.path.exists(model_dir): os.makedirs(model_dir)

    early_stop_path = os.path.join(model_dir, "checkpoint.pt")
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, path=early_stop_path)

    train_dataset = SiteLevelDataset(args.train_file, min_cov=min_cov)
    valid_dataset = SiteLevelDataset(args.valid_file, min_cov=min_cov)
    train_dl = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                          prefetch_factor=2)
    val_dl = DataLoader(dataset=valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        prefetch_factor=2)

    curr_best_epoch_acc = 0
    for epoch in range(args.epochs):
        print(f"\nEpoch: [{epoch + 1}/{args.epochs}]")
        train_res = train_epoch(model, train_dl, optimizer, loss_func, emb_dict, args.clip_grad)
        val_res = val_epoch(model, val_dl, loss_func, emb_dict)

        print("Train Loss: {:.3f}\t ROC AUC: {:.3f}\t PR AUC: {:.3f}\t ACC: {:.3f}\t SEN: {:.3f}".format(
            train_res['avg_loss'], train_res['roc_auc'], train_res['pr_auc'], train_res['acc'],
            train_res['sensitivity']))
        print("Val   Loss: {:.3f}\t ROC AUC: {:.3f}\t PR AUC: {:.3f}\t ACC: {:.3f}\t SEN: {:.3f}".format(
            val_res['avg_loss'], val_res['roc_auc'], val_res['pr_auc'], val_res['acc'], val_res['sensitivity']))

        state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict()}
        torch.save(state, os.path.join(model_dir, "model_states_{}.pt".format(epoch + 1)))

        early_stopping(val_res['acc'], model, optimizer)
        if val_res['acc'] > curr_best_epoch_acc:
            curr_best_epoch_acc = val_res['acc']
            print(f"  >>> New Best Accuracy: {curr_best_epoch_acc:.4f} at Epoch {epoch + 1}")

        if early_stopping.early_stop:
            if epoch + 1 >= args.min_epoch:
                break
            else:
                early_stopping.recount()


def argparser():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--valid_file", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--embedding_file", required=True)

    parser.add_argument("--use_cross_attention", type=str2bool, default=True,
                        help="Whether to enable bidirectional cross-attention fusion (True/False)")
    parser.add_argument("--num_heads", default=4, type=int)
    parser.add_argument("--dropout_attn", default=0.1, type=float)

    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--seed", default=48, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--min_epoch", default=10, type=int)
    parser.add_argument("--weight_decay", default=1e-5, type=float)
    parser.add_argument("--patience", default=20, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--clip_grad", default=0.5, type=float)
    parser.add_argument("--dropout_rate", default=0.5, type=float)
    parser.add_argument("--hidden_size", default=128, type=int)

    parser.add_argument("--seq_lens", default=5, type=int)
    parser.add_argument("--signal_lens", default=65, type=int)
    parser.add_argument("--embedding_size", default=768, type=int)
    parser.add_argument("--model_type", default='comb', type=str, choices=["basecall", "raw_signals", "comb"])
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--resume", type=str, default=None)

    return parser


if __name__ == '__main__':
    args = argparser().parse_args()
    train(args)