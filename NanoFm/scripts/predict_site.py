import os
import sys
import torch
import numpy as np
import time
import pandas as pd
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, ArgumentTypeError
import torch.multiprocessing as mp
from torch.multiprocessing import Queue
from sklearn.metrics import (accuracy_score, roc_auc_score, average_precision_score,
                             confusion_matrix, recall_score, precision_score, f1_score)

sys.path.append(os.getcwd())
from NanoFM.model.models_new import SiteLevelModel
from NanoFM.utils.constants import use_cuda, min_cov

base2code = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
SIGNAL_LEN = 65
queue_size_border_batch = 100


def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')


def adjust_features(features):
    kmer, qual, mis, ins, dele = features[:, :, 0:1, :], features[:, :, 4:5, :], features[:, :, 5:6, :], features[:, :,
                                                                                                         6:7,
                                                                                                         :], features[:,
                                                                                                             :, 7:8, :]
    sigs = features[:, :, 8:, :]
    return torch.cat([kmer, qual, mis, ins, dele, sigs], dim=2)


def _get_zero_feature(k_len):
    return np.zeros((8 + SIGNAL_LEN, k_len), dtype=np.float32)


def parse_site_line_standalone(line, target_min_cov=20):
    line = line.strip()
    if not line: return None
    parts = line.split("\t")
    label = 0
    last_part = parts[-1].strip()

    if last_part in ['0', '1']:
        label = int(last_part)
        reads_block = parts[3] if len(parts) >= 5 else parts[3]
    else:
        last_tokens = last_part.split(' ')
        if len(last_tokens) > 1 and last_tokens[-1] in ['0', '1']:
            label = int(last_tokens[-1])
            reads_block = last_part[:-2].strip()
        else:
            label = 0
            reads_block = last_part

    chrom, pos, kmer_str = parts[0], parts[1], parts[2]
    sample_info = f"{chrom}\t{pos}\t{kmer_str}"

    try:
        kmer_code = np.array([base2code[x] for x in kmer_str])
    except KeyError:
        return None

    current_kmer_len = len(kmer_code)
    raw_reads = reads_block.split("/") if reads_block else []

    if len(raw_reads) == 0:
        return sample_info, np.array([_get_zero_feature(current_kmer_len)] * target_min_cov, dtype=np.float32), label

    selected_indices = np.random.choice(len(raw_reads), target_min_cov, replace=(len(raw_reads) < target_min_cov))
    selected_reads = [raw_reads[i] for i in selected_indices]
    batch_features = []

    for read_str in selected_reads:
        tokens = read_str.strip().split(" ")
        if len(tokens) < 9:
            batch_features.append(_get_zero_feature(current_kmer_len))
            continue
        try:
            base_means = np.array([float(x) for x in tokens[0].split("|")])
            base_stds = np.array([float(x) for x in tokens[1].split("|")])
            base_median = np.array([float(x) for x in tokens[2].split("|")])
            qual = np.array([int(x) for x in tokens[4].split("|")])
            mis = np.array([int(x) for x in tokens[-3].split("|")])
            ins = np.array([int(x) for x in tokens[-2].split("|")])
            dele = np.array([int(x) for x in tokens[-1].split("|")])

            k_signals_raw = [[float(y) for y in blk.split("|")] for blk in tokens[5: 5 + current_kmer_len]]
            k_signals_processed = np.zeros((len(k_signals_raw), SIGNAL_LEN))
            for i, sig_arr in enumerate(k_signals_raw):
                sig_len = len(sig_arr)
                if sig_len == 0: continue
                if sig_len < SIGNAL_LEN:
                    pad0s = SIGNAL_LEN - sig_len
                    padl = pad0s // 2
                    k_signals_processed[i, padl: padl + sig_len] = sig_arr
                else:
                    start = (sig_len - SIGNAL_LEN) // 2
                    k_signals_processed[i, :] = sig_arr[start: start + SIGNAL_LEN]

            k_signals = k_signals_processed.transpose()
            batch_features.append(np.concatenate((
                kmer_code.reshape(-1, current_kmer_len), base_means.reshape(-1, current_kmer_len),
                base_median.reshape(-1, current_kmer_len), base_stds.reshape(-1, current_kmer_len),
                qual.reshape(-1, current_kmer_len), mis.reshape(-1, current_kmer_len),
                ins.reshape(-1, current_kmer_len), dele.reshape(-1, current_kmer_len), k_signals
            ), axis=0))
        except Exception:
            batch_features.append(_get_zero_feature(current_kmer_len))

    return sample_info, np.array(batch_features, dtype=np.float32), label


def load_embedding_dict(parquet_path):
    print(f"Loading embeddings from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    emb_dict = {}
    for _, row in df.iterrows():
        key = f"{row['chr']}_{row['pos']}"
        v5 = row['center_5mer_emb']
        if hasattr(v5, 'tolist'): v5 = v5.tolist()
        emb_dict[key] = np.array(v5, dtype=np.float32)
    return emb_dict


def get_embeddings_for_batch(sample_infos, emb_dict, device):
    e5 = []
    for info in sample_infos:
        parts = str(info).split('\t')
        key = f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else str(info)

        if key in emb_dict:
            e5.append(emb_dict[key])
        else:
            e5.append(np.zeros((5, 768), dtype=np.float32))

    e5_t = torch.tensor(np.array(e5)).float()
    if use_cuda: e5_t = e5_t.cuda(device)
    return e5_t


def read_feature_file(feature_file, features_batch_q, batch_size=64):
    sample_infos, features_list, labels_list = [], [], []
    with open(feature_file, "r") as f:
        for line in f:
            res = parse_site_line_standalone(line, target_min_cov=min_cov)
            if res is None: continue

            s_info, feat, lbl = res
            sample_infos.append(s_info)
            features_list.append(feat)
            labels_list.append(lbl)

            if len(features_list) == batch_size:
                features_batch_q.put((sample_infos, np.stack(features_list, axis=0), labels_list))
                while features_batch_q.qsize() > queue_size_border_batch: time.sleep(0.1)
                sample_infos, features_list, labels_list = [], [], []

        if len(features_list) > 0:
            features_batch_q.put((sample_infos, np.stack(features_list, axis=0), labels_list))

    features_batch_q.put("kill")


def predict(model_path, features_batch_q, pred_str_q, args, device=0):
    emb_dict = load_embedding_dict(args.embedding_file)
    model = SiteLevelModel(args.model_type, args.dropout_rate, args.hidden_size,
                           args.seq_lens, args.signal_lens, embedding_size=768,
                           use_cross_attention=args.use_cross_attention,
                           num_heads=args.num_heads, dropout_attn=args.dropout_attn)

    checkpoint = torch.load(model_path, map_location=f'cuda:{device}' if use_cuda else 'cpu', weights_only=True)
    try:
        model.load_state_dict(checkpoint['net'])
    except RuntimeError:
        from collections import OrderedDict
        new_state_dict = OrderedDict((k[7:] if k.startswith('module.') else k, v) for k, v in checkpoint['net'].items())
        model.load_state_dict(new_state_dict)
    except KeyError:
        model.load_state_dict(checkpoint)

    if use_cuda: model = model.cuda(device)
    model.eval()

    with torch.no_grad():
        while True:
            if features_batch_q.empty():
                time.sleep(0.1)
                continue

            batch_data = features_batch_q.get()
            if batch_data == "kill":
                features_batch_q.put("kill")
                break

            sample_infos, features_np, labels = batch_data
            features = adjust_features(
                torch.FloatTensor(features_np).cuda(device) if use_cuda else torch.FloatTensor(features_np))

            e5 = get_embeddings_for_batch(sample_infos, emb_dict, device)
            probs = model(features, e5).cpu().numpy().flatten()

            pred_strs = [f"{sample_infos[i]}\t{probs[i]:.6f}\t{1 if probs[i] >= 0.5 else 0}\t{int(labels[i])}" for i in
                         range(len(sample_infos))]
            pred_str_q.put(pred_strs)

            while pred_str_q.qsize() > queue_size_border_batch: time.sleep(0.1)


def _write_predstr_to_file(write_fp, predstr_q):
    with open(write_fp, 'w') as wf:
        wf.write("Chrom\tPos\tKmer\tProbability\tPredLabel\tTrueLabel\n")
        while True:
            if predstr_q.empty():
                time.sleep(0.1)
                continue

            pred_strs = predstr_q.get()
            if pred_strs == "kill": break
            for s in pred_strs: wf.write(s + "\n")
            wf.flush()


def _calc_and_print_metrics(y_true, y_pred_label, y_prob, region_name):
    if len(y_true) == 0: return None

    # Basic metrics
    acc = accuracy_score(y_true, y_pred_label)
    try:
        auc_val = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    except ValueError:
        auc_val = 0.0
    try:
        pr_auc = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    except:
        pr_auc = 0.0

    # Confusion-matrix-related metrics
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_label, labels=[0, 1]).ravel()
    except ValueError:
        tn, fp, fn, tp = 0, 0, 0, 0

    # [Added metrics]
    sen = recall_score(y_true, y_pred_label, zero_division=0)  # Sensitivity (Recall)
    prec = precision_score(y_true, y_pred_label, zero_division=0)  # Precision
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # Specificity
    f1 = f1_score(y_true, y_pred_label, zero_division=0)  # F1-score

    return {
        "Region": region_name,
        "Samples": len(y_true),
        "Accuracy": acc,
        "ROC_AUC": auc_val,
        "PR_AUC": pr_auc,
        "Sensitivity_Recall": sen,
        "Precision": prec,
        "Specificity": spec,
        "F1_score": f1,
        "TP": tp, "FP": fp, "TN": tn, "FN": fn
    }


def calculate_metrics(output_file):
    if not os.path.exists(output_file): return
    try:
        df = pd.read_csv(output_file, sep='\t')
        if len(df) == 0: return
        metrics_list = []

        # Compute overall metrics
        global_metrics = _calc_and_print_metrics(df['TrueLabel'].values, df['PredLabel'].values,
                                                 df['Probability'].values, "Overall")
        if global_metrics: metrics_list.append(global_metrics)

        # Save the results as CSV
        if metrics_list:
            output_csv = output_file.replace('.tsv', '_metrics.csv') if output_file.endswith(
                '.tsv') else output_file + "_metrics.csv"
            pd.DataFrame(metrics_list).to_csv(output_csv, index=False)
            print(f"Metrics saved to {output_csv}")
    except Exception as e:
        print(f"Error calculating metrics: {e}")


def _get_gpus():
    num_gpus = torch.cuda.device_count()
    return list(range(num_gpus)) * 1000 if num_gpus > 0 else [0] * 1000


def argparser():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", required=True)
    #model_states_3.pt checkpoint.pt
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--embedding_file", required=True)

    parser.add_argument("--use_cross_attention", type=str2bool, default=True)
    parser.add_argument("--num_heads", default=4, type=int)
    parser.add_argument("--dropout_attn", default=0.1, type=float)

    parser.add_argument("--seq_lens", default=5, type=int)
    parser.add_argument("--signal_lens", default=65, type=int)
    parser.add_argument("--embedding_size", default=768, type=int)
    parser.add_argument("--hidden_size", default=128, type=int)
    parser.add_argument("--dropout_rate", default=0.5, type=float)
    parser.add_argument("--model_type", default='comb', type=str, choices=["basecall", "raw_signals", "comb"])

    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--nproc", default=4, type=int)
    parser.add_argument("--seed", default=48, type=int)
    return parser


def main():
    args = argparser().parse_args()
    if not os.path.exists(args.model) or not os.path.exists(args.input_file):
        return

    # Automatically create the directory corresponding to output_file
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    features_batch_q, pred_str_q = Queue(), Queue()

    p_read = mp.Process(target=read_feature_file, args=(args.input_file, features_batch_q, args.batch_size))
    p_read.daemon = True
    p_read.start()

    predict_procs = []
    for i in range(max(1, args.nproc - 2)):
        p = mp.Process(target=predict, args=(args.model, features_batch_q, pred_str_q, args, _get_gpus()[i]))
        p.daemon = True
        p.start()
        predict_procs.append(p)

    p_write = mp.Process(target=_write_predstr_to_file, args=(args.output_file, pred_str_q))
    p_write.daemon = True
    p_write.start()

    for p in predict_procs:
        p.join()
    pred_str_q.put("kill")
    p_read.join()
    p_write.join()

    calculate_metrics(args.output_file)

if __name__ == '__main__':
    main()