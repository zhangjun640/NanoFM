import torch
from torch import nn
from torch.nn import MultiheadAttention
from NanoFM.utils.constants import min_cov

from .BiLstm import BiLSTM_Basecaller
from .util import Full_NN, FlattenLayer, one_hot_embedding
from .raw_signal_model import RawSignal_Hybrid_Model


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim=512, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.attn_y_to_x = MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn_y = nn.Sequential(nn.Linear(embed_dim, ffn_dim), nn.ReLU(), nn.Dropout(dropout),
                                   nn.Linear(ffn_dim, embed_dim))
        self.norm_y1 = nn.LayerNorm(embed_dim)
        self.norm_y2 = nn.LayerNorm(embed_dim)
        self.dropout_y1 = nn.Dropout(dropout)
        self.dropout_y2 = nn.Dropout(dropout)

        self.attn_x_to_y = MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn_x = nn.Sequential(nn.Linear(embed_dim, ffn_dim), nn.ReLU(), nn.Dropout(dropout),
                                   nn.Linear(ffn_dim, embed_dim))
        self.norm_x1 = nn.LayerNorm(embed_dim)
        self.norm_x2 = nn.LayerNorm(embed_dim)
        self.dropout_x1 = nn.Dropout(dropout)
        self.dropout_x2 = nn.Dropout(dropout)

    def forward(self, feat_y, feat_x):
        y_q = feat_y.unsqueeze(1)
        x_kv = feat_x.unsqueeze(1)
        attn_output_y, _ = self.attn_y_to_x(y_q, x_kv, x_kv)
        y = feat_y + self.dropout_y1(attn_output_y.squeeze(1))
        y = self.norm_y1(y)
        ffn_output_y = self.ffn_y(y)
        y = self.norm_y2(y + self.dropout_y2(ffn_output_y))

        x_q = feat_x.unsqueeze(1)
        y_kv = feat_y.unsqueeze(1)
        attn_output_x, _ = self.attn_x_to_y(x_q, y_kv, y_kv)
        x = feat_x + self.dropout_x1(attn_output_x.squeeze(1))
        x = self.norm_x1(x)
        ffn_output_x = self.ffn_x(x)
        x = self.norm_x2(x + self.dropout_x2(ffn_output_x))

        return y, x


class ReadLevelModel(nn.Module):
    def __init__(self, model_type, dropout_rate=0.5, hidden_size=128,
                 seq_len=5, signal_lens=65, embedding_size=768, device=0,
                 use_cross_attention=True, num_heads=4, dropout_attn=0.1):
        super(ReadLevelModel, self).__init__()
        self.seq_len = seq_len
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.model_type = model_type
        self.use_cross_attention = use_cross_attention
        self.raw, self.basecall = False, False

        if self.model_type in ["raw_signals", "comb"]: self.raw = True
        if self.model_type in ["basecall", "comb"]: self.basecall = True

        self.proj_dim = 16

        if self.basecall:

            self.llm_projection = nn.Sequential(
                nn.LayerNorm(embedding_size),
                nn.Linear(embedding_size, self.proj_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            )
            # 32 + 4 = 36 dimensions
            self.basecall_model = BiLSTM_Basecaller(input_size=self.proj_dim + 4, hidden_size=128, output_size=256)

        if self.raw:
            self.raw_model = RawSignal_Hybrid_Model(in_channels=5, out_channels=256)

        if self.raw and self.basecall:
            if self.use_cross_attention:
                self.cross_attention = CrossAttention(embed_dim=256, num_heads=num_heads, dropout=dropout_attn)
                self.full = Full_NN(input_size=256 + 256, hidden_size=hidden_size, num_classes=1,
                                    dropout_rate=dropout_rate)
            else:
                self.full = Full_NN(input_size=256 + 256, hidden_size=hidden_size, num_classes=1,
                                    dropout_rate=dropout_rate)
        else:
            self.full = Full_NN(input_size=256, hidden_size=hidden_size, num_classes=1, dropout_rate=dropout_rate)

    # [Core modification]: removed the site_seq_emb parameter
    def forward(self, features, site_5mer_emb):
        kmer = features[:, 0, :]
        qual = features[:, 1, :]
        mis = features[:, 2, :]
        ins = features[:, 3, :]
        dele = features[:, 4, :]
        signals = torch.transpose(features[:, 5:, :], 1, 2)

        y = None
        if self.basecall:
            y_kmer_embed = self.llm_projection(site_5mer_emb)
            qual = torch.reshape(qual, (-1, self.seq_len, 1)).float()
            mis = torch.reshape(mis, (-1, self.seq_len, 1)).float()
            ins = torch.reshape(ins, (-1, self.seq_len, 1)).float()
            dele = torch.reshape(dele, (-1, self.seq_len, 1)).float()
            y = torch.cat((y_kmer_embed, qual, mis, ins, dele), 2)
            y = self.basecall_model(y)

        x = None
        if self.raw:
            signals = signals.float()
            signals_len = signals.shape[2]

            # [Core modification]: reverted to the pure one_hot_embedding implementation
            kmer_embed = one_hot_embedding(kmer.long(), signals_len)  # (N, seq_len*signal_len, 4)
            signals_ex = signals.reshape(signals.shape[0], -1, 1)  # (N, seq_len*signal_len, 1)

            x_cat = torch.cat((kmer_embed, signals_ex), -1)  # (N, 325, 5)
            x_transposed = torch.transpose(x_cat, 1, 2)  # (N, 5, 325)
            x = self.raw_model(x_transposed)

        if self.raw and self.basecall:
            if self.use_cross_attention:
                y_attended, x_attended = self.cross_attention(feat_y=y, feat_x=x)
                z = torch.cat((y_attended, x_attended), 1)
            else:
                z = torch.cat((x, y), 1)
        elif self.raw:
            z = x
        else:
            z = y

        z_classified = self.full(z)
        out_ = self.sigmoid(z_classified)
        return out_, z


class SiteLevelModel(nn.Module):
    def __init__(self, model_type, dropout_rate=0.5, hidden_size=128,
                 seq_len=5, signal_lens=65, embedding_size=768, device=0,
                 use_cross_attention=True, num_heads=4, dropout_attn=0.1):
        super(SiteLevelModel, self).__init__()
        self.read_level_model = ReadLevelModel(model_type, dropout_rate, hidden_size, seq_len, signal_lens,
                                               embedding_size, device=device,
                                               use_cross_attention=use_cross_attention,
                                               num_heads=num_heads, dropout_attn=dropout_attn)

    # [Core modification]: removed struct_seq_emb
    def forward(self, features, struct_5mer_emb):
        batch_size = features.shape[0]
        min_cov = features.shape[1]

        features = features.view(-1, features.shape[2], features.shape[3])
        struct_5mer_emb_repeated = struct_5mer_emb.unsqueeze(1).expand(-1, min_cov, -1, -1).reshape(-1, 5, 768)

        # Sequence embedding is no longer passed
        probs, _ = self.read_level_model(features, struct_5mer_emb_repeated)
        probs = probs.view(-1, min_cov)
        return 1 - torch.prod(1 - probs, axis=1)
