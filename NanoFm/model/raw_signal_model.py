import torch
from torch import nn
import torch.nn.functional as F

from .resnet_new1 import Residual



class BahdanauAttention(nn.Module):
    """
    Attention aggregation after BiLSTM, following the TandemMod-style design.
    """

    def __init__(self, in_features, hidden_units, num_task=1):
        super(BahdanauAttention, self).__init__()
        self.W1 = nn.Linear(in_features=in_features, out_features=hidden_units)
        self.W2 = nn.Linear(in_features=in_features, out_features=hidden_units)
        self.V = nn.Linear(in_features=hidden_units, out_features=num_task)

    def forward(self, hidden_states, values):
        """
        hidden_states: (Batch, Hidden_Dim) - the final hidden state of the LSTM
        values: (Batch, Seq_Len, Hidden_Dim) - outputs from all LSTM time steps
        """
        hidden_with_time_axis = torch.unsqueeze(hidden_states, dim=1)

        score = self.V(torch.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))

        attention_weights = nn.Softmax(dim=1)(score)

        values = torch.transpose(values, 1, 2)
        context_vector = torch.matmul(values, attention_weights)  # (B, H, 1)
        context_vector = torch.transpose(context_vector, 1, 2)  # (B, 1, H)

        return context_vector, attention_weights

class RawSignal_Hybrid_Model(nn.Module):
    def __init__(self, in_channels=5, out_channels=256):
        super(RawSignal_Hybrid_Model, self).__init__()

        # --- A. Shallow feature extraction (Shallow ResNet/CNN) ---
        # Purpose: extract local features from raw signals and reduce dimensionality appropriately
        self.cnn_extractor = nn.Sequential(

            nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),

            # Layer 1: residual block (32 -> 64)
            Residual(32, 64, use_1x1conv=True, stride=1),

            # Layer 2: residual block (64 -> 128)
            Residual(64, 128, use_1x1conv=True, stride=1),

        )


        self.lstm = nn.LSTM(input_size=128, hidden_size=128,
                            batch_first=True, bidirectional=True)


        self.attention = BahdanauAttention(in_features=256, hidden_units=10, num_task=1)


        self.fc = nn.Sequential(
            nn.Linear(256, out_channels),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        # x shape: (Batch, Channels, Length)

        # 1. CNN feature extraction
        x = self.cnn_extractor(x)  # Out: (Batch, 128, Reduced_Length)

        # 2. Dimension adjustment (prepare for LSTM)
        # Permute: (N, C, L) -> (N, L, C)
        x = x.permute(0, 2, 1)  # Out: (Batch, Reduced_Length, 128)

        # 3. BiLSTM
        # self.lstm.flatten_parameters() # Optional: speed up execution
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: (Batch, L, 256)

        # 4. Attention aggregation
        # h_n shape: (num_layers*num_directions, batch, hidden_size)
        # We concatenate the last two hidden states (bidirectional) as the query
        h_n_view = h_n.view(h_n.shape[1], -1)  # (Batch, 256) (assuming num_layers=1)
        if h_n_view.shape[1] != 256:
            h_n_view = lstm_out[:, -1, :]

        context_vector, _ = self.attention(h_n_view, lstm_out)  # Out: (Batch, 1, 256)

        # 5. FC projection
        out = context_vector.squeeze(1)  # (Batch, 256)
        out = self.fc(out)

        return out
