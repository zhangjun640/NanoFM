import torch
from torch import nn



# Creating Fully Connected Network
class Full_NN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout_rate=0.5):
        super(Full_NN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dropout(x)
        x = self.dropout(self.fc1(x))
        x = self.relu(x)
        x = self.fc2(x)
        return x


class FlattenLayer(torch.nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):  # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)


def one_hot_embedding(kmer, signal_lens):
    # print(kmer.shape)  # torch.Size([N, 5])
    # print(signal_lens)
    # print("kmer", kmer[0])

    expand_kmer = torch.repeat_interleave(kmer, signal_lens, dim=1)
    # print("DD", expand_kmer.shape)  # torch.Size([N, 150])

    # expand_kmer = expand_kmer.view(kmer.shape[0], -1, signal_lens)
    # print("DD", expand_kmer.shape)  # torch.Size([N, 5, 30])

    embed = nn.functional.one_hot(expand_kmer, 4)
    # print(embed.shape)  # torch.Size([N, 5, 30, 4])

    return embed



def augment_features(features, noise_std=0.02, scale_range=0.1, p=0.5):
    """
    Perform data augmentation on the feature tensor (add noise and scaling).
    Note: this function assumes that 'signals' starts from index 8: in the features tensor.
    """
    # Apply augmentation with probability p
    if torch.rand(1).item() > p:
        return features

    features_aug = features.clone()

    # Extract the signal part (according to your function, starting from index 8)
    # (N, 70, 5) -> (N, 62, 5)
    signals = features_aug[:, 8:, :]

    # 1. Add Gaussian noise
    noise = torch.randn_like(signals) * noise_std
    signals += noise

    # 2. Random scaling
    # (N, 1, 1) tensor, one scaling factor for each sample
    scaler = torch.empty(signals.shape[0], 1, 1, device=signals.device).uniform_(1 - scale_range, 1 + scale_range)
    signals *= scaler

    # Put the augmented signals back into the feature tensor
    features_aug[:, 8:, :] = signals

    return features_aug
