import torch
import torch.nn as nn

class CNNLSTMClassifier(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=128, num_layers=1, num_outputs=3, output_dims=(8, 10, 6)):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(32, hidden_dim, num_layers, batch_first=True)
        self.fc_route = nn.Linear(hidden_dim, output_dims[0])
        self.fc_play = nn.Linear(hidden_dim, output_dims[1])
        self.fc_form = nn.Linear(hidden_dim, output_dims[2])

    def forward(self, x):  # x: (B, P, T, F)
        B, P, T, F = x.shape
        x = x.view(B * P, T, F).permute(0, 2, 1)  # (B*P, F, T)
        x = self.relu(self.conv1(x))  # (B*P, 32, T)
        x = x.permute(0, 2, 1)  # (B*P, T, 32)
        _, (h_n, _) = self.lstm(x)  # (1, B*P, hidden)
        h_n = h_n[-1].view(B, P, -1).mean(dim=1)  # (B, hidden)
        return torch.sigmoid(self.fc_route(h_n)), torch.sigmoid(self.fc_play(h_n)), torch.sigmoid(self.fc_form(h_n))