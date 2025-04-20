import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        support = self.linear(x)
        out = torch.matmul(adj, support)
        return F.relu(out)

class GCNLSTMClassifier(nn.Module):
    def __init__(self, input_dim=6, gcn_dim=32, lstm_dim=128, output_dims=(8, 10, 6)):
        super().__init__()
        self.gcn = GCNLayer(input_dim, gcn_dim)
        self.lstm = nn.LSTM(gcn_dim, lstm_dim, batch_first=True)
        self.fc_route = nn.Linear(lstm_dim, output_dims[0])
        self.fc_play = nn.Linear(lstm_dim, output_dims[1])
        self.fc_form = nn.Linear(lstm_dim, output_dims[2])

    def forward(self, x, adj):  # x: (B, T, P, F), adj: (P, P)
        B, T, P, F = x.shape
        out = []
        for t in range(T):
            xt = x[:, t]  # (B, P, F)
            xt = self.gcn(xt, adj)  # (B, P, gcn_dim)
            out.append(xt)
        out = torch.stack(out, dim=1)  # (B, T, P, gcn_dim)
        out = out.mean(dim=2)  # (B, T, gcn_dim)
        _, (h_n, _) = self.lstm(out)
        h_n = h_n[-1]  # (B, lstm_dim)
        return torch.sigmoid(self.fc_route(h_n)), torch.sigmoid(self.fc_play(h_n)), torch.sigmoid(self.fc_form(h_n))