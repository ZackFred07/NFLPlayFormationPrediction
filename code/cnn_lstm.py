import torch
import torch.nn as nn

class CNNLSTMClassifier(nn.Module):
    def __init__(self, input_channels=27, lstm_hidden=128, num_outputs=202, output_seq=False):
        super().__init__()
        self.output_seq = output_seq

        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # output: [32, 1, 1]
        )

        self.lstm = nn.LSTM(input_size=32, hidden_size=lstm_hidden, batch_first=False)

        self.static_fc = nn.Linear(73, 32)

        self.head = nn.Sequential(
            nn.Linear(lstm_hidden + 32, 128),
            nn.ReLU(),
            nn.Linear(128, num_outputs),
        )

    # Forward function (only takes non-batched inputs)
    def forward(self, x_dynamic, x_static):
        dtype = next(self.parameters()).dtype
        if x_dynamic.ndim == 5:
            x_dynamic = x_dynamic.squeeze(0)  # [T, C, H, W]
        if x_static.ndim == 2:
            x_static = x_static.squeeze(0)    # [F]

        T, C, H, W = x_dynamic.shape

        cnn_feats = []
        for t in range(T):
            ft = self.cnn(x_dynamic[t].unsqueeze(0)).squeeze()  # [32]
            cnn_feats.append(ft)

        cnn_feats = torch.stack(cnn_feats)  # [T, 32]

        lstm_out, _ = self.lstm(cnn_feats.unsqueeze(1))  # [T, 1, hidden_states]
        lstm_out = lstm_out.squeeze(1)  # [T, hidden_states]

        x_static_proj = self.static_fc(x_static)  # [32]

        outputs = []
        for t in range(T):
            fused = torch.cat([lstm_out[t], x_static_proj], dim=-1)
            logits = self.head(fused)  # [num_labels]
            outputs.append(logits)

        return outputs  # List[T] of [num_labels]
