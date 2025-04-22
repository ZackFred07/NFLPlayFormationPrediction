import torch
import os
import numpy as np
import time
import gc
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score, auc

class TwoDimensionalTensorDataset(torch.utils.data.Dataset):    
    def __init__(self, tensor_path_or_dir):
        self.max_timesteps = None  # use full length by default
        self.lazy_mode = os.path.isdir(tensor_path_or_dir)

        if self.lazy_mode:
            self.file_paths = [
                os.path.join(tensor_path_or_dir, fname)
                for fname in sorted(os.listdir(tensor_path_or_dir))
                if fname.endswith(".pt")
            ]
            self.index_map = []
            self.seq_lens = []
            for file_idx, path in enumerate(self.file_paths):
                h, _, _ = torch.load(path, map_location="cpu")
                for i in range(h.size(0)):
                    self.index_map.append((file_idx, i))
                    self.seq_lens.append(h[i].shape[0])
        else:
            self.heatmaps, self.global_feats, self.labels = torch.load(tensor_path_or_dir)
            self.seq_lens = [hm.shape[0] for hm in self.heatmaps]

    def __len__(self):
        if self.lazy_mode:
            return len(self.index_map)
        return self.heatmaps.size(0)

    def __getitem__(self, idx):
        if self.lazy_mode:
            file_idx, local_idx = self.index_map[idx]
            h, g, l = torch.load(self.file_paths[file_idx], map_location="cpu")
            heatmap_seq = h[local_idx]
            global_feat = g[local_idx]
            label = l[local_idx]
        else:
            heatmap_seq = self.heatmaps[idx]
            global_feat = self.global_feats[idx]
            label = self.labels[idx]

        if self.max_timesteps is not None:
            heatmap_seq = heatmap_seq[:self.max_timesteps]

        return heatmap_seq, global_feat, label

    def truncate_timesteps(self, max_timesteps):
        self.max_timesteps = max_timesteps
        return self

    def truncate_by_percentile(self, percentile):
        if not self.seq_lens:
            raise RuntimeError("Sequence lengths not available.")
        self.max_timesteps = int(np.percentile(self.seq_lens, percentile))
        return self

def evaluate(model, dataset, batch_size=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size)
    model.eval()
    total_acc = 0
    y_true, y_pred = [], []
    ttc_scores = []
    acc_time_curve = []

    with torch.no_grad():
        for x, _, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            T = len(outputs)
            batch_preds = [torch.sigmoid(p) > 0.5 for p in outputs]
            final_preds = batch_preds[-1]
            y_true.append(y.cpu())
            y_pred.append(final_preds.cpu())
            acc = final_preds.eq(y.bool()).float().mean()
            total_acc += acc.item()

            for i in range(y.size(0)):
                for t in range(T):
                    if batch_preds[t][i].eq(y[i].bool()).all():
                        ttc_scores.append(t / T)
                        break
                else:
                    ttc_scores.append(1.0)

            for t in range(T):
                acc_t = batch_preds[t].eq(y.bool()).float().mean().item()
                acc_time_curve.append((t / T, acc_t))

    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()
    precision = precision_score(y_true, y_pred, average='micro')
    recall = recall_score(y_true, y_pred, average='micro')
    f1 = f1_score(y_true, y_pred, average='micro')
    auatc = auc(*zip(*acc_time_curve)) if acc_time_curve else 0.0
    avg_ttc = sum(ttc_scores) / len(ttc_scores) if ttc_scores else 1.0

    print(f"Eval Accuracy = {total_acc / len(loader):.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}, F1 = {f1:.4f}, Avg TTC = {avg_ttc:.4f}, AUATC = {auatc:.4f}")

def curriculum_train(model, dataset, epochs=10, batch_size=8, lr=1e-3, percentile_schedule=[10, 25, 50, 70, 90, 100]):
    dataset_size = len(dataset)
    split_idx = int(0.8 * dataset_size)
    train_dataset, test_dataset = random_split(dataset, [split_idx, dataset_size - split_idx])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_acc = 0
        start_time = time.time()
        torch.cuda.reset_peak_memory_stats()

        percentile = percentile_schedule[min(epoch, len(percentile_schedule) - 1)]
        train_dataset.dataset.truncate_by_percentile(percentile)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        for x, _, y in train_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            pred = torch.cat(outputs, dim=1)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc = (pred.sigmoid() > 0.5).eq(y).float().mean()
            total_loss += loss.item()
            total_acc += acc.item()

        end_time = time.time()
        print(f"Epoch {epoch + 1}: Loss = {total_loss:.4f}, Accuracy = {total_acc / len(train_loader):.4f}, GPU = {torch.cuda.max_memory_allocated() / 1e6:.2f}MB, Time = {end_time - start_time:.2f}s")
        gc.collect()

        print("Evaluating on validation set after epoch...")
        test_dataset.dataset.truncate_by_percentile(percentile)
        evaluate(model, test_dataset, batch_size=batch_size)
