import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import os
import time
import gc
from models.cnn_lstm import CNNLSTMClassifier2D
from data.heatmap_tensor_dataset import HeatmapTensorDataset


def compute_accuracy(preds, labels):
    preds_bin = (preds > 0.5).float()
    correct = (preds_bin == labels).float()
    return correct.sum() / correct.numel()


def memory_usage():
    torch.cuda.synchronize()
    gpu = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0
    return gpu


def train(model, dataset, epochs=10, batch_size=8, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_acc = 0
        start_time = time.time()
        torch.cuda.reset_peak_memory_stats()

        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            pred = torch.cat(outputs, dim=1)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc = compute_accuracy(pred, y)
            total_loss += loss.item()
            total_acc += acc.item()

        end_time = time.time()
        gpu = memory_usage()
        print(f"Epoch {epoch + 1}: Loss = {total_loss:.4f}, Accuracy = {total_acc / len(loader):.4f}, GPU = {gpu:.2f}MB, Time = {end_time - start_time:.2f}s")
        gc.collect()


def evaluate(model, dataset, batch_size=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size)
    model.eval()
    total_acc = 0
    start_time = time.time()
    torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            pred = torch.cat(outputs, dim=1)
            acc = compute_accuracy(pred, y)
            total_acc += acc.item()

    end_time = time.time()
    gpu = memory_usage()
    print(f"Eval Accuracy = {total_acc / len(loader):.4f}, GPU = {gpu:.2f}MB, Time = {end_time - start_time:.2f}s")