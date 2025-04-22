from collections import defaultdict
import json
import os
import numpy as np
import os
import gc
import time
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split, Subset
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score, auc
from tqdm import tqdm
from st_dataset import TwoDimensionalTensorDataset, TestTwoDimensionalTensorDataset
from cnn_lstm import CNNLSTMClassifier


def compute_group_ttc(ttc_scores, label_names, group_map):
    ttc_by_group = defaultdict(list)
    for class_idx, ttc in ttc_scores:
        label = label_names[class_idx]
        for group, label_list in group_map.items():
            if label in label_list:
                ttc_by_group[group].append(ttc)
    return {g: round(sum(v) / len(v), 4) if v else 1.0 for g, v in ttc_by_group.items()}


# Train through different percentiles of the timeline
def train_eval(
    model, dataset, epochs=10, lr=1e-3, percentile_schedule=[10, 20, 40, 50, 75]
):
    print("Start training")
    # Train test split using torch
    dataset_size = len(dataset)
    split_idx = int(0.8 * dataset_size)
    train_dataset, test_dataset = random_split(
        dataset, [split_idx, dataset_size - split_idx]
    )

    # Send model to gpu
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

        # Percentile schedule to as part of curiculum
        percentile = percentile_schedule[min(epoch, len(percentile_schedule) - 1)]

        # train_dataset.dataset.truncate_by_percentile(percentile)
        # test_dataset.dataset.truncate_by_percentile(percentile)

        train_dataset.dataset.truncate_by_percentile(0.01)
        test_dataset.dataset.truncate_by_percentile(0.01)

        train_loader = DataLoader(train_dataset, batch_size=None, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=None)

        for x, x_static, y in tqdm(train_loader, desc=f"Training"):
            x, x_static, y = (
                x.to(device),
                x_static.to(device),
                y.to(device),
            )

            # Make predictions
            outputs = model(x, x_static)  # List[T] of [num_labels]
            pred = torch.stack(outputs, dim=0)[-1]  # shape: [num_labels]

            # Backpropagate + Optimizer adjustments
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Train stats
            acc = (pred.sigmoid() > 0.5).eq(y).float().mean()
            total_loss += loss.item()
            total_acc += acc.item()

        end_time = time.time()
        print(
            f"Epoch {epoch + 1}: Loss = {total_loss:.4f}, Accuracy = {total_acc / len(train_loader):.4f}, GPU = {torch.cuda.max_memory_allocated() / 1e6:.2f}MB, Time = {end_time - start_time:.2f}s"
        )
        gc.collect()

        if epoch % 1 == 0:

            model.eval()

            y_true, y_pred = [], []
            ttc_scores = []
            acc_time_curve_by_class = defaultdict(
                list
            )  # {class_idx: [(t/T, acc), ...]}

            with torch.no_grad():
                # For each data point
                for x, x_static, y in tqdm(test_loader, "Testing"):

                    x, x_static, y = x.to(device), x_static.to(device), y.to(device)

                    # run inference
                    outputs = model(x, x_static)

                    # Get timesteps
                    T = len(outputs)

                    # Get sequnce of  predictions
                    preds = [
                        (torch.sigmoid(o) > 0.5) for o in outputs
                    ]  # List[T] of [num_labels] boolean tensors

                    # Get latest prediction
                    final_pred = preds[-1]

                    # Add true and predicted
                    y_true.append(y.cpu())
                    y_pred.append(final_pred.cpu())

                    # Time to correct (TTC)
                    for t in range(T):
                        # If the entire thing got it all right set the frame score
                        if preds[t].eq(y.bool()).all():
                            ttc_scores.append((None, t / T))
                            break
                    else:
                        # Else it got none of it right
                        ttc_scores.append((None, 1.0))

                    # TTC per class
                    for class_idx in range(y.size(0)):
                        for t in range(T):
                            if preds[t][class_idx] == y[class_idx]:
                                ttc_scores.append((class_idx, t / T))
                                break

                    for t in range(T):
                        pred_t = preds[t]  # shape: [num_labels]
                        correct_t = pred_t.eq(y.bool())  # shape: [num_labels]

                        for class_idx, correct in enumerate(correct_t):
                            acc_time_curve_by_class[class_idx].append(
                                (t / T, float(correct))
                            )

                # Metric computation
                y_true = torch.stack(y_true).to(torch.float32).numpy()
                y_pred = torch.stack(y_pred).to(torch.float32).numpy()

                precision_micro = precision_score(y_true, y_pred, average="micro")
                recall_micro = recall_score(y_true, y_pred, average="micro")
                f1_micro = f1_score(y_true, y_pred, average="micro")

                auatc_by_class = {}
                for class_idx, curve in acc_time_curve_by_class.items():
                    
                    # Group by timestep
                    acc_dict = defaultdict(list)
                    for t, acc in curve:
                        acc_dict[t].append(acc)

                    # Average at each t, then compute AUC
                    x_vals, y_vals = zip(
                        *sorted((t, np.mean(accs)) for t, accs in acc_dict.items())
                    )

                    if len(set(x_vals)) > 1:  # Need at least two distinct x-values
                        auatc_by_class[class_idx] = auc(x_vals, y_vals)
                    else:
                        auatc_by_class[class_idx] = 0.0

                print("\n--- Overall Per-Label (Micro) ---")
                print(
                    f"Precision: {precision_micro:.4f}, Recall: {recall_micro:.4f}, F1: {f1_micro:.4f}"
                )

                # Load the json with label information
                with open("label_metadata.json") as f:
                    label_metadata = json.load(f)

                label_names = list(label_metadata.keys())

                # Create mappings from the metadata json file
                onehot_class_map = defaultdict(list)
                for label, info in label_metadata.items():
                    if info.get("type") == "onehot" and "onehot_class" in info:
                        onehot_class_map[info["onehot_class"]].append(label)

                # Calculation of onehot metrics
                # 🧠 Get raw predictions before thresholding
                y_pred_raw = torch.stack([
                    torch.cat([out.cpu().unsqueeze(0) for out in model(x.to(device), x_static.to(device))], dim=0)[-1]
                    for x, x_static, _ in test_loader
                ]).to(torch.float32).numpy()

                # 📦 Softmax-Based Onehot Metrics
                onehot_metrics = {}
                for group, label_list in onehot_class_map.items():
                    indices = [label_names.index(lbl) for lbl in label_list if lbl in label_names]
                    if not indices:
                        continue

                    y_true_group = y_true[:, indices]
                    y_pred_group_raw = y_pred_raw[:, indices]

                    # 🧠 Softmax across one-hot class group
                    softmaxed = torch.softmax(torch.tensor(y_pred_group_raw), dim=1).numpy()

                    # 🎯 Convert to 1-hot prediction (argmax)
                    y_pred_group = np.zeros_like(softmaxed)
                    y_pred_group[np.arange(len(softmaxed)), np.argmax(softmaxed, axis=1)] = 1

                    onehot_metrics[group] = {
                        "precision": precision_score(y_true_group, y_pred_group, average="micro", zero_division=0),
                        "recall": recall_score(y_true_group, y_pred_group, average="micro", zero_division=0),
                        "f1": f1_score(y_true_group, y_pred_group, average="micro", zero_division=0),
                    }

                # Save CSVs

if __name__ == "__main__":
    model = CNNLSTMClassifier().to(torch.bfloat16)
    dataset = TwoDimensionalTensorDataset()
    dataset = TestTwoDimensionalTensorDataset(dataset) # This is for debugging
    train_eval(model=model, dataset=dataset)
