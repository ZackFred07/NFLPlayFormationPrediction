from collections import defaultdict
import json
import os
import numpy as np
import os
import gc
import time
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score, auc

def compute_group_ttc(ttc_scores, label_names, group_map):
    ttc_by_group = defaultdict(list)
    for class_idx, ttc in ttc_scores:
        label = label_names[class_idx]
        for group, label_list in group_map.items():
            if label in label_list:
                ttc_by_group[group].append(ttc)
    return {g: round(sum(v) / len(v), 4) if v else 1.0 for g, v in ttc_by_group.items()}

# Function that will run evaluation on the model
def evaluate(
    model, dataset, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    model.to(device)
    loader = DataLoader(dataset, batch_size=None)
    model.eval()

    y_true, y_pred = [], []
    ttc_scores = []
    acc_time_curve = []

    with torch.no_grad():
        # For each data point
        for x, x_static, y in loader:
            # TODO: Check dim for both x's may need to unsqueeze
            x, x_static, y = x.to(device), x_static.to(device), y.to(device)

            # run inference
            outputs = model(x, x_static)

            # Get timesteps
            T = len(outputs)

            # Get sequnce of  predictions
            preds_seq = [torch.sigmoid(o[0]) > 0.5 for o in outputs]

            # Get latest prediction
            final_pred = preds_seq[-1]

            # Add true and predicted
            y_true.append(y.cpu())
            y_pred.append(final_pred.cpu())

            # Time to correct (TTC)
            for t in range(T):
                # If the entire thing got it all right set the frame score
                if preds_seq[t].eq(y.bool()).all():
                    ttc_scores.append((None, t / T))
                    break
            else:
                # Else it got none of it right
                ttc_scores.append((None, 1.0))

            # TTC per class
            for class_idx in range(y.size(0)):
                for t in range(T):
                    if preds_seq[t][class_idx] == y[class_idx]:
                        ttc_scores.append((class_idx, t / T))
                        break
                    
            # AUATC
            for t in range(T):
                acc = preds_seq[t].eq(y.bool()).float().mean().item()
                acc_time_curve.append((t / T, acc))

    # Metric computation
    y_true = torch.stack(y_true).numpy()
    y_pred = torch.stack(y_pred).numpy()

    precision_micro = precision_score(y_true, y_pred, average='micro')
    recall_micro = recall_score(y_true, y_pred, average='micro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    
    auatc = auc(*zip(*acc_time_curve)) if acc_time_curve else 0.0
    avg_ttc = sum([ttc for idx, ttc in ttc_scores if idx is None]) / max(1, sum(1 for idx, _ in ttc_scores if idx is None))

    print("\n--- Overall Per-Label (Micro) ---")
    print(f"Precision: {precision_micro:.4f}, Recall: {recall_micro:.4f}, F1: {f1_micro:.4f}")
    print(f"Avg TTC: {avg_ttc:.4f}, AUATC: {auatc:.4f}")
    
    # Load the json with label information
    with open("label_metadata.json") as f:
        label_metadata = json.load(f)
    
    label_names = list(label_metadata.keys())
    
    # Create maps
    onehot_class_map = defaultdict(list)
    super_cat_map = defaultdict(list)
    for label, info in label_metadata.items():
        if info.get("type") == "onehot" and "onehot_class" in info:
            onehot_class_map[info["onehot_class"]].append(label)
        if "superCategory" in info:
            super_cat_map[info["superCategory"]].append(label)
    
    # Calculation of onehot metrics
    onehot_metrics = {}
    for group, label_list in onehot_class_map.items():
        indices = [label_names.index(lbl) for lbl in label_list if lbl in label_names]
        if not indices:
            continue
        y_true_group = y_true[:, indices]
        y_pred_group = y_pred[:, indices]
        onehot_metrics[group] = {
            "precision": precision_score(y_true_group, y_pred_group, average='micro', zero_division=0),
            "recall": recall_score(y_true_group, y_pred_group, average='micro', zero_division=0),
            "f1": f1_score(y_true_group, y_pred_group, average='micro', zero_division=0),
        }

    
    # Calculation of super metrics
    super_metrics = {}
    for group, label_list in super_cat_map.items():
        indices = [label_names.index(lbl) for lbl in label_list if lbl in label_names]
        if not indices:
            continue
        y_true_group = y_true[:, indices]
        y_pred_group = y_pred[:, indices]
        super_metrics[group] = {
            "precision": precision_score(y_true_group, y_pred_group, average='micro', zero_division=0),
            "recall": recall_score(y_true_group, y_pred_group, average='micro', zero_division=0),
            "f1": f1_score(y_true_group, y_pred_group, average='micro', zero_division=0),
        }
    
    # TTCs calculations
    classwise_ttc = [(idx, ttc) for idx, ttc in ttc_scores if idx is not None]
    onehot_ttc = compute_group_ttc(classwise_ttc, label_names, onehot_metrics)
    super_ttc = compute_group_ttc(classwise_ttc, label_names, super_metrics)
    
    # Save everything as a csv
    super_df = pd.DataFrame(super_metrics).T
    super_df["ttc"] = pd.Series(super_ttc)
    onehot_df = pd.DataFrame(onehot_metrics).T
    onehot_df["ttc"] = pd.Series(onehot_ttc)

    super_csv_path = "super_category_metrics.csv"
    onehot_csv_path = "onehot_class_metrics.csv"
    super_df.to_csv(super_csv_path)
    onehot_df.to_csv(onehot_csv_path)

    print(f"\nSaved CSVs:\n  {super_csv_path}\n  {onehot_csv_path}")
        
        
# Train through different percentiles of the timeline
def curriculum_train(
    model,
    dataset,
    epochs=10,
    lr=1e-3,
    percentile_schedule=[5, 10, 15, 20, 25, 30, 40, 50, 70, 90, 100],
):
    # Train test split using torch
    dataset_size = len(dataset)
    split_idx = int(0.8 * dataset_size)
    train_dataset, test_dataset = random_split(
        dataset, [split_idx, dataset_size - split_idx]
    )

    # Send model to gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Set optimizer and loss TODO: set to that custom loss
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
        train_dataset.dataset.truncate_by_percentile(percentile)
        train_loader = DataLoader(train_dataset, batch_size=None, shuffle=True)

        for x, x_static, y in train_loader:
            x, x_static, y = (
                x.to(device),
                x_static.to(device),
                y.to(device),
            )

            # Make predictions
            outputs = model(x, x_static)
            pred = torch.cat(outputs, dim=1)

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

        if (epoch % 1 == 0):
            test_dataset.dataset.truncate_by_percentile(percentile)
        
