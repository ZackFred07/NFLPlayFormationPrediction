from collections import defaultdict
import csv
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
from sklearn.metrics import precision_score, recall_score, f1_score, auc, accuracy_score
from tqdm import tqdm
from st_dataset import TwoDimensionalTensorDataset, TestTwoDimensionalTensorDataset
from cnn_lstm import CNNLSTMClassifier


output_dir = os.environ["OUTPUT_DIR"]
best_loss = float("inf")


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
    model,
    dataset,
    epochs=10,
    lr=1e-3,
    percentile_schedule=[10, 20, 40, 50, 75, 100],
    name="default",
    device="cuda"
):
    print("Start training")
    best_loss = np.inf
    # Train test split using torch
    dataset_size = len(dataset)
    split_idx = int(0.8 * dataset_size)
    train_dataset, test_dataset = random_split(
        dataset, [split_idx, dataset_size - split_idx]
    )

    # Send model to gpu
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        train_loss = 0
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
            train_loss += loss.item()
        train_loss /= len(train_loader)

        end_time = time.time()
        gc.collect()

        model.eval()

        y_true, y_pred = [], []

        with torch.no_grad():
            y_true_list, y_pred_list, y_logits_list = [], [], []
            test_loss = 0
            # For each data point
            for x, x_static, y in tqdm(test_loader, "Testing"):

                x, x_static, y = x.to(device), x_static.to(device), y.to(device)

                # run inference
                outputs = model(x, x_static)

                final_output = outputs[-1]  # [batch_size, num_labels] - raw logits

                loss = criterion(final_output, y)
                test_loss += loss.item()

                # Add true and predicted
                y_true_list.append(y.cpu())
                y_pred_list.append((final_output.sigmoid() > 0.5).cpu())
                y_logits_list.append(final_output.cpu())
            
            test_loss /= len(test_loader)
            latest_path = os.path.join(
                output_dir, name, "checkpoints", "model_latest.pt"
            )
            torch.save(model.state_dict(), latest_path)

            if test_loss < best_loss:
                best_loss = test_loss
                best_path = os.path.join(
                    output_dir, name, "checkpoints", "model_best.pt"
                )
                torch.save(model.state_dict(), best_path)

            # Stack all predictions
            y_true = torch.stack(y_true_list).to(torch.float32).numpy()
            y_pred = torch.stack(y_pred_list).to(torch.float32).numpy()
            y_logits = torch.stack(y_logits_list).to(torch.float32).numpy()

            # Load the json with label information
            with open("label_metadata.json") as f:
                label_metadata = json.load(f)

            label_names = list(label_metadata.keys())

            # Create mappings from the metadata json file
            onehot_class_map = defaultdict(list)
            for label, info in label_metadata.items():
                if info.get("type") == "onehot" and "onehot_class" in info:
                    onehot_class_map[info["onehot_class"]].append(label)

            # Per label Metrics
            per_label_metrics = []
            label_names = list(label_metadata.keys())

            for i, label in enumerate(label_names):
                p = precision_score(y_true[:, i], y_pred[:, i], zero_division=0)
                r = recall_score(y_true[:, i], y_pred[:, i], zero_division=0)
                f = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
                per_label_metrics.append(
                    {"label": label, "accuracy":a, "precision": p, "recall": r, "f1": f}
                )

            # Softmax-Based Onehot Metrics
            onehot_metrics = []

            for group, label_list in onehot_class_map.items():
                indices = [
                    label_names.index(lbl) for lbl in label_list if lbl in label_names
                ]
                if not indices:
                    continue

                logits_group = y_logits[:, indices]
                true_group = y_true[:, indices]

                softmaxed = torch.softmax(torch.tensor(logits_group), dim=1).numpy()
                pred_onehot = np.zeros_like(softmaxed)
                pred_onehot[np.arange(len(softmaxed)), np.argmax(softmaxed, axis=1)] = 1
                a = accuracy_score(
                    true_group, pred_onehot
                )
                p = precision_score(
                    true_group, pred_onehot, average="macro", zero_division=0
                )
                r = recall_score(
                    true_group, pred_onehot, average="macro", zero_division=0
                )
                f = f1_score(true_group, pred_onehot, average="macro", zero_division=0)

                onehot_metrics.append(
                    {"group": group, "accuracy":a, "precision": p, "recall": r, "f1": f}
                )

        # print eveything
        print(
            f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}, Train GPU = {torch.cuda.max_memory_allocated() / 1e6:.2f}MB, Train Time = {end_time - start_time:.2f}s"
        )

        # Save epoch-level metrics (append mode)
        with open(
            os.path.join(output_dir, name, "epoch_metrics.csv"), mode="a", newline=""
        ) as file:
            writer = csv.writer(file)
            if epoch == 0:  # Write header only once
                writer.writerow(
                    [
                        "epoch",
                        "train_loss",
                        "test_loss",
                        "gpu_memory_mb",
                        "epoch_time_sec",
                    ]
                )
            writer.writerow(
                [
                    epoch + 1,
                    train_loss,
                    test_loss,
                    round(torch.cuda.max_memory_allocated() / 1e6, 2),
                    round(end_time - start_time, 2),
                ]
            )

        # Save per-label metrics
        with open(
            os.path.join(output_dir, name, "per_label_metrics.csv"), "a", newline=""
        ) as f:
            fieldnames = ["epoch", "label", "precision", "recall", "f1"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if epoch == 0:
                writer.writeheader()
            for row in per_label_metrics:
                row["epoch"] = epoch + 1
                writer.writerow(row)

        # Save per-onehot-class metrics
        with open(
            os.path.join(output_dir, name, "per_onehot_class_metrics.csv"),
            "a",
            newline="",
        ) as f:
            fieldnames = ["epoch", "group", "accuracy", "precision", "recall", "f1"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if epoch == 0:
                writer.writeheader()
            for row in onehot_metrics:
                row["epoch"] = epoch + 1
                writer.writerow(row)


def main(args):
    # make the directories
    os.makedirs(os.path.join(output_dir, args.name, "logs"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, args.name, "checkpoints"), exist_ok=True)

    # Load dataset
    model = CNNLSTMClassifier().to(torch.bfloat16)
    dataset = TwoDimensionalTensorDataset()
    # dataset = TestTwoDimensionalTensorDataset(dataset)  # This is for debugging

    train_eval(
        model=model,
        dataset=dataset,
        epochs=args.epochs,
        lr=args.lr,
        percentile_schedule=args.percentiles,
        name=args.name,
        device=args.device
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train CNN-LSTM model")

    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--percentiles",
        type=int,
        nargs="+",
        default=[10, 20, 40, 50, 75],
        help="List of percentile values for curriculum schedule",
    )
    parser.add_argument("--name", type=str, default="unnamed")
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    main(args)