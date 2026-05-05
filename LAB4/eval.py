import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    accuracy = float((y_true == y_pred).mean())
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    f1_scores: list[float] = []
    precisions: list[float] = []
    recalls: list[float] = []

    for label in labels:
        tp = int(((y_true == label) & (y_pred == label)).sum())
        fp = int(((y_true != label) & (y_pred == label)).sum())
        fn = int(((y_true == label) & (y_pred != label)).sum())
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    return {
        "accuracy": accuracy,
        "precision_macro": float(np.mean(precisions)),
        "recall_macro": float(np.mean(recalls)),
        "f1_macro": float(np.mean(f1_scores)),
    }


def evaluate_classifier(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_targets: list[np.ndarray] = []
    all_preds: list[np.ndarray] = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            loss = criterion(logits, targets)
            preds = logits.argmax(dim=1)

            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
            all_targets.append(targets.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    metrics = classification_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / total_samples
    return metrics


def save_history(rows: list[dict], output_path: Path) -> None:
    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_json(data: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
