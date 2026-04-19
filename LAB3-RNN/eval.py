import time
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> tuple[float, list[int], list[int]]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    y_true: list[int] = []
    y_pred: list[int] = []

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        logits = model(inputs)
        loss = criterion(logits, labels)
        preds = (torch.sigmoid(logits) >= 0.5).long()

        total_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)
        y_true.extend(labels.long().cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    return total_loss / total_samples, y_true, y_pred


def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> dict[str, float]:
    loss, y_true, y_pred = predict(model, dataloader, criterion, device)
    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = loss
    return metrics


def save_metrics(metrics: dict[str, float], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(output_path, index=False)


@torch.no_grad()
def benchmark_inference(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    warmup_batches: int = 1,
    measure_batches: int = 5,
) -> dict[str, float]:
    model.eval()
    batch_times: list[float] = []
    total_samples = 0
    measured = 0

    for batch_idx, (inputs, _) in enumerate(dataloader):
        if batch_idx >= warmup_batches + measure_batches:
            break

        inputs = inputs.to(device)
        start = time.perf_counter()
        _ = model(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start

        if batch_idx >= warmup_batches:
            batch_times.append(elapsed)
            total_samples += inputs.size(0)
            measured += 1

    if measured == 0:
        return {
            "inference_batches": 0.0,
            "inference_time_sec": 0.0,
            "avg_batch_inference_ms": 0.0,
            "samples_per_sec": 0.0,
        }

    total_time = sum(batch_times)
    avg_batch_time = total_time / measured
    return {
        "inference_batches": float(measured),
        "inference_time_sec": total_time,
        "avg_batch_inference_ms": avg_batch_time * 1000.0,
        "samples_per_sec": total_samples / total_time if total_time > 0 else 0.0,
    }
