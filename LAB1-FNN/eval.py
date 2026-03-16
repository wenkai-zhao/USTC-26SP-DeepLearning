from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader


def evaluate_model(
    model: nn.Module, dataloader: DataLoader, device: torch.device
) -> Tuple[float, float, float]:
    model.eval()
    criterion = nn.MSELoss()
    losses: List[float] = []
    all_preds: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []

    with torch.no_grad():
        for features, targets in dataloader:
            features = features.to(device)
            targets = targets.to(device)
            preds = model(features)
            loss = criterion(preds, targets)
            losses.append(loss.item())
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    y_pred = np.concatenate(all_preds, axis=0).reshape(-1)
    y_true = np.concatenate(all_targets, axis=0).reshape(-1)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    avg_loss = float(np.mean(losses)) if losses else float("inf")
    return avg_loss, mse, r2
