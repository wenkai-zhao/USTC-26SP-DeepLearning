import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from arguments import parse_args
from dataloader import DataBundle, load_and_split_data
from eval import evaluate_model
from model import FNNRegressor
from utils import build_optimizer, set_seed


def train_model(
    data: DataBundle,
    run_name: str,
    hidden_dims: List[int],
    activation: str,
    lr: float,
    optimizer_name: str,
    batch_size: int,
    epochs: int,
    output_dir: Path | str,
    device: torch.device,
) -> Dict:
    train_loader = DataLoader(data.train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(data.val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(data.test_dataset, batch_size=batch_size, shuffle=False)

    model = FNNRegressor(data.input_dim, hidden_dims, activation).to(device)
    criterion = nn.MSELoss()
    optimizer = build_optimizer(model, optimizer_name, lr)

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
    }

    best_val = float("inf")
    run_output_dir = Path(output_dir) / run_name
    best_path = run_output_dir / "checkpoints" / f"{run_name}.pt"
    best_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            preds = model(features)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train = float(np.mean(train_losses)) if train_losses else float("inf")
        val_loss, _, _ = evaluate_model(model, val_loader, device)

        history["epoch"].append(epoch)
        history["train_loss"].append(avg_train)
        history["val_loss"].append(val_loss)

        if epoch % 10 == 0 or epoch == epochs:
            print(
                f"[{run_name}] Epoch {epoch}/{epochs} "
                f"train_loss={avg_train:.4f} val_loss={val_loss:.4f}"
            )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=device))

    train_final, train_mse, train_r2 = evaluate_model(model, train_loader, device)
    val_final, val_mse, val_r2 = evaluate_model(model, val_loader, device)
    test_final, test_mse, test_r2 = evaluate_model(model, test_loader, device)

    history_df = pd.DataFrame(history)
    history_csv = run_output_dir / "history.csv"
    history_csv.parent.mkdir(parents=True, exist_ok=True)
    history_df.to_csv(history_csv, index=False, encoding="utf-8")

    # Save arguments/configuration
    args_dict = {
        "run_name": run_name,
        "hidden_dims": hidden_dims,
        "activation": activation,
        "learning_rate": lr,
        "optimizer": optimizer_name,
        "batch_size": batch_size,
        "epochs": epochs,
    }
    args_json = run_output_dir / "arguments.json"
    with open(args_json, "w", encoding="utf-8") as f:
        json.dump(args_dict, f, indent=4, ensure_ascii=False)

    return {
        "run_name": run_name,
        "hidden_dims": str(hidden_dims),
        "activation": activation,
        "optimizer": optimizer_name,
        "learning_rate": lr,
        "batch_size": batch_size,
        "epochs": epochs,
        "best_val_loss": best_val,
        "train_loss": train_final,
        "val_loss": val_final,
        "test_loss": test_final,
        "train_mse": train_mse,
        "val_mse": val_mse,
        "test_mse": test_mse,
        "train_r2": train_r2,
        "val_r2": val_r2,
        "test_r2": test_r2,
        "history_csv": str(history_csv),
        "checkpoint": str(best_path),
    }


def main():
    args = parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_and_split_data(seed=args.seed)

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"Using device: {device}")
    print(f"Training model with config:")
    print(f"  Hidden dims: {args.hidden_dims}")
    print(f"  Activation: {args.activation}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Optimizer: {args.optimizer}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")

    result = train_model(
        data=data,
        run_name=run_name,
        hidden_dims=args.hidden_dims,
        activation=args.activation,
        lr=args.learning_rate,
        optimizer_name=args.optimizer,
        batch_size=args.batch_size,
        epochs=args.epochs,
        output_dir=args.output_dir,
        device=device,
    )

    print(f"\nTraining completed!")
    print(f"Test loss: {result['test_loss']:.4f}")
    print(f"Test MSE: {result['test_mse']:.4f}")
    print(f"Test R²: {result['test_r2']:.4f}")
    print(f"Checkpoint saved to: {result['checkpoint']}")
    print(f"History saved to: {result['history_csv']}")


if __name__ == "__main__":
    main()
