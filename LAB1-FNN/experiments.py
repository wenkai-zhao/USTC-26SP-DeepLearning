import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn

from arguments import parse_args
from dataloader import DataBundle
from train import train_model
from utils import plot_histories, set_seed
from dataloader import load_and_split_data


def run_depth_experiment(
    data: DataBundle,
    base_cfg: Dict,
    output_dir: Path,
    device: torch.device,
) -> Tuple[pd.DataFrame, Path]:
    candidates = {
        "shallow_1layer": [32],
        "medium_2layer": [64, 32],
        "deep_3layer": [128, 64, 32],
    }

    rows: List[Dict] = []
    for name, dims in candidates.items():
        row = train_model(
            data=data,
            run_name=f"depth_{name}",
            hidden_dims=dims,
            activation=base_cfg["activation"],
            lr=base_cfg["learning_rate"],
            optimizer_name=base_cfg["optimizer"],
            batch_size=base_cfg["batch_size"],
            epochs=base_cfg["epochs"],
            output_dir=output_dir,
            device=device,
        )
        rows.append(row)

    fig_path = plot_histories(rows, "depth", output_dir)
    df = pd.DataFrame(rows).sort_values(by="test_loss")
    out_csv = output_dir / "results" / "depth_experiment.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8")
    return df, fig_path


def run_lr_experiment(
    data: DataBundle,
    base_cfg: Dict,
    output_dir: Path,
    device: torch.device,
) -> Tuple[pd.DataFrame, Path]:
    lrs = [0.1, 0.01, 0.001, 0.0001]

    rows: List[Dict] = []
    for lr in lrs:
        row = train_model(
            data=data,
            run_name=f"lr_{lr}",
            hidden_dims=base_cfg["hidden_dims"],
            activation=base_cfg["activation"],
            lr=lr,
            optimizer_name=base_cfg["optimizer"],
            batch_size=base_cfg["batch_size"],
            epochs=base_cfg["epochs"],
            output_dir=output_dir,
            device=device,
        )
        rows.append(row)

    fig_path = plot_histories(rows, "learning_rate", output_dir)
    df = pd.DataFrame(rows).sort_values(by="test_loss")
    out_csv = output_dir / "results" / "learning_rate_experiment.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8")
    return df, fig_path


def run_activation_experiment(
    data: DataBundle,
    base_cfg: Dict,
    output_dir: Path,
    device: torch.device,
) -> Tuple[pd.DataFrame, Path]:
    activations = ["sigmoid", "tanh", "relu", "leakyrelu", "swish"]

    rows: List[Dict] = []
    for activation in activations:
        row = train_model(
            data=data,
            run_name=f"act_{activation}",
            hidden_dims=base_cfg["hidden_dims"],
            activation=activation,
            lr=base_cfg["learning_rate"],
            optimizer_name=base_cfg["optimizer"],
            batch_size=base_cfg["batch_size"],
            epochs=base_cfg["epochs"],
            output_dir=output_dir,
            device=device,
        )
        rows.append(row)

    fig_path = plot_histories(rows, "activation", output_dir)
    df = pd.DataFrame(rows).sort_values(by="test_loss")
    out_csv = output_dir / "results" / "activation_experiment.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8")
    return df, fig_path


def main():
    args = parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    experiment_dir = (
        output_dir / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    experiment_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_and_split_data(seed=args.seed)

    config = {
        "hidden_dims": args.hidden_dims,
        "activation": args.activation,
        "learning_rate": args.learning_rate,
        "optimizer": args.optimizer,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "seed": args.seed,
        "device": str(device),
    }

    print(f"Using device: {device}")
    print(f"Experiment output dir: {experiment_dir}")
    print("Start depth experiment...")
    run_depth_experiment(data, config, experiment_dir, device)

    print("Start learning-rate experiment...")
    run_lr_experiment(data, config, experiment_dir, device)

    print("Start activation experiment...")
    run_activation_experiment(data, config, experiment_dir, device)

    results_dir = experiment_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    config_path = results_dir / "train_config.json"
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nAll experiments finished.")
    print(f"Saved train config: {config_path}")


if __name__ == "__main__":
    main()
