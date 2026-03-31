import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch

from load_dataset import DataBundle, load_and_split_data
from train import parse_args, set_seed, train_model

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def save_experiment_csv(rows: List[Dict], output_path: Path) -> None:
    if not rows:
        return

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_histories(result_rows: List[Dict], experiment_name: str, output_dir: Path) -> Path | None:
    if plt is None or not result_rows:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cmap = plt.get_cmap("tab10")

    for i, row in enumerate(result_rows):
        color = cmap(i % 10)
        history_path = Path(row["history_csv"])
        epochs: List[int] = []
        train_loss: List[float] = []
        val_loss: List[float] = []
        train_acc: List[float] = []
        val_acc: List[float] = []

        with history_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for history_row in reader:
                epochs.append(int(history_row["epoch"]))
                train_loss.append(float(history_row["train_loss"]))
                val_loss.append(float(history_row["val_loss"]))
                train_acc.append(float(history_row["train_acc"]))
                val_acc.append(float(history_row["val_acc"]))

        axes[0].plot(
            epochs,
            val_loss,
            linestyle="-",
            color=color,
            linewidth=2,
            label=f"{row['run_name']} val",
        )
        axes[0].plot(
            epochs,
            train_loss,
            linestyle="--",
            color=color,
            alpha=0.6,
            label=f"{row['run_name']} train",
        )

        axes[1].plot(
            epochs,
            val_acc,
            linestyle="-",
            color=color,
            linewidth=2,
            label=f"{row['run_name']} val",
        )
        axes[1].plot(
            epochs,
            train_acc,
            linestyle="--",
            color=color,
            alpha=0.6,
            label=f"{row['run_name']} train",
        )

    axes[0].set_title(f"{experiment_name} Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("CrossEntropy Loss")
    axes[0].legend(fontsize=8, ncol=2)

    axes[1].set_title(f"{experiment_name} Accuracy Curves")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend(fontsize=8, ncol=2)

    fig.tight_layout()
    figure_path = output_dir / "figures" / f"{experiment_name}_curves.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    return figure_path


def run_depth_experiment(
    data: DataBundle,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    base_cfg: Dict,
    output_dir: Path,
    device: torch.device,
) -> List[Dict]:
    candidates = [2, 3, 4]
    rows: List[Dict] = []

    for conv_blocks in candidates:
        row = train_model(
            data=data,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            run_name=f"depth_{conv_blocks}",
            conv_blocks=conv_blocks,
            kernel_size=base_cfg["kernel_size"],
            pool_type=base_cfg["pool_type"],
            base_channels=base_cfg["base_channels"],
            dropout=base_cfg["dropout"],
            learning_rate=base_cfg["learning_rate"],
            epochs=base_cfg["epochs"],
            output_dir=output_dir,
            device=device,
        )
        rows.append(row)

    rows.sort(key=lambda row: row["test_acc"], reverse=True)
    return rows


def run_kernel_experiment(
    data: DataBundle,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    base_cfg: Dict,
    output_dir: Path,
    device: torch.device,
) -> List[Dict]:
    candidates = [3, 5, 7]
    rows: List[Dict] = []

    for kernel_size in candidates:
        row = train_model(
            data=data,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            run_name=f"kernel_{kernel_size}",
            conv_blocks=base_cfg["conv_blocks"],
            kernel_size=kernel_size,
            pool_type=base_cfg["pool_type"],
            base_channels=base_cfg["base_channels"],
            dropout=base_cfg["dropout"],
            learning_rate=base_cfg["learning_rate"],
            epochs=base_cfg["epochs"],
            output_dir=output_dir,
            device=device,
        )
        rows.append(row)

    rows.sort(key=lambda row: row["test_acc"], reverse=True)
    return rows


def run_pool_experiment(
    data: DataBundle,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    base_cfg: Dict,
    output_dir: Path,
    device: torch.device,
) -> List[Dict]:
    candidates = ["max", "avg"]
    rows: List[Dict] = []

    for pool_type in candidates:
        row = train_model(
            data=data,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            run_name=f"pool_{pool_type}",
            conv_blocks=base_cfg["conv_blocks"],
            kernel_size=base_cfg["kernel_size"],
            pool_type=pool_type,
            base_channels=base_cfg["base_channels"],
            dropout=base_cfg["dropout"],
            learning_rate=base_cfg["learning_rate"],
            epochs=base_cfg["epochs"],
            output_dir=output_dir,
            device=device,
        )
        rows.append(row)

    rows.sort(key=lambda row: row["test_acc"], reverse=True)
    return rows


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, train_loader, val_loader, test_loader = load_and_split_data(
        data_dir=args.data_dir,
        val_ratio=args.val_ratio,
        batch_size=args.batch_size,
        seed=args.seed,
        sample_ratio=args.sample_ratio,
        num_workers=args.num_workers,
    )

    output_dir = Path(args.output_dir)
    experiment_dir = output_dir / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "conv_blocks": args.conv_blocks,
        "base_channels": args.base_channels,
        "kernel_size": args.kernel_size,
        "pool_type": args.pool_type,
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "seed": args.seed,
        "sample_ratio": args.sample_ratio,
        "num_workers": args.num_workers,
        "device": str(device),
    }

    print(f"Using device: {device}")
    print(f"Experiment output dir: {experiment_dir}")

    print("Start depth experiment...")
    depth_rows = run_depth_experiment(
        data, train_loader, val_loader, test_loader, config, experiment_dir, device
    )
    plot_histories(depth_rows, "depth", experiment_dir)

    print("Start kernel-size experiment...")
    kernel_rows = run_kernel_experiment(
        data, train_loader, val_loader, test_loader, config, experiment_dir, device
    )
    plot_histories(kernel_rows, "kernel", experiment_dir)

    print("Start pooling experiment...")
    pool_rows = run_pool_experiment(
        data, train_loader, val_loader, test_loader, config, experiment_dir, device
    )
    plot_histories(pool_rows, "pool", experiment_dir)

    results_dir = experiment_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    save_experiment_csv(depth_rows, results_dir / "depth_experiment.csv")
    save_experiment_csv(kernel_rows, results_dir / "kernel_experiment.csv")
    save_experiment_csv(pool_rows, results_dir / "pool_experiment.csv")

    all_rows = depth_rows + kernel_rows + pool_rows
    save_experiment_csv(all_rows, results_dir / "experiment_results.csv")

    config_path = results_dir / "experiment_config.json"
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nAll experiments finished.")
    print(f"Saved experiment results to: {results_dir / 'experiment_results.csv'}")


if __name__ == "__main__":
    main()
