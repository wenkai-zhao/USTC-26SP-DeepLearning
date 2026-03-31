import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn

from eval import evaluate_classifier, save_confusion_matrix, save_misclassified_samples
from load_dataset import DataBundle, describe_data, load_and_split_data
from model import CNNClassifier

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CNN classification on Fashion-MNIST")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--sample-ratio", type=float, default=1.0)
    parser.add_argument("--conv-blocks", type=int, default=2)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--pool-type", choices=["max", "avg"], default="max")
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        total_loss += loss.item() * images.size(0)
        total_correct += (preds == targets).sum().item()
        total_samples += images.size(0)

    return total_loss / total_samples, total_correct / total_samples


def save_history_csv(history: List[Dict[str, float]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"],
        )
        writer.writeheader()
        writer.writerows(history)


def save_history_plot(history: List[Dict[str, float]], output_dir: Path) -> None:
    if plt is None:
        return

    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]
    train_acc = [row["train_acc"] for row in history]
    val_acc = [row["val_acc"] for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(epochs, train_loss, label="train")
    axes[0].plot(epochs, val_loss, label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, train_acc, label="train")
    axes[1].plot(epochs, val_acc, label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=200)
    plt.close(fig)


def train_model(
    data: DataBundle,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    run_name: str,
    conv_blocks: int,
    kernel_size: int,
    pool_type: str,
    base_channels: int,
    dropout: float,
    learning_rate: float,
    epochs: int,
    output_dir: Path | str,
    device: torch.device,
) -> Dict:
    model = CNNClassifier(
        num_classes=len(data.class_names),
        conv_blocks=conv_blocks,
        base_channels=base_channels,
        kernel_size=kernel_size,
        pool_type=pool_type,
        dropout=dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history: List[Dict[str, float]] = []
    best_val_acc = 0.0
    run_output_dir = Path(output_dir) / run_name
    best_path = run_output_dir / "checkpoints" / f"{run_name}.pt"
    best_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, _, _ = evaluate_classifier(
            model, val_loader, criterion, device
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

        print(
            f"[{run_name}] Epoch {epoch}/{epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=device))
    test_loss, test_acc, y_true, y_pred = evaluate_classifier(
        model, test_loader, criterion, device
    )

    history_csv = run_output_dir / "history.csv"
    save_history_csv(history, history_csv)
    save_history_plot(history, run_output_dir)
    save_confusion_matrix(y_true, y_pred, data.class_names, run_output_dir / "confusion_matrix.png")
    save_misclassified_samples(
        model,
        test_loader,
        data.class_names,
        device,
        run_output_dir / "misclassified_samples.png",
    )

    args_dict = {
        "run_name": run_name,
        "conv_blocks": conv_blocks,
        "kernel_size": kernel_size,
        "pool_type": pool_type,
        "base_channels": base_channels,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "epochs": epochs,
    }
    args_json = run_output_dir / "arguments.json"
    with args_json.open("w", encoding="utf-8") as f:
        json.dump(args_dict, f, indent=2, ensure_ascii=False)

    return {
        "run_name": run_name,
        "conv_blocks": conv_blocks,
        "kernel_size": kernel_size,
        "pool_type": pool_type,
        "base_channels": base_channels,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "history_csv": str(history_csv),
        "checkpoint": str(best_path),
        "arguments_json": str(args_json),
    }


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

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"Using device: {device}")
    print("Training model with config:")
    print(f"  Conv blocks: {args.conv_blocks}")
    print(f"  Base channels: {args.base_channels}")
    print(f"  Kernel size: {args.kernel_size}")
    print(f"  Pool type: {args.pool_type}")
    print(f"  Dropout: {args.dropout}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Output dir: {Path(args.output_dir) / run_name}")
    describe_data(data)

    result = train_model(
        data=data,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        run_name=run_name,
        conv_blocks=args.conv_blocks,
        kernel_size=args.kernel_size,
        pool_type=args.pool_type,
        base_channels=args.base_channels,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        output_dir=args.output_dir,
        device=device,
    )

    summary = {
        "run_name": result["run_name"],
        "device": str(device),
        "best_val_acc": result["best_val_acc"],
        "test_loss": result["test_loss"],
        "test_acc": result["test_acc"],
        "config": vars(args),
        "checkpoint": result["checkpoint"],
    }
    summary_path = Path(args.output_dir) / run_name / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nTraining completed!")
    print(f"Best validation accuracy: {result['best_val_acc']:.4f}")
    print(f"Test loss: {result['test_loss']:.4f}")
    print(f"Test accuracy: {result['test_acc']:.4f}")
    print(f"Checkpoint saved to: {result['checkpoint']}")
    print(f"History saved to: {result['history_csv']}")


if __name__ == "__main__":
    main()
