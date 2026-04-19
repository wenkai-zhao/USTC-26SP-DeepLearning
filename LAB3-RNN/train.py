import argparse
import csv
import json
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from eval import benchmark_inference, evaluate_model, save_metrics
from load_dataset import PAD_TOKEN, create_dataloaders, describe_data, load_and_split_data
from model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RNN or attention model on IMDB")
    parser.add_argument("--data-path", type=str, default="data/IMDB Dataset.csv")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--model", choices=["attention", "rnn"], default="attention")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-vocab-size", type=int, default=30000)
    parser.add_argument("--max-seq-len", type=int, default=200)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--disable-positional-encoding", action="store_true")
    parser.add_argument("--sample-ratio", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
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

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        preds = (torch.sigmoid(logits) >= 0.5).long()
        total_loss += loss.item() * inputs.size(0)
        total_correct += (preds == labels.long()).sum().item()
        total_samples += inputs.size(0)

    return total_loss / total_samples, total_correct / total_samples


def save_history(history: list[dict[str, float]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_accuracy",
                "val_loss",
                "val_accuracy",
                "val_precision",
                "val_recall",
                "val_f1",
                "epoch_time_sec",
            ],
        )
        writer.writeheader()
        writer.writerows(history)


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def train(args: argparse.Namespace) -> dict[str, float | str | int]:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_and_split_data(
        csv_path=args.data_path,
        max_vocab_size=args.max_vocab_size,
        max_seq_len=args.max_seq_len,
        sample_ratio=args.sample_ratio,
        seed=args.seed,
    )
    train_loader, val_loader, test_loader = create_dataloaders(
        data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    describe_data(data)
    print(f"Using device: {device}")

    model = build_model(
        model_name=args.model,
        vocab_size=len(data.vocab),
        max_seq_len=args.max_seq_len,
        pad_idx=data.vocab[PAD_TOKEN],
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_heads=args.num_heads,
        use_positional_encoding=not args.disable_positional_encoding,
    ).to(device)
    parameter_count = count_parameters(model)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    run_name = f"{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    run_dir = Path(args.output_dir) / run_name
    checkpoint_path = run_dir / "best.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, float]] = []
    best_val_f1 = -1.0
    epoch_times: list[float] = []
    train_start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.perf_counter()
        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        val_metrics = evaluate_model(model, val_loader, criterion, device)
        epoch_time_sec = time.perf_counter() - epoch_start_time
        epoch_times.append(epoch_time_sec)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_metrics["f1"],
                "epoch_time_sec": epoch_time_sec,
            }
        )

        print(
            f"Epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['f1']:.4f} epoch_time={epoch_time_sec:.2f}s"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), checkpoint_path)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    total_train_time_sec = time.perf_counter() - train_start_time
    test_metrics = evaluate_model(model, test_loader, criterion, device)
    inference_metrics = benchmark_inference(model, test_loader, device)
    test_metrics.update(inference_metrics)

    save_history(history, run_dir / "history.csv")
    save_metrics(test_metrics, run_dir / "test_metrics.csv")
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    summary = {
        "run_name": run_name,
        "model": args.model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "embed_dim": args.embed_dim,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "dropout": args.dropout,
        "use_positional_encoding": not args.disable_positional_encoding,
        "sample_ratio": args.sample_ratio,
        "parameter_count": parameter_count,
        "train_time_sec": total_train_time_sec,
        "avg_epoch_time_sec": sum(epoch_times) / len(epoch_times),
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
        "test_avg_batch_inference_ms": test_metrics["avg_batch_inference_ms"],
        "test_samples_per_sec": test_metrics["samples_per_sec"],
        "output_dir": str(run_dir),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
