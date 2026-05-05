import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from eval import evaluate_classifier, save_history, save_json
from load_dataset import create_eval_loaders, create_simclr_loader, describe_data, load_datasets
from losses import build_contrastive_loss
from model import LinearClassifier, SimCLRModel, build_encoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SimCLR for CIFAKE real/fake detection")
    parser.add_argument("--stage", choices=["pretrain", "linear_eval", "baseline"], required=True)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--encoder", choices=["resnet18", "mobilenet_v2"], default="resnet18")
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--projection-variant", choices=["plain", "batchnorm", "wide"], default="plain")
    parser.add_argument("--loss", choices=["ntxent", "contrastive", "triplet"], default="ntxent")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--augmentation", choices=["strong", "weak"], default="strong")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--linear-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--labeled-fraction", type=float, default=0.1)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--sample-ratio", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def run_name(args: argparse.Namespace) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [args.stage, args.encoder, f"sr{args.sample_ratio}", timestamp]
    if args.stage == "pretrain":
        parts.insert(2, args.loss)
        parts.insert(3, f"t{args.temperature}")
        parts.insert(4, args.projection_variant)
    else:
        parts.insert(2, f"lf{args.labeled_fraction}")
    return "_".join(str(part).replace(".", "p") for part in parts)


def pretrain(args: argparse.Namespace, device: torch.device) -> dict:
    data = load_datasets(args.data_dir, args.val_ratio, args.sample_ratio, args.seed)
    describe_data(data)
    loader = create_simclr_loader(data, args.batch_size, args.num_workers, args.augmentation)
    model = SimCLRModel(
        encoder=args.encoder,
        feature_dim=args.feature_dim,
        projection_dim=args.projection_dim,
        projection_variant=args.projection_variant,
    ).to(device)
    loss_fn = build_contrastive_loss(args.loss)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    out_dir = Path(args.output_dir) / run_name(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    best_loss = float("inf")
    best_path = out_dir / "best_encoder.pt"
    start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        total_samples = 0
        epoch_start = time.perf_counter()

        for view_i, view_j in loader:
            view_i = view_i.to(device, non_blocking=True)
            view_j = view_j.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            _, projections = model(torch.cat([view_i, view_j], dim=0))
            z_i, z_j = projections.chunk(2, dim=0)
            loss = loss_fn(z_i, z_j, args.temperature)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * view_i.size(0)
            total_samples += view_i.size(0)

        avg_loss = epoch_loss / total_samples
        row = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "epoch_time_sec": time.perf_counter() - epoch_start,
        }
        history.append(row)
        print(f"Epoch {epoch}/{args.epochs} pretrain_loss={avg_loss:.4f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.encoder.state_dict(), best_path)

    summary = {
        "stage": "pretrain",
        "encoder": args.encoder,
        "loss": args.loss,
        "temperature": args.temperature,
        "projection_variant": args.projection_variant,
        "augmentation": args.augmentation,
        "sample_ratio": args.sample_ratio,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "parameter_count": count_parameters(model),
        "best_loss": best_loss,
        "train_time_sec": time.perf_counter() - start,
        "checkpoint": str(best_path),
        "output_dir": str(out_dir),
    }
    save_history(history, out_dir / "history.csv")
    save_json(vars(args), out_dir / "config.json")
    save_json(summary, out_dir / "summary.json")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def train_classifier(args: argparse.Namespace, device: torch.device, baseline: bool) -> dict:
    data = load_datasets(args.data_dir, args.val_ratio, args.sample_ratio, args.seed)
    describe_data(data)
    train_loader, val_loader, test_loader = create_eval_loaders(
        data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        labeled_fraction=args.labeled_fraction,
        seed=args.seed,
    )

    encoder = build_encoder(args.encoder, args.feature_dim)
    if not baseline:
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for linear_eval.")
        state_dict = torch.load(args.checkpoint, map_location="cpu")
        encoder.load_state_dict(state_dict)

    model = LinearClassifier(
        encoder=encoder,
        feature_dim=args.feature_dim,
        num_classes=len(data.class_names),
        freeze_encoder=not baseline,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    out_dir = Path(args.output_dir) / run_name(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best_classifier.pt"
    best_f1 = -1.0
    history: list[dict] = []
    start = time.perf_counter()

    for epoch in range(1, args.linear_epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        epoch_start = time.perf_counter()

        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            total_loss += loss.item() * images.size(0)
            total_correct += (preds == targets).sum().item()
            total_samples += images.size(0)

        train_loss = total_loss / total_samples
        train_acc = total_correct / total_samples
        val_metrics = evaluate_classifier(model, val_loader, criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_f1_macro": val_metrics["f1_macro"],
            "epoch_time_sec": time.perf_counter() - epoch_start,
        }
        history.append(row)
        print(
            f"Epoch {epoch}/{args.linear_epochs} train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['f1_macro']:.4f}"
        )
        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=device))
    test_metrics = evaluate_classifier(model, test_loader, criterion, device)
    summary = {
        "stage": "baseline" if baseline else "linear_eval",
        "encoder": args.encoder,
        "checkpoint": "" if baseline else args.checkpoint,
        "labeled_fraction": args.labeled_fraction,
        "sample_ratio": args.sample_ratio,
        "linear_epochs": args.linear_epochs,
        "batch_size": args.batch_size,
        "parameter_count": count_parameters(model),
        "train_time_sec": time.perf_counter() - start,
        "test_accuracy": test_metrics["accuracy"],
        "test_precision_macro": test_metrics["precision_macro"],
        "test_recall_macro": test_metrics["recall_macro"],
        "test_f1_macro": test_metrics["f1_macro"],
        "test_loss": test_metrics["loss"],
        "classifier_checkpoint": str(best_path),
        "output_dir": str(out_dir),
    }
    save_history(history, out_dir / "history.csv")
    save_json(vars(args), out_dir / "config.json")
    save_json(summary, out_dir / "summary.json")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.stage == "pretrain":
        pretrain(args, device)
    elif args.stage == "linear_eval":
        train_classifier(args, device, baseline=False)
    elif args.stage == "baseline":
        train_classifier(args, device, baseline=True)
    else:
        raise ValueError(f"Unknown stage: {args.stage}")


if __name__ == "__main__":
    main()
