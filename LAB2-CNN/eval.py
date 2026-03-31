import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from load_dataset import load_and_split_data
from model import CNNClassifier

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def evaluate_classifier(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
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
            total_correct += (preds == targets).sum().item()
            total_samples += images.size(0)

            all_targets.append(targets.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    return avg_loss, accuracy, y_true, y_pred


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    output_path: str | Path,
) -> None:
    num_classes = len(class_names)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        cm[int(true_label), int(pred_label)] += 1

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_path.with_suffix(".csv"), cm, fmt="%d", delimiter=",")

    if plt is None:
        return

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_misclassified_samples(
    model: torch.nn.Module,
    dataloader: DataLoader,
    class_names: list[str],
    device: torch.device,
    output_path: str | Path,
    max_samples: int = 16,
) -> None:
    if plt is None:
        return

    model.eval()
    images_to_plot: list[np.ndarray] = []
    labels_to_plot: list[str] = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            preds = model(images).argmax(dim=1)

            mismatches = preds != targets
            if mismatches.any():
                wrong_images = images[mismatches].cpu().numpy()
                wrong_targets = targets[mismatches].cpu().numpy()
                wrong_preds = preds[mismatches].cpu().numpy()
                for image, target, pred in zip(wrong_images, wrong_targets, wrong_preds):
                    images_to_plot.append(image[0])
                    labels_to_plot.append(
                        f"true={class_names[target]}\npred={class_names[pred]}"
                    )
                    if len(images_to_plot) >= max_samples:
                        break
            if len(images_to_plot) >= max_samples:
                break

    if not images_to_plot:
        return

    rows = int(np.ceil(len(images_to_plot) / 4))
    fig, axes = plt.subplots(rows, 4, figsize=(10, 2.5 * rows))
    axes = np.atleast_1d(axes).reshape(rows, 4)

    for ax in axes.flat:
        ax.axis("off")

    for ax, image, label in zip(axes.flat, images_to_plot, labels_to_plot):
        ax.imshow(image, cmap="gray")
        ax.set_title(label, fontsize=9)
        ax.axis("off")

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained Fashion-MNIST CNN.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--sample-ratio", type=float, default=1.0)
    parser.add_argument("--conv-blocks", type=int, default=2)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--pool-type", choices=["max", "avg"], default="max")
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="./outputs/eval")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data, _, _, test_loader = load_and_split_data(
        data_dir=args.data_dir,
        val_ratio=args.val_ratio,
        batch_size=args.batch_size,
        seed=args.seed,
        sample_ratio=args.sample_ratio,
        num_workers=args.num_workers,
    )

    model = CNNClassifier(
        num_classes=len(data.class_names),
        conv_blocks=args.conv_blocks,
        base_channels=args.base_channels,
        kernel_size=args.kernel_size,
        pool_type=args.pool_type,
        dropout=args.dropout,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    criterion = torch.nn.CrossEntropyLoss()
    test_loss, test_acc, y_true, y_pred = evaluate_classifier(
        model, test_loader, criterion, device
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_confusion_matrix(y_true, y_pred, data.class_names, output_dir / "confusion_matrix.png")
    save_misclassified_samples(
        model,
        test_loader,
        data.class_names,
        device,
        output_dir / "misclassified_samples.png",
    )

    summary = {
        "checkpoint": args.checkpoint,
        "test_loss": test_loss,
        "test_acc": test_acc,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Saved evaluation outputs to: {output_dir}")


if __name__ == "__main__":
    main()
