import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import torch

from train import pretrain, set_seed, train_classifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LAB4 SimCLR experiment suites.")
    parser.add_argument("--suite", choices=["main", "ablations", "all"], default="main")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--sample-ratio", type=float, default=0.1)
    parser.add_argument("--pretrain-epochs", type=int, default=50)
    parser.add_argument("--linear-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def make_train_args(base: argparse.Namespace, **overrides) -> argparse.Namespace:
    cfg = {
        "stage": "pretrain",
        "data_dir": base.data_dir,
        "output_dir": base.output_dir,
        "checkpoint": "",
        "encoder": "resnet18",
        "feature_dim": 128,
        "projection_dim": 64,
        "projection_variant": "plain",
        "loss": "ntxent",
        "temperature": 0.5,
        "augmentation": "strong",
        "epochs": base.pretrain_epochs,
        "linear_epochs": base.linear_epochs,
        "batch_size": base.batch_size,
        "learning_rate": base.learning_rate,
        "weight_decay": base.weight_decay,
        "labeled_fraction": 0.1,
        "val_ratio": 0.1,
        "sample_ratio": base.sample_ratio,
        "num_workers": base.num_workers,
        "seed": base.seed,
    }
    cfg.update(overrides)
    return argparse.Namespace(**cfg)


def save_rows(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def run_main_suite(base: argparse.Namespace, device: torch.device) -> list[dict]:
    rows: list[dict] = []
    for encoder in ["resnet18", "mobilenet_v2"]:
        pre_args = make_train_args(base, stage="pretrain", encoder=encoder)
        pre_summary = pretrain(pre_args, device)
        rows.append(pre_summary)

        for fraction in [0.01, 0.10]:
            eval_args = make_train_args(
                base,
                stage="linear_eval",
                encoder=encoder,
                checkpoint=pre_summary["checkpoint"],
                labeled_fraction=fraction,
            )
            rows.append(train_classifier(eval_args, device, baseline=False))

            baseline_args = make_train_args(
                base,
                stage="baseline",
                encoder=encoder,
                labeled_fraction=fraction,
            )
            rows.append(train_classifier(baseline_args, device, baseline=True))
    return rows


def run_ablation_suite(base: argparse.Namespace, device: torch.device) -> list[dict]:
    rows: list[dict] = []
    settings = [
        {"loss": "contrastive"},
        {"loss": "triplet"},
        {"projection_variant": "batchnorm"},
        {"projection_variant": "wide"},
        {"temperature": 0.2},
        {"temperature": 1.0},
        {"augmentation": "weak"},
    ]

    for setting in settings:
        pre_args = make_train_args(base, stage="pretrain", encoder="resnet18", **setting)
        pre_summary = pretrain(pre_args, device)
        rows.append(pre_summary)
        eval_args = make_train_args(
            base,
            stage="linear_eval",
            encoder="resnet18",
            checkpoint=pre_summary["checkpoint"],
            labeled_fraction=0.10,
        )
        rows.append(train_classifier(eval_args, device, baseline=False))
    return rows


def main() -> None:
    args = parse_args()
    if args.sample_ratio < 0.1:
        raise ValueError("sample_ratio should not be less than 0.1 for this assignment.")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = Path(args.output_dir) / f"suite_{args.suite}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using device: {device}")
    print(f"Suite output: {exp_dir}")

    all_rows: list[dict] = []
    if args.suite in {"main", "all"}:
        main_rows = run_main_suite(args, device)
        save_rows(main_rows, exp_dir / "main_results.csv")
        all_rows.extend(main_rows)
    if args.suite in {"ablations", "all"}:
        ablation_rows = run_ablation_suite(args, device)
        save_rows(ablation_rows, exp_dir / "ablation_results.csv")
        all_rows.extend(ablation_rows)

    save_rows(all_rows, exp_dir / "all_results.csv")
    (exp_dir / "suite_config.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved aggregate results to: {exp_dir / 'all_results.csv'}")


if __name__ == "__main__":
    main()
