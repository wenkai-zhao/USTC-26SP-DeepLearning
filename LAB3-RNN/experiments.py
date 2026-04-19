import argparse
import csv
from argparse import Namespace
from datetime import datetime
from pathlib import Path

from train import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run grouped LAB3-RNN experiments")
    parser.add_argument("--data-path", type=str, default="data/IMDB Dataset.csv")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sample-ratio", type=float, default=0.1)
    parser.add_argument("--max-vocab-size", type=int, default=30000)
    parser.add_argument("--max-seq-len", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--suite",
        choices=["full", "main", "heads", "vocab", "embed", "position", "lr", "layers"],
        default="full",
    )
    return parser.parse_args()


def build_base_args(cli_args: argparse.Namespace) -> dict:
    return {
        "data_path": cli_args.data_path,
        "output_dir": cli_args.output_dir,
        "epochs": cli_args.epochs,
        "batch_size": cli_args.batch_size,
        "learning_rate": 1e-3,
        "max_vocab_size": cli_args.max_vocab_size,
        "max_seq_len": cli_args.max_seq_len,
        "embed_dim": 128,
        "hidden_dim": 256,
        "num_layers": 1,
        "num_heads": 4,
        "dropout": 0.2,
        "disable_positional_encoding": False,
        "sample_ratio": cli_args.sample_ratio,
        "num_workers": cli_args.num_workers,
        "seed": cli_args.seed,
    }


def build_runs(cli_args: argparse.Namespace) -> list[Namespace]:
    common = build_base_args(cli_args)
    suites: dict[str, list[dict]] = {
        "main": [
            {
                "experiment_group": "main_compare",
                "variant": "rnn_baseline",
                "model": "rnn",
                "hidden_dim": 128,
            },
            {
                "experiment_group": "main_compare",
                "variant": "attention_baseline",
                "model": "attention",
                "embed_dim": 128,
                "hidden_dim": 256,
                "num_heads": 4,
            },
        ],
        "heads": [
            {
                "experiment_group": "attention_heads",
                "variant": "heads_2",
                "model": "attention",
                "num_heads": 2,
            },
            {
                "experiment_group": "attention_heads",
                "variant": "heads_4",
                "model": "attention",
                "num_heads": 4,
            },
            {
                "experiment_group": "attention_heads",
                "variant": "heads_8",
                "model": "attention",
                "num_heads": 8,
            },
        ],
        "vocab": [
            {
                "experiment_group": "vocab_size",
                "variant": "vocab_10000",
                "model": "attention",
                "max_vocab_size": 10000,
            },
            {
                "experiment_group": "vocab_size",
                "variant": "vocab_20000",
                "model": "attention",
                "max_vocab_size": 20000,
            },
            {
                "experiment_group": "vocab_size",
                "variant": "vocab_30000",
                "model": "attention",
                "max_vocab_size": 30000,
            },
        ],
        "embed": [
            {
                "experiment_group": "embed_dim",
                "variant": "embed_64",
                "model": "attention",
                "embed_dim": 64,
                "hidden_dim": 128,
                "num_heads": 4,
            },
            {
                "experiment_group": "embed_dim",
                "variant": "embed_128",
                "model": "attention",
                "embed_dim": 128,
                "hidden_dim": 256,
                "num_heads": 4,
            },
            {
                "experiment_group": "embed_dim",
                "variant": "embed_256",
                "model": "attention",
                "embed_dim": 256,
                "hidden_dim": 256,
                "num_heads": 8,
            },
        ],
        "position": [
            {
                "experiment_group": "position_encoding",
                "variant": "pos_on",
                "model": "attention",
                "disable_positional_encoding": False,
            },
            {
                "experiment_group": "position_encoding",
                "variant": "pos_off",
                "model": "attention",
                "disable_positional_encoding": True,
            },
        ],
        "lr": [
            {
                "experiment_group": "learning_rate",
                "variant": "lr_1e-4",
                "model": "attention",
                "learning_rate": 1e-4,
            },
            {
                "experiment_group": "learning_rate",
                "variant": "lr_1e-3",
                "model": "attention",
                "learning_rate": 1e-3,
            },
            {
                "experiment_group": "learning_rate",
                "variant": "lr_1e-2",
                "model": "attention",
                "learning_rate": 1e-2,
            },
        ],
        "layers": [
            {
                "experiment_group": "num_layers",
                "variant": "layers_1",
                "model": "attention",
                "num_layers": 1,
            },
            {
                "experiment_group": "num_layers",
                "variant": "layers_2",
                "model": "attention",
                "num_layers": 2,
            },
            {
                "experiment_group": "num_layers",
                "variant": "layers_3",
                "model": "attention",
                "num_layers": 3,
            },
        ],
    }

    selected_configs: list[dict] = []
    if cli_args.suite == "full":
        for key in ["main", "heads", "vocab", "embed", "position", "lr", "layers"]:
            selected_configs.extend(suites[key])
    else:
        selected_configs.extend(suites[cli_args.suite])

    return [Namespace(**(common | config)) for config in selected_configs]


def save_summary(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    cli_args = parse_args()
    results = []
    for args in build_runs(cli_args):
        result = train(args)
        result["experiment_group"] = args.experiment_group
        result["variant"] = args.variant
        results.append(result)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_summary(results, Path(cli_args.output_dir) / f"experiment_summary_{timestamp}.csv")


if __name__ == "__main__":
    main()
