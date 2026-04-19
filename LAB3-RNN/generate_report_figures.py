from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = ROOT / "outputs"
ASSETS_DIR = ROOT / "assets"


def load_experiment_results() -> pd.DataFrame:
    summary_paths = sorted(OUTPUTS_DIR.glob("experiment_summary_*.csv"))
    if not summary_paths:
        raise FileNotFoundError("No experiment_summary_*.csv files found in outputs/")

    frames = [pd.read_csv(path) for path in summary_paths]
    results = pd.concat(frames, ignore_index=True)
    results = results[(results["epochs"] == 20) & (results["sample_ratio"] == 1.0)].copy()
    results = results.sort_values("run_name")
    results = results.groupby(["experiment_group", "variant"], as_index=False).tail(1)
    return results


def load_main_histories(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    main = results[results["experiment_group"] == "main_compare"]
    rnn_row = main[main["variant"] == "rnn_baseline"].iloc[-1]
    attention_row = main[main["variant"] == "attention_baseline"].iloc[-1]

    rnn_history = pd.read_csv(ROOT / rnn_row["output_dir"] / "history.csv")
    attention_history = pd.read_csv(ROOT / attention_row["output_dir"] / "history.csv")
    return rnn_history, attention_history


def setup_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["figure.dpi"] = 160
    plt.rcParams["savefig.dpi"] = 200
    plt.rcParams["axes.unicode_minus"] = False


def save_training_curves(rnn_history: pd.DataFrame, attention_history: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(rnn_history["epoch"], rnn_history["val_f1"], marker="o", label="RNN")
    axes[0].plot(
        attention_history["epoch"], attention_history["val_f1"], marker="o", label="Attention"
    )
    axes[0].set_title("Validation F1 Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Validation F1")
    axes[0].legend()

    axes[1].plot(rnn_history["epoch"], rnn_history["val_loss"], marker="o", label="RNN")
    axes[1].plot(
        attention_history["epoch"], attention_history["val_loss"], marker="o", label="Attention"
    )
    axes[1].set_title("Validation Loss Curves")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation Loss")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "training_curves.png", bbox_inches="tight")
    plt.close(fig)


def save_main_comparison(results: pd.DataFrame) -> None:
    main = results[results["experiment_group"] == "main_compare"].copy()
    main = main.set_index("model").loc[["rnn", "attention"]].reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].bar(main["model"], main["test_f1"], color=["#4C78A8", "#F58518"])
    axes[0].set_title("Main Experiment: Test F1")
    axes[0].set_ylabel("F1-score")
    axes[0].set_ylim(0.0, 1.0)

    axes[1].bar(main["model"], main["avg_epoch_time_sec"], color=["#4C78A8", "#F58518"])
    axes[1].set_title("Main Experiment: Avg Epoch Time")
    axes[1].set_ylabel("Seconds")

    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "main_comparison.png", bbox_inches="tight")
    plt.close(fig)


def save_group_bar_chart(
    results: pd.DataFrame,
    experiment_group: str,
    order: list[str],
    labels: list[str],
    title: str,
    filename: str,
    metric: str = "test_f1",
) -> None:
    subset = results[results["experiment_group"] == experiment_group].copy()
    subset["variant"] = pd.Categorical(subset["variant"], categories=order, ordered=True)
    subset = subset.sort_values("variant")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bars = ax.bar(labels, subset[metric], color="#4C78A8")
    ax.set_title(title)
    ax.set_ylabel(metric.replace("_", " ").title())
    if metric.startswith("test_") and metric not in {"test_avg_batch_inference_ms", "test_samples_per_sec"}:
        ax.set_ylim(0.0, 1.0)

    for bar, value in zip(bars, subset[metric]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.4f}", ha="center")

    fig.tight_layout()
    fig.savefig(ASSETS_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def save_embed_tradeoff(results: pd.DataFrame) -> None:
    subset = results[results["experiment_group"] == "embed_dim"].copy()
    subset["label"] = subset["embed_dim"].astype(str)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    scatter = ax.scatter(
        subset["parameter_count"],
        subset["test_f1"],
        s=100,
        c=subset["avg_epoch_time_sec"],
        cmap="viridis",
    )
    for _, row in subset.iterrows():
        ax.annotate(
            f"embed={int(row['embed_dim'])}",
            (row["parameter_count"], row["test_f1"]),
            textcoords="offset points",
            xytext=(5, 5),
        )

    ax.set_title("Embedding Size Trade-off")
    ax.set_xlabel("Parameter Count")
    ax.set_ylabel("Test F1-score")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Avg Epoch Time (s)")

    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "embed_tradeoff.png", bbox_inches="tight")
    plt.close(fig)


def save_learning_rate_chart(results: pd.DataFrame) -> None:
    save_group_bar_chart(
        results,
        experiment_group="learning_rate",
        order=["lr_1e-4", "lr_1e-3", "lr_1e-2"],
        labels=["1e-4", "1e-3", "1e-2"],
        title="Learning Rate vs Test F1",
        filename="learning_rate_f1.png",
    )


def save_num_layers_chart(results: pd.DataFrame) -> None:
    save_group_bar_chart(
        results,
        experiment_group="num_layers",
        order=["layers_1", "layers_2", "layers_3"],
        labels=["1", "2", "3"],
        title="Number of Layers vs Test F1",
        filename="num_layers_f1.png",
    )


def main() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()

    results = load_experiment_results()
    rnn_history, attention_history = load_main_histories(results)

    save_training_curves(rnn_history, attention_history)
    save_main_comparison(results)
    save_group_bar_chart(
        results,
        experiment_group="attention_heads",
        order=["heads_2", "heads_4", "heads_8"],
        labels=["2", "4", "8"],
        title="Attention Heads vs Test F1",
        filename="attention_heads_f1.png",
    )
    save_group_bar_chart(
        results,
        experiment_group="vocab_size",
        order=["vocab_10000", "vocab_20000", "vocab_30000"],
        labels=["10000", "20000", "30000"],
        title="Vocabulary Size vs Test F1",
        filename="vocab_size_f1.png",
    )
    save_group_bar_chart(
        results,
        experiment_group="position_encoding",
        order=["pos_on", "pos_off"],
        labels=["With PE", "Without PE"],
        title="Positional Encoding vs Test F1",
        filename="position_encoding_f1.png",
    )
    save_embed_tradeoff(results)
    save_learning_rate_chart(results)
    save_num_layers_chart(results)

    print(f"Saved figures to {ASSETS_DIR}")


if __name__ == "__main__":
    main()
