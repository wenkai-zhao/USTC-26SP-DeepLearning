import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
FIG_DIR = ROOT / "assets" / "report_figures"


def latest_suite_dir(prefix: str) -> Path:
    candidates = sorted(
        [path for path in OUTPUTS.glob(f"{prefix}_*") if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"No output directory matches {prefix}_*")
    return candidates[-1]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(value: str, default: float = 0.0) -> float:
    return float(value) if value not in {"", None} else default


def save_current_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def set_style() -> None:
    sns.set_theme(
        context="paper",
        style="whitegrid",
        font="DejaVu Sans",
        rc={
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 120,
        },
    )


def make_main_metric_chart() -> None:
    main_dir = latest_suite_dir("suite_main")
    rows = [
        row
        for row in read_csv(main_dir / "main_results.csv")
        if row["stage"] in {"linear_eval", "baseline"}
    ]
    plot_rows = []
    for row in rows:
        encoder = "ResNet-18" if row["encoder"] == "resnet18" else "MobileNetV2"
        method = "SimCLR" if row["stage"] == "linear_eval" else "Baseline"
        fraction = "1%" if fnum(row["labeled_fraction"]) < 0.02 else "10%"
        group = f"{encoder}\n{method}\n{fraction}"
        plot_rows.append({"Setting": group, "Metric": "Accuracy", "Score": fnum(row["test_accuracy"])})
        plot_rows.append({"Setting": group, "Metric": "Macro F1", "Score": fnum(row["test_f1_macro"])})

    plt.figure(figsize=(11, 4.8))
    ax = sns.barplot(data=pd.DataFrame(plot_rows), x="Setting", y="Score", hue="Metric", palette="Set2")
    ax.set_title("Main Results on CIFAKE Subset")
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_ylim(0.30, 0.85)
    ax.legend(loc="upper left", frameon=True)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=7, padding=2)
    save_current_figure(FIG_DIR / "main_accuracy_f1.png")


def make_ablation_chart() -> None:
    ablation_dir = latest_suite_dir("suite_ablations")
    rows = read_csv(ablation_dir / "ablation_results.csv")
    eval_rows = [row for row in rows if row["stage"] == "linear_eval"]
    labels = [
        "Contrastive",
        "Triplet",
        "BN head",
        "Wide head",
        "Temp 0.2",
        "Temp 1.0",
        "Weak aug",
    ]
    plot_rows = []
    for label, row in zip(labels, eval_rows):
        plot_rows.append({"Setting": label, "Metric": "Accuracy", "Score": fnum(row["test_accuracy"])})
        plot_rows.append({"Setting": label, "Metric": "Macro F1", "Score": fnum(row["test_f1_macro"])})

    plt.figure(figsize=(9.5, 4.8))
    ax = sns.barplot(data=pd.DataFrame(plot_rows), x="Setting", y="Score", hue="Metric", palette="Set1")
    ax.set_title("Ablation Results with ResNet-18 and 10% Labels")
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_ylim(0.60, 0.80)
    ax.legend(loc="upper left", frameon=True)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8, padding=2)
    save_current_figure(FIG_DIR / "ablation_accuracy_f1.png")


def history_series(run_dir: Path, metric: str) -> tuple[list[int], list[float]]:
    rows = read_csv(run_dir / "history.csv")
    epochs = [int(fnum(row["epoch"])) for row in rows if metric in row]
    values = [fnum(row[metric]) for row in rows if metric in row]
    return epochs, values


def make_pretrain_loss_chart() -> None:
    main_dir = latest_suite_dir("suite_main")
    rows = read_csv(main_dir / "main_results.csv")
    pretrain_rows = {row["encoder"]: row for row in rows if row["stage"] == "pretrain"}
    series = [
        (
            "ResNet-18",
            ROOT / pretrain_rows["resnet18"]["output_dir"],
        ),
        (
            "MobileNetV2",
            ROOT / pretrain_rows["mobilenet_v2"]["output_dir"],
        ),
    ]
    plt.figure(figsize=(8, 4.6))
    for label, run_dir in series:
        epochs, values = history_series(run_dir, "train_loss")
        sns.lineplot(x=epochs, y=values, label=label, linewidth=2.2)
    plt.title("SimCLR Pretraining Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.ylim(5.0, 7.1)
    plt.legend(frameon=True)
    save_current_figure(FIG_DIR / "pretrain_loss_curves.png")


def make_linear_eval_chart() -> None:
    main_dir = latest_suite_dir("suite_main")
    rows = read_csv(main_dir / "main_results.csv")

    def find_run(stage: str, encoder: str) -> Path:
        for row in rows:
            if (
                row["stage"] == stage
                and row["encoder"] == encoder
                and abs(fnum(row["labeled_fraction"]) - 0.10) < 1e-9
            ):
                return ROOT / row["output_dir"]
        raise FileNotFoundError(f"Missing {stage} {encoder} 10% run")

    series = [
        (
            "SimCLR ResNet-18 10%",
            find_run("linear_eval", "resnet18"),
        ),
        (
            "Baseline ResNet-18 10%",
            find_run("baseline", "resnet18"),
        ),
        (
            "SimCLR MobileNetV2 10%",
            find_run("linear_eval", "mobilenet_v2"),
        ),
        (
            "Baseline MobileNetV2 10%",
            find_run("baseline", "mobilenet_v2"),
        ),
    ]
    plt.figure(figsize=(8.8, 4.8))
    for label, run_dir in series:
        epochs, values = history_series(run_dir, "val_f1_macro")
        sns.lineplot(x=epochs, y=values, label=label, linewidth=2.2)
    plt.title("Linear Evaluation Validation F1")
    plt.xlabel("Epoch")
    plt.ylabel("Macro F1")
    plt.ylim(0.25, 0.90)
    plt.legend(frameon=True, ncol=2)
    save_current_figure(FIG_DIR / "linear_eval_f1_curves.png")


def main() -> None:
    set_style()
    make_main_metric_chart()
    make_ablation_chart()
    make_pretrain_loss_chart()
    make_linear_eval_chart()
    print(f"Saved matplotlib/seaborn figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
