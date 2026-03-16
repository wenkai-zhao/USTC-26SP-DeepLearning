import random
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer(model: nn.Module, name: str, lr: float) -> torch.optim.Optimizer:
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr)
    raise ValueError(f"Unsupported optimizer: {name}")


def plot_histories(
    result_rows: List[Dict], experiment_name: str, output_dir: Path
) -> Path:
    plt.figure(figsize=(10, 6))
    cmap = plt.get_cmap("tab10")
    for i, row in enumerate(result_rows):
        color = cmap(i % 10)
        hist_path = Path(row["history_csv"])
        hist_df = pd.read_csv(hist_path, encoding="utf-8")
        plt.plot(
            hist_df["epoch"],
            hist_df["train_loss"],
            linestyle="-",
            color=color,
            alpha=0.6,
            label=f"{row['run_name']} train",
        )
        plt.plot(
            hist_df["epoch"],
            hist_df["val_loss"],
            linestyle="--",
            color=color,
            alpha=0.9,
            label=f"{row['run_name']} val",
        )

    plt.title(f"{experiment_name} Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()

    figure_path = output_dir / "figures" / f"{experiment_name}_loss.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(figure_path, dpi=180)
    plt.close()
    return figure_path
