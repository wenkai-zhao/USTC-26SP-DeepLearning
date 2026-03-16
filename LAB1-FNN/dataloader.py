from dataclasses import dataclass

import torch
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset


@dataclass
class DataBundle:
    train_dataset: TensorDataset
    val_dataset: TensorDataset
    test_dataset: TensorDataset
    input_dim: int


def load_and_split_data(
    test_ratio: float = 0.2,
    val_ratio_from_train: float = 0.2,
    seed: int = 42,
) -> DataBundle:
    diabetes = load_diabetes()
    x = diabetes.data
    y = diabetes.target.reshape(-1, 1)

    x_train_full, x_test, y_train_full, y_test = train_test_split(
        x, y, test_size=test_ratio, random_state=seed
    )

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full,
        y_train_full,
        test_size=val_ratio_from_train,
        random_state=seed,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    train_dataset = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_dataset = TensorDataset(
        torch.tensor(x_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )
    test_dataset = TensorDataset(
        torch.tensor(x_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
    )

    return DataBundle(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        input_dim=x.shape[1],
    )
