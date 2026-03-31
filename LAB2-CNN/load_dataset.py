from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms


@dataclass
class DataBundle:
    train_dataset: Dataset
    val_dataset: Dataset
    test_dataset: Dataset
    class_names: list[str]
    input_shape: tuple[int, int, int]


def compute_dataset_stats(data_dir: str | Path) -> tuple[float, float]:
    dataset = datasets.FashionMNIST(
        root=str(data_dir),
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    loader = DataLoader(dataset, batch_size=1024, shuffle=False)

    pixel_sum = 0.0
    squared_pixel_sum = 0.0
    num_pixels = 0

    for images, _ in loader:
        pixel_sum += images.sum().item()
        squared_pixel_sum += (images ** 2).sum().item()
        num_pixels += images.numel()

    mean = pixel_sum / num_pixels
    variance = squared_pixel_sum / num_pixels - mean**2
    std = variance**0.5
    return mean, std


def build_transform(data_dir: str | Path) -> transforms.Compose:
    mean, std = compute_dataset_stats(data_dir)
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((mean,), (std,)),
        ]
    )


def _maybe_subsample(dataset: Dataset, sample_ratio: float, seed: int) -> Dataset:
    if sample_ratio >= 1.0:
        return dataset

    subset_size = max(1, int(len(dataset) * sample_ratio))
    generator = torch.Generator().manual_seed(seed)
    subset, _ = random_split(dataset, [subset_size, len(dataset) - subset_size], generator)
    return subset


def load_and_split_data(
    data_dir: str | Path = "./data",
    val_ratio: float = 0.1,
    batch_size: int = 64,
    seed: int = 42,
    sample_ratio: float = 1.0,
    num_workers: int = 0,
) -> tuple[DataBundle, DataLoader, DataLoader, DataLoader]:
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1.")
    if not 0 < sample_ratio <= 1:
        raise ValueError("sample_ratio must be between 0 and 1.")

    # 设置随机种子
    torch.manual_seed(seed)

    # 数据预处理
    transform = build_transform(data_dir)

    # 加载完整数据集（全量）
    full_train = datasets.FashionMNIST(
        root=str(data_dir), train=True, download=True, transform=transform
    )
    full_test = datasets.FashionMNIST(
        root=str(data_dir), train=False, download=True, transform=transform
    )

    full_train = _maybe_subsample(full_train, sample_ratio=sample_ratio, seed=seed)
    full_test = _maybe_subsample(full_test, sample_ratio=sample_ratio, seed=seed + 1)

    # 将训练集划分为训练集和验证集
    val_size = max(1, int(len(full_train) * val_ratio))
    if val_size >= len(full_train):
        val_size = len(full_train) - 1
    train_size = len(full_train) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full_train, [train_size, val_size], generator)

    # 创建DataLoader
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        full_test, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    class_names = (
        full_train.dataset.classes if isinstance(full_train, Subset) else full_train.classes
    )
    sample_image, _ = full_train[0]
    data = DataBundle(
        train_dataset=train_set,
        val_dataset=val_set,
        test_dataset=full_test,
        class_names=list(class_names),
        input_shape=tuple(sample_image.shape),
    )
    return data, train_loader, val_loader, test_loader


def describe_data(data: DataBundle) -> None:
    print(f"训练样本数: {len(data.train_dataset)}")
    print(f"验证样本数: {len(data.val_dataset)}")
    print(f"测试样本数: {len(data.test_dataset)}")
    print(f"类别标签: {data.class_names}")
    print(f"类别数量: {len(data.class_names)}")
    print(f"\n图像形状: {data.input_shape}")


if __name__ == "__main__":
    bundle, _, _, _ = load_and_split_data()
    describe_data(bundle)
