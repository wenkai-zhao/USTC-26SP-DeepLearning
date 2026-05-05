from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms


CIFAKE_MEAN = (0.4914, 0.4822, 0.4465)
CIFAKE_STD = (0.2470, 0.2435, 0.2616)


@dataclass
class DataBundle:
    train_dataset: Dataset
    val_dataset: Dataset
    test_dataset: Dataset
    class_names: list[str]
    input_shape: tuple[int, int, int]


class TwoViewDataset(Dataset):
    def __init__(self, dataset: Dataset, transform: Callable) -> None:
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = load_pil_image(self.dataset, index)
        return self.transform(image), self.transform(image)


def build_eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(CIFAKE_MEAN, CIFAKE_STD),
        ]
    )


def build_simclr_transform(augmentation: str = "strong") -> transforms.Compose:
    if augmentation == "weak":
        color_jitter = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        blur_prob = 0.0
        gray_prob = 0.05
    elif augmentation == "strong":
        color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        blur_prob = 0.2
        gray_prob = 0.2
    else:
        raise ValueError("augmentation must be 'weak' or 'strong'.")

    return transforms.Compose(
        [
            transforms.RandomResizedCrop(32, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=gray_prob),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=blur_prob),
            transforms.ToTensor(),
            transforms.Normalize(CIFAKE_MEAN, CIFAKE_STD),
        ]
    )


def _maybe_subsample(dataset: Dataset, sample_ratio: float, seed: int) -> Dataset:
    if not 0 < sample_ratio <= 1:
        raise ValueError("sample_ratio must be between 0 and 1.")
    if sample_ratio >= 1:
        return dataset

    subset_size = max(1, int(len(dataset) * sample_ratio))
    generator = torch.Generator().manual_seed(seed)
    subset, _ = random_split(dataset, [subset_size, len(dataset) - subset_size], generator)
    return subset


def get_targets(dataset: Dataset) -> list[int]:
    if isinstance(dataset, Subset):
        parent_targets = get_targets(dataset.dataset)
        return [parent_targets[i] for i in dataset.indices]
    if hasattr(dataset, "targets"):
        return list(dataset.targets)
    return [int(dataset[i][1]) for i in range(len(dataset))]


def load_pil_image(dataset: Dataset, index: int) -> Image.Image:
    if isinstance(dataset, Subset):
        return load_pil_image(dataset.dataset, int(dataset.indices[index]))
    if isinstance(dataset, datasets.ImageFolder):
        path, _ = dataset.samples[index]
        with Image.open(path) as image:
            return image.convert("RGB")
    image, _ = dataset[index]
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    raise TypeError("TwoViewDataset requires PIL images or an ImageFolder/Subset source.")


def stratified_labeled_subset(dataset: Dataset, fraction: float, seed: int) -> Subset:
    if not 0 < fraction <= 1:
        raise ValueError("labeled fraction must be between 0 and 1.")
    targets = get_targets(dataset)
    generator = torch.Generator().manual_seed(seed)
    selected: list[int] = []

    for label in sorted(set(targets)):
        indices = [i for i, y in enumerate(targets) if y == label]
        count = max(1, int(len(indices) * fraction))
        perm = torch.randperm(len(indices), generator=generator).tolist()
        selected.extend(indices[i] for i in perm[:count])

    selected.sort()
    return Subset(dataset, selected)


def load_datasets(
    data_dir: str | Path = "data",
    val_ratio: float = 0.1,
    sample_ratio: float = 1.0,
    seed: int = 42,
) -> DataBundle:
    data_dir = Path(data_dir)
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1.")

    transform = build_eval_transform()
    full_train = datasets.ImageFolder(data_dir / "train", transform=transform)
    full_test = datasets.ImageFolder(data_dir / "test", transform=transform)
    full_train = _maybe_subsample(full_train, sample_ratio, seed)
    full_test = _maybe_subsample(full_test, sample_ratio, seed + 1)

    val_size = max(1, int(len(full_train) * val_ratio))
    val_size = min(val_size, len(full_train) - 1)
    train_size = len(full_train) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full_train, [train_size, val_size], generator)
    sample_image, _ = full_train[0]

    base_dataset = full_train.dataset if isinstance(full_train, Subset) else full_train
    return DataBundle(
        train_dataset=train_set,
        val_dataset=val_set,
        test_dataset=full_test,
        class_names=list(base_dataset.classes),
        input_shape=tuple(sample_image.shape),
    )


def create_eval_loaders(
    data: DataBundle,
    batch_size: int,
    num_workers: int = 0,
    labeled_fraction: float = 1.0,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = stratified_labeled_subset(data.train_dataset, labeled_fraction, seed)
    kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2
    return (
        DataLoader(train_dataset, shuffle=True, **kwargs),
        DataLoader(data.val_dataset, shuffle=False, **kwargs),
        DataLoader(data.test_dataset, shuffle=False, **kwargs),
    )


def create_simclr_loader(
    data: DataBundle,
    batch_size: int,
    num_workers: int = 0,
    augmentation: str = "strong",
) -> DataLoader:
    dataset = TwoViewDataset(data.train_dataset, build_simclr_transform(augmentation))
    kwargs = {
        "batch_size": batch_size,
        "shuffle": True,
        "drop_last": True,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(
        dataset,
        **kwargs,
    )


def describe_data(data: DataBundle) -> None:
    print(f"train samples: {len(data.train_dataset)}")
    print(f"val samples: {len(data.val_dataset)}")
    print(f"test samples: {len(data.test_dataset)}")
    print(f"class names: {data.class_names}")
    print(f"input shape: {data.input_shape}")
