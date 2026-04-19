import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


@dataclass
class TextDataBundle:
    train_texts: list[str]
    val_texts: list[str]
    test_texts: list[str]
    train_labels: list[int]
    val_labels: list[int]
    test_labels: list[int]
    vocab: dict[str, int]
    max_seq_len: int


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_data(csv_path: str | Path = "data/IMDB Dataset.csv") -> pd.DataFrame:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"review": "text", "sentiment": "label"})
    df["text"] = df["text"].astype(str).map(normalize_text)
    df["label"] = df["label"].map({"negative": 0, "positive": 1}).astype(int)
    return df[["text", "label"]]


def build_vocab(texts: Iterable[str], max_vocab_size: int) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(text.split())

    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, _ in counter.most_common(max_vocab_size - len(vocab)):
        vocab[token] = len(vocab)
    return vocab


def text_to_indices(text: str, vocab: dict[str, int], max_seq_len: int) -> list[int]:
    indices = [vocab.get(token, vocab[UNK_TOKEN]) for token in text.split()]
    indices = indices[:max_seq_len]
    if len(indices) < max_seq_len:
        pad_len = max_seq_len - len(indices)
        indices = [vocab[PAD_TOKEN]] * pad_len + indices
    return indices


def encode_texts(
    texts: Iterable[str],
    vocab: dict[str, int],
    max_seq_len: int,
) -> torch.Tensor:
    encoded = [text_to_indices(text, vocab, max_seq_len) for text in texts]
    return torch.tensor(encoded, dtype=torch.long)


def load_and_split_data(
    csv_path: str | Path = "data/IMDB Dataset.csv",
    max_vocab_size: int = 30000,
    max_seq_len: int = 200,
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
    seed: int = 42,
    sample_ratio: float = 1.0,
) -> TextDataBundle:
    df = load_data(csv_path)

    if sample_ratio < 1.0:
        df = df.sample(frac=sample_ratio, random_state=seed).reset_index(drop=True)

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df["text"].tolist(),
        df["label"].tolist(),
        test_size=test_ratio,
        random_state=seed,
        stratify=df["label"].tolist(),
    )
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts,
        train_labels,
        test_size=val_ratio,
        random_state=seed,
        stratify=train_labels,
    )

    vocab = build_vocab(train_texts, max_vocab_size)
    return TextDataBundle(
        train_texts=train_texts,
        val_texts=val_texts,
        test_texts=test_texts,
        train_labels=train_labels,
        val_labels=val_labels,
        test_labels=test_labels,
        vocab=vocab,
        max_seq_len=max_seq_len,
    )


def create_dataloaders(
    data: TextDataBundle,
    batch_size: int = 64,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = TensorDataset(
        encode_texts(data.train_texts, data.vocab, data.max_seq_len),
        torch.tensor(data.train_labels, dtype=torch.float32),
    )
    val_dataset = TensorDataset(
        encode_texts(data.val_texts, data.vocab, data.max_seq_len),
        torch.tensor(data.val_labels, dtype=torch.float32),
    )
    test_dataset = TensorDataset(
        encode_texts(data.test_texts, data.vocab, data.max_seq_len),
        torch.tensor(data.test_labels, dtype=torch.float32),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader


def describe_data(data: TextDataBundle) -> None:
    lengths = [len(text.split()) for text in data.train_texts]
    print("Dataset summary:")
    print(f"  Train/Val/Test: {len(data.train_texts)}/{len(data.val_texts)}/{len(data.test_texts)}")
    print(f"  Vocabulary size: {len(data.vocab)}")
    print(f"  Sequence length: {data.max_seq_len}")
    print(f"  Mean train length: {sum(lengths) / len(lengths):.2f}")
    print(f"  Max train length: {max(lengths)}")


if __name__ == "__main__":
    bundle = load_and_split_data()
    describe_data(bundle)
