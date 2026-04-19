import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 200) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class AttentionClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.2,
        max_seq_len: int = 200,
        pad_idx: int = 0,
        use_positional_encoding: bool = True,
    ) -> None:
        super().__init__()
        self.pad_idx = pad_idx
        self.use_positional_encoding = use_positional_encoding
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(embed_dim, max_seq_len)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "attn": nn.MultiheadAttention(
                            embed_dim=embed_dim,
                            num_heads=num_heads,
                            dropout=dropout,
                            batch_first=True,
                        ),
                        "norm1": nn.LayerNorm(embed_dim),
                        "ffn": nn.Sequential(
                            nn.Linear(embed_dim, hidden_dim),
                            nn.ReLU(),
                            nn.Dropout(dropout),
                            nn.Linear(hidden_dim, embed_dim),
                        ),
                        "norm2": nn.LayerNorm(embed_dim),
                    }
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        padding_mask = x.eq(self.pad_idx)
        x = self.embedding(x)
        if self.use_positional_encoding:
            x = self.pos_encoder(x)
        x = self.dropout(x)

        for layer in self.layers:
            attn_out, _ = layer["attn"](x, x, x, key_padding_mask=padding_mask, need_weights=False)
            x = layer["norm1"](x + attn_out)
            ffn_out = layer["ffn"](x)
            x = layer["norm2"](x + ffn_out)

        pooled = x[:, -1, :]
        return self.classifier(pooled).squeeze(1)


class RNNClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.2,
        pad_idx: int = 0,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        output_dim = hidden_dim * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(output_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        outputs, _ = self.rnn(x)
        pooled = self.dropout(outputs[:, -1, :])
        return self.classifier(pooled).squeeze(1)


def build_model(
    model_name: str,
    vocab_size: int,
    max_seq_len: int,
    pad_idx: int,
    embed_dim: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    num_heads: int = 4,
    use_positional_encoding: bool = True,
) -> nn.Module:
    if model_name == "attention":
        return AttentionClassifier(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            max_seq_len=max_seq_len,
            pad_idx=pad_idx,
            use_positional_encoding=use_positional_encoding,
        )
    if model_name == "rnn":
        return RNNClassifier(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            pad_idx=pad_idx,
        )
    raise ValueError(f"Unsupported model: {model_name}")
