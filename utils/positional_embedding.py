from typing import Optional

import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    def __init__(self, sequence_length: int, input_dim: int, embed_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.token_embeddings = nn.Embedding(num_embeddings=input_dim, embedding_dim=embed_dim, padding_idx=0)
        self.position_embeddings = nn.Embedding(num_embeddings=sequence_length, embedding_dim=embed_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.sequence_length = sequence_length
        self.embed_dim = embed_dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(inputs.size(1), device=inputs.device)
        positions = positions.unsqueeze(0).expand_as(inputs)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return self.dropout(embedded_tokens + embedded_positions)

    def extra_repr(self) -> str:
        return f"sequence_length={self.sequence_length}, embed_dim={self.embed_dim}"
