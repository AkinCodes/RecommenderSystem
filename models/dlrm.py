"""Deep Learning Recommendation Model (DLRM)."""

from __future__ import annotations

import logging
import os

import torch
import torch.nn as nn

torch.set_num_threads(1)
os.environ["TORCHREC_DISABLE_FBGEMM"] = "1"

logger = logging.getLogger(__name__)


class DLRMModel(nn.Module):
    """Deep Learning Recommendation Model."""

    def __init__(self, num_features: int, embedding_sizes: list[int], mlp_layers: list[int], dropout: float = 0.2):
        super().__init__()

        self.continuous_layer = nn.Linear(num_features, mlp_layers[0])

        self.embeddings = nn.ModuleList(
            [nn.Embedding(size, mlp_layers[0]) for size in embedding_sizes]
        )

        total_embedding_size = sum(emb.embedding_dim for emb in self.embeddings)
        interaction_size = self.embeddings[0].embedding_dim if len(self.embeddings) >= 2 else 0
        mlp_input_dim = self.continuous_layer.out_features + total_embedding_size + interaction_size

        layers: list[nn.Module] = [nn.Linear(mlp_input_dim, mlp_layers[1]), nn.ReLU(), nn.Dropout(dropout)]
        for i in range(1, len(mlp_layers) - 1):
            layers.append(nn.Linear(mlp_layers[i], mlp_layers[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        self.mlp = nn.Sequential(*layers)
        self.output_layer = nn.Linear(mlp_layers[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, continuous_features: torch.Tensor, categorical_features: torch.Tensor) -> torch.Tensor:
        if continuous_features is None or categorical_features is None:
            raise ValueError("Forward pass inputs cannot be None.")

        x = self.continuous_layer(continuous_features)

        batch_size, num_categorical = categorical_features.shape
        if num_categorical > len(self.embeddings):
            raise ValueError(
                f"Too many categorical features: got {num_categorical}, "
                f"expected at most {len(self.embeddings)}"
            )

        cat_embeds = [
            emb(categorical_features[:, i])
            for i, emb in enumerate(self.embeddings[:num_categorical])
        ]

        logger.debug("Continuous shape: %s", x.shape)
        logger.debug("Categorical embeddings: %d tensors", len(cat_embeds))

        if len(cat_embeds) >= 2:
            interaction = cat_embeds[0] * cat_embeds[1]
            x = torch.cat([x] + cat_embeds + [interaction], dim=1)
        else:
            x = torch.cat([x] + cat_embeds, dim=1)

        if x.shape[-1] != self.mlp[0].in_features:
            raise ValueError(
                f"MLP input shape mismatch: expected {self.mlp[0].in_features}, "
                f"got {x.shape[-1]}"
            )

        x = self.mlp(x)
        return self.sigmoid(self.output_layer(x))
