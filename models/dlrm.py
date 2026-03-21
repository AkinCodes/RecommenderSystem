"""Deep Learning Recommendation Model (DLRM).

A PyTorch implementation of the DLRM architecture that combines continuous
and categorical features through embeddings and an MLP to produce a
recommendation score between 0 and 1.
"""

import logging
import os

import torch
import torch.nn as nn

# Force Torch to use CPU mode
torch.set_num_threads(1)
os.environ["TORCHREC_DISABLE_FBGEMM"] = "1"

logger = logging.getLogger(__name__)


class DLRMModel(nn.Module):
    """Deep Learning Recommendation Model.

    Args:
        num_features: Number of continuous (dense) input features.
        embedding_sizes: List of vocabulary sizes for each categorical feature.
        mlp_layers: List of hidden-layer dimensions for the interaction MLP.
    """

    def __init__(self, num_features: int, embedding_sizes: list[int], mlp_layers: list[int]):
        super().__init__()

        self.continuous_layer = nn.Linear(num_features, mlp_layers[0])

        self.embeddings = nn.ModuleList(
            [nn.Embedding(size, mlp_layers[0]) for size in embedding_sizes]
        )

        total_embedding_size = sum(emb.embedding_dim for emb in self.embeddings)
        mlp_input_dim = self.continuous_layer.out_features + total_embedding_size

        layers: list[nn.Module] = [nn.Linear(mlp_input_dim, mlp_layers[1]), nn.ReLU()]
        for i in range(1, len(mlp_layers) - 1):
            layers.append(nn.Linear(mlp_layers[i], mlp_layers[i + 1]))
            layers.append(nn.ReLU())

        self.mlp = nn.Sequential(*layers)
        self.output_layer = nn.Linear(mlp_layers[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, continuous_features: torch.Tensor, categorical_features: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through the DLRM.

        Args:
            continuous_features: Float tensor of shape ``(batch, num_features)``.
            categorical_features: Long tensor of shape ``(batch, num_categorical)``.

        Returns:
            Tensor of shape ``(batch, 1)`` with values in ``[0, 1]``.

        Raises:
            ValueError: If inputs are ``None`` or if the number of categorical
                features exceeds the number of embedding tables.
        """
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

        x = torch.cat([x] + cat_embeds, dim=1)

        if x.shape[-1] != self.mlp[0].in_features:
            raise ValueError(
                f"MLP input shape mismatch: expected {self.mlp[0].in_features}, "
                f"got {x.shape[-1]}"
            )

        x = self.mlp(x)
        return self.sigmoid(self.output_layer(x))
