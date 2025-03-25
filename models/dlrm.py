import torch
import torch.nn as nn
import os

torch.set_num_threads(1)
os.environ["TORCHREC_DISABLE_FBGEMM"] = "1"

import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F


class DLRMModel(nn.Module):
    def __init__(self, num_features, embedding_sizes, mlp_layers):
        super(DLRMModel, self).__init__()

        # Linear layer for continuous features
        self.continuous_layer = nn.Linear(num_features, mlp_layers[0])

        # Embeddings for categorical features
        self.embeddings = nn.ModuleList(
            [nn.Embedding(size, mlp_layers[0]) for size in embedding_sizes]
        )

        total_embedding_size = sum([emb.embedding_dim for emb in self.embeddings])
        mlp_input_dim = self.continuous_layer.out_features + total_embedding_size

        layers = [nn.Linear(mlp_input_dim, mlp_layers[1]), nn.ReLU()]

        for i in range(1, len(mlp_layers) - 1):
            layers.append(nn.Linear(mlp_layers[i], mlp_layers[i + 1]))
            layers.append(nn.ReLU())

        self.mlp = nn.Sequential(*layers)

        self.output_layer = nn.Linear(mlp_layers[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, continuous_features, categorical_features):
        if continuous_features is None or categorical_features is None:
            raise ValueError("🔴 Forward Pass Error: Inputs cannot be None!")

        x = self.continuous_layer(continuous_features)

        batch_size, num_categorical_features = categorical_features.shape
        if num_categorical_features > len(self.embeddings):
            raise ValueError(
                f"🛑 Too many categorical features: {num_categorical_features}, expected {len(self.embeddings)}"
            )

        offsets = torch.zeros_like(categorical_features[:, 0])

        cat_embeds = [
            emb(categorical_features[:, i])
            for i, emb in enumerate(self.embeddings[:num_categorical_features])
        ]

        print(f"🔵 Continuous Features Shape: {x.shape}")
        print(f"🔵 Number of categorical embeddings: {len(cat_embeds)}")
        print(f"🔵 Categorical Embeddings Shapes: {[e.shape for e in cat_embeds]}")
        print(f"🔵 Total Embedding Size: {sum(e.shape[-1] for e in cat_embeds)}")
        print(f"🔵 Expected Input Size for MLP: {self.mlp[0].in_features}")

        x = torch.cat([x] + cat_embeds, dim=1)

        print(f"🔵 Shape of input to MLP: {x.shape}")

        if x.shape[-1] != self.mlp[0].in_features:
            raise ValueError(
                f"🛑 Shape mismatch: Expected {self.mlp[0].in_features}, but got {x.shape[-1]}"
            )

        x = self.mlp(x)
        return self.sigmoid(self.output_layer(x))
