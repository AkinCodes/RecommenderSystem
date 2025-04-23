import os
import torch
import torch.nn as nn

# Optional: Disable multi-threading and FBGEMM if needed
torch.set_num_threads(1)
os.environ["TORCHREC_DISABLE_FBGEMM"] = "1"


class DLRMModel(nn.Module):
    def __init__(
        self, num_continuous_features, embedding_sizes, mlp_layers, debug=False
    ):
        super(DLRMModel, self).__init__()

        self.debug = debug
        self.continuous_layer = nn.Linear(num_continuous_features, mlp_layers[0])
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings, mlp_layers[0])
                for num_embeddings in embedding_sizes
            ]
        )

        total_embedding_dim = sum(emb.embedding_dim for emb in self.embeddings)
        mlp_input_dim = self.continuous_layer.out_features + total_embedding_dim

        layers = [nn.Linear(mlp_input_dim, mlp_layers[1]), nn.ReLU()]
        for i in range(1, len(mlp_layers) - 1):
            layers += [nn.Linear(mlp_layers[i], mlp_layers[i + 1]), nn.ReLU()]
        self.mlp = nn.Sequential(*layers)

        self.output_layer = nn.Linear(mlp_layers[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, continuous_features, categorical_features):
        if continuous_features is None or categorical_features is None:
            raise ValueError("🔴 Forward Pass Error: Inputs cannot be None!")

        x = self.continuous_layer(continuous_features)

        if categorical_features.shape[1] > len(self.embeddings):
            raise ValueError(
                f"🛑 Too many categorical features: {categorical_features.shape[1]}, "
                f"expected {len(self.embeddings)}"
            )

        # Embed categorical features
        cat_embeds = [
            emb(categorical_features[:, i])
            for i, emb in enumerate(self.embeddings[: categorical_features.shape[1]])
        ]

        # Optional debug information
        if self.debug:
            print(f"🔵 Continuous Shape: {x.shape}")
            print(f"🔵 Embedding Shapes: {[e.shape for e in cat_embeds]}")
            print(
                f"🔵 Combined Input Size: {x.shape[-1] + sum(e.shape[-1] for e in cat_embeds)}"
            )

        # Concatenate continuous and embedded categorical features
        x = torch.cat([x] + cat_embeds, dim=1)

        if x.shape[-1] != self.mlp[0].in_features:
            raise ValueError(
                f"🛑 Shape mismatch: Expected {self.mlp[0].in_features}, got {x.shape[-1]}"
            )

        x = self.mlp(x)
        return self.sigmoid(self.output_layer(x))
