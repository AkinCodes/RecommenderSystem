import torch
from models.dlrm import DLRMModel

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.dlrm import DLRMModel


def test_dlrm_forward():
    model = DLRMModel(
        num_features=10, embedding_sizes=[10, 10, 10, 10, 10], mlp_layers=[64, 32, 16]
    )

    continuous_features = torch.randn(1, 10)
    categorical_features = torch.randint(0, 5, (1, 5))

    output = model(continuous_features, categorical_features)

    assert output.shape == torch.Size([1, 1]), ""
    assert 0 <= output.item() <= 1, ""


if __name__ == "__main__":
    test_dlrm_forward()
