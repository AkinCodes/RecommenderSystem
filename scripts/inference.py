import logging
import os
import sys

import torch

# Ensure Python can find the 'models' directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.dlrm import DLRMModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Load trained model
model = DLRMModel(
    num_features=10, embedding_sizes=[10, 10, 10, 10, 10], mlp_layers=[64, 32, 16]
)
model.load_state_dict(torch.load("trained_model.pth"))
model.eval()

# Sample Input
continuous_features = torch.randn(1, 10)
categorical_features = torch.randint(0, 5, (1, 5))

# Run Inference
output = model(continuous_features, categorical_features)
logger.info("Recommendation Score: %.4f", output.item())
