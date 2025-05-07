import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from models.dlrm import DLRMModel

model = DLRMModel(
    num_continuous_features=2,
    embedding_sizes=[2, 18],
    mlp_layers=[64, 32, 16],
)

# Load weights
model.load_state_dict(torch.load("trained_model.pth", map_location=torch.device("cpu")))
model.eval()

# Sample Input
# release_year (normalized), parsed_duration (normalized)
continuous_features = torch.tensor([[0.3, -0.2]], dtype=torch.float32)

# categorical: type_index (0 = Movie), rating_index (e.g., 4)
categorical_features = torch.tensor([[0, 4]], dtype=torch.int64)

# Inference
with torch.no_grad():
    output = model(continuous_features, categorical_features)
    print(f"✅ Recommendation Score: {output.item():.4f}")
