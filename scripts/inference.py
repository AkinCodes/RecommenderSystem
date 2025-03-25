import torch
from models.dlrm import DLRMModel

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
print(f"Recommendation Score: {output.item()}")
