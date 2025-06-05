import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


print("✅ Starting test...")

from models.dlrm import DLRMModel

print("✅ Imported DLRMModel")

from scripts.preprocessing import load_and_apply_scaler, load_encoders

print("✅ Imported preprocessing utils")

import torch

print("✅ Imported torch")

x = torch.tensor([1.0, 2.0])
print("✅ Created tensor:", x)

model = DLRMModel(
    num_continuous_features=2, embedding_sizes=[2, 18], mlp_layers=[64, 32]
)
print("✅ Model created")

print("✅ All done.")
