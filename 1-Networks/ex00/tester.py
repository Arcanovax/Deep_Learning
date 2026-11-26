import torch
from mlp import MLP


torch.manual_seed(42)
model = MLP(4, 8, 3)
print(model)
out = model(torch.randn(5, 4))
print("Output shape:", out.shape)
