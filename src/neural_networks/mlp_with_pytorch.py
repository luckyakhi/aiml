import torch
import numpy as np

# Check if PyTorch is installed
print(f"PyTorch Version: {torch.__version__}")

# 1. Create a Tensor from a list (Like Arrays.asList)
data = [[1, 2], [3, 4]]
x_tensor = torch.tensor(data)

# 2. Create a Tensor of random numbers (Like constructing a Model)
weights = torch.rand(2, 2)  # 2x2 matrix of random floats

print("--- My First Tensor ---")
print(x_tensor)
print(f"Shape: {x_tensor.shape}")
print(f"Type: {x_tensor.dtype}")

# Check if GPU is available (In Colab, go to Runtime -> Change runtime type -> T4 GPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nRunning on: {device}")

# Move the tensor to the device
x_gpu = x_tensor.to(device)

print(f"Tensor is now on: {x_gpu.device}")