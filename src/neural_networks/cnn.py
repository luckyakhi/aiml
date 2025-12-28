import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # Layer 1: Convolution
        # in_channels=1 (Black & White image) so one number per pixel, out_channels=32 (32 different filters)
        # kernel_size=3 (3x3 scanning window)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        
        # Layer 2: Convolution
        # Input is now 32 (from previous layer), output is 64 filters
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # Pooling Layer (2x2 window)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully Connected Layer (Classifier)
        # Math: 7 * 7 image size * 64 filters
        self.fc1 = nn.Linear(7 * 7 * 64, 128)
        self.fc2 = nn.Linear(128, 10) # 10 digits

    def forward(self, x):
        # 1. Conv1 -> ReLU -> Pool
        # Input: [Batch, 1, 28, 28] -> Output: [Batch, 32, 14, 14]
        x = self.pool(F.relu(self.conv1(x)))
        
        # 2. Conv2 -> ReLU -> Pool
        # Input: [Batch, 32, 14, 14] -> Output: [Batch, 64, 7, 7]
        x = self.pool(F.relu(self.conv2(x)))
        
        # 3. Flatten (Unroll the 3D cube into 1D vector)
        # -1 means "calculate this dimension automatically based on batch size"
        x = x.view(-1, 7 * 7 * 64)
        
        # 4. Classification
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = ConvNet()
print(model)

# Create a dummy image (Batch size 1, 1 Color Channel, 28x28 pixels)
dummy_input = torch.randn(1, 1, 28, 28)

# Pass it through the model
output = model(dummy_input)

print(f"Input Shape:  {dummy_input.shape}")
print(f"Output Shape: {output.shape}") 
# Expected Output: [1, 10] -> Probabilities for digits 0-9