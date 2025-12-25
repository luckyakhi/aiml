import torch.nn as nn
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        # Layer 1: Input (784) -> Hidden (128)
        self.fc1 = nn.Linear(28 * 28, 128) 
        # Layer 2: Hidden (128) -> Output (10)
        self.fc2 = nn.Linear(128, 10)
        # Activation Function
        self.relu = nn.ReLU()

    def forward(self, x):
        # 1. Flatten the image (convert 28x28 matrix to 784 vector)
        x = x.view(-1, 28 * 28) 
        # 2. Pass through first layer + Activation
        x = self.relu(self.fc1(x))
        # 3. Pass through output layer (No activation here, CrossEntropyLoss handles it)
        x = self.fc2(x)
        return x
