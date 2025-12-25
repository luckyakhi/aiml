import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. Define Transformations (Data Preprocessing)
# ToTensor(): Converts images to PyTorch Tensors and scales pixels from 0-255 to 0-1.
transform = transforms.Compose([transforms.ToTensor()])

# 2. Download and Load Data
# train=True (Training Data), train=False (Test Data)
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 3. Create DataLoaders (The Batch Processor)
# batch_size=64: Instead of processing 1 image at a time, we process 64 in parallel.
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

print(f"Training Data: {len(train_dataset)} images")
print(f"Test Data: {len(test_dataset)} images")

from simple_net import SimpleNet

# Instantiate the model
model = SimpleNet()
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01) # lr = Learning Rate (Step size)

# Train for 5 epochs (go through the entire dataset 5 times)
epochs = 5

for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        # 1. Zero the gradients (Reset error accumulation from previous batch)
        optimizer.zero_grad()
        
        # 2. Forward Pass (Make predictions)
        outputs = model(images)
        
        # 3. Calculate Loss (Compare predictions vs actual labels)
        loss = criterion(outputs, labels)
        
        # 4. Backward Pass (Calculate gradients/slopes)
        loss.backward()
        
        # 5. Optimization (Update weights)
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

print("Training Complete!")

correct = 0
total = 0

# torch.no_grad(): Tell PyTorch "We are testing, don't calculate gradients/math for updates" (Saves Memory)
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        # torch.max returns (max_value, index_of_max_value)
        # The index (0-9) IS our prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy of the network on the 10,000 test images: {100 * correct / total:.2f}%")