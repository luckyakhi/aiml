import numpy as np
from sklearn.neural_network import MLPClassifier

# The XOR Data (0,0)->0, (1,1)->0, (0,1)->1, (1,0)->1
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [0, 1, 1, 0]

# 1. Instantiate the Network
# We use 'lbfgs' solver because it works better for tiny datasets like this
mlp = MLPClassifier(hidden_layer_sizes=(2,), activation='relu', solver='lbfgs', random_state=42)

# 2. Train
mlp.fit(X, y)

# 3. Test
print("--- Predictions for XOR ---")
predictions = mlp.predict(X)
for i, input_data in enumerate(X):
    print(f"Input: {input_data} -> Predicted: {predictions[i]} (Actual: {y[i]})")

