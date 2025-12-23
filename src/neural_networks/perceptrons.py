import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=100):
        # Initialize weights to zeros (or small random numbers)
        # We need one weight per input, plus one for Bias (which we treat as weight[0])
        self.weights = np.zeros(input_size + 1) 
        self.lr = learning_rate
        self.epochs = epochs

    def activation(self, x):
        # Simple Step Function: Return 1 if x >= 0, else 0
        return 1 if x >= 0 else 0

    def predict(self, x):
        # Calculate Weighted Sum: z = w1*x1 + w2*x2 + ... + bias
        # We add [1] to x to handle the bias term easily (Standard trick)
        z = np.dot(x, self.weights[1:]) + self.weights[0]
        return self.activation(z)

    def fit(self, X, y):
        # The Training Loop
        for _ in range(self.epochs):
            for i in range(y.shape[0]):
                # 1. Forward Pass (Make a guess)
                prediction = self.predict(X[i])
                
                # 2. Calculate Error (Actual - Predicted)
                # If correct (1-1=0), error is 0. No update.
                # If wrong (1-0=1), error is 1. Weights increase.
                error = y[i] - prediction
                
                # 3. Backward Pass (Update Weights)
                # Weight = Weight + (LearningRate * Error * Input)
                self.weights[1:] += self.lr * error * X[i]
                self.weights[0]  += self.lr * error # Update Bias

# 1. Define Data (OR Gate Logic)
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([0, 1, 1, 1]) # The labels for OR

# 2. Instantiate and Train
neuron = Perceptron(input_size=2)
neuron.fit(X, y)

# 3. Test it
print("--- Final Weights ---")
print(f"Bias: {neuron.weights[0]}")
print(f"W1:   {neuron.weights[1]}")
print(f"W2:   {neuron.weights[2]}")

print("\n--- Predictions ---")
for inputs in X:
    print(f"Input: {inputs} -> Predicted: {neuron.predict(inputs)}")