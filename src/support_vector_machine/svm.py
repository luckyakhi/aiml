from sklearn.datasets import make_moons
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# 1. Generate "Moons" data (two interlocking crescents)
# noise=0.3 makes it messy and hard to separate
X, y = make_moons(n_samples=200, noise=0.3, random_state=42)

# Visualize it - verify a straight line would fail here!
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm')
plt.title("Data that needs a CURVE, not a LINE")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid (The Menu)
param_grid = {
    'C': [0.1, 1, 10, 100],          # Try 4 different strengths
    'gamma': [1, 0.1, 0.01, 0.001],  # Controls how "curvy" the RBF kernel is
    'kernel': ['rbf', 'linear']      # Try both Straight Line and Curves
}

# 1. Instantiate the Grid Search
# refit=True means "Once you find the best one, retrain it on ALL data so I can use it immediately"
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)

# 2. Run the brute force loop
grid.fit(X_train, y_train)

# 3. The Results
print(f"\n The Best Params were: {grid.best_params_}")
print(f" The Best Accuracy was: {grid.best_score_:.2%}")

from sklearn.metrics import accuracy_score

predictions = grid.predict(X_test)
print(f"Test Set Accuracy: {accuracy_score(y_test, predictions):.2%}")