import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

# 1. Load Data (8x8 pixel images of handwritten digits)
digits = load_digits()
X = digits.data
y = digits.target

# Visualize what the data looks like
plt.gray()
plt.matshow(digits.images[0])
plt.title(f"Target: {digits.target[0]}")
plt.show()

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Instantiate
# n_estimators: Number of trees
# n_jobs=-1: Use ALL CPU cores (Java ForkJoinPool equivalent)
# random_state: For reproducibility explain:  
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# 2. Fit (This trains 100 trees in parallel)
model.fit(X_train, y_train)

# 3. Predict
predictions = model.predict(X_test)

# 4. Evaluate
print(f"Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")

# The model tracks which pixels contributed most to the correct votes
import pandas as pd

feature_importances = pd.Series(model.feature_importances_)

# Visualize the importance as an 8x8 heatmap (just like the image)
plt.figure(figsize=(6, 6))
sns.heatmap(feature_importances.values.reshape(8, 8), cmap='hot', square=True)
plt.title("Pixel Importance Heatmap")
plt.show()