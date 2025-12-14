import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Load Data
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
class_names = iris.target_names

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Features: {feature_names}")
print(f"Classes: {class_names}")

# 1. Instantiate
# criterion='gini' is the default metric for measuring purity
model = DecisionTreeClassifier(max_depth=3, criterion='gini', random_state=42)

# 2. Fit
model.fit(X_train, y_train)

# 3. Predict & Evaluate
predictions = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")

plt.figure(figsize=(12, 8))
plot_tree(model, 
          feature_names=feature_names, 
          class_names=list(class_names), # Convert numpy array to list
          filled=True, 
          rounded=True)
plt.show()