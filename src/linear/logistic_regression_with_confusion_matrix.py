import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load the dataset
data = load_breast_cancer()

# Convert to DataFrame for easier viewing
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target  # 0 = Malignant, 1 = Benign

# Check the first few rows
print(df[['mean radius', 'mean texture', 'target']].head())
print(f"Dataset shape: {df.shape}")

# 1. Split Data
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Instantiate Model
# max_iter=3000 ensures the solver has enough time to find the optimal weights
model = LogisticRegression(max_iter=3000)

# 3. Fit (Train)
model.fit(X_train, y_train)

# 4. Predict
predictions = model.predict(X_test)

# 5. Evaluate
# Accuracy = (Correct Predictions) / (Total Predictions)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy * 100:.2f}%")



# 1. Generate the Confusion Matrix
cm = confusion_matrix(y_test, predictions)

# 2. Visualize it nicely
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Malignant (0)', 'Benign (1)'],
            yticklabels=['Malignant (0)', 'Benign (1)'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# 3. Print the full report
print(classification_report(y_test, predictions, target_names=['Malignant', 'Benign']))