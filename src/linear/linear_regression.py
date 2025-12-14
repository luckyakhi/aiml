import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 1. Generate synthetic data (50 employees)
# Random experience between 1 and 20 years
np.random.seed(42) # ensures we get same random numbers every time
X = 20 * np.random.rand(50, 1)  # An 1D array of 50 random numbers between 0 and 20

# Base salary 30k + 4k per year of experience + some random noise
y = 30000 + (4000 * X) + (np.random.randn(50, 1) * 5000) # An 1D array of 50 random numbers for salary

# Convert to DataFrame just so it looks familiar
df = pd.DataFrame({'Years_Exp': X.flatten(), 'Salary': y.flatten()}) # A data frame is row and column of data with keys as column names

# Visualize it
plt.scatter(df['Years_Exp'], df['Salary'], color='blue')
print("Data type of panda dataframe column lookup "+ str(type(df['Years_Exp'])))
plt.title("Employee Data")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

# 1. Split the data (80% training, 20% testing)
# X_train, y_train: The textbooks the student studies
# X_test, y_test: The exam questions
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #Meaning of random_state=42 is that it will always give same result, 42 has no special meaning

# 2. Instantiate the model
model = LinearRegression()

# 3. Fit (Train) the model
# This is where the computer calculates the best 'm' and 'c'
model.fit(X_train, y_train)

# 4. Inspect what it learned
print(f"Base Salary (Bias/Intercept): ${model.intercept_[0]:.2f}")
print(f"Raise per Year (Weight/Slope): ${model.coef_[0][0]:.2f}")

# Make predictions on the test set
predictions = model.predict(X_test)

# Compare prediction vs actual for the first 5 people in test set
comparison = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': predictions.flatten()})
print(comparison.head())

# Visualizing the Regression Line
plt.scatter(X_test, y_test, color='black', label='Actual Data')
plt.plot(X_test, predictions, color='red', linewidth=3, label='Model Prediction')
plt.title("Linear Regression Model")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.legend()
plt.show()