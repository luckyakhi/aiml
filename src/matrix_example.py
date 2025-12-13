import pandas as pd
import numpy as np

# Creating a dummy dataset (dictionary of lists)
data = {
    'Employee_ID': [101, 102, 103, 104, 105],
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Department': ['IT', 'HR', 'IT', 'Finance', 'IT'],
    'Salary': [85000, 60000, 92000, 75000, 88000],
    'Experience_Years': [5, 2, 7, 4, 6]
}

# Convert the dictionary to a DataFrame (The ML equivalent of a List of Objects)
df = pd.DataFrame(data)

#Multiply the Salary column by 1.1 (10% raise)
df['Salary'] = df['Salary'] * 1.1

# Display the first few rows
print(df.head())