#Generate a data frame from a dictionary
import pandas as pd
import numpy as np


#Create a dictionary
emp_data = {
    'Employee_ID': [101, 102, 103, 104, 105],
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Department': ['IT', 'HR', 'IT', 'Finance', 'IT'],
    'Salary': [85000, 60000, 92000, 75000, 88000],
    'Experience_Years': [5, 2, 7, 4, 6]
}
employee_df = pd.DataFrame(emp_data)
print(employee_df)
# Get numpy array from the data frame
employee_array = employee_df.to_numpy()
print(employee_array)