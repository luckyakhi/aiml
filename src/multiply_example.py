import pandas as pd
import numpy as np

def multiply_columns(df, multiplier):
    """
    Multiplies all numerical columns in the DataFrame by the given multiplier.
    """
    # Select numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    # Multiply
    df[numerical_cols] = df[numerical_cols] * multiplier
    return df

def main():
    # Create a sample DataFrame
    data = {
        'A': [1, 2, 3],
        'B': [4, 5, 6],
        'C': ['x', 'y', 'z'] # Non-numerical column to demonstrate safety
    }
    df = pd.DataFrame(data)
    
    print("Original DataFrame:")
    print(df)
    
    multiplier = 2
    print(f"\nMultiplying numerical columns by {multiplier}...")
    
    result_df = multiply_columns(df.copy(), multiplier)
    
    print("\nResult DataFrame:")
    print(result_df)

if __name__ == "__main__":
    main()
