import pandas as pd
import sys

with open("initial_exploration.txt", "w") as f:
    sys.stdout = f
    
    df = pd.read_csv("Employee Turnover Dataset.csv")
    
    print("Dataset Info:\n")
    df.info(buf=f)
    
    print("\nSummary Statistics (Numerical):\n")
    print(df.describe(include=['number']))
    
    print("\nSummary Statistics (Categorical):\n")
    print(df.describe(include=['object']))

    bool_cols = df.select_dtypes(include=['bool'])
    if not bool_cols.empty:
        print("\nSummary Statistics (Boolean):\n")
        print(bool_cols.describe())

    print("\nSample Rows (Transposed):\n")
    print(df.head().T)

sys.stdout = sys.__stdout__

print("Initial exploration results saved to 'initial_exploration.txt'.")
