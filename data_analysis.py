import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_and_inspect(filename):
    # Load dataset
    df = pd.read_csv(filename)
    
    print("--- Step 1: Inspection ---")
    print("Columns:", df.columns.tolist())
    print("\nData Types:")
    print(df.dtypes)
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nBasic Stats:")
    print(df.describe())
    
    return df

def clean_data(df):
    print("\n--- Step 2: Data Cleaning ---")
    # Handle missing values: Drop rows with missing data for simplicity
    df_clean = df.dropna().copy()
    
    # Remove unusable columns (IDs are not features)
    if 'Person ID' in df_clean.columns:
        df_clean = df_clean.drop(columns=['Person ID'])
        print("Dropped 'Person ID' column.")
        
    print(f"Data shape after cleaning: {df_clean.shape}")
    return df_clean
