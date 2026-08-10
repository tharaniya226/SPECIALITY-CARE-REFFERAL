import pandas as pd

DATA_PATH = "data/ReferAI_10000_Synthetic_Patient_Dataset.xlsx"
df = pd.read_excel(DATA_PATH, sheet_name="Dataset")

print("===== SHAPE =====")
print(df.shape)
print("\n===== COLUMN NAMES =====")
print(list(df.columns))
print("\n===== DATA TYPES =====")
print(df.dtypes)
print("\n===== FIRST 5 ROWS =====")
print(df.head())
print("\n===== MISSING VALUES PER COLUMN =====")
print(df.isnull().sum())
print("\n===== TARGET COLUMN DISTRIBUTION (Specialist) =====")
print(df['Specialist'].value_counts())
print("\n===== NUMERIC COLUMN SUMMARY =====")
print(df.describe())
