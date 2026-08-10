import pandas as pd


df = pd.read_excel("data/ReferAI_10000_Synthetic_Patient_Dataset.xlsx", sheet_name="Dataset")


df = df.drop(columns=["Patient_ID"])


yes_no_columns = df.columns[df.isin(['Yes', 'No']).any()].tolist()
print("Yes/No columns found:", yes_no_columns)

for col in yes_no_columns:
    df[col] = df[col].map({'Yes': 1, 'No': 0})


categorical_columns = df.select_dtypes(include='object').columns.tolist()
categorical_columns.remove('Specialist')  # keep target separate for now
print("\nCategorical columns to encode:", categorical_columns)

df = pd.get_dummies(df, columns=categorical_columns)


print("\nFinal shape after preprocessing:", df.shape)
print("\nColumn names after preprocessing:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())


df.to_csv("data/processed_dataset.csv", index=False)
print("\nSaved cleaned dataset to data/processed_dataset.csv")