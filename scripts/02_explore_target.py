import pandas as pd

df = pd.read_excel("data/ReferAI_10000_Synthetic_Patient_Dataset.xlsx", sheet_name="Dataset")


print("Unique values in 'Chest_Pain':", df['Chest_Pain'].unique())
print("Unique values in 'Fever':", df['Fever'].unique())
print("Unique values in 'Smoking_Status':", df['Smoking_Status'].unique())
print("Unique values in 'Diabetes':", df['Diabetes'].unique())


print("\nSpecialist value counts:")
print(df['Specialist'].value_counts())


print("\nAverage Age and BMI by Specialist:")
print(df.groupby('Specialist')[['Age', 'BMI']].mean())