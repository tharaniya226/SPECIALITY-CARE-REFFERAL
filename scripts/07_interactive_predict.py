import pandas as pd
import joblib


model = joblib.load("models/referai_model.pkl")


reference = pd.read_csv("data/processed_dataset.csv")
feature_columns = reference.drop(columns=["Specialist"]).columns


numeric_cols = ["Age", "Height_cm", "Weight_kg", "BMI", "Systolic_BP",
                 "Diastolic_BP", "Heart_Rate", "Temperature", "Oxygen_Saturation"]
onehot_prefixes = ["Gender_", "Smoking_Status_", "Alcohol_Use_"]

symptom_cols = [
    col for col in feature_columns
    if col not in numeric_cols and not any(col.startswith(p) for p in onehot_prefixes)
]

print("=== ReferAI: Patient Intake ===\n")


patient = {}
patient["Age"] = int(input("Age: "))
patient["Height_cm"] = float(input("Height (cm): "))
patient["Weight_kg"] = float(input("Weight (kg): "))
patient["BMI"] = round(patient["Weight_kg"] / ((patient["Height_cm"] / 100) ** 2), 1)
patient["Systolic_BP"] = int(input("Systolic BP: "))
patient["Diastolic_BP"] = int(input("Diastolic BP: "))
patient["Heart_Rate"] = int(input("Heart Rate: "))
patient["Temperature"] = float(input("Temperature (C): "))
patient["Oxygen_Saturation"] = int(input("Oxygen Saturation (%): "))


gender = input("Gender (Male/Female): ").strip().capitalize()
patient["Gender_Male"] = (gender == "Male")
patient["Gender_Female"] = (gender == "Female")


smoking = input("Smoking Status (Never/Current/Former): ").strip().capitalize()
patient["Smoking_Status_Never"] = (smoking == "Never")
patient["Smoking_Status_Current"] = (smoking == "Current")
patient["Smoking_Status_Former"] = (smoking == "Former")


alcohol = input("Alcohol Use (Never/Occasional/Regular): ").strip().capitalize()
patient["Alcohol_Use_Never"] = (alcohol == "Never")
patient["Alcohol_Use_Occasional"] = (alcohol == "Occasional")
patient["Alcohol_Use_Regular"] = (alcohol == "Regular")


print("\n--- Symptoms (answer y/n) ---")
for symptom in symptom_cols:
    answer = input(f"{symptom.replace('_', ' ')}? (y/n): ").strip().lower()
    patient[symptom] = 1 if answer == "y" else 0


patient_df = pd.DataFrame([patient])[feature_columns]
prediction = model.predict(patient_df)[0]
probabilities = model.predict_proba(patient_df)[0]

print("\n=== Result ===")
print("Predicted Specialist:", prediction)
print("\nConfidence per specialist:")
for specialist, prob in sorted(zip(model.classes_, probabilities), key=lambda x: -x[1]):
    print(f"  {specialist}: {prob:.2%}")