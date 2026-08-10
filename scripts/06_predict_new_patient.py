import pandas as pd
import joblib


model = joblib.load("models/referai_model.pkl")


reference = pd.read_csv("data/processed_dataset.csv")
feature_columns = reference.drop(columns=["Specialist"]).columns


new_patient = {
    "Age": 45,
    "Height_cm": 170,
    "Weight_kg": 75,
    "BMI": 26.0,
    "Systolic_BP": 130,
    "Diastolic_BP": 85,
    "Heart_Rate": 88,
    "Temperature": 37.0,
    "Oxygen_Saturation": 97,
    "Gender_Male": True,
    "Gender_Female": False,
    "Smoking_Status_Never": True,
    "Smoking_Status_Current": False,
    "Smoking_Status_Former": False,
    "Alcohol_Use_Never": True,
    "Alcohol_Use_Occasional": False,
    "Alcohol_Use_Regular": False,
}


for col in feature_columns:
    if col not in new_patient:
        new_patient[col] = 0


new_patient["Chest_Pain"] = 1
new_patient["Breathlessness"] = 1


patient_df = pd.DataFrame([new_patient])[feature_columns]

# 5. Predict
prediction = model.predict(patient_df)[0]
probabilities = model.predict_proba(patient_df)[0]

print("Predicted Specialist:", prediction)
print("\nConfidence per specialist:")
for specialist, prob in zip(model.classes_, probabilities):
    print(f"  {specialist}: {prob:.2%}")