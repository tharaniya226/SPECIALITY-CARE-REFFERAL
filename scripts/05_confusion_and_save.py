import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


df = pd.read_csv("data/processed_dataset.csv")

X = df.drop(columns=["Specialist"])
y = df["Specialist"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
labels = sorted(y.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)

cm_df = pd.DataFrame(cm, index=labels, columns=labels)
print("Confusion Matrix (rows = actual, columns = predicted):")
print(cm_df)


joblib.dump(model, "models/referai_model.pkl")
print("\nModel saved to models/referai_model.pkl")