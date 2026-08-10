import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


df = pd.read_csv("data/processed_dataset.csv")

# 1. Split into features (X) and target (y)
X = df.drop(columns=["Specialist"])
y = df["Specialist"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training rows:", X_train.shape[0])
print("Testing rows:", X_test.shape[0])


model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("\nDetailed report:")
print(classification_report(y_test, y_pred))

# 5. Show which features matter most
importances = pd.Series(model.feature_importances_, index=X.columns)
print("\nTop 10 most important features:")
print(importances.sort_values(ascending=False).head(10))