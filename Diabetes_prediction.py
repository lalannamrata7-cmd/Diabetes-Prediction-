# Diabetes Prediction Project
# Author: Santoshi Metkel

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1️⃣ Load Dataset
data = pd.read_csv("Dataset_50_rows.csv")

print("First 5 Rows of Dataset:")
print(data.head())

# 2️⃣ Separate Features & Target
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# 3️⃣ Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4️⃣ Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5️⃣ Predictions
y_pred = model.predict(X_test)

# 6️⃣ Model Evaluation
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 7️⃣ Custom Prediction
print("\nEnter Patient Details for Prediction:")

pregnancies = int(input("Pregnancies: "))
glucose = int(input("Glucose Level: "))
blood_pressure = int(input("Blood Pressure: "))
skin_thickness = int(input("Skin Thickness: "))
insulin = int(input("Insulin Level: "))
bmi = float(input("BMI: "))
dpf = float(input("Diabetes Pedigree Function: "))
age = int(input("Age: "))

# Create input array
input_data = [[pregnancies, glucose, blood_pressure,
               skin_thickness, insulin, bmi, dpf, age]]

prediction = model.predict(input_data)

if prediction[0] == 1:
    print("\n🔴 The person is Diabetic")
else:
    print("\n🟢 The person is Not Diabetic")
