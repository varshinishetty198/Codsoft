# titanic_project.py

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ---------------------------------------------------
# 1️⃣ Load Dataset
# ---------------------------------------------------
df = pd.read_csv("cleaned_titanic.csv")
print("✅ Dataset loaded successfully!")

# ---------------------------------------------------
# 2️⃣ Remove Unnecessary Columns
# ---------------------------------------------------
if "PassengerId" in df.columns:
    df.drop("PassengerId", axis=1, inplace=True)

# ---------------------------------------------------
# 3️⃣ Define Features and Target
# ---------------------------------------------------
X = df.drop("Survived", axis=1)
y = df["Survived"]

# ---------------------------------------------------
# 4️⃣ Train-Test Split
# ---------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------
# 5️⃣ Train Random Forest Model
# ---------------------------------------------------
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
print("✅ Model trained successfully!")

# ---------------------------------------------------
# 6️⃣ Save Model
# ---------------------------------------------------
joblib.dump(model, "titanic_random_forest_model.pkl")
print("✅ Model saved successfully!")

# ---------------------------------------------------
# 7️⃣ Evaluate Model
# ---------------------------------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n📊 Model Accuracy: {accuracy:.2f}")

# Clear Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\n📌 Confusion Matrix:")
print(f"True Negatives  (Actual 0, Predicted 0): {cm[0][0]}")
print(f"False Positives (Actual 0, Predicted 1): {cm[0][1]}")
print(f"False Negatives (Actual 1, Predicted 0): {cm[1][0]}")
print(f"True Positives  (Actual 1, Predicted 1): {cm[1][1]}")

# Classification Report
print("\n📌 Classification Report:")
print(classification_report(y_test, y_pred))

# ---------------------------------------------------
# 8️⃣ Show ONLY Model Predicted Survival Count
# ---------------------------------------------------
print("\n📊 Model Predicted Survival Count:")
pred_counts = pd.Series(y_pred).value_counts().sort_index()

print(f"0 (Predicted Did Not Survive): {pred_counts.get(0, 0)}")
print(f"1 (Predicted Survived): {pred_counts.get(1, 0)}")

# ---------------------------------------------------
# 9️⃣ Feature Importance
# ---------------------------------------------------
print("\n📌 Feature Importance:")
importance = pd.Series(model.feature_importances_, index=X.columns)
print(importance.sort_values(ascending=False))

# ---------------------------------------------------
# 🔟 Prediction Function (Returns 0 or 1)
# ---------------------------------------------------
def predict_survival(passenger):
    df_new = pd.DataFrame([passenger])
    df_new = df_new[X.columns]  # Ensure correct order
    return model.predict(df_new)[0]

# ---------------------------------------------------
# 1️⃣1️⃣ Example: Predict New Passenger
# ---------------------------------------------------
new_passenger = {
    'Pclass': 3,
    'Sex': 1,          # 0 = Female, 1 = Male
    'Age': 22.0,
    'SibSp': 1,
    'Parch': 0,
    'Fare': 7.25,
    'FamilySize': 2,
    'IsAlone': 0,
    'Embarked_Q': 0,
    'Embarked_S': 1,
    'Deck_B': 0,
    'Deck_C': 0,
    'Deck_D': 0,
    'Deck_E': 0,
    'Deck_F': 0,
    'Deck_G': 0,
    'Deck_T': 0,
    'Deck_U': 1
}

result = predict_survival(new_passenger)

print("\n🎯 Prediction (0 = Did Not Survive, 1 = Survived):", result)