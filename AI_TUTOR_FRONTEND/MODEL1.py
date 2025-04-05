import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score

# Load dataset
df = pd.read_csv("k12_students_data.csv")

# Handle missing values
df.ffill(inplace=True)

# Custom logic for 'Promoted' column based on Assessment Score, Time, and IQ
def custom_promotion(row):
    if row["Assessment Score"] < 60 or row["Time per Day (min)"] < 60 or row["IQ of Student"] < 85:
        return 0
    return 1

df["Promoted"] = df.apply(custom_promotion, axis=1)

# Handle categorical using one-hot encoding (no label encoder used)
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
categorical_cols.remove("Material Name")  # Keep this as the target for material model

df_encoded = pd.get_dummies(df, columns=categorical_cols)

# Save material label mapping
material_mapping = {name: idx for idx, name in enumerate(df["Material Name"].unique())}
df_encoded["Material Name"] = df["Material Name"].map(material_mapping)

# Save mapping for decoding later
with open("material_mapping.pkl", "wb") as f:
    pickle.dump(material_mapping, f)

# Define targets
y_score = df_encoded["Assessment Score"]
y_promotion = df_encoded["Promoted"]
y_material = df_encoded["Material Name"]

# Define features
X = df_encoded.drop(columns=["Assessment Score", "Promoted", "Material Name"])

# Scale numeric data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Save feature names
feature_names = X_scaled_df.columns.tolist()

# --- Train Assessment Score Model (Regression) ---
X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y_score, test_size=0.2, random_state=42)
model_score = RandomForestRegressor(n_estimators=100, random_state=42)
model_score.fit(X_train, y_train)
mae = mean_absolute_error(y_test, model_score.predict(X_test))
print(f"[OK] Assessment Score MAE: {mae:.2f}")

# --- Train Promotion Model (Binary Classification) ---
X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y_promotion, test_size=0.2, random_state=42)
model_promotion = RandomForestClassifier(n_estimators=100, random_state=42)
model_promotion.fit(X_train, y_train)
acc1 = accuracy_score(y_test, model_promotion.predict(X_test))
print(f"[OK] Promotion Prediction Accuracy: {acc1:.2f}")

# --- Train Material Level Model (Multi-Class Classification) ---
X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y_material, test_size=0.2, random_state=42)
model_material = RandomForestClassifier(n_estimators=100, random_state=42)
model_material.fit(X_train, y_train)
acc2 = accuracy_score(y_test, model_material.predict(X_test))
print(f"[OK] Material Prediction Accuracy: {acc2:.2f}")

# --- Save all models and objects ---
pickle.dump(model_score, open("model_assessment.pkl", "wb"))
pickle.dump(model_promotion, open("model_promotion.pkl", "wb"))
pickle.dump(model_material, open("model_material.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(feature_names, open("feature_names.pkl", "wb"))

print("[SAVED] All models and files saved successfully.")
