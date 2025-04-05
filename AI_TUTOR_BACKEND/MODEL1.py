import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.impute import SimpleImputer

# Load the dataset (adjust the path to your dataset)
df = pd.read_csv('k12_students_data.csv')  # Replace with your dataset path

# Handle missing values by imputing with the mean (for numeric) or mode (for categorical)
numeric_columns = df.select_dtypes(include=[np.number]).columns
categorical_columns = df.select_dtypes(include=[object]).columns

# Impute numeric columns with mean
imputer = SimpleImputer(strategy='mean')
df[numeric_columns] = imputer.fit_transform(df[numeric_columns])

# Impute categorical columns with mode
imputer_cat = SimpleImputer(strategy='most_frequent')
df[categorical_columns] = imputer_cat.fit_transform(df[categorical_columns])

# Encoding categorical columns
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Feature Engineering (add any additional feature transformations here)
# You can also create features based on combinations of existing ones, if needed

# Scaling numeric features
scaler = StandardScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Separate features and target variables
X = df.drop(columns=['Assessment Score', 'Promoted', 'Material Name'])  # Features (excluding target columns)
y_score = df['Assessment Score']  # Target variable for prediction of score
y_promotion = df['Promoted']  # Target variable for promotion decision (binary)
y_material = df['Material Name']  # Target variable for material prediction (categorical)

# Split data into training and testing sets
X_train, X_test, y_train_score, y_test_score = train_test_split(X, y_score, test_size=0.2, random_state=42)
X_train, X_test, y_train_promotion, y_test_promotion = train_test_split(X, y_promotion, test_size=0.2, random_state=42)
X_train, X_test, y_train_material, y_test_material = train_test_split(X, y_material, test_size=0.2, random_state=42)

# --- Task 1: Predict Assessment Score ---
# Train the model for predicting assessment score (regression task)
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train_score)

# Predict assessment scores
y_pred_score = rf_regressor.predict(X_test)

# Evaluate model performance using Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test_score, y_pred_score)
print(f"Mean Absolute Error for Assessment Score Prediction: {mae:.2f}")

# --- Task 2: Predict Promotion Decision ---
# Train the model for predicting promotion decision (classification task)
rf_classifier_promotion = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier_promotion.fit(X_train, y_train_promotion)

# Predict promotion status
y_pred_promotion = rf_classifier_promotion.predict(X_test)

# Evaluate model performance using accuracy
accuracy_promotion = accuracy_score(y_test_promotion, y_pred_promotion)
print(f"Accuracy for Promotion Prediction: {accuracy_promotion * 100:.2f}%")

# --- Task 3: Predict Material Type ---
# Train the model for predicting material type (multi-class classification)
rf_classifier_material = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier_material.fit(X_train, y_train_material)

# Predict material type
y_pred_material = rf_classifier_material.predict(X_test)

# Evaluate model performance using accuracy
accuracy_material = accuracy_score(y_test_material, y_pred_material)
print(f"Accuracy for Material Prediction: {accuracy_material * 100:.2f}%")

