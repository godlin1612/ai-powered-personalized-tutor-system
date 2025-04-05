import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, confusion_matrix

# Assuming you have a dataframe df already loaded from a CSV or other source
# df = pd.read_csv("path_to_your_dataset.csv") 

# Replace 'df' with your actual dataset
df = pd.read_csv('k12_students_data.csv')  # Load your actual dataset here

# Define features (X) and target variables
X = df.drop(columns=['Name', 'Assessment Score'])
y_assessment = df['Assessment Score']

# Create binary target for promotion decision (1 for promoted, 0 for not)
promotion_threshold = 60
y_promotion = (y_assessment > promotion_threshold).astype(int)  # 1 if promoted, 0 if not

# Create multi-class labels for material level prediction
y_material = df['Material Level']

# Split data into training and testing sets for assessment score regression
X_train, X_test, y_train_assessment, y_test_assessment = train_test_split(X, y_assessment, test_size=0.2, random_state=42)

# Split data into training and testing sets for promotion decision classification
X_train_promo, X_test_promo, y_train_promo, y_test_promo = train_test_split(X, y_promotion, test_size=0.2, random_state=42)

# Split data into training and testing sets for material level prediction classification
X_train_material, X_test_material, y_train_material, y_test_material = train_test_split(X, y_material, test_size=0.2, random_state=42)

# Define preprocessing pipeline for numerical and categorical features
numerical_cols = ['Age', 'Time per Day (min)', 'IQ of Student']
categorical_cols = ['Gender', 'Country', 'State', 'City', 'Parent Occupation', 'Earning Class', 
                    'Level of Student', 'Level of Course', 'Course Name', 'Material Name', 'Material Level']

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), numerical_cols),
        
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_cols)
    ]
)

# 1. Model for Predicting Assessment Score (Regression)
model_regressor = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train the model for assessment score prediction
model_regressor.fit(X_train, y_train_assessment)

# Predict on the test set for assessment score
y_pred_assessment = model_regressor.predict(X_test)

# Evaluate the regression model
mae_assessment = mean_absolute_error(y_test_assessment, y_pred_assessment)
r2_assessment = r2_score(y_test_assessment, y_pred_assessment)

print(f"Assessment Score Prediction (Regression) - MAE: {mae_assessment}, R²: {r2_assessment}")

# 2. Model for Predicting Promotion Decision (Classification)
model_classifier_promotion = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train the model for promotion prediction
model_classifier_promotion.fit(X_train_promo, y_train_promo)

# Predict on the test set for promotion decision
y_pred_promotion = model_classifier_promotion.predict(X_test_promo)

# Evaluate the classification model for promotion
accuracy_promotion = accuracy_score(y_test_promo, y_pred_promotion)
conf_matrix_promotion = confusion_matrix(y_test_promo, y_pred_promotion)

print(f"Promotion Decision Prediction (Classification) - Accuracy: {accuracy_promotion}")
print(f"Confusion Matrix for Promotion Prediction: \n{conf_matrix_promotion}")

# 3. Model for Predicting Material Level (Classification)
model_classifier_material = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train the model for material level prediction
model_classifier_material.fit(X_train_material, y_train_material)

# Predict on the test set for material level prediction
y_pred_material = model_classifier_material.predict(X_test_material)

# Evaluate the classification model for material level prediction
accuracy_material = accuracy_score(y_test_material, y_pred_material)

print(f"Material Level Prediction (Classification) - Accuracy: {accuracy_material}")