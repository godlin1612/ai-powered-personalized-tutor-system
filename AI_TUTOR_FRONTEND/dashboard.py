import streamlit as st
import numpy as np
import pandas as pd
import joblib
import pickle

# Load saved models and files
model_score = joblib.load("model_assessment.pkl")
model_promotion = joblib.load("model_promotion.pkl")
model_material = joblib.load("model_material.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")
material_mapping = pickle.load(open("material_mapping.pkl", "rb"))

# Inverse mapping for material name
material_inverse_mapping = {v: k for k, v in material_mapping.items()}

st.title("📚 AI-Powered Personalized Tutor")

st.subheader("👤 Student Info")
name = st.text_input("Name")
age = st.number_input("Age", min_value=4, max_value=25, value=10)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
country = st.selectbox("Country", ["India", "USA", "UK", "Other"])
state = st.text_input("State", "Tamil Nadu")
city = st.text_input("City", "Chennai")
parent_occupation = st.selectbox("Parent Occupation", ["Doctor", "Engineer", "Teacher", "Other"])
earning_class = st.selectbox("Earning Class", ["Low", "Medium", "High"])

st.subheader("🎓 Course Info")
level_student = st.selectbox("Level of Student", ["Beginner", "Intermediate", "Advanced"])
level_course = st.selectbox("Level of Course", ["Beginner", "Intermediate", "Advanced"])
course_name = st.selectbox("Course Name", ["Math", "Science", "English"])
material_level = st.selectbox("Material Level", ["Easy", "Medium", "Hard"])
time_per_day = st.slider("Time spent per day (min)", 0, 300, 60)
iq = st.slider("IQ of Student", 70, 160, 100)

if st.button("🔍 Predict"):

    # Form input dictionary
    input_dict = {
        "Age": age,
        "Time per Day (min)": time_per_day,
        "IQ of Student": iq,
        "Gender_" + gender: 1,
        "Country_" + country: 1,
        "State_" + state: 1,
        "City_" + city: 1,
        "Parent Occupation_" + parent_occupation: 1,
        "Earning Class_" + earning_class: 1,
        "Level of Student_" + level_student: 1,
        "Level of Course_" + level_course: 1,
        "Course Name_" + course_name: 1,
        "Material Level_" + material_level: 1,
    }

    # Ensure all feature names are in input_dict
    input_vector = []
    for feature in feature_names:
        input_vector.append(input_dict.get(feature, 0))

    input_array = np.array(input_vector).reshape(1, -1)

    # Scale numeric features
    input_scaled = scaler.transform(input_array)

    # Predict assessment score
    assessment_score = model_score.predict(input_scaled)[0]

    # Predict promotion (override by rules)
    promote = model_promotion.predict(input_scaled)[0]
    if assessment_score < 60 or time_per_day < 60 or iq < 85:
        promote = 0

    # Predict material name (level)
    material_pred = model_material.predict(input_scaled)[0]
    material_name = material_inverse_mapping.get(material_pred, "Unknown")

    st.subheader("📊 Predictions")
    st.write(f"🎯 **Assessment Score:** {assessment_score:.2f}")
    st.write(f"🎓 **Promotion Status:** {'Promoted' if promote == 1 else 'Not Promoted'}")
    st.write(f"📘 **Recommended Material:** {material_name}")
