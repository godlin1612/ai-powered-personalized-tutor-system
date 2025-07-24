# AI-Powered Personal Tutor System

## 📌 Overview
The **AI-Powered Personal Tutor System** is a machine learning-based platform designed to personalize learning experiences for students.  
It predicts assessment scores, determines promotion eligibility, and recommends study materials using **Random Forest models**.  
The project includes both **backend (model training)** and **frontend (Streamlit dashboard)** modules.

---

## 🎯 Key Features
- **Student Performance Prediction:** Predicts assessment scores with Random Forest Regressor.
- **Promotion Eligibility:** Classification model evaluates whether a student can be promoted.
- **Material Recommendation:** Multi-class classification for personalized content.
- **Streamlit Dashboard:** A user-friendly web interface for predictions and recommendations.
- **Data Generation:** Custom script to generate simulated K-12 student data.

---

## 🛠 Tech Stack
- **Languages:** Python
- **Libraries:** Scikit-learn, Pandas, NumPy, Matplotlib, Streamlit
- **Models:** Random Forest (Regression & Classification)
- **Tools:** Git, VS Code
- **Dataset:** `k12_students_data.csv`

---

## 📂 Project Structure
AI-Powered-Personal-Tutor-System/
│
├── AI_TUTOR_BACKEND/ # Model training and data generation
│ ├── analysis/
│ ├── MODEL1.py
│ ├── MODEL2.py
│ ├── generate_k12_data.py
│ └── k12_students_data.csv
│
├── AI_TUTOR_FRONTEND/ # Streamlit app and ML models
│ ├── Dashboard Output/
│ ├── MODEL1.py
│ ├── MODEL2.py
│ ├── dashboard.py # Streamlit dashboard
│ ├── feature_columns.pkl
│ ├── feature_names.pkl
│ ├── material_mapping.pkl
│ ├── model_assessment.pkl
│ ├── model_material.pkl
│ ├── model_promotion.pkl
│ ├── scaler.pkl
│ ├── requirements.txt # Dependencies
│ └── k12_students_data.csv
│
└── README.md
