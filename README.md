Stroke Prediction Powered By Machine Learning
A machine learning–based web application that predicts the risk of stroke (Low Risk / High Risk) using clinical and demographic data. The system is designed to assist in early risk assessment and support preventive healthcare decisions.

📌 Project Overview

Stroke is one of the leading causes of death and long-term disability worldwide. Early identification of high-risk individuals can significantly reduce its impact. This project leverages machine learning algorithms to analyze patient data and predict the likelihood of stroke.

The application provides:

A user-friendly web interface for data entry

A trained ML classification model

Clear risk-level output (Low Risk / High Risk)
Technologies Used

Programming Language: Python

Machine Learning: Scikit-learn (SVC)

Web Framework: Flask

Frontend: HTML, CSS
Dataset Description

The dataset consists of clinical and lifestyle attributes such as:

Age

Gender

Hypertension

Heart Disease

Average Glucose Level

BMI

Smoking Status

Data preprocessing steps include:

Handling missing values

Encoding categorical variables

Feature scaling
Machine Learning Model

Algorithm Used: Support Vector Classifier (SVC)

Why SVC?

Effective in high-dimensional spaces

Robust to overfitting with proper kernel selection

Model Evaluation Metrics

Accuracy

Precision

Recall

F1-score

🌐 Web Application Workflow

User enters health details through the web form.

Input data is preprocessed and scaled.

The trained ML model predicts stroke risk.

The result is displayed as Low Risk or High Risk.
Results

The model successfully classifies users into low-risk and high-risk categories with reliable performance.

Data Handling: Pandas, NumPy

Development Tools: Jupyter Notebook, VS Code
