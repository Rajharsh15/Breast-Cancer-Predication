🩺 Breast Cancer Prediction Web App
A Flask-based web application that predicts whether a tumor is benign or malignant based on 30 input features from the Breast Cancer Wisconsin Diagnostic Dataset. Built using Python, Machine Learning (Scikit-learn), Flask, and a responsive HTML/CSS frontend.

🔧 Technologies Used
Python 3.8+

Flask - Web framework

Scikit-learn - ML model training

Pandas / NumPy - Data processing

HTML5 + CSS3 - Frontend

Jinja2 - Dynamic HTML templating

📊 Model
Dataset: Breast Cancer Wisconsin Diagnostic Dataset

Algorithm: Logistic Regression

Features Used: 30 input numerical features (like mean radius, mean texture, worst area, etc.)

🔁 How It Works
User enters 30 numeric input features via web form

The Flask backend receives the data and feeds it into a pre-trained ML model

The app returns a prediction:

✅ Benign

⚠️ Malignant

Also displays training and testing accuracy scores for model transparency
