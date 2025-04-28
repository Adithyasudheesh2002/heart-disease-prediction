# main.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import smtplib
from email.mime.text import MIMEText
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
import os
import joblib

# Email Alert Function
def send_email_alert(subject, body):
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = "your_email@gmail.com"        # <-- Replace with your email
    msg["To"] = "recipient_email@gmail.com"      # <-- Replace with recipient email

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login("your_email@gmail.com", "your_app_password")  # <-- App Password
            server.sendmail(msg["From"], msg["To"], msg.as_string())

        print("✅ Email sent successfully!")
    except Exception as e:
        print(f"❌ Email error: {e}")

# Load Dataset
df = pd.read_csv('data/heart.csv')
print("✅ Data loaded successfully!")

# Features and Target
X = df.drop('target', axis=1)
y = df['target']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression Model
logreg = LogisticRegression()
logreg.fit(X_train_scaled, y_train)
print("✅ Logistic Regression model trained.")

# Random Forest Model
rf = RandomForestClassifier()
rf.fit(X_train_scaled, y_train)
print("✅ Random Forest model trained.")

# Save models
os.makedirs('models', exist_ok=True)
joblib.dump(logreg, 'models/logistic_regression_model.pkl')
joblib.dump(rf, 'models/random_forest_model.pkl')
print("✅ Models saved!")

# ROC Curve - Logistic Regression
fpr_logreg, tpr_logreg, _ = roc_curve(y_test, logreg.predict_proba(X_test_scaled)[:,1])
roc_auc_logreg = auc(fpr_logreg, tpr_logreg)

# ROC Curve - Random Forest
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf.predict_proba(X_test_scaled)[:,1])
roc_auc_rf = auc(fpr_rf, tpr_rf)

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr_logreg, tpr_logreg, color='blue', lw=2, label=f'Logistic Regression (AUC = {roc_auc_logreg:.2f})')
plt.plot(fpr_rf, tpr_rf, color='green', lw=2, label=f'Random Forest (AUC = {roc_auc_rf:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
os.makedirs('outputs', exist_ok=True)
plt.savefig('outputs/roc_curve.png')
plt.show()
print("✅ ROC curve saved!")

# Send Email Alert
send_email_alert(
    subject="Heart Disease Prediction Model Completed",
    body="The machine learning model training and evaluation have been completed successfully. Check the ROC curve and saved models."
)
