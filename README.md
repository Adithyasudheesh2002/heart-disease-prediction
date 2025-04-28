Heart Disease Prediction & Monitoring
This project builds a machine learning model to predict the presence of heart disease using patient data. It compares the performance of Logistic Regression and Random Forest Classifier using ROC curves and includes a system to send email alerts after deployment.

🚀 Project Workflow
Load and preprocess the heart disease dataset.

Handle missing values and encode categorical variables.

Train Logistic Regression and Random Forest models.

Evaluate models using ROC Curve and AUC (Area Under Curve) score.

Send an email alert when needed (for monitoring and notification purposes).

🛠 Technologies Used
Python

pandas, numpy

scikit-learn

matplotlib

smtplib (for sending emails)

📊 Model Evaluation
Plotted ROC curves for both models.

Compared model performance using AUC score.

Random Forest and Logistic Regression both evaluated to find the better model.

📬 Email Alert System
After model training and evaluation, an automated email alert is sent using Gmail SMTP server with a secure App Password.

📁 Project Structure
css
Copy
Edit
heart-disease-prediction/
│
├── README.md
├── requirements.txt
├── main.py
├── data/
│   └── heart.csv
├── models/
│   └── logistic_regression_model.pkl
│   └── random_forest_model.pkl
├── outputs/
│   └── roc_curve.png
└── utils/
    └── email_alert.py
⚡ Setup Instructions
Clone this repository:

bash
Copy
Edit
git clone https://github.com/your-username/heart-disease-prediction.git
Install required packages:

nginx
Copy
Edit
pip install -r requirements.txt
Create a Gmail App Password if using email alerts.
(Follow this guide to generate it.)

Run the project:

css
Copy
Edit
python main.py
📢 Important Notes
Make sure you do not expose your real Gmail password. Always use App Passwords.

Replace your email address and app password in the email script.

If ROC curves or models are saved, ensure folders like models/ and outputs/ exist.

✨ Future Enhancements
Deploy the model using Flask API and monitor predictions.

Integrate with real-time dashboards.

Add logging and advanced error handling.

❤️ Thank you!
