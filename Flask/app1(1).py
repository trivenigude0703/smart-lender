from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Resolve absolute paths for pickeled model and scaler
model_path = os.path.join(os.path.dirname(__file__), 'rdf.pkl')
scaler_path = os.path.join(os.path.dirname(__file__), 'scale1.pkl')

print(f"Loading model from {model_path}...")
with open(model_path, 'rb') as f:
    model = pickle.load(f)

print(f"Loading scaler from {scaler_path}...")
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/input')
def input_page():
    return render_template('input.html')

@app.route('/input(1)')
def input_dup_page():
    return render_template('input(1).html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from form
        gender = int(request.form.get('gender', 1))
        married = int(request.form.get('married', 1))
        dependents = int(request.form.get('dependents', 0))
        education = int(request.form.get('education', 1))
        self_employed = int(request.form.get('self_employed', 0))
        applicant_income = float(request.form.get('applicant_income', 0))
        coapplicant_income = float(request.form.get('coapplicant_income', 0))
        loan_amount = float(request.form.get('loan_amount', 0))
        loan_amount_term = float(request.form.get('loan_amount_term', 360))
        credit_history = float(request.form.get('credit_history', 1.0))
        property_area = int(request.form.get('property_area', 2))

        # Build DataFrame with exact column names used during training
        features = pd.DataFrame([{
            'Gender': gender,
            'Married': married,
            'Dependents': dependents,
            'Education': education,
            'Self_Employed': self_employed,
            'ApplicantIncome': applicant_income,
            'CoapplicantIncome': coapplicant_income,
            'LoanAmount': loan_amount,
            'Loan_Amount_Term': loan_amount_term,
            'Credit_History': credit_history,
            'Property_Area': property_area
        }])

        # Scale features
        scaled_features = scaler.transform(features)

        # Predict
        prediction = model.predict(scaled_features)
        
        # Prediction output
        result = "Approved" if prediction[0] == 1 else "Rejected"
        
        # Decide if rendering output.html or output(1).html based on form parameter
        use_dup = request.form.get('use_dup', 'false') == 'true'
        template_name = 'output(1).html' if use_dup else 'output.html'
        
        return render_template(template_name, 
                               result=result, 
                               gender='Male' if gender == 1 else 'Female',
                               married='Yes' if married == 1 else 'No',
                               dependents=str(dependents),
                               education='Graduate' if education == 1 else 'Not Graduate',
                               self_employed='Yes' if self_employed == 1 else 'No',
                               applicant_income=applicant_income,
                               coapplicant_income=coapplicant_income,
                               loan_amount=loan_amount,
                               loan_amount_term=loan_amount_term,
                               credit_history='Good' if credit_history == 1.0 else 'Bad',
                               property_area='Urban' if property_area == 2 else ('Semiurban' if property_area == 1 else 'Rural'))
    except Exception as e:
        return f"<div style='color:red; font-family:sans-serif;'>An error occurred: {str(e)}</div>"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
