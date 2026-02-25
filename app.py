import pickle

import gradio as gr
import numpy as np
import pandas as pd

# ^ Main Logic

# & Load Model
with open("loan_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)


# & Predict Function
def predict_loan(
    gender,
    married,
    dependents,
    education,
    self_employed,
    applicant_income,
    coapplicant_income,
    loan_amount,
    loan_amount_term,
    credit_history,
    property_area,
):

    # Encode inputs same way as training data
    gender_enc = 1 if gender == "Male" else 0
    married_enc = 1 if married == "Yes" else 0

    dep_map = {"0": 0, "1": 1, "2": 2, "3+": 3}
    dependents_enc = dep_map.get(dependents, 0)

    education_enc = 0 if education == "Graduate" else 1
    self_employed_enc = 1 if self_employed == "Yes" else 0

    area_map = {"Urban": 2, "Semiurban": 1, "Rural": 0}
    property_area_enc = area_map.get(property_area, 0)

    # Create engineered features
    total_income = applicant_income + coapplicant_income
    emi = loan_amount / loan_amount_term if loan_amount_term > 0 else 0
    balance_income = total_income - (emi * 1000)

    # Pack inputs into a DataFrame
    input_df = pd.DataFrame(
        [
            [
                gender_enc,
                married_enc,
                dependents_enc,
                education_enc,
                self_employed_enc,
                applicant_income,
                coapplicant_income,
                loan_amount,
                loan_amount_term,
                credit_history,
                property_area_enc,
                total_income,
                emi,
                balance_income,
            ]
        ],
        columns=feature_names,
    )

    # Predict
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    if prediction == 1:
        return f"✅ Loan APPROVED (Confidence: {probability[1]*100:.1f}%)"
    else:
        return f"❌ Loan NOT APPROVED (Confidence: {probability[0]*100:.1f}%)"


# ^ Interface
inputs = [
    gr.Dropdown(["Male", "Female"], label="Gender"),
    gr.Dropdown(["Yes", "No"], label="Married"),
    gr.Dropdown(["0", "1", "2", "3+"], label="Dependents"),
    gr.Dropdown(["Graduate", "Not Graduate"], label="Education"),
    gr.Dropdown(["Yes", "No"], label="Self Employed"),
    gr.Number(label="Applicant Income", value=5000),
    gr.Number(label="Coapplicant Income", value=0),
    gr.Number(label="Loan Amount (in thousands)", value=150),
    gr.Number(label="Loan Amount Term (in months)", value=360),
    gr.Dropdown([1.0, 0.0], label="Credit History (1=Good, 0=Bad)", value=1.0),
    gr.Dropdown(["Urban", "Semiurban", "Rural"], label="Property Area"),
]

app = gr.Interface(
    fn=predict_loan,
    inputs=inputs,
    outputs="text",
    title="🏦 Loan Approval Predictor",
    description="Enter your details below to check if your loan will be approved or not.",
)

app.launch()
