# 1 - Good (Lower risk)
# 0 - Bad (Higher risk)

import streamlit as st
import pandas as pd
import joblib

model = joblib.load("rf_model_credit.pkl")
encoders = {col: joblib.load(f"{col}_encoder.pkl") for col in ["Sex", "Housing", "Saving accounts", "Checking account" ]}

st.title("Credit Risk Prediction Model")
st.write("Enter informations to predict if the credit risk is low or high")

age = st.number_input("Age", min_value=18, max_value=80, value=30)
sex = st.selectbox("Sex", ["male", "female"])
job = st.number_input("Job (0-3)", min_value=0, max_value=3, value=1)
housing = st.selectbox("Housing", ["free", "own", "rent"])
saving_account = st.selectbox("Saving accounts", ["little", "moderate", "quite rich", "rich"])
checking_account = st.selectbox("Checking accounts", ["little", "moderate", "rich"])
credit_amount = st.number_input("Credit Amount", min_value=0, value=1000)
duration = st.number_input("Duration", min_value=1, value=12)

input_df = pd.DataFrame({
    "Age": [age],
    "Sex": [encoders["Sex"].transform([sex])[0]],
    "Job": [job],
    "Housing": [encoders["Housing"].transform([housing])[0]],
    "Saving accounts": [encoders["Saving accounts"].transform([saving_account])[0]],
    "Checking account": [encoders["Checking account"].transform([checking_account])[0]],
    "Credit amount": [credit_amount],
    "Duration": [duration]
})

if st.button("Predict"):
    pred = model.predict(input_df)[0]

    if pred == 1:
        st.success("The predicted credit risk is Low (Good)")
    else:
        st.error("The predicted credit risk is High (Bad)")