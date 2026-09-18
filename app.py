import streamlit as st
import joblib

model = joblib.load("polynomial_regression_electricity_ac.pkl")

st.title("⚡ AC Units vs Electric Bill")
st.write("Predict Electric Bill using Polynomial Regression")

ac_units = st.number_input(
    "Enter AC Electricity Consumption (Units):",
    min_value=0.0,
    value=100.0,
    step=1.0
)

if st.button("Predict Electric Bill"):

    prediction = model.predict([[ac_units]])

    st.success(f"Expected Electric Bill: ₹{prediction[0]:.2f}")

