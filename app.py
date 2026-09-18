import streamlit as st
import joblib
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)

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
    input_data=[[ac_units]]
    input_poly=poly.fit_transform(input_data)
    prediction = model.predict(input_poly)
    st.success(f"Expected Electric Bill: ₹{prediction[0]:.2f}")

