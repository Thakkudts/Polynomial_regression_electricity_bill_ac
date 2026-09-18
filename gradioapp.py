import gradio as gr
import joblib
import pandas as pd
import os

model = joblib.load("polynomial_regression_electricity_ac.pkl")


def predict_bill(ac_units):

    input_data = pd.DataFrame({
        "AC_Units": [ac_units]
    })

    prediction = model.predict(input_data)[0]

    return f"Expected Electric Bill: ₹{prediction:.2f}"


demo = gr.Interface(
    fn=predict_bill,

    inputs=gr.Number(
        label="Enter AC Electricity Consumption (Units)",
        minimum=0,
        maximum=105,
        value=50
    ),

    outputs=gr.Textbox(label="Predicted Electric Bill"),

    title="⚡ Electric Bill Prediction",

    description="Predict the Electric Bill based on AC electricity consumption using Linear Regression."
)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
