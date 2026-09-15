import streamlit as st
import pandas as pd
import joblib
model_package = joblib.load("bodyfat_model.pkl")

model = model_package["model"]
features = model_package["features"]
st.set_page_config(
    page_title="Body Fat Predictor",
    page_icon="📊",
    layout="centered"
)

st.title("Body Fat Percentage Predictor")

st.write(
    "Enter your measurements below to generate an estimated body fat percentage."
)

st.info(
    "Educational machine-learning project only. "
    "This model was trained on a small male-only dataset and should not be used for medical decisions."
)
st.subheader("Personal Measurements")

age = st.number_input(
    "Age (years)",
    min_value=18,
    max_value=100,
    value=30
)

weight = st.number_input(
    "Weight (lb)",
    min_value=80.0,
    max_value=400.0,
    value=175.0
)

height = st.number_input(
    "Height (inches)",
    min_value=50.0,
    max_value=85.0,
    value=70.0
)
st.subheader("Circumference Measurements")

st.caption("Enter circumference measurements in centimeters.")

neck = st.number_input(
    "Neck (cm)",
    min_value=20.0,
    max_value=70.0,
    value=38.0
)

chest = st.number_input(
    "Chest (cm)",
    min_value=60.0,
    max_value=160.0,
    value=100.0
)

abdomen = st.number_input(
    "Abdomen (cm)",
    min_value=50.0,
    max_value=170.0,
    value=90.0
)

hip = st.number_input(
    "Hip (cm)",
    min_value=60.0,
    max_value=170.0,
    value=100.0
)
thigh = st.number_input(
    "Thigh (cm)",
    min_value=30.0,
    max_value=100.0,
    value=60.0
)

knee = st.number_input(
    "Knee (cm)",
    min_value=20.0,
    max_value=70.0,
    value=38.0
)

ankle = st.number_input(
    "Ankle (cm)",
    min_value=10.0,
    max_value=50.0,
    value=23.0
)

biceps = st.number_input(
    "Biceps (cm)",
    min_value=15.0,
    max_value=70.0,
    value=32.0
)

forearm = st.number_input(
    "Forearm (cm)",
    min_value=15.0,
    max_value=60.0,
    value=29.0
)

wrist = st.number_input(
    "Wrist (cm)",
    min_value=10.0,
    max_value=40.0,
    value=18.0
)
bmi = (
    weight * 703
    / (height ** 2)
)

abdomen_height_ratio = (
    abdomen / height
)
input_data = pd.DataFrame({
    "Age": [age],
    "Weight": [weight],
    "Height": [height],
    "Neck": [neck],
    "Chest": [chest],
    "Abdomen": [abdomen],
    "Hip": [hip],
    "Thigh": [thigh],
    "Knee": [knee],
    "Ankle": [ankle],
    "Biceps": [biceps],
    "Forearm": [forearm],
    "Wrist": [wrist],
    "BMI": [bmi],
    "AbdomenHeightRatio": [abdomen_height_ratio]
})

input_data = input_data[features]
if st.button("Predict Body Fat"):

    prediction = model.predict(input_data)[0]

    st.subheader("Model Estimate")

    st.metric(
        "Estimated Body Fat",
        f"{prediction:.1f}%"
    )

    st.write(
        f"Calculated BMI: **{bmi:.1f}**"
    )

    st.caption(
    "This is a statistical estimate, not a clinical measurement."
)
    st.divider()

st.subheader("About the Model")

st.write(
    """
    This project compared Linear Regression, Ridge Regression,
    Lasso Regression, PCA-based regression, Random Forest,
    and Gradient Boosting.

    Gradient Boosting produced the best cross-validated performance.

    **Held-out test performance:**

    - MAE: 3.49 percentage points
    - RMSE: 4.31 percentage points
    - R²: 0.683
    """
)