import os
import sys
import pandas as pd
import streamlit as st

from Earthquake_Magnitude_Estimation.exception.exception import (
    Earthquake_Magnitude_EstimationException
)
from Earthquake_Magnitude_Estimation.utils.main_utils.utils import load_object


# Streamlit Page Config
st.set_page_config(
    page_title="Earthquake Magnitude Prediction",
    layout="centered"
)

st.title(" Earthquake Magnitude Prediction")
st.markdown(
    "Predict earthquake magnitude using a trained ML model and preprocessing pipeline."
)

# Absolute Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PREPROCESSOR_PATH = os.path.join(
    BASE_DIR,
    "Artifacts",
    "11_10_2025_15_41_39",
    "data_transformation",
    "transformed_object",
    "preprocessing.pkl"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "Artifacts",
    "11_10_2025_15_41_39",
    "model_trainer",
    "trained_model",
    "model.pkl"
)


# Load Model & Preprocessor (ONCE)

@st.cache_resource
def load_artifacts():
    preprocessor = load_object(PREPROCESSOR_PATH)
    model = load_object(MODEL_PATH)
    return preprocessor, model

try:
    preprocessor, model = load_artifacts()
except Exception as e:
    st.error(" Failed to load model artifacts")
    raise Earthquake_Magnitude_EstimationException(e, sys)


# Input Form
st.subheader(" Enter Earthquake Parameters")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        latitude = st.number_input("Latitude", value=37.7749, format="%.6f")
        depth = st.number_input("Depth (km)", value=8.5)
        gap = st.number_input("Azimuthal Gap (gap)", value=62.3)
        rms = st.number_input("RMS Travel Time (rms)", value=0.42)
        depthError = st.number_input("Depth Error", value=1.4)
        magNst = st.number_input("Magnitude Stations (magNst)", value=32)

    with col2:
        longitude = st.number_input("Longitude", value=-122.4194, format="%.6f")
        nst = st.number_input("Number of Stations (nst)", value=45)
        dmin = st.number_input("Minimum Distance (dmin)", value=0.38)
        horizontalError = st.number_input("Horizontal Error", value=1.1)
        magError = st.number_input("Magnitude Error", value=0.18)

    submit = st.form_submit_button(" Predict Magnitude")


# Prediction Logic
if submit:
    try:
        # Base numeric input
        input_dict = {
            "latitude": latitude,
            "longitude": longitude,
            "depth": depth,
            "nst": nst,
            "gap": gap,
            "dmin": dmin,
            "rms": rms,
            "horizontalError": horizontalError,
            "depthError": depthError,
            "magError": magError,
            "magNst": magNst,

            # REQUIRED numeric dummy columns
            "time": 0.0,
            "updated": 0.0,
            "id": 0.0,
            "net": 0.0,
            "magType": 0.0,
            "place": 0.0
        }

        input_df = pd.DataFrame([input_dict])

        # Preprocess + Predict
        X = preprocessor.transform(input_df)
        prediction = model.predict(X)[0]

        st.success(f" **Predicted Earthquake Magnitude: {prediction:.2f}**")

    except Exception as e:
        st.error(" Prediction failed")
        raise Earthquake_Magnitude_EstimationException(e, sys)
