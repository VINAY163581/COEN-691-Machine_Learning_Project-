import os
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from Earthquake_Magnitude_Estimation.utils.main_utils.utils import load_object

# =========================
# BASE DIRECTORY
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================
# PATHS (MATCH YOUR FOLDERS)
# =========================
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

TEST_NPY_PATH = os.path.join(
    BASE_DIR,
    "Artifacts",
    "11_10_2025_15_41_39",
    "data_transformation",
    "transformed",
    "test.npy"
)

# =========================
# LOAD ARTIFACTS
# =========================
print("Loading preprocessor and model...")
preprocessor = load_object(PREPROCESSOR_PATH)
model_wrapper = load_object(MODEL_PATH)

# Extract sklearn model if wrapped
model = model_wrapper.model if hasattr(model_wrapper, "model") else model_wrapper

# =========================
# FEATURE NAMES
# =========================
feature_names = preprocessor.get_feature_names_out()
print(f"\nTotal transformed features: {len(feature_names)}")
print("First 10 feature names:")
for f in feature_names[:10]:
    print(" ", f)

# =========================
# LOAD TRANSFORMED TEST DATA
# =========================
print("\nLoading transformed test data...")
test_arr = np.load(TEST_NPY_PATH)

X_test = test_arr[:, :-1]   # features
y_test = test_arr[:, -1]    # target

X_test_df = pd.DataFrame(X_test, columns=feature_names)

# =========================
# SHAP ANALYSIS
# =========================
print("\nComputing SHAP values...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_df)

# =========================
# TOP 10 IMPORTANT FEATURES
# =========================
mean_abs_shap = np.abs(shap_values).mean(axis=0)
top_idx = np.argsort(mean_abs_shap)[::-1][:10]

print("\nTOP 10 MOST INFLUENTIAL FEATURES:")
for i in top_idx:
    print(f"{feature_names[i]}  ->  mean(|SHAP|) = {mean_abs_shap[i]:.4f}")

# =========================
# SHAP SUMMARY PLOT
# =========================
print("\nDisplaying SHAP summary plot...")
shap.summary_plot(shap_values, X_test_df, show=True)
