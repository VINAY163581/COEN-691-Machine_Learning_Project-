import sys
import pandas as pd
import numpy as np

from Earthquake_Magnitude_Estimation.exception.exception import (
    Earthquake_Magnitude_EstimationException
)

class EarthQuakeModel:
    def __init__(self, preprocessor, model):
        try:
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            raise Earthquake_Magnitude_EstimationException(e, sys)

    def predict(self, x):
        try:
            MODEL_FEATURES = [
                "latitude",
                "longitude",
                "depth",
                "nst",
                "gap",
                "dmin",
                "rms",
                "horizontalError",
                "depthError",
                "magError",
                "magNst"
            ]

            # CASE 1: x is a DataFrame (normal inference)
            if isinstance(x, pd.DataFrame):
                missing_cols = set(MODEL_FEATURES) - set(x.columns)
                if missing_cols:
                    raise ValueError(f"Missing required columns: {missing_cols}")

                x = x[MODEL_FEATURES]  # enforce order
                x_transformed = self.preprocessor.transform(x)

            # CASE 2: x is already a NumPy array (preprocessed)
            elif isinstance(x, np.ndarray):
                x_transformed = x

            else:
                raise TypeError("Input must be a pandas DataFrame or numpy array")

            predictions = self.model.predict(x_transformed)
            return predictions

        except Exception as e:
            raise Earthquake_Magnitude_EstimationException(e, sys)
