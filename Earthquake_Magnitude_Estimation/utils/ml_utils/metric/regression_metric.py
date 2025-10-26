from dataclasses import dataclass
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

@dataclass
class RegressionMetric:
    rmse: float
    mae: float
    r2_score: float

def get_regression_score(y_true, y_pred) -> RegressionMetric:
    try:
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))
        
        regression_metric = RegressionMetric(
            rmse=rmse,
            mae=mae,
            r2_score=r2
        )
        return regression_metric
    except Exception as e:
        raise e