import os
import sys
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from Earthquake_Magnitude_Estimation.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact
)
from Earthquake_Magnitude_Estimation.entity.config_entity import ModelTrainerConfig
from Earthquake_Magnitude_Estimation.exception.exception import Earthquake_Magnitude_EstimationException
from Earthquake_Magnitude_Estimation.logging.logger import logging
from Earthquake_Magnitude_Estimation.utils.main_utils.utils import (
    save_object,
    load_object,
    load_numpy_array_data,
    evaluate_models
)
from Earthquake_Magnitude_Estimation.utils.ml_utils.model.estimator import EarthQuakeModel
from Earthquake_Magnitude_Estimation.utils.ml_utils.metric.regression_metric import get_regression_score


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig,
                 data_transformation_artifact: DataTransformationArtifact):
        try:
            logging.info(f"{'>>'*10} Model Trainer Initiated {'<<'*10}")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise Earthquake_Magnitude_EstimationException(e, sys)

    def train_model(self, X_train, y_train, X_test, y_test):
        try:
            # Define models
            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "ElasticNet": ElasticNet(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "Support Vector Regression": Pipeline([("scaler", StandardScaler()), ("svr", SVR())])
            }

            # Hyperparameters
            params = {
                "Linear Regression": {},
                "Lasso": {'alpha': [0.1, 0.5, 1.0]},
                "Ridge": {'alpha': [0.1, 1.0, 10.0]},
                "ElasticNet": {'alpha': [0.1, 1.0], 'l1_ratio': [0.2, 0.5, 0.8]},
                "Decision Tree": {'max_depth': [3, 5, None]},
                "Random Forest": {'n_estimators': [50, 100], 'max_depth': [None, 10]},
                "Gradient Boosting": {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]},
                "AdaBoost": {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]},
                "Support Vector Regression": {'svr__C': [0.1, 1, 10], 'svr__kernel': ['linear', 'rbf']}
            }

            # Remove NaNs
            nan_mask_train = ~np.isnan(y_train)
            X_train, y_train = X_train[nan_mask_train], y_train[nan_mask_train]
            nan_mask_test = ~np.isnan(y_test)
            X_test, y_test = X_test[nan_mask_test], y_test[nan_mask_test]

            logging.info(f"NaNs in y_train after cleaning: {np.isnan(y_train).sum()}")
            logging.info(f"NaNs in y_test after cleaning: {np.isnan(y_test).sum()}")

            # Evaluate models
            model_report = evaluate_models(
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
                models=models, param=params
            )

            # Best model selection
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]

            # Predictions & metrics
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            train_metric = get_regression_score(y_true=y_train, y_pred=y_train_pred)
            test_metric = get_regression_score(y_true=y_test, y_pred=y_test_pred)

            # Load preprocessor & save model
            preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path, exist_ok=True)

            earthquake_model = EarthQuakeModel(preprocessor=preprocessor, model=best_model)
            save_object(self.model_trainer_config.trained_model_file_path, obj=earthquake_model)

            # Artifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=train_metric,
                test_metric_artifact=test_metric
            )
            logging.info(f"Model Trainer Artifact: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise Earthquake_Magnitude_EstimationException(e, sys)

    # ✅ Re-add initiate_model_trainer method
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            # Load numpy arrays
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            return self.train_model(X_train, y_train, X_test, y_test)

        except Exception as e:
            raise Earthquake_Magnitude_EstimationException(e, sys)
