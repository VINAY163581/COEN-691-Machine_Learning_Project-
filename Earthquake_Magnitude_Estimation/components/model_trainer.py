import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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


try:
    import shap
    _SHAP_AVAILABLE = True
except Exception as _:
    _SHAP_AVAILABLE = False


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig,
                 data_transformation_artifact: DataTransformationArtifact):
        try:
            logging.info(f"{'>>'*10} Model Trainer Initiated {'<<'*10}")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise Earthquake_Magnitude_EstimationException(e, sys)

    # ----------------------- Functions for the Grapg generation-----------------------
    def compute_extra_metrics(self, y_true, y_pred):
        """Compute extra evaluation metrics."""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        bias = float(np.mean(y_pred - y_true))
        variance = float(np.var(y_pred))
        return mse, rmse, mae, bias, variance

    def _ensure_dirs(self):
        eval_dir = os.path.join(
            os.path.dirname(self.model_trainer_config.trained_model_file_path),
            "evaluation"
        )
        plots_dir = os.path.join(eval_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        return eval_dir, plots_dir

    def _save_barh(self, df, metric, outpath, title):
        """Generic horizontal bar plot for a metric."""
        try:
            ordered = df.sort_values(by=metric, ascending=(metric not in ["R2_Score"]))  # R2 high=good; others low=good
            plt.figure(figsize=(10, 6))
            plt.barh(ordered["Model"], ordered[metric])
            plt.xlabel(metric)
            plt.title(title)
            plt.tight_layout()
            plt.savefig(outpath, dpi=150)
            plt.close()
            logging.info(f"Saved plot: {outpath}")
        except Exception as e:
            logging.error(f"Failed to save plot {metric}: {e}")

    def _plot_predicted_vs_actual(self, y_true, y_pred, outpath):
        try:
            plt.figure(figsize=(6, 6))
            plt.scatter(y_true, y_pred, s=18)
            minv = float(min(np.min(y_true), np.min(y_pred)))
            maxv = float(max(np.max(y_true), np.max(y_pred)))
            plt.plot([minv, maxv], [minv, maxv])  # y=x line
            plt.xlabel("Actual")
            plt.ylabel("Predicted")
            plt.title("Predicted vs Actual (Best Model)")
            plt.tight_layout()
            plt.savefig(outpath, dpi=150)
            plt.close()
            logging.info(f"Saved Predicted vs Actual: {outpath}")
        except Exception as e:
            logging.error(f"Failed Predicted vs Actual plot: {e}")

    def _plot_residuals(self, y_true, y_pred, scatter_out, hist_out):
        try:
            residuals = y_true - y_pred

            # Residual scatter vs predicted
            plt.figure(figsize=(8, 5))
            plt.scatter(y_pred, residuals, s=14)
            plt.axhline(0.0)
            plt.xlabel("Predicted")
            plt.ylabel("Residual (y_true - y_pred)")
            plt.title("Residuals vs Predicted")
            plt.tight_layout()
            plt.savefig(scatter_out, dpi=150)
            plt.close()
            logging.info(f"Saved Residual scatter: {scatter_out}")

            # Residual histogram
            plt.figure(figsize=(8, 5))
            plt.hist(residuals, bins=30)
            plt.xlabel("Residual")
            plt.ylabel("Count")
            plt.title("Residuals Histogram")
            plt.tight_layout()
            plt.savefig(hist_out, dpi=150)
            plt.close()
            logging.info(f"Saved Residual histogram: {hist_out}")

        except Exception as e:
            logging.error(f"Failed residual plots: {e}")

    def _try_shap(self, model, X_test, plots_dir):
        """Generate SHAP summary plots for tree-based models if SHAP is available."""
        try:
            if not _SHAP_AVAILABLE:
                logging.warning("SHAP not installed. Skipping SHAP plots. Run `pip install shap` to enable.")
                return

            # Only proceed for tree-based models
            tree_types = (RandomForestRegressor, GradientBoostingRegressor, DecisionTreeRegressor)
            if not isinstance(model, tree_types):
                logging.info("Best model is not a tree-based estimator. Skipping SHAP plots.")
                return

            # Subsample for speed if necessary
            n = min(400, X_test.shape[0])
            X_sample = X_test[:n]

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)

            # SHAP summary dot plot
            shap_sum_path = os.path.join(plots_dir, "shap_summary.png")
            plt.figure()
            shap.summary_plot(shap_values, X_sample, show=False)
            plt.tight_layout()
            plt.savefig(shap_sum_path, dpi=150, bbox_inches="tight")
            plt.close()
            logging.info(f"Saved SHAP summary: {shap_sum_path}")

            # SHAP summary bar plot
            shap_bar_path = os.path.join(plots_dir, "shap_summary_bar.png")
            plt.figure()
            shap.summary_plot(shap_values, X_sample, show=False, plot_type="bar")
            plt.tight_layout()
            plt.savefig(shap_bar_path, dpi=150, bbox_inches="tight")
            plt.close()
            logging.info(f"Saved SHAP bar summary: {shap_bar_path}")

        except Exception as e:
            logging.error(f"SHAP plotting failed: {e}")

    # ----------------------- main train -----------------------
    def train_model(self, X_train, y_train, X_test, y_test):
        try:
            logging.info("Starting model training and hyperparameter tuning...")

            # Candidate models
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

            # Hyperparameters (used by your evaluate_models)
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

            # Clean NaNs
            mask_train = ~np.isnan(y_train)
            X_train, y_train = X_train[mask_train], y_train[mask_train]
            mask_test = ~np.isnan(y_test)
            X_test, y_test = X_test[mask_test], y_test[mask_test]

            logging.info(f"Cleaned NaNs - train: {np.isnan(y_train).sum()} | test: {np.isnan(y_test).sum()}")

            # Evaluate models via your utility (returns dict of R2)
            model_report = evaluate_models(
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
                models=models, param=params
            )

            # Compute full metrics for all models
            all_results = []
            for model_name, model in models.items():
                try:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    r2 = r2_score(y_test, y_pred)
                    mse, rmse, mae, bias, variance = self.compute_extra_metrics(y_test, y_pred)
                    all_results.append([model_name, r2, mse, rmse, mae, bias, variance])
                except Exception as err:
                    logging.error(f"Model failed during evaluation: {model_name} | {err}")

            df_eval = pd.DataFrame(
                all_results,
                columns=["Model", "R2_Score", "MSE", "RMSE", "MAE", "Bias", "Variance"]
            )

            # Save evaluation report
            evaluation_dir, plots_dir = self._ensure_dirs()
            evaluation_report_path = os.path.join(evaluation_dir, "model_evaluation_report.csv")
            df_eval.to_csv(evaluation_report_path, index=False)
            logging.info(f" Model Evaluation Report saved at: {evaluation_report_path}")

            # ==== Comparison plots (bar charts) ====
            self._save_barh(df_eval, "R2_Score", os.path.join(plots_dir, "r2_score.png"), "R² Score by Model (higher is better)")
            self._save_barh(df_eval, "RMSE", os.path.join(plots_dir, "rmse.png"), "RMSE by Model (lower is better)")
            self._save_barh(df_eval, "MAE", os.path.join(plots_dir, "mae.png"), "MAE by Model (lower is better)")
            self._save_barh(df_eval, "Bias", os.path.join(plots_dir, "bias.png"), "Bias by Model (closer to 0 is better)")
            self._save_barh(df_eval, "Variance", os.path.join(plots_dir, "variance.png"), "Prediction Variance by Model")

            
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]
            logging.info(f" Best model selected: {best_model_name} (R² = {best_model_score:.4f})")

         
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            # Best-model plots
            self._plot_predicted_vs_actual(y_test, y_test_pred, os.path.join(plots_dir, "predicted_vs_actual.png"))
            self._plot_residuals(y_test, y_test_pred,
                                 os.path.join(plots_dir, "residuals_scatter.png"),
                                 os.path.join(plots_dir, "residuals_hist.png"))

            # SHAP interpretability (only for tree-based best models)
            self._try_shap(best_model, X_test, plots_dir)

            # Compute and log final metrics (unchanged)
            train_metric = get_regression_score(y_true=y_train, y_pred=y_train_pred)
            test_metric = get_regression_score(y_true=y_test, y_pred=y_test_pred)
            logging.info(f"Training metrics: {train_metric}")
            logging.info(f"Testing metrics: {test_metric}")

            # Saving the final model
            preprocessor = load_object(self.data_transformation_artifact.transformed_object_file_path)
            earthquake_model = EarthQuakeModel(preprocessor=preprocessor, model=best_model)
            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)
            save_object(self.model_trainer_config.trained_model_file_path, earthquake_model)
            logging.info(" Best model saved successfully.")

            
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=train_metric,
                test_metric_artifact=test_metric
            )
            return model_trainer_artifact

        except Exception as e:
            raise Earthquake_Magnitude_EstimationException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """Executes the full model trainer pipeline."""
        try:
            logging.info("Initiating model trainer pipeline...")
            train_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_test_file_path)

            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            return self.train_model(X_train, y_train, X_test, y_test)

        except Exception as e:
            raise Earthquake_Magnitude_EstimationException(e, sys)
