import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from Earthquake_Magnitude_Estimation.exception.exception import Earthquake_Magnitude_EstimationException
from Earthquake_Magnitude_Estimation.entity.artifact_entity import (
    DataTransformationArtifact,
    DataVisualizationArtifact,
    ModelTrainerArtifact
)
from Earthquake_Magnitude_Estimation.entity.config_entity import DataVisualizationConfig, ModelTrainerConfig
from Earthquake_Magnitude_Estimation.logging.logger import logging
from Earthquake_Magnitude_Estimation.components.model_trainer import ModelTrainer


class DataVisualization:
    def __init__(self,
                 data_transformation_artifact: DataTransformationArtifact,
                 data_visualization_config: DataVisualizationConfig,
                 model_trainer_config: ModelTrainerConfig):

        try:
            self.data_transformation_artifact = data_transformation_artifact
            self.data_visualization_config = data_visualization_config
            self.model_trainer_config = model_trainer_config
        except Exception as e:
            raise Earthquake_Magnitude_EstimationException(e, sys)

    # --------------------------
    # ✅ Load Transformed Data
    # --------------------------
    def read_transformed_data(self, file_path: str) -> pd.DataFrame:
        try:
            logging.info(f"Loading transformed data from: {file_path}")
            arr = np.load(file_path)
            n_features = arr.shape[1] - 1
            columns = [f"feature_{i}" for i in range(n_features)] + ["target"]
            return pd.DataFrame(arr, columns=columns)
        except Exception as e:
            raise Earthquake_Magnitude_EstimationException(e, sys)

    # --------------------------
    # ✅ Basic Plots
    # --------------------------
    def plot_target_distribution(self, df, path):
        try:
            plt.figure(figsize=(8,6))
            sns.histplot(df["target"], bins=30, kde=True)
            plt.title("Target Distribution")
            plt.savefig(path)
            plt.close()
            logging.info(f"✅ Target distribution saved at: {path}")
        except Exception as e:
            raise Earthquake_Magnitude_EstimationException(e, sys)

    def plot_top_correlations(self, df, path, top_n=10):
        try:
            corr = df.corr()["target"].drop("target").sort_values(ascending=False)
            top = corr.head(top_n)

            plt.figure(figsize=(10,6))
            sns.barplot(x=top.values, y=top.index)
            plt.title("Top Feature Correlations With Target")
            plt.savefig(path)
            plt.close()
            logging.info(f"✅ Correlation barplot saved at: {path}")
        except Exception as e:
            raise Earthquake_Magnitude_EstimationException(e, sys)

    def plot_correlation_heatmap(self, df, path, top_n=10):
        try:
            corr = df.corr()
            important = corr["target"].abs().sort_values(ascending=False).head(top_n).index
            sub_corr = df[important].corr()

            plt.figure(figsize=(12,8))
            sns.heatmap(sub_corr, annot=True, cmap="coolwarm")
            plt.title("Correlation Heatmap")
            plt.savefig(path)
            plt.close()
            logging.info(f"✅ Heatmap saved at: {path}")
        except Exception as e:
            raise Earthquake_Magnitude_EstimationException(e, sys)

    # --------------------------
    # ✅ NEW: Pairplot
    # --------------------------
    def plot_pairplot(self, df, path, top_n=5):
        try:
            corr = df.corr()["target"].abs().sort_values(ascending=False)
            top_features = corr.head(top_n).index.tolist()

            sns.pairplot(df[top_features], diag_kind="kde")
            plt.savefig(path)
            plt.close()
            logging.info(f"✅ Pairplot saved at: {path}")
        except Exception as e:
            raise Earthquake_Magnitude_EstimationException(e, sys)

    # --------------------------
    # ✅ NEW: Scatter plots
    # --------------------------
    def plot_feature_vs_target(self, df, path_dir, top_n=6):
        try:
            corr = df.corr()["target"].drop("target").abs().sort_values(ascending=False)
            best = corr.head(top_n).index

            os.makedirs(path_dir, exist_ok=True)

            for feat in best:
                plt.figure(figsize=(6,4))
                sns.scatterplot(x=df[feat], y=df["target"])
                plt.title(f"{feat} vs Target")
                save_path = os.path.join(path_dir, f"{feat}_scatter.png")
                plt.savefig(save_path)
                plt.close()

            logging.info(f"✅ Feature vs Target scatterplots saved in {path_dir}")
        except Exception as e:
            raise Earthquake_Magnitude_EstimationException(e, sys)

    # --------------------------
    # ✅ NEW: Boxplots
    # --------------------------
    def plot_boxplots(self, df, path_dir):
        try:
            os.makedirs(path_dir, exist_ok=True)
            for col in df.columns:
                plt.figure(figsize=(6,4))
                sns.boxplot(x=df[col])
                plt.title(f"Boxplot: {col}")
                plt.savefig(os.path.join(path_dir, f"{col}_boxplot.png"))
                plt.close()

            logging.info(f"✅ Boxplots saved in: {path_dir}")
        except Exception as e:
            raise Earthquake_Magnitude_EstimationException(e, sys)

    # --------------------------
    # ✅ Pipeline Entry Point
    # --------------------------
    def initiate_data_visualization(self):
        try:
            logging.info("📊 Starting Visualization Stage...")

            train_df = self.read_transformed_data(self.data_transformation_artifact.transformed_train_file_path)

            root = self.data_visualization_config.root_dir
            os.makedirs(root, exist_ok=True)

            # Main plots
            self.plot_target_distribution(train_df, os.path.join(root, "target_distribution.png"))
            self.plot_top_correlations(train_df, os.path.join(root, "top_corr.png"))
            self.plot_correlation_heatmap(train_df, os.path.join(root, "heatmap.png"))

            # Additional plots
            self.plot_pairplot(train_df, os.path.join(root, "pairplot.png"))
            self.plot_feature_vs_target(train_df, os.path.join(root, "scatter_plots"))
            self.plot_boxplots(train_df, os.path.join(root, "boxplots"))

            artifact = DataVisualizationArtifact(
                train_correlation_heatmap_path=os.path.join(root, "heatmap.png"),
                test_correlation_heatmap_path=None,
                train_numeric_distribution_dir=None,
                test_numeric_distribution_dir=None,
                train_categorical_distribution_dir=None,
                test_categorical_distribution_dir=None
            )

            logging.info("✅ Visualization completed. Auto-triggering Model Trainer...")

            # ---------------------------
            # ✅ Auto-trigger ModelTrainer
            # ---------------------------
            model_trainer = ModelTrainer(
                model_trainer_config=self.model_trainer_config,
                data_transformation_artifact=self.data_transformation_artifact
            )
            model_trainer_artifact = model_trainer.initiate_model_trainer()

            logging.info("✅ Model Trainer completed successfully.")

            return artifact

        except Exception as e:
            raise Earthquake_Magnitude_EstimationException(e, sys)
