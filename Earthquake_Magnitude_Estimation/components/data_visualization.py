import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from Earthquake_Magnitude_Estimation.exception.exception import Earthquake_Magnitude_EstimationException
from Earthquake_Magnitude_Estimation.entity.artifact_entity import (
    DataTransformationArtifact,
    DataVisualizationArtifact
)
from Earthquake_Magnitude_Estimation.entity.config_entity import DataVisualizationConfig
from Earthquake_Magnitude_Estimation.logging.logger import logging


class DataVisualization:
    def __init__(self,
                 data_transformation_artifact: DataTransformationArtifact,
                 data_visualization_config: DataVisualizationConfig):
        """
        Initialize DataVisualization with artifacts and config.
        """
        try:
            self.data_transformation_artifact = data_transformation_artifact
            self.data_visualization_config = data_visualization_config
        except Exception as e:
            raise Earthquake_Magnitude_EstimationException(e, sys)

    def read_transformed_data(self, file_path: str) -> pd.DataFrame:
        """
        Reads transformed numpy array and converts it into a DataFrame
        with placeholder feature column names.
        """
        try:
            logging.info(f"Loading transformed data from: {file_path}")
            arr = np.load(file_path)
            n_features = arr.shape[1] - 1  # Last column is assumed to be target
            feature_columns = [f"feature_{i}" for i in range(n_features)]
            feature_columns.append("target")
            df = pd.DataFrame(arr, columns=feature_columns)
            return df
        except Exception as e:
            raise Earthquake_Magnitude_EstimationException(e, sys)

    def plot_correlation_heatmap(self, df: pd.DataFrame, save_path: str):
        """
        Generates and saves a correlation heatmap for the given dataframe.
        """
        try:
            plt.figure(figsize=(12, 10))
            sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
            plt.title("Correlation Heatmap")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close()
            logging.info(f"Correlation heatmap saved at: {save_path}")
        except Exception as e:
            raise Earthquake_Magnitude_EstimationException(e, sys)

    def initiate_data_visualization(self) -> DataVisualizationArtifact:
        """
        Creates visualizations (heatmaps, numeric/categorical distribution placeholders)
        and returns DataVisualizationArtifact paths.
        """
        try:
            logging.info("Initiated the data visualization component")

            # Load transformed data arrays
            train_df = self.read_transformed_data(
                self.data_transformation_artifact.transformed_train_file_path
            )
            test_df = self.read_transformed_data(
                self.data_transformation_artifact.transformed_test_file_path
            )

            # Define correlation heatmap paths
            train_heatmap_path = self.data_visualization_config.train_correlation_heatmap_path
            test_heatmap_path = self.data_visualization_config.test_correlation_heatmap_path

            # Define placeholder directories for numeric & categorical distributions
            root_dir = self.data_visualization_config.root_dir
            train_numeric_dir = os.path.join(root_dir, "train_numeric_distribution")
            test_numeric_dir = os.path.join(root_dir, "test_numeric_distribution")
            train_categorical_dir = os.path.join(root_dir, "train_categorical_distribution")
            test_categorical_dir = os.path.join(root_dir, "test_categorical_distribution")

            # Ensure directories exist
            os.makedirs(train_numeric_dir, exist_ok=True)
            os.makedirs(test_numeric_dir, exist_ok=True)
            os.makedirs(train_categorical_dir, exist_ok=True)
            os.makedirs(test_categorical_dir, exist_ok=True)

            # Generate correlation heatmaps
            self.plot_correlation_heatmap(train_df, train_heatmap_path)
            self.plot_correlation_heatmap(test_df, test_heatmap_path)

            logging.info("Data visualization completed successfully.")

            # ✅ Return DataVisualizationArtifact with all required attributes
            return DataVisualizationArtifact(
                train_correlation_heatmap_path=train_heatmap_path,
                test_correlation_heatmap_path=test_heatmap_path,
                train_numeric_distribution_dir=train_numeric_dir,
                test_numeric_distribution_dir=test_numeric_dir,
                train_categorical_distribution_dir=train_categorical_dir,
                test_categorical_distribution_dir=test_categorical_dir
            )

        except Exception as e:
            raise Earthquake_Magnitude_EstimationException(e, sys)
