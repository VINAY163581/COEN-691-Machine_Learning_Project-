import sys
import os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from Earthquake_Magnitude_Estimation.constant.training_pipeline import TARGET_COLUMN
from Earthquake_Magnitude_Estimation.constant.training_pipeline import DATA_TRANSFORMATION_IMPUTER_PARAMS

from Earthquake_Magnitude_Estimation.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact
)

from Earthquake_Magnitude_Estimation.entity.config_entity import DataTransformationConfig
from Earthquake_Magnitude_Estimation.exception.exception import Earthquake_Magnitude_EstimationException 
from Earthquake_Magnitude_Estimation.logging.logger import logging
from Earthquake_Magnitude_Estimation.utils.main_utils.utils import save_numpy_array_data, save_object


class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise Earthquake_Magnitude_EstimationException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        """Read CSV file and return a DataFrame."""
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise Earthquake_Magnitude_EstimationException(e, sys)

    @staticmethod
    def preprocess_datetime(df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert datetime-like columns (e.g., '1930-12-08T08:01:02.000Z') to numeric UNIX timestamps.
        Non-datetime columns are left unchanged.
        """
        try:
            df_copy = df.copy()
            object_cols = df_copy.select_dtypes(include='object').columns
            for col in object_cols:
                # Try to parse datetime
                parsed = pd.to_datetime(df_copy[col], errors='coerce', format='%Y-%m-%dT%H:%M:%S.%fZ')
                if parsed.notna().any():
                    df_copy[col] = parsed.astype(np.int64) // 10**9
            return df_copy
        except Exception as e:
            raise Earthquake_Magnitude_EstimationException(e, sys)

    @staticmethod
    def get_data_transformer_object(numeric_columns, categorical_columns) -> ColumnTransformer:
        """
        Create and return a ColumnTransformer with pipelines for numeric and categorical features.
        """
        try:
            logging.info("Initializing numeric and categorical transformation pipelines")

            numeric_pipeline = Pipeline([
                ("imputer", KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)),
                ("scaler", StandardScaler())
            ])

            categorical_pipeline = Pipeline([
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numeric_pipeline, numeric_columns),
                    ("cat", categorical_pipeline, categorical_columns)
                ],
                remainder='drop'
            )

            return preprocessor
        except Exception as e:
            raise Earthquake_Magnitude_EstimationException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        logging.info("Entered initiate_data_transformation method of DataTransformation class")
        try:
            # Load train and test datasets
            train_df = self.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = self.read_data(self.data_validation_artifact.valid_test_file_path)

            # Split features and target
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN].replace(-1, 0)

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN].replace(-1, 0)

            # Convert datetime columns to numeric
            input_feature_train_df = self.preprocess_datetime(input_feature_train_df)
            input_feature_test_df = self.preprocess_datetime(input_feature_test_df)

            # Identify column types
            numeric_columns = input_feature_train_df.select_dtypes(include=np.number).columns.tolist()
            categorical_columns = input_feature_train_df.select_dtypes(exclude=np.number).columns.tolist()

            logging.info(f"Numeric columns: {numeric_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")

            # Build preprocessor
            preprocessor = self.get_data_transformer_object(numeric_columns, categorical_columns)

            # Transform data
            transformed_input_train_feature = preprocessor.fit_transform(input_feature_train_df)
            transformed_input_test_feature = preprocessor.transform(input_feature_test_df)

            # Combine with target
            train_arr = np.c_[transformed_input_train_feature, np.array(target_feature_train_df)]
            test_arr = np.c_[transformed_input_test_feature, np.array(target_feature_test_df)]

            # Ensure directories exist
            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_train_file_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_test_file_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_object_file_path), exist_ok=True)

            # Save numpy arrays
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, test_arr)

            # Save as CSV for inspection
            train_csv_path = self.data_transformation_config.transformed_train_file_path.replace(".npy", ".csv")
            test_csv_path = self.data_transformation_config.transformed_test_file_path.replace(".npy", ".csv")
            pd.DataFrame(train_arr).to_csv(train_csv_path, index=False)
            pd.DataFrame(test_arr).to_csv(test_csv_path, index=False)

            # Save preprocessor
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
            save_object("final_model/preprocessor.pkl", preprocessor)

            logging.info(f"Saved transformed train CSV at: {train_csv_path}")
            logging.info(f"Saved transformed test CSV at: {test_csv_path}")
            logging.info(f"Data transformation completed successfully")

            # Return artifact
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

            return data_transformation_artifact

        except Exception as e:
            raise Earthquake_Magnitude_EstimationException(e, sys)