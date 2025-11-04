import sys
import os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from Earthquake_Magnitude_Estimation.constant.training_pipeline import TARGET_COLUMN, DATA_TRANSFORMATION_IMPUTER_PARAMS
from Earthquake_Magnitude_Estimation.entity.artifact_entity import DataValidationArtifact, DataTransformationArtifact
from Earthquake_Magnitude_Estimation.entity.config_entity import DataTransformationConfig
from Earthquake_Magnitude_Estimation.exception.exception import Earthquake_Magnitude_EstimationException
from Earthquake_Magnitude_Estimation.logging.logger import logging
from Earthquake_Magnitude_Estimation.utils.main_utils.utils import save_object, save_numpy_array_data


class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise Earthquake_Magnitude_EstimationException(e, sys)

    def read_data(self, file_path: str) -> pd.DataFrame:
        try:
            logging.info(f"Reading data from file: {file_path}")
            return pd.read_csv(file_path)
        except Exception as e:
            raise Earthquake_Magnitude_EstimationException(e, sys)

    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logging.info("Preprocessing dataframe...")

            # Convert numeric columns
            numeric_cols = ['latitude', 'longitude', 'depth', 'mag', 'nst', 'gap', 'dmin', 'rms']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Convert datetime columns to numeric timestamps
            datetime_cols = ['time', 'updated']
            for col in datetime_cols:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    df[col] = df[col].astype('int64') // 10**9  # convert to seconds since epoch

            # Convert categorical columns to string and handle missing values
            for col in df.select_dtypes(include=['object']).columns:
                df[col] = df[col].fillna("missing").astype(str)

            logging.info("Dataframe preprocessing completed.")
            return df

        except Exception as e:
            raise Earthquake_Magnitude_EstimationException(e, sys)

    def get_data_transformer_object(self, numerical_columns: list, categorical_columns: list) -> ColumnTransformer:
        try:
            logging.info("Creating data transformer object...")

            numeric_pipeline = Pipeline([
                ("imputer", KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)),
                ("scaler", StandardScaler())
            ])

            categorical_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("onehot", OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])

            preprocessor = ColumnTransformer([
                ("num", numeric_pipeline, numerical_columns),
                ("cat", categorical_pipeline, categorical_columns)
            ])

            logging.info("Data transformer object created successfully.")
            return preprocessor

        except Exception as e:
            raise Earthquake_Magnitude_EstimationException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Starting data transformation process...")

            # Load validated train and test data
            train_df = self.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = self.read_data(self.data_validation_artifact.valid_test_file_path)

            # Preprocess dataframes
            train_df = self.preprocess_dataframe(train_df)
            test_df = self.preprocess_dataframe(test_df)

            # Split into input and target
            X_train = train_df.drop(columns=[TARGET_COLUMN])
            y_train = train_df[TARGET_COLUMN].values.reshape(-1, 1)

            X_test = test_df.drop(columns=[TARGET_COLUMN])
            y_test = test_df[TARGET_COLUMN].values.reshape(-1, 1)

            # Identify numeric and categorical columns
            numerical_columns = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_columns = X_train.select_dtypes(include=['object']).columns.tolist()

            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")

            # Preprocessor
            preprocessor = self.get_data_transformer_object(numerical_columns, categorical_columns)

            # Fit and transform train data
            logging.info("Fitting and transforming training data...")
            X_train_arr = preprocessor.fit_transform(X_train)

            # Transform test data
            logging.info("Transforming test data...")
            X_test_arr = preprocessor.transform(X_test)

            # Combine with target
            train_arr = np.hstack([X_train_arr, y_train])
            test_arr = np.hstack([X_test_arr, y_test])

            # Save numpy arrays
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, test_arr)

            # Save preprocessor
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)

            logging.info("Data transformation completed successfully.")

            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

        except Exception as e:
            raise Earthquake_Magnitude_EstimationException(e, sys)
