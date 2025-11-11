from Earthquake_Magnitude_Estimation.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from Earthquake_Magnitude_Estimation.entity.config_entity import DataValidationConfig
from Earthquake_Magnitude_Estimation.exception.exception import Earthquake_Magnitude_EstimationException
from Earthquake_Magnitude_Estimation.logging.logger import logging
from Earthquake_Magnitude_Estimation.constant.training_pipeline import SCHEMA_FILE_PATH
from scipy.stats import ks_2samp
import pandas as pd
import os, sys
from Earthquake_Magnitude_Estimation.utils.main_utils.utils import read_yaml_file, write_yaml_file


class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_config: DataValidationConfig):
        print("test")
        """
        Initialize DataValidation component with artifacts and configurations.
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
            logging.info("DataValidation initialized successfully.")
        except Exception as e:
            raise Earthquake_Magnitude_EstimationException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        """Reads a CSV file into a pandas DataFrame."""
        try:
            logging.info(f"Reading data from file: {file_path}")
            return pd.read_csv(file_path)
        except Exception as e:
            raise Earthquake_Magnitude_EstimationException(e, sys)

    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        """
        Validate if dataframe contains all required columns from schema.
        Extra columns in the dataset will be ignored.
        """
        try:
            required_columns = list(self._schema_config["columns"].keys())
            dataframe_columns = list(dataframe.columns)

            missing_columns = [col for col in required_columns if col not in dataframe_columns]

            if missing_columns:
                logging.error(f"❌ Missing required columns: {missing_columns}")
                return False
            else:
                if len(dataframe_columns) > len(required_columns):
                    extra_columns = [col for col in dataframe_columns if col not in required_columns]
                    logging.warning(f"⚠️ Ignoring extra columns: {extra_columns}")
                return True

        except Exception as e:
            raise Earthquake_Magnitude_EstimationException(e, sys)

    def detect_dataset_drift(self, base_df: pd.DataFrame, current_df: pd.DataFrame, threshold=0.05) -> bool:
        """
        Detect dataset drift using KS-Test for each numeric column.
        Returns True if no drift found, False otherwise.
        """
        try:
            logging.info("Starting dataset drift detection...")
            status = True
            report = {}

            for column in base_df.columns:
                if column not in current_df.columns:
                    continue

                if pd.api.types.is_numeric_dtype(base_df[column]) and pd.api.types.is_numeric_dtype(current_df[column]):
                    d1, d2 = base_df[column], current_df[column]
                    ks_result = ks_2samp(d1, d2)
                    drift_found = ks_result.pvalue < threshold

                    report[column] = {
                        "p_value": float(ks_result.pvalue),
                        "drift_status": drift_found
                    }

                    if drift_found:
                        logging.warning(f"Drift detected in column: {column}")
                        status = False

            drift_report_file_path = self.data_validation_config.drift_report_file_path
            os.makedirs(os.path.dirname(drift_report_file_path), exist_ok=True)
            write_yaml_file(file_path=drift_report_file_path, content=report)

            logging.info(f"Drift report generated at: {drift_report_file_path}")
            return status

        except Exception as e:
            raise Earthquake_Magnitude_EstimationException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Executes the full validation process:
        1. Read train/test data
        2. Keep only required columns
        3. Validate schema
        4. Detect drift
        5. Save valid files and drift report
        """
        try:
            logging.info("🚀 Starting data validation process...")

            train_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            train_df = self.read_data(train_file_path)
            test_df = self.read_data(test_file_path)

            # Keep only schema columns (ignore extras)
            required_columns = list(self._schema_config["columns"].keys())
            train_df = train_df[[col for col in train_df.columns if col in required_columns]]
            test_df = test_df[[col for col in test_df.columns if col in required_columns]]

            # Validate column presence
            if not self.validate_number_of_columns(train_df):
                raise Exception("Train dataset does not contain all required columns.")
            if not self.validate_number_of_columns(test_df):
                raise Exception("Test dataset does not contain all required columns.")

            # Detect dataset drift
            drift_status = self.detect_dataset_drift(base_df=train_df, current_df=test_df)

            # Save validated datasets
            os.makedirs(os.path.dirname(self.data_validation_config.valid_train_file_path), exist_ok=True)
            train_df.to_csv(self.data_validation_config.valid_train_file_path, index=False, header=True)
            test_df.to_csv(self.data_validation_config.valid_test_file_path, index=False, header=True)

            logging.info(f"✅ Validated train file saved at: {self.data_validation_config.valid_train_file_path}")
            logging.info(f"✅ Validated test file saved at: {self.data_validation_config.valid_test_file_path}")

            # Create artifact
            data_validation_artifact = DataValidationArtifact(
                validation_status=drift_status,
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )

            logging.info(f"🏁 Data Validation completed successfully.")
            logging.info(f"Data Validation Artifact: {data_validation_artifact}")
            return data_validation_artifact

        except Exception as e:
            raise Earthquake_Magnitude_EstimationException(e, sys)
