from Earthquake_Magnitude_Estimation.components.data_ingestion import DataIngestion
from Earthquake_Magnitude_Estimation.components.data_validation import DataValidation
from Earthquake_Magnitude_Estimation.exception.exception import Earthquake_Magnitude_EstimationException
from Earthquake_Magnitude_Estimation.logging.logger import logging
from Earthquake_Magnitude_Estimation.entity.config_entity import DataIngestionConfig, DataValidationConfig
from Earthquake_Magnitude_Estimation.entity.config_entity import TrainingPipelineConfig
import sys


if __name__ == "__main__":
    try:
        training_pipeline_config = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        logging.info(f"Initiated the data ingestion component")
        data_ingestion_artifact = DataIngestion(data_ingestion_config=data_ingestion_config).initiate_data_ingestion()
        logging.info(f"Completed the data ingestion component")
        print(data_ingestion_artifact)
        data_validation_config = DataValidationConfig(training_pipeline_config)
        data_validation = DataValidation(data_ingestion_artifact,data_validation_config)
        logging.info(f"Initiated the data validation component")
        data_validation_artifact = data_validation.initiate_data_validation()
        print(data_validation_artifact)

    except Exception as e:
        raise Earthquake_Magnitude_EstimationException(e, sys) 