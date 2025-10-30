from Earthquake_Magnitude_Estimation.components.data_ingestion import DataIngestion
from Earthquake_Magnitude_Estimation.components.data_validation import DataValidation
from Earthquake_Magnitude_Estimation.components.data_transformation import DataTransformation
from Earthquake_Magnitude_Estimation.exception.exception import Earthquake_Magnitude_EstimationException
from Earthquake_Magnitude_Estimation.logging.logger import logging
from Earthquake_Magnitude_Estimation.entity.config_entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig
from Earthquake_Magnitude_Estimation.entity.config_entity import TrainingPipelineConfig

from Earthquake_Magnitude_Estimation.components.model_trainer import ModelTrainer
from Earthquake_Magnitude_Estimation.entity.config_entity import ModelTrainerConfig

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
        data_transformation_config = DataTransformationConfig(training_pipeline_config)
        logging.info("Initiated the data transformation component")
        data_transformation = DataTransformation(data_validation_artifact, data_transformation_config)
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        print(data_transformation_artifact)
        logging.info("Completed the data transformation component")

        logging.info("Model Training sstared")
        model_trainer_config=ModelTrainerConfig(training_pipeline_config)
        model_trainer=ModelTrainer(model_trainer_config=model_trainer_config,data_transformation_artifact=data_transformation_artifact)
        model_trainer_artifact=model_trainer.initiate_model_trainer()

        logging.info("Model Training artifact created")

    except Exception as e:
        raise Earthquake_Magnitude_EstimationException(e, sys)