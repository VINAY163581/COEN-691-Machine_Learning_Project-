from Earthquake_Magnitude_Estimation.components.data_ingestion import DataIngestion
from Earthquake_Magnitude_Estimation.components.data_validation import DataValidation
from Earthquake_Magnitude_Estimation.components.data_transformation import DataTransformation
from Earthquake_Magnitude_Estimation.components.data_visualization import DataVisualization
from Earthquake_Magnitude_Estimation.exception.exception import Earthquake_Magnitude_EstimationException
from Earthquake_Magnitude_Estimation.logging.logger import logging
from Earthquake_Magnitude_Estimation.entity.config_entity import (
    DataIngestionConfig, DataValidationConfig, DataTransformationConfig,
    DataVisualizationConfig, TrainingPipelineConfig
)
from Earthquake_Magnitude_Estimation.components.model_trainer import ModelTrainer
from Earthquake_Magnitude_Estimation.entity.config_entity import ModelTrainerConfig

from Earthquake_Magnitude_Estimation.components.data_ingestion import DataIngestion
from Earthquake_Magnitude_Estimation.components.data_validation import DataValidation
from Earthquake_Magnitude_Estimation.components.data_transformation import DataTransformation
from Earthquake_Magnitude_Estimation.components.data_visualization import DataVisualization
from Earthquake_Magnitude_Estimation.components.model_trainer import ModelTrainer

from Earthquake_Magnitude_Estimation.exception.exception import Earthquake_Magnitude_EstimationException
from Earthquake_Magnitude_Estimation.logging.logger import logging

from Earthquake_Magnitude_Estimation.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    DataVisualizationConfig,
    ModelTrainerConfig,
    TrainingPipelineConfig
)

import sys


if __name__ == "__main__":
    try:
        # PIPELINE CONFIG  #
        training_pipeline_config = TrainingPipelineConfig()

        # DATA INGESTION #
        logging.info("===== Starting Data Ingestion =====")
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        logging.info(f"Data Ingestion completed:\n{data_ingestion_artifact}\n")

        # DATA VALIDATION #
        logging.info("===== Starting Data Validation =====")
        data_validation_config = DataValidationConfig(training_pipeline_config)
        data_validation = DataValidation(
            data_ingestion_artifact=data_ingestion_artifact,
            data_validation_config=data_validation_config
        )
        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info(f"Data Validation completed:\n{data_validation_artifact}\n")

        #  DATA TRANSFORMATION #
        logging.info("===== Starting Data Transformation =====")
        data_transformation_config = DataTransformationConfig(training_pipeline_config)
        data_transformation = DataTransformation(
            data_validation_artifact=data_validation_artifact,
            data_transformation_config=data_transformation_config
        )
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        logging.info(f"Data Transformation completed:\n{data_transformation_artifact}\n")

        # DATA VISUALIZATION (AUTO-TRIGGERS MODEL TRAINER) #
        logging.info("===== Starting Data Visualization =====")
        data_visualization_config = DataVisualizationConfig(training_pipeline_config)
        model_trainer_config = ModelTrainerConfig(training_pipeline_config)

        data_visualization = DataVisualization(
            data_transformation_artifact=data_transformation_artifact,
            data_visualization_config=data_visualization_config,
            model_trainer_config=model_trainer_config
        )

        data_visualization_artifact = data_visualization.initiate_data_visualization()
        logging.info(f"Data Visualization completed:\n{data_visualization_artifact}\n")

        logging.info(" Pipeline execution completed successfully.")

    except Exception as e:
        raise Earthquake_Magnitude_EstimationException(e, sys)

