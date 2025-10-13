from Earthquake_Magnitude_Estimation.components.data_ingestion import DataIngestion
from Earthquake_Magnitude_Estimation.exception.exception import Earthquake_Magnitude_EstimationException
from Earthquake_Magnitude_Estimation.logging.logger import logging
from Earthquake_Magnitude_Estimation.entity.config_entity import DataIngestionConfig
from Earthquake_Magnitude_Estimation.entity.config_entity import TrainingPipelineConfig
import sys


if __name__ == "__main__":
    try:
        training_pipeline_config = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        logging.info(f"Initiated the data ingestion component")
        dataingestionartifact = DataIngestion(data_ingestion_config=data_ingestion_config).initiate_data_ingestion()
        print(dataingestionartifact)
    except Exception as e:
        raise Earthquake_Magnitude_EstimationException(e, sys) 