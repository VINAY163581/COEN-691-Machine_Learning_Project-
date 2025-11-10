from Earthquake_Magnitude_Estimation.exception.exception import Earthquake_Magnitude_EstimationException
from Earthquake_Magnitude_Estimation.logging.logger import logging

from Earthquake_Magnitude_Estimation.entity.config_entity import DataIngestionConfig
from Earthquake_Magnitude_Estimation.entity.artifact_entity import DataIngestionArtifact

import os
import sys
import numpy as np
import pandas as pd
import pymongo
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

load_dotenv()
MONGO_DB_URL = os.getenv("MONGO_DB_URL")


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise Earthquake_Magnitude_EstimationException(e, sys)

    def export_collection_as_dataframe(self):
        """
        Read data from MongoDB and return as DataFrame
        """
        try:
            logging.info("Connecting to MongoDB...")

            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name
            
            mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            collection = mongo_client[database_name][collection_name]

            logging.info(f"Reading data from MongoDB collection: {database_name}.{collection_name}")
            df = pd.DataFrame(list(collection.find()))

            if df.empty:
                raise Earthquake_Magnitude_EstimationException(
                    f"MongoDB collection '{collection_name}' is EMPTY.",
                    sys
                )

            logging.info(f"Data fetched from MongoDB: {df.shape} rows")

            if "_id" in df.columns:
                df = df.drop(columns=["_id"], axis=1)

            df.replace({"na": np.nan}, inplace=True)

            return df

        except Exception as e:
            raise Earthquake_Magnitude_EstimationException(e, sys)

    def export_data_into_feature_store(self, dataframe: pd.DataFrame):
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path

            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)

            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            logging.info(f"Feature store saved at: {feature_store_file_path}")

            return dataframe

        except Exception as e:
            raise Earthquake_Magnitude_EstimationException(e, sys)

    def split_data_as_train_test(self, dataframe: pd.DataFrame):
        try:
            logging.info(f"Splitting dataset: {dataframe.shape}")

            if dataframe.shape[0] == 0:
                raise Earthquake_Magnitude_EstimationException(
                    "Dataset is EMPTY. Cannot perform train-test split.",
                    sys
                )

            train_set, test_set = train_test_split(
                dataframe, 
                test_size=self.data_ingestion_config.train_test_split_ratio
            )

            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)

            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False)

            logging.info("Train and test files generated successfully.")

        except Exception as e:
            raise Earthquake_Magnitude_EstimationException(e, sys)

    def initiate_data_ingestion(self):
        try:
            logging.info(">>>> Starting Data Ingestion <<<<")

            dataframe = self.export_collection_as_dataframe()
            dataframe = self.export_data_into_feature_store(dataframe)

            self.split_data_as_train_test(dataframe)

            artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path
            )

            logging.info(f"Data Ingestion Artifact Created: {artifact}")

            return artifact

        except Exception as e:
            raise Earthquake_Magnitude_EstimationException(e, sys)
