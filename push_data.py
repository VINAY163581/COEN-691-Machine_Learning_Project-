import os
import sys
import json
from io import BytesIO
 
from dotenv import load_dotenv
load_dotenv()
 
MONGO_DB_URL = os.getenv("MONGO_DB_URL")
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
 
AZURE_CONTAINER_NAME = os.getenv("AZURE_CONTAINER_NAME")
AZURE_BLOB_NAME = os.getenv("AZURE_BLOB_NAME")
 
print("Using MONGO_DB_URL:", bool(MONGO_DB_URL))
print("Using AZURE_STORAGE_CONNECTION_STRING:", bool(AZURE_STORAGE_CONNECTION_STRING))
 
import certifi
ca = certifi.where()
 
import pandas as pd
import numpy as np
import pymongo
 
 
from Earthquake_Magnitude_Estimation.exception.exception import (
    Earthquake_Magnitude_EstimationException,
)
from Earthquake_Magnitude_Estimation.logging.logger import logging
 
 
from azure.storage.blob import BlobServiceClient
 
 
class Earthquake_Data_Extract:
    def __init__(self):
        try:
            
            pass
        except Exception as e:
            raise Earthquake_Magnitude_EstimationException(e, sys)
 
    def read_csv_from_azure(self, connection_string: str,
                            container_name: str,
                            blob_name: str) -> pd.DataFrame:
        """
        Read a CSV file from Azure Blob Storage into a pandas DataFrame.
        """
        try:
            if not connection_string:
                raise ValueError("Azure connection string is missing.")
            logging.info(f"Connecting to Azure Blob container '{container_name}', blob '{blob_name}'")
            blob_service_client = BlobServiceClient.from_connection_string(connection_string)
            container_client = blob_service_client.get_container_client(container_name)
            blob_client = container_client.get_blob_client(blob_name)
 
            stream = blob_client.download_blob().readall()
            
            df = pd.read_csv(BytesIO(stream))
            df.reset_index(drop=True, inplace=True)
            logging.info(f"Read {len(df)} rows from Azure blob '{blob_name}'")
            return df
        except Exception as e:
            raise Earthquake_Magnitude_EstimationException(e, sys)
 
    def dataframe_to_records(self, df: pd.DataFrame) -> list:
        """
        Convert a DataFrame to a list-of-dicts suitable for insert_many.
        """
        try:
            records = list(json.loads(df.T.to_json()).values())
            return records
        except Exception as e:
            raise Earthquake_Magnitude_EstimationException(e, sys)
 
    def insert_data_mongodb(self, records: list, database: str, collection: str) -> int:
        """
        Insert a list of dict records into MongoDB Atlas.
        Returns number of inserted records.
        """
        try:
            if not MONGO_DB_URL:
                raise ValueError("MONGO_DB_URL environment variable is not set.")
            logging.info(f"Connecting to MongoDB and inserting into {database}.{collection}")
            mongo_client = pymongo.MongoClient(MONGO_DB_URL, tlsCAFile=ca)
            db = mongo_client[database]
            coll = db[collection]
            if not records:
                logging.warning("No records to insert.")
                return 0
            
            result = coll.insert_many(records)
            inserted = len(result.inserted_ids)
            logging.info(f"Inserted {inserted} documents into MongoDB collection '{collection}'")
            return inserted
        except Exception as e:
            
            raise Earthquake_Magnitude_EstimationException(e, sys)
 
 
if __name__ == "__main__":
    try:
        
        FILE_CONTAINER = AZURE_CONTAINER_NAME or "your-container-name"
        FILE_BLOB = AZURE_BLOB_NAME or "dataset.csv"
        DATABASE = "Earthquake_Data"
        COLLECTION = "earthquake_collection"
 
        extractor = Earthquake_Data_Extract()
 
        
        df = extractor.read_csv_from_azure(
            connection_string=AZURE_STORAGE_CONNECTION_STRING,
            container_name=FILE_CONTAINER,
            blob_name=FILE_BLOB,
        )
 
       
        records = extractor.dataframe_to_records(df)
      
        print("sample record:", records[0] if records else "no records")
 
        
        no_of_records = extractor.insert_data_mongodb(records, DATABASE, COLLECTION)
        print(f"Inserted {no_of_records} records into MongoDB")
    except Exception as e:
        
        print("Failed:", e)
        raise