import os
import sys
import json
import sklearn 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL=os.getenv("MONGO_DB_URL")
print(MONGO_DB_URL)
import certifi
ca=certifi.where()

import pandas as pd
import numpy as np
import pymongo
from Earthquake_Magnitude_Estimation.exception.exception import Earthquake_Magnitude_EstimationException
from Earthquake_Magnitude_Estimation.logging.logger import logging

class Earthquake_Data_Extract():
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise Earthquake_Magnitude_EstimationException(e,sys)
        
    def csv_to_json_convertor(self,file_path):
        try:
            data=pd.read_csv(file_path)
            data.reset_index(drop=True,inplace=True)
            records=list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            raise Earthquake_Magnitude_EstimationException(e,sys)
        
    def insert_data_mongodb(self,records,database,collection):
        try:
            self.database=database
            self.collection=collection
            self.records=records

            self.mongo_client=pymongo.MongoClient(MONGO_DB_URL)
            self.database = self.mongo_client[self.database]
            
            self.collection=self.database[self.collection]
            self.collection.insert_many(self.records)
            return(len(self.records))
        except Exception as e:
            raise (e,sys)
        
if __name__=='__main__':
    FILE_PATH=""
    DATABASE=""
    Collection=""
    networkobj=Earthquake_Data_Extract()
    records=networkobj.csv_to_json_convertor(file_path=FILE_PATH)
    print(records)
    no_of_records=networkobj.insert_data_mongodb(records,DATABASE,Collection)
    print(no_of_records)