import os
import sys

from Earthquake_Magnitude_Estimation.exception.exception import Earthquake_Magnitude_EstimationException 
from Earthquake_Magnitude_Estimation.logging.logger import logging 

from Earthquake_Magnitude_Estimation.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
from Earthquake_Magnitude_Estimation.entity.config_entity import ModelTrainerConfig



from Earthquake_Magnitude_Estimation.utils.ml_utils.model.estimator import EarthQuakeModel
from Earthquake_Magnitude_Estimation.utils.main_utils.utils import save_object,load_object
from Earthquake_Magnitude_Estimation.utils.main_utils.utils import load_numpy_array_data,evaluate_models
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
import mlflow
from urllib.parse import urlparse

import dagshub
# Configure MLflow tracking
os.environ["MLFLOW_TRACKING_URI"]="https://dagshub.com/Mohammad-Riyazuddin/COEN-691-Machine_Learning_Project-.mlflow"





class ModelTrainer:
    def __init__(self,model_trainer_config:ModelTrainerConfig,data_transformation_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact
        except Exception as e:
            raise Earthquake_Magnitude_EstimationException(e,sys)
        
    def track_mlflow(self, best_model, regressionmetric):
        mlflow.set_registry_uri("https://dagshub.com/krishnaik06/networksecurity.mlflow")
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        with mlflow.start_run():
            rmse = regressionmetric.rmse
            mae = regressionmetric.mae
            r2 = regressionmetric.r2_score

            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2_score", r2)
            
            mlflow.sklearn.log_model(best_model, "model")
            
            # Model registry does not work with file store
            if tracking_url_type_store != "file":
                # Register the model
                mlflow.sklearn.log_model(best_model, "model", registered_model_name=best_model)
            else:
                mlflow.sklearn.log_model(best_model, "model")


        
    def train_model(self,X_train,y_train,x_test,y_test):
        models = {
                "Random Forest": RandomForestRegressor(verbose=1),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(verbose=1),
                "Linear Regression": LinearRegression(),
                "AdaBoost": AdaBoostRegressor(),
            }
        params={
            "Decision Tree": {
                'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                'max_depth': [None, 5, 10, 15],
                'min_samples_split': [2, 5, 10],
            },
            "Random Forest":{
                'n_estimators': [8,16,32,64,128,256],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
            },
            "Gradient Boosting":{
                'learning_rate':[.1,.01,.05,.001],
                'subsample':[0.6,0.7,0.75,0.85,0.9],
                'n_estimators': [8,16,32,64,128,256],
                'max_depth': [3, 4, 5, 6],
            },
            "Linear Regression":{},
            "AdaBoost":{
                'learning_rate':[.1,.01,.001],
                'n_estimators': [8,16,32,64,128,256],
                'loss': ['linear', 'square', 'exponential']
            }
        }
        model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=x_test,y_test=y_test,
                                          models=models,param=params)
        
        ## To get best model score from dict
        best_model_score = max(sorted(model_report.values()))

        ## To get best model name from dict

        best_model_name = list(model_report.keys())[
            list(model_report.values()).index(best_model_score)
        ]
        best_model = models[best_model_name]
        
        # Get predictions
        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(x_test)
        
        # Calculate regression metrics
        from Earthquake_Magnitude_Estimation.utils.ml_utils.metric.regression_metric import get_regression_score
        
        train_metric = get_regression_score(y_true=y_train, y_pred=y_train_pred)
        test_metric = get_regression_score(y_true=y_test, y_pred=y_test_pred)
        
        ## Track the experiments with mlflow
        # self.track_mlflow(best_model, test_metric)

        preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            
        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path,exist_ok=True)

        EarthQuake_Model=EarthQuakeModel(preprocessor=preprocessor,model=best_model)
        save_object(self.model_trainer_config.trained_model_file_path,obj=EarthQuakeModel)
        #model pusher
        # save_object("final_model/model.pkl",best_model)
        

        ## Model Trainer Artifact
        model_trainer_artifact=ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                             train_metric_artifact=train_metric,
                             test_metric_artifact=test_metric
                             )
        logging.info(f"Model trainer artifact: {model_trainer_artifact}")
        return model_trainer_artifact


        


       
    
    
        
    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            #loading training array and testing array
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            model_trainer_artifact=self.train_model(x_train,y_train,x_test,y_test)
            return model_trainer_artifact

            
        except Exception as e:
            raise Earthquake_Magnitude_EstimationException(e,sys)