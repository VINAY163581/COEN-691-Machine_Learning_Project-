from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    trained_file_path:str
    test_file_path:str
    
@dataclass
class DataValidationArtifact:
    validation_status: bool
    valid_train_file_path: str
    valid_test_file_path: str
    invalid_train_file_path: str
    invalid_test_file_path: str
    drift_report_file_path: str

@dataclass
class DataTransformationArtifact:
    transformed_object_file_path: str
    transformed_train_file_path: str
    transformed_test_file_path: str

@dataclass
class DataVisualizationArtifact:
    train_correlation_heatmap_path: str
    test_correlation_heatmap_path: str
    train_numeric_distribution_dir: str
    test_numeric_distribution_dir: str
    train_categorical_distribution_dir: str
    test_categorical_distribution_dir: str

@dataclass
class RegressionMetricArtifact:
    rmse: float
    mae: float
    r2_score: float
    
@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str
    train_metric_artifact: RegressionMetricArtifact
    test_metric_artifact: RegressionMetricArtifact

