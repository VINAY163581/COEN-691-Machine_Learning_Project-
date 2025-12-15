from Earthquake_Magnitude_Estimation.components.data_visualization import DataVisualization
from Earthquake_Magnitude_Estimation.entity.config_entity import DataVisualizationConfig, ModelTrainerConfig, TrainingPipelineConfig
from Earthquake_Magnitude_Estimation.entity.artifact_entity import DataTransformationArtifact

#  Load pipeline configs
training_pipeline_config = TrainingPipelineConfig()
data_visualization_config = DataVisualizationConfig(training_pipeline_config)
model_trainer_config = ModelTrainerConfig(training_pipeline_config)

# Use previously saved transformed files
data_transformation_artifact = DataTransformationArtifact(
    transformed_train_file_path="artifacts/data_transformation/transformed_train.npy",
    transformed_test_file_path="artifacts/data_transformation/transformed_test.npy",
    preprocessor_object_file_path="artifacts/data_transformation/preprocessor.pkl"
)

# Create visualizer instance and run visualization
visualizer = DataVisualization(
    data_transformation_artifact=data_transformation_artifact,
    data_visualization_config=data_visualization_config,
    model_trainer_config=model_trainer_config
)

visualizer.initiate_data_visualization()
