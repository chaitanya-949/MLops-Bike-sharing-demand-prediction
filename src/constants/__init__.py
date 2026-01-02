import os
from datetime import date
import os
from dotenv import load_dotenv
load_dotenv('.env')




DATABRICKS_HOST = os.environ.get("DATABRICKS_HOST")
DATABRICKS_HTTP_PATH = os.environ.get("DATABRICKS_HTTP_PATH")
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN")




DEFAULT_CATALOG = "workspace"
DEFAULT_SCHEMA = "default"
DEFAULT_TABLE = "seoul_bike_data"




PIPELINE_NAME: str = ""
ARTIFACT_DIR: str = "artifact"


MODEL_FILE_NAME = "model.pkl"




TARGET_COLUMN = "Rented Bike Count"
CURRENT_YEAR = date.today().year
# PREPROCESSING_OBJECT_FILE_NAME = "preprocessing.pkl"




FILE_NAME: str = "data.csv"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"
SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")


AWS_ACCESS_KEY_ID_ENV_KEY = "AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY_ENV_KEY = "AWS_SECRET_ACCESS_KEY"
REGION_NAME = "us-east-1"


"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_COLLECTION_NAME: str = "Proj1-Data"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.20






"""
Data Validation realted contant start with DATA_VALIDATION VAR NAME
"""
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_REPORT_FILE_NAME: str = "report.yaml"






"""
Data Transformation ralated constant start with DATA_TRANSFORMATION VAR NAME
"""


DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
# DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"








"""
MODEL TRAINER related constant start with MODEL_TRAINER var name
"""
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.6
MODEL_TRAINER_MODEL_CONFIG_FILE_PATH: str = os.path.join("config", "model.yaml")
objective: str = "reg:squarederror"
MODEL_TRAINER_N_ESTIMATORS: int=300
gamma: float = 0.1
learning_rate: float = 0.05
subsample: float = 0.8
colsample_bytree: float = 0.8
reg_lambda: float = 1.5
reg_alpha: float = 0.5
n_jobs: int = -1
random_state: int = 42
MODEL_TRAINER_MIN_CHILD_WEIGHT: int = 1
#MODEL_TRAINER_MIN_SAMPLES_LEAF: int = 6
MIN_SAMPLES_SPLIT_MAX_DEPTH: int = 5
#MIN_SAMPLES_SPLIT_CRITERION: str = 'entropy'
#MIN_SAMPLES_SPLIT_RANDOM_STATE: int = 101








"""
MODEL Evaluation related constants
"""
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.02
MODEL_BUCKET_NAME = "mlopsproj949"
MODEL_PUSHER_S3_KEY = "model-registry"



APP_HOST = "0.0.0.0"
APP_PORT = 5000