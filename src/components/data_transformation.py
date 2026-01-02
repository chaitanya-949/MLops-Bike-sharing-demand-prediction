import numpy as np
import pandas as pd
import sys
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer


from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH, CURRENT_YEAR
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file
from src.utils.main_utils import write_yaml_file
import json




class dataTransformation:
    def __init__(self,data_ingestion_artifact:DataIngestionArtifact,
                 data_transformation_config:DataTransformationConfig,
                 data_validation_artifact:DataValidationArtifact):
        try:
            self.data_ingestion_artifact=data_ingestion_artifact
            self.data_transformation_config=data_transformation_config
            self.data_validation_artifact=data_validation_artifact
            self.schema_config=read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e,sys)
   
   
    @staticmethod
    def read_csv(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys) from e


       
   
   
    def handle_date(self,df):
        """ handling date column because it is object data type"""
        logging.info("handling date column as month,day,day_of_week")
        df['Date'] = pd.to_datetime(df['Date'])
        df['month'] = pd.to_datetime(df['Date']).dt.month
        df['day'] = pd.to_datetime(df['Date']).dt.day
        #df['dayofweek']= pd.to_datetime(df['Date']).dt.dayofweek
        return df
       
       
       
    def _drop_datecolumn(self, df):
        """Drop the 'date' column if it exists."""
        logging.info("Dropping 'date' column")
        drop_cols = self.schema_config['drop_columns']
       
        # Handle both single column (string) and multiple columns (list)
        if isinstance(drop_cols, list):
            df = df.drop(columns=[col for col in drop_cols if col in df.columns],axis=1, errors='ignore')
        else:
            df = df.drop(columns=[drop_cols], errors='ignore',axis=1)
       
        return df
       
       
       
    def _create_dummy_columns(self, df):
        """Create dummy variables for categorical features."""
        logging.info("Creating dummy variables for categorical features")
        df = pd.get_dummies(data=df).astype(int)
        return df
           
           
    def square_root_transformation(self, df):
        """transforming data to reduce skewness"""
        logging.info("applying square root transformation")
        cols = self.schema_config['1columns']
       
        # Handle both list and dict cases
        if isinstance(cols, dict):
            cols = list(cols.keys())
        elif not isinstance(cols, list):
            cols = [cols]
       
        # Apply sqrt to each numeric column in the list
        for col in cols:
            if col in df.columns:
                df[col] = np.sqrt(df[col])
       
        return df
   


    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Initiates the data transformation component for the pipeline.
        """
        try:
            logging.info("Data Transformation Started !!!")
            if not self.data_validation_artifact.validation_status:
                logging.info("Data validation failed. Stopping transformation.")
                return None


            # Load train and test data
            train_df = self.read_csv(file_path=self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_csv(file_path=self.data_ingestion_artifact.test_file_path)
            logging.info("Train-Test data loaded")
           
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN])
            target_feature_train_df = train_df[TARGET_COLUMN]


            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN])
            target_feature_test_df = test_df[TARGET_COLUMN]
            logging.info("Input and Target cols defined for both train and test df.")


           
           
            input_feature_train_df = self.handle_date(input_feature_train_df)
            input_feature_train_df = self._drop_datecolumn(input_feature_train_df)
            input_feature_train_df = self._create_dummy_columns(input_feature_train_df)
           # input_feature_train_df = self.square_root_transformation(input_feature_train_df)
           
           
            input_feature_test_df = self.handle_date(input_feature_test_df)
            input_feature_test_df = self._drop_datecolumn(input_feature_test_df)
            input_feature_test_df = self._create_dummy_columns(input_feature_test_df)
           # input_feature_test_df = self.square_root_transformation(input_feature_test_df)
            logging.info("Custom transformations applied to train and test data")


            # # # Get transformer and fit on train data
            # preprocessor = self.get_data_transformer_object()
            # input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            # input_feature_test_arr = preprocessor.transform(input_feature_test_df)
            # logging.info("Preprocessing pipeline applied")


            # Save transformed arrays. The training pipeline expects arrays where the
            # last column is the target (so downstream trainer can split train[:, :-1], train[:, -1]).
            # Create numpy arrays with features + target as last column.
            train_combined = None
            test_combined = None

            try:
                train_combined = pd.concat([input_feature_train_df.reset_index(drop=True), target_feature_train_df.reset_index(drop=True)], axis=1).to_numpy()
                test_combined = pd.concat([input_feature_test_df.reset_index(drop=True), target_feature_test_df.reset_index(drop=True)], axis=1).to_numpy()
            except Exception:
                # fallback: if target series not available, save features only
                train_combined = input_feature_train_df.to_numpy()
                test_combined = input_feature_test_df.to_numpy()

            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, train_combined)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, test_combined)

            # Save the transformed feature names (including target as last entry) to a YAML file
            feature_names = list(input_feature_train_df.columns) + [TARGET_COLUMN]
            feature_names_path = os.path.join(self.data_transformation_config.data_transformation_dir, 'transformed', 'feature_names.yaml')
            write_yaml_file(feature_names_path, feature_names, replace=True)
           
            # #Save preprocessor object
            # save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
            logging.info("Transformed data and preprocessor saved")


            data_transformation_artifact = DataTransformationArtifact(
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                # transformed_object_file_path=self.data_transformation_config.transformed_object_file_path
            )
           
            logging.info(f"Data Transformation Artifact: {data_transformation_artifact}")
            return data_transformation_artifact


        except Exception as e:
            logging.exception("Exception in initiate_data_transformation")
            raise MyException(e, sys) from e

