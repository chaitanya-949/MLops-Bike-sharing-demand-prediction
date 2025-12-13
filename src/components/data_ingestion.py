import os
import sys


import pandas as pd
from sklearn.model_selection import train_test_split


from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.exception import MyException
from src.logger import logging
from src.data_access.proj1_data import Source_Connectors


class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig=DataIngestionConfig()):
        """
        :param data_ingestion_config: configuration for data ingestion
        """
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise MyException(e,sys)
       


    def export_data_into_feature_store(self)->pd.DataFrame:
        """
        Fetch data from the source connector and return as a DataFrame.
        Production behavior: keep data in-memory and return it to downstream components.
        (No CSV write in this method.)
        """
        try:
            logging.info(f"Fetching data from source connector")
            connector = Source_Connectors()
            dataframe = connector.fetch_dataframe()


            logging.info(f"Shape of dataframe fetched: {dataframe.shape}")


            # In production we avoid writing a raw CSV feature-store here.
            # If you still want to persist a copy for debugging, add explicit logic
            # or a config flag in DataIngestionConfig to control it.


            return dataframe


        except Exception as e:
            raise MyException(e,sys)


    def split_data_as_train_test(self,dataframe: pd.DataFrame) ->None:
        """
        Method Name :   split_data_as_train_test
        Description :   This method splits the dataframe into train set and test set based on split ratio
        """
        logging.info("Entered split_data_as_train_test method of Data_Ingestion class")


        try:
            train_set, test_set = train_test_split(dataframe, test_size=self.data_ingestion_config.train_test_split_ratio)
            logging.info("Performed train test split on the dataframe")
            logging.info(
                "Exited split_data_as_train_test method of Data_Ingestion class"
            )
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path,exist_ok=True)
           
            logging.info(f"Exporting train and test file path.")
            train_set.to_csv(self.data_ingestion_config.training_file_path,index=False,header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path,index=False,header=True)


            logging.info(f"Exported train and test file path.")
        except Exception as e:
            raise MyException(e, sys) from e


    def initiate_data_ingestion(self) ->DataIngestionArtifact:
        """
        Method Name :   initiate_data_ingestion
        Description :   This method initiates the data ingestion components of training pipeline
        """
        logging.info("Entered initiate_data_ingestion method of Data_Ingestion class")


        try:
            dataframe = self.export_data_into_feature_store()


            logging.info("Got the data from source connector")


            self.split_data_as_train_test(dataframe)


            logging.info("Performed train test split on the dataset")


            logging.info(
                "Exited initiate_data_ingestion method of Data_Ingestion class"
            )


            data_ingestion_artifact = DataIngestionArtifact(trained_file_path=self.data_ingestion_config.training_file_path,
            test_file_path=self.data_ingestion_config.testing_file_path)
           
            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise MyException(e, sys) from e

