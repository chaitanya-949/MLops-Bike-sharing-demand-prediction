import sys
from src.entity.config_entity import VehiclePredictorConfig
from src.entity.s3_estimator import Proj1Estimator
from src.exception import MyException
from src.logger import logging
from pandas import DataFrame
import os
import glob
from src.utils.main_utils import read_yaml_file
from src.constants import TARGET_COLUMN




class VehicleData:
    def __init__(self,
                Hour,
                Temperature,
                Humidity,
                Wind_speed,
                Visibility,
                dew_point_temperature,
                Solar_Radiation,
                Rainfall,
                snowfall,
                month,
                day,
                Seasons_Spring,
                Seasons_Summer,
                Seasons_Winter,
                Holiday_No_Holiday,
                Functioning_Day_Yes
                # Age,


                ):
        """
        Vehicle Data constructor
        Input: all features of the trained model for prediction
        """
        try:
            self.Hour  = Hour
            self.Temperature = Temperature
            self.Humidity = Humidity
            self.Wind_speed = Wind_speed
            self.Visibility = Visibility
            self.dew_point_temperature = dew_point_temperature
            self.Solar_Radiation = Solar_Radiation
            self.Rainfall = Rainfall
            self.snowfall = snowfall
            self.month = month
            self.day = day
            self.Seasons_Spring=Seasons_Spring
            self.Seasons_Summer=Seasons_Summer
            self.Seasons_Winter=Seasons_Winter
            self.Holiday_No_Holiday=Holiday_No_Holiday
            self.Functioning_Day_Yes=Functioning_Day_Yes
           


        except Exception as e:
            raise MyException(e, sys) from e


    def get_vehicle_input_data_frame(self)-> DataFrame:
        """
        This function returns a DataFrame from USvisaData class input
        """
        try:
           
            vehicle_input_dict = self.get_vehicle_data_as_dict()
            return DataFrame(vehicle_input_dict)
       
        except Exception as e:
            raise MyException(e, sys) from e




    def get_vehicle_data_as_dict(self):
        """
        This function returns a dictionary from VehicleData class input
        """
        logging.info("Entered get_usvisa_data_as_dict method as VehicleData class")


        try:
            # Build input dict in the same order as transformed training features (excluding target)
            input_data = {
                "Hour": [self.Hour],
                "Temperature": [self.Temperature],
                "Humidity": [self.Humidity],
                "Wind_speed": [self.Wind_speed],
                "Visibility": [self.Visibility],
                "dew_point_temperature": [self.dew_point_temperature],
                "Solar_Radiation": [self.Solar_Radiation],
                "Rainfall": [self.Rainfall],
                "snowfall": [self.snowfall],
                "month": [self.month],
                "day": [self.day],
                "Seasons_Spring": [self.Seasons_Spring],
                "Seasons_Summer": [self.Seasons_Summer],
                "Seasons_Winter": [self.Seasons_Winter],
                "Holiday_No_Holiday": [self.Holiday_No_Holiday],
                "Functioning_Day_Yes": [self.Functioning_Day_Yes]
            }


            logging.info("Created vehicle data dict")
            logging.info("Exited get_vehicle_data_as_dict method as VehicleData class")
            return input_data


        except Exception as e:
            raise MyException(e, sys) from e


class VehicleDataClassifier:
    def __init__(self,prediction_pipeline_config: VehiclePredictorConfig = VehiclePredictorConfig(),) -> None:
        """
        :param prediction_pipeline_config: Configuration for prediction the value
        """
        try:
            self.prediction_pipeline_config = prediction_pipeline_config
        except Exception as e:
            raise MyException(e, sys)


    def predict(self, dataframe) -> str:
        """
        This is the method of VehicleDataClassifier
        Returns: Prediction in string format
        """
        try:
            logging.info("Entered predict method of VehicleDataClassifier class")
            model = Proj1Estimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path,
            )

            # Attempt to find transformed feature names from the latest artifact so
            # we can reorder/select the incoming DataFrame columns to match training.
            try:
                artifact_dirs = glob.glob(os.path.join("artifact", "*"))
                artifact_dirs = [d for d in artifact_dirs if os.path.isdir(d)]
                artifact_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                feature_names = None
                for d in artifact_dirs:
                    candidate = os.path.join(d, "data_transformation", "transformed", "feature_names.yaml")
                    if os.path.exists(candidate):
                        feature_names = read_yaml_file(candidate)
                        break

                if feature_names:
                    # feature_names saved include the TARGET_COLUMN as last entry; remove it
                    if feature_names and feature_names[-1] == TARGET_COLUMN:
                        expected_features = feature_names[:-1]
                    else:
                        expected_features = feature_names

                    # Build ordered DataFrame matching expected_features, pad missing with zeros
                    import pandas as _pd
                    ordered = _pd.DataFrame()
                    for feat in expected_features:
                        if feat in dataframe.columns:
                            ordered[feat] = dataframe[feat]
                        else:
                            ordered[feat] = 0
                    dataframe = ordered
                    logging.info(f"Reordered/padded input to match {len(expected_features)} training features")
            except Exception:
                logging.warning("Could not find or load feature_names.yaml — falling back to given dataframe")

            result = model.predict(dataframe)
            return result
       
        except Exception as e:
            raise MyException(e, sys)



