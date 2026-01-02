import sys


import pandas as pd
from pandas import DataFrame
from sklearn.pipeline import Pipeline


from src.exception import MyException
from src.logger import logging


# class TargetValueMapping:
#     def __init__(self):
#         self.yes:int = 0
#         self.no:int = 1
#     def _asdict(self):
#         return self.__dict__
#     def reverse_mapping(self):
#         mapping_response = self._asdict()
#         return dict(zip(mapping_response.values(),mapping_response.keys()))


class MyModel:
    def __init__(self, trained_model_object: object):
    #def __init__(self, preprocessing_object: Pipeline, trained_model_object: object):
        """
        :param preprocessing_object: Input Object of preprocesser
        :param trained_model_object: Input Object of trained model
        """
       # self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object


    def predict(self, dataframe: pd.DataFrame) -> DataFrame:
        """
        Function accepts preprocessed inputs (with all custom transformations already applied),
        applies scaling using preprocessing_object, and performs prediction on transformed features.
        """
        try:
            logging.info("Starting prediction process.")

            # Use the dataframe as-is unless custom preprocessing is provided
            transformed_feature = dataframe

            # Determine expected number of features from the trained estimator
            expected = None
            try:
                expected = getattr(self.trained_model_object, "n_features_in_", None)
            except Exception:
                expected = None

            # Fallback: some models expose feature_importances_ length
            if expected is None:
                try:
                    fi = getattr(self.trained_model_object, 'feature_importances_', None)
                    if fi is not None:
                        expected = len(fi)
                except Exception:
                    expected = None

            # If expected is known, align the input
            if expected is not None:
                try:
                    current = transformed_feature.shape[1]
                except Exception:
                    current = None

                if current is not None and current != expected:
                    logging.warning(f"Feature count mismatch: input has {current}, model expects {expected}. Attempting to align input.")

                    # If too many columns, drop excess columns (keep left-most columns)
                    if current > expected:
                        transformed_feature = transformed_feature.iloc[:, :expected]
                        logging.info(f"Dropped {current-expected} extra input columns to match expected feature count {expected}.")

                    # If too few columns, pad with zeros columns (unnamed) to match expected count
                    elif current < expected:
                        n_missing = expected - current
                        import numpy as _np
                        pad = _np.zeros((transformed_feature.shape[0], n_missing))
                        # create DataFrame for padding with generic column names
                        import pandas as _pd
                        pad_df = _pd.DataFrame(pad, columns=[f"__pad_{i}" for i in range(n_missing)])
                        transformed_feature = _pd.concat([transformed_feature.reset_index(drop=True), pad_df], axis=1)
                        logging.info(f"Padded input with {n_missing} zero-columns to match expected feature count {expected}.")

            # Step 2: Perform prediction using the trained model
            logging.info("Using the trained model to get predictions")
            try:
                # sklearn accepts numpy arrays; convert if DataFrame
                inp = transformed_feature.values if hasattr(transformed_feature, 'values') else transformed_feature
                predictions = self.trained_model_object.predict(inp)
            except Exception as e:
                logging.error("Prediction failed after input alignment", exc_info=True)
                raise MyException(e, sys) from e

            return predictions


        except Exception as e:
            logging.error("Error occurred in predict method", exc_info=True)
            raise MyException(e, sys) from e




    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"


    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"



