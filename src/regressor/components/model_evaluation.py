import pandas as pd
import os
from regressor import logger
from regressor.entity.config_entity import ModelEvaluationConfig
from regressor.Mylib import myfuncs
from sklearn import metrics
import numpy as np



class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def evaluate_model(self):
        df = myfuncs.load_python_object(self.config.test_data_path)
        preprocessor = myfuncs.load_python_object(self.config.preprocessor_path)
        df_transformed = preprocessor.transform(df)  # chỗ này bị lỗi nè
        df_feature = df_transformed.drop(columns=[self.config.target_col])
        df_target = df_transformed[self.config.target_col]

        model = myfuncs.load_python_object(self.config.model_path)

        df_feature_prediction = model.predict(df_feature)
        score = self.evaluate(df_target, df_feature_prediction)

        print(f"{self.config.evaluated_model_name}: {score}\n\n")

        with open(self.config.result, "a") as file:
            file.write(f"{self.config.evaluated_model_name}: {score}\n\n")

    def evaluate(self, df_target, df_feature_prediction):
        while True:
            if self.config.scoring == "neg_mean_squared_error":
                return np.sqrt(
                    metrics.mean_squared_error(df_target, df_feature_prediction)
                )

            if self.config.scoring == "neg_mean_absolute_error":

                return metrics.mean_absolute_error(df_target, df_feature_prediction)

            return
