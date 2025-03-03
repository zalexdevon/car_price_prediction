import pandas as pd
import os
from regressor import logger
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit, GridSearchCV
from regressor.entity.config_entity import ModelTrainerConfig
from regressor.Mylib import myfuncs
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from xgboost import XGBClassifier
from scipy.stats import randint
import random
from lightgbm import LGBMClassifier
from sklearn.model_selection import ParameterSampler
from sklearn import metrics
from sklearn.base import clone
import time


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train_randomised_train_val(self):
        random_search = RandomizedSearchCV(
            self.base_model,
            param_distributions=self.config.param_grid_model,
            n_iter=self.config.N_ITER,
            cv=self.trainval_splitter,
            random_state=42,
            scoring=self.config.scoring,
            return_train_score=True,
            verbose=2,
        )

        self.find_best_model_and_save_result(random_search)

    def train_grid_train_val(self):
        grid_search = GridSearchCV(
            self.base_model,
            param_grid=self.config.param_grid_model,
            cv=self.trainval_splitter,
            scoring=self.config.scoring,
            return_train_score=True,
            verbose=2,
        )

        self.find_best_model_and_save_result(grid_search)

    def train_randomisedcv(self):
        random_search = RandomizedSearchCV(
            self.base_model,
            param_distributions=self.config.param_grid_model,
            n_iter=self.config.N_ITER,
            cv=5,
            random_state=42,
            scoring=self.config.scoring,
            return_train_score=True,
            verbose=2,
        )

        self.find_best_model_and_save_result(random_search)

    def train_gridcv(self):
        grid_search = GridSearchCV(
            self.base_model,
            param_grid=self.config.param_grid_model,
            cv=5,
            scoring=self.config.scoring,
            return_train_score=True,
            verbose=2,
        )

        self.find_best_model_and_save_result(grid_search)

    def find_best_model_and_save_result(self, searcher):
        searcher.fit(self.features, self.target)
        best_model = searcher.best_estimator_
        cv_results = searcher.cv_results_

        scores = list(
            zip(cv_results["mean_train_score"], cv_results["mean_test_score"])
        )
        scores = sorted(scores, key=lambda x: x[1], reverse=True)

        self.update_best_val_accuracy(scores)

        myfuncs.save_python_object(self.config.best_model_path, best_model)

    def update_best_val_accuracy(self, scores):
        while True:

            if self.config.scoring == "neg_mean_squared_error":
                self.best_val_score = np.sqrt(-scores[0][1])
                self.train_score_follow_best_val = np.sqrt(-scores[0][0])
                return

            if self.config.scoring == "neg_mean_absolute_error":
                self.best_val_score = -scores[0][1]
                self.train_score_follow_best_val = -scores[0][0]
                return

            return

    def load_data_to_train(self):
        self.train_feature_data = myfuncs.load_python_object(
            self.config.train_feature_path
        )
        self.train_target_data = myfuncs.load_python_object(
            self.config.train_target_path
        )
        self.val_feature_data = myfuncs.load_python_object(self.config.val_feature_path)
        self.val_target_data = myfuncs.load_python_object(self.config.val_target_path)

        self.features, self.target, self.trainval_splitter = (
            myfuncs.get_features_target_spliter_for_CV_train_val(
                self.train_feature_data,
                self.train_target_data,
                self.val_feature_data,
                self.val_target_data,
            )
        )

        self.base_model = myfuncs.get_base_model(self.config.model_name)

        result = f"P: {self.config.data_transformation}<br>{self.config.model_name}<br>"
        result += myfuncs.get_monitor_desc(self.config.param_grid_model_desc)
        self.monitor_desc = result

    def save_list_monitor_components(self):
        with open("artifacts/run_count.txt", "r") as file:
            run_count = int(file.read())
        with open("artifacts/run_count.txt", "w") as file:
            file.write(str(run_count + 1))

        if run_count > 0:

            self.list_monitor_components = myfuncs.load_python_object(
                self.config.list_monitor_components_path
            )

            # debug
            print("list_monitor_components da ton tai !!!!")
            # d

        else:
            self.list_monitor_components = []

        self.list_monitor_components += [
            (
                self.monitor_desc,
                self.train_score_follow_best_val,
                self.best_val_score,
            )
        ]

        myfuncs.save_python_object(
            self.config.list_monitor_components_path, self.list_monitor_components
        )
