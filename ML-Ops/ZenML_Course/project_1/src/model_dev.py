import logging 
from abc import ABC, abstractmethod

import optuna

import pandas as pd
import numpy as np
import xgboost as xgb

from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


class Model(ABC):
    """
    Abstract base class for all models.
    """

    @abstractmethod
    def train(self, x_train, y_train):
        """
        Trains the model on the given data.

        Args:
            x_train: Training data
            y_train: Target data
        """
        pass

    @abstractmethod
    def optimize(self, trial, x_train, y_train, x_test, y_test):
        """
        Optimizes the hyperparameters of the model.

        Args:
            trial: Optuna trial object
            x_train: Training data
            y_train: Target data
            x_test: Testing data
            y_test: Testing target
        """
        pass
    
    
class RandomForestModel(Model):
    """
    RandomForestModel that implements the Model interface.
    """
    def train(self, x_train, y_train, **kwargs):
        reg = RandomForestRegressor(**kwargs)
        reg.fit(x_train, y_train)
        
        return reg
    
class LightGBMModel(Model):
    """
    LightGBMModel that implements the Model interface.
    """
    def train(self, x_train, y_train, **kwargs):
        reg = LightGBMModel(**kwargs)
        reg.fit(x_train, y_train)
        
        return reg
    
class XGBoostModel(Model):
    """
    XGBoostModel that implements the Model interface.
    """
    def train(self, x_train, y_train, **kwargs):
        reg = xgb.XGBRegressor(**kwargs)
        reg.fit(x_train, y_train)
        
        return reg

class LinearRegressionModel(Model):
    """
    LinearRegressionModel that implements the Model interface.
    """
    def train(self, x_train, y_train, **kwargs):
        reg = LinearRegression(**kwargs)
        reg.fit(x_train, y_train)
        
        return reg

class HyperparameterTuner:
    """
    Class for performing hyperparameter tuning. It uses Model strategy to perform tuning.
    """

    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def optimize(self, n_trials=100):
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: self.model.optimize(trial, self.x_train, self.y_train, self.x_test, self.y_test), n_trials=n_trials)
        return study.best_trial.params