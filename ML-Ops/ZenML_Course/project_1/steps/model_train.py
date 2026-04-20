import logging
import yaml

import pandas as pd
from src.model_dev import (
    HyperparameterTuner,
    LightGBMModel,
    LinearRegressionModel,
    RandomForestModel,
    XGBoostModel,
)
from sklearn.base import RegressorMixin

from zenml import step

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

@step(experiment_tracker=experiment_tracker.name)
def train_model(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config: dict,
) -> RegressorMixin:
    """
    Args:
        x_train: pd.DataFrame
        x_test: pd.DataFrame
        y_train: pd.Series
        y_test: pd.Series
    Returns:
        model: RegressorMixin
    """
    try:
        model = None
        tuner = None

        if config.model_name == "lightgbm":
            model = LightGBMModel()
            
        elif config.model_name == "randomforest":
            model = RandomForestModel()
            
        elif config.model_name == "xgboost":
            model = XGBoostModel()
            
        elif config.model_name == "linear_regression":
            model = LinearRegressionModel()
            
        else:
            raise ValueError("Model name not supported")

        tuner = HyperparameterTuner(model, x_train, y_train, x_test, y_test)

        if config.fine_tuning:
            best_params = tuner.optimize()
            trained_model = model.train(x_train, y_train, **best_params)
        else:
            trained_model = model.train(x_train, y_train)
            
        return trained_model
    
    except Exception as e:
        logging.error(e)
        raise e
