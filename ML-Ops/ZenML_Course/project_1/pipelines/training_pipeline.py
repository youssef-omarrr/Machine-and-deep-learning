import pandas as pd
from zenml import pipeline

from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.model_train import train_model
from steps.evaluation import eval_model


@pipeline(enable_cache=False) # True by default
def training_pipeline(data_path: str):
    df = ingest_df(data_path=data_path)
    x_train, x_test, y_train, y_test = clean_data(df)
    model = model_train(x_train, x_test, y_train, y_test)
    mse, rmse = evaluation(model, x_test, y_test)