import pandas as pd
from zenml import pipeline

from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.model_train import train_model
from steps.evaluation import eval_model


@pipeline(enable_cache=False) # True by default
def training_pipeline(data_path: str):
    df = ingest_df(data_path=data_path)
    clean_df = clean_df(df)
    train_model(clean_df)
    eval_model(clean_df)