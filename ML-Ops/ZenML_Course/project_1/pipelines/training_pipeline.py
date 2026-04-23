import pandas as pd

from zenml import pipeline
from zenml.config import DockerSettings
from zenml.integrations.constants import MLFLOW


from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.model_train import train_model
from steps.evaluation import eval_model


docker_settings = DockerSettings(required_integrations = [MLFLOW]) # Tells docker that mlflow integrations is required

@pipeline(enable_cache=True, settings={'docker': docker_settings}) # Cache is True by default
def training_pipeline(data_path: str):
    df = ingest_df(data_path=data_path)
    x_train, x_test, y_train, y_test = clean_df(df)
    model = train_model(x_train, x_test, y_train, y_test)
    mse, rmse = eval_model(model, x_test, y_test)