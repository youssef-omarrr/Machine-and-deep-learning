import pandas as pd
from zenml import step


@step
def train_model(clean_data) -> None:
    """ 
    Takes the clean data and trains the model on it
    """
    pass