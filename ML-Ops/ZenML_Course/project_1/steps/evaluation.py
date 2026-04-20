import pandas as pd
from zenml import step


@step
def eval_model(clean_data) -> None:
    """ 
    Evaluatees the model
    """
    pass