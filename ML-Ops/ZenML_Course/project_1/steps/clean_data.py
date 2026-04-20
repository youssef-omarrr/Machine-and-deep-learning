import pandas as pd
from zenml import step

@step
def clean_df(data: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the data in the dataframe 
    and then retruns the cleaned data
    """
    unclean_data = data
    clean_data = data
    
    return clean_data