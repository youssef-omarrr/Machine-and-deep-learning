import pandas as pd
from zenml import step

from src.data_cleaning import DataCleaning, DataDivideStrategy, DataPreprocessStrategy
from typing import Tuple
from typing_extensions import Annotated

@step
def clean_df(data: pd.DataFrame
            ) -> Tuple[
                Annotated[pd.DataFrame, "x_train"],
                Annotated[pd.DataFrame, "x_test"],
                Annotated[pd.DataFrame, "y_train"],
                Annotated[pd.DataFrame, "y_test"],
            ]:
    """
    Cleans the data in the dataframe 
    and then retruns the cleaned data
    """
    try:
        preprocess_strat = DataPreprocessStrategy()
        data_cleaning = DataCleaning(data, preprocess_strat)
        preprocessed_data = data_cleaning.handle_data()
        
        divide_strat = DataDivideStrategy()
        data_cleaning = DataCleaning(preprocessed_data, divide_strat)
        x_train, x_test, y_train, y_test = data_cleaning.handle_data()
        
        return x_train, x_test, y_train, y_test
    
    except Exception as e:
        logging.error(e)
        raise e