import logging

import pandas as pd
from zenml import step

class IngestData:
    """
    Class object that takes the path of the cv data 
    ands return it as pandas dataframe
    """
    def __init__(self, data_path: str):
        self.data_path = data_path
        
    def get_data(self):
        logging.info(f"Ingesting data from {self.data_path}")
        return pd.read_csv(self.data_path)

@step
def ingest_df(data_path: str) -> pd.DataFrame:
    """
    Ingests data from data_path
    Returns:
        pd.DataFrame: the ingested data
    """
    try:
        ingest_data = IngestData(data_path= data_path)
        df = ingest_data.get_data()
        
        return df
    
    except Exception as e:
        logging.error(f"Data not ingested, maybe path not found")
        raise e