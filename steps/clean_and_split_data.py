import pandas as pd
from typing import Tuple
from zenml.steps import BaseParameters, step
from src.data_cleaning import DataPreProcessStrategy
from src.data_divide import DataDivideStrategy

@step
def clean_and_split_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    preprocessor = DataPreProcessStrategy()
    processed_data = preprocessor.handle_data(data)
    divider = DataDivideStrategy()
    return divider.handle_data(processed_data)

