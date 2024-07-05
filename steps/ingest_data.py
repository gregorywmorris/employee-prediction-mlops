import pandas as pd
from zenml.steps import BaseParameters, step

class IngestDataParams(BaseParameters):
    data_path: str

@step
def ingest_data(params: IngestDataParams) -> pd.DataFrame:
    return pd.read_csv(params.data_path)

