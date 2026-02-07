import pandas as pd
from langchain.tools import tool

@tool
def get_column_stats(df: pd.DataFrame) -> dict:
    """Returns basic statistics for each column."""
    return df.describe().to_dict()

@tool
def list_columns(df: pd.DataFrame) -> list:
    """Returns a list of column names."""
    return df.columns.tolist()
