import pandas as pd
from langchain.tools import tool

@tool
def load_csv(file_path: str) -> pd.DataFrame:
    """Loads a CSV file into a pandas DataFrame."""
    return pd.read_csv(file_path)

@tool
def load_excel(file_path: str) -> pd.DataFrame:
    """Loads an Excel file into a pandas DataFrame."""
    return pd.read_excel(file_path)

@tool
def load_txt(file_path: str) -> pd.DataFrame:
    """Loads a TXT file into a pandas DataFrame (assuming tab-delimited or similar)."""
    return pd.read_table(file_path)
