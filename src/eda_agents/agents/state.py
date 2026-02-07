import pandas as pd
from typing import TypedDict, Annotated, List, Union, Dict, Any
import operator

class AgentState(TypedDict):
    """
    State for the EDA Agent.
    """
    messages: Annotated[List[Any], operator.add]
    dataframe: pd.DataFrame
    file_path: str
    analysis_results: Dict[str, Any]
    next_step: str
