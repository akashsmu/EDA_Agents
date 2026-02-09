import plotly.io as pio
import plotly.graph_objects as go
import json

def plotly_from_dict(fig_dict: dict) -> go.Figure:
    """
    Converts a dictionary representation of a Plotly figure back to a go.Figure object.
    """
    return pio.from_json(json.dumps(fig_dict, cls=json.JSONEncoder))

def fig_to_dict(fig: go.Figure) -> dict:
    """
    Converts a go.Figure object to a dictionary.
    """
    return json.loads(pio.to_json(fig))
