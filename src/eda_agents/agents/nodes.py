from eda_agents.agents.state import AgentState
from langchain_core.messages import HumanMessage, AIMessage

def ingest_node(state: AgentState):
    """
    Node responsible for ingesting data.
    """
    # Logic to load data if not already loaded would go here
    # For now, we assume the UI handles the initial load and passes the DF in the state
    return {"messages": [AIMessage(content="Data ingested successfully.")]}

def analyze_node(state: AgentState):
    """
    Node responsible for performing analysis.
    """
    # Logic to call analysis tools
    return {"messages": [AIMessage(content="Analysis complete.")]}

def report_node(state: AgentState):
    """
    Node responsible for generating a report.
    """
    # Logic to summarize findings
    return {"messages": [AIMessage(content="Report generated.")]}
