from langgraph.graph import StateGraph, END
from eda_agents.agents.state import AgentState
from eda_agents.agents.nodes import ingest_node, analyze_node, report_node

def create_graph():
    """
    Creates the LangGraph for the EDA Agent.
    """
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("ingest", ingest_node)
    workflow.add_node("analyze", analyze_node)
    workflow.add_node("report", report_node)

    # Define edges
    workflow.set_entry_point("ingest")
    workflow.add_edge("ingest", "analyze")
    workflow.add_edge("analyze", "report")
    workflow.add_edge("report", END)

    # Compile the graph
    app = workflow.compile()
    return app
