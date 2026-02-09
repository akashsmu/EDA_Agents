from typing import TypedDict, Annotated, Literal
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import pandas as pd

from eda_agents.agents.visualization import DataVisualizationAgent
from eda_agents.agents.wrangling import DataWranglingAgent

class MultiAgentState(TypedDict):
    messages: Annotated[list[BaseMessage], "messages"]
    data_raw: dict
    next_agent: str
    final_output: dict

def create_eda_graph(model):
    
    # Initialize sub-agents
    viz_agent = DataVisualizationAgent(model)
    wrangling_agent = DataWranglingAgent(model)

    def router(state: MultiAgentState):
        messages = state["messages"]
        last_message = messages[-1].content.lower()
        
        if "plot" in last_message or "graph" in last_message or "visualize" in last_message or "chart" in last_message:
            return "visualization_node"
        elif "clean" in last_message or "transform" in last_message or "wrangle" in last_message:
            return "wrangling_node"
        else:
            return "visualization_node" # Default to viz? or maybe a general chat node?

    def visualization_node(state: MultiAgentState):
        result = viz_agent.invoke({
            "messages": state["messages"],
            "data_raw": state["data_raw"]
        })
        return {"final_output": result}

    def wrangling_node(state: MultiAgentState):
        result = wrangling_agent.invoke({
            "messages": state["messages"],
            "data_raw": state["data_raw"]
        })
        return {
            "final_output": result, 
            "data_raw": result.get("wrangled_data", state["data_raw"]) # Update data if wrangled
        }

    workflow = StateGraph(MultiAgentState)

    workflow.add_node("visualization_node", visualization_node)
    workflow.add_node("wrangling_node", wrangling_node)

    workflow.set_conditional_entry_point(
        router,
        {
            "visualization_node": "visualization_node",
            "wrangling_node": "wrangling_node"
        }
    )

    workflow.add_edge("visualization_node", END)
    workflow.add_edge("wrangling_node", END)

    return workflow.compile(checkpointer=MemorySaver())
