from typing import TypedDict, Annotated, Literal, Union, List
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import pandas as pd

from eda_agents.agents.visualization import DataVisualizationAgent
from eda_agents.agents.wrangling import DataWranglingAgent


def _normalize_messages(messages: list) -> List[BaseMessage]:
    """Convert mixed message formats (dicts, tuples, BaseMessage) to BaseMessage objects."""
    normalized = []
    for msg in messages:
        if isinstance(msg, BaseMessage):
            normalized.append(msg)
        elif isinstance(msg, dict):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role in ("user", "human"):
                normalized.append(HumanMessage(content=content))
            elif role == "system":
                normalized.append(SystemMessage(content=content))
            elif role in ("assistant", "ai"):
                normalized.append(AIMessage(content=content))
            else:
                normalized.append(HumanMessage(content=content))
        elif isinstance(msg, (list, tuple)) and len(msg) == 2:
            role, content = msg
            if role in ("user", "human"):
                normalized.append(HumanMessage(content=content))
            elif role == "system":
                normalized.append(SystemMessage(content=content))
            else:
                normalized.append(AIMessage(content=content))
        else:
            normalized.append(HumanMessage(content=str(msg)))
    return normalized


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
        messages = _normalize_messages(state["messages"])
        last_message = messages[-1].content.lower() if messages else ""

        viz_keywords = ["plot", "graph", "visualize", "chart", "histogram", "scatter", "bar", "line", "heatmap"]
        wrangle_keywords = ["clean", "transform", "wrangle", "drop", "fill", "merge", "pivot", "rename", "filter", "remove", "replace"]

        if any(kw in last_message for kw in viz_keywords):
            return "visualization_node"
        elif any(kw in last_message for kw in wrangle_keywords):
            return "wrangling_node"
        else:
            return "visualization_node"

    def visualization_node(state: MultiAgentState):
        messages = _normalize_messages(state["messages"])
        result = viz_agent.invoke({
            "messages": messages,
            "data_raw": state["data_raw"]
        })
        return {"final_output": result}

    def wrangling_node(state: MultiAgentState):
        messages = _normalize_messages(state["messages"])
        result = wrangling_agent.invoke({
            "messages": messages,
            "data_raw": state["data_raw"]
        })
        return {
            "final_output": result,
            "data_raw": result.get("wrangled_data", state["data_raw"])
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
