from typing import TypedDict, Annotated, Literal, Union, List
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
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
    router_decision: dict


def create_eda_graph(model):

    # Initialize sub-agents
    viz_agent = DataVisualizationAgent(model)
    wrangling_agent = DataWranglingAgent(model)

    # --- Router Node Logic ---
    def router_node(state: MultiAgentState):
        messages = _normalize_messages(state["messages"])
        user_input = messages[-1].content if messages else ""

        parser = JsonOutputParser()
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert intent classifier for a data analysis system.
            Route the user's request to one of the following agents:
            - 'visualization_node': For requests to plot, graph, chart, or visualize data.
            - 'wrangling_node': For requests to clean, transform, modify, fill missing values, drop columns, or filter data.

            Also, refine the user's instructions to be precise for the target agent.
            If the request is ambiguous, default to 'visualization_node'.

            Return a valid JSON object with keys: 'route' and 'refined_instructions'.
            Example: {{"route": "visualization_node", "refined_instructions": "Plot a scatter plot of Age vs Fare"}}
            """),
            ("user", "{input}")
        ])
        
        chain = prompt | model | parser
        try:
            decision = chain.invoke({"input": user_input})
        except Exception:
            # Fallback if JSON parsing fails
            decision = {"route": "visualization_node", "refined_instructions": user_input}

        return {"router_decision": decision}

    def router_condition(state: MultiAgentState):
        decision = state.get("router_decision", {})
        return decision.get("route", "visualization_node")

    def visualization_node(state: MultiAgentState):
        messages = _normalize_messages(state["messages"])
        # Inject refined instructions if available
        decision = state.get("router_decision", {})
        if decision.get("refined_instructions"):
             # Add the refined instruction as a system note or just replace the last message contextually
             # For now, we append it to history so the agent sees it clearly
             messages.append(HumanMessage(content=f"Refined Instructions: {decision['refined_instructions']}"))
        
        result = viz_agent.invoke({
            "messages": messages,
            "data_raw": state["data_raw"]
        })
        return {"final_output": result}

    def wrangling_node(state: MultiAgentState):
        messages = _normalize_messages(state["messages"])
        decision = state.get("router_decision", {})
        if decision.get("refined_instructions"):
             messages.append(HumanMessage(content=f"Refined Instructions: {decision['refined_instructions']}"))

        result = wrangling_agent.invoke({
            "messages": messages,
            "data_raw": state["data_raw"]
        })
        return {
            "final_output": result,
            "data_raw": result.get("wrangled_data", state["data_raw"])
        }

    workflow = StateGraph(MultiAgentState)

    workflow.add_node("router_node", router_node)
    workflow.add_node("visualization_node", visualization_node)
    workflow.add_node("wrangling_node", wrangling_node)

    workflow.set_entry_point("router_node")
    
    workflow.add_conditional_edges(
        "router_node",
        router_condition,
        {
            "visualization_node": "visualization_node",
            "wrangling_node": "wrangling_node"
        }
    )

    workflow.add_edge("visualization_node", END)
    workflow.add_edge("wrangling_node", END)

    return workflow.compile(checkpointer=MemorySaver())
