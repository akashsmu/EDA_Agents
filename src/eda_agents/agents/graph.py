from typing import TypedDict, Annotated, Literal, Union, List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from eda_agents.agents.visualization import DataVisualizationAgent
from eda_agents.agents.wrangling import DataWranglingAgent
from eda_agents.utils.logger import logger


def _normalize_messages(messages: List[Union[BaseMessage, dict, tuple]]) -> List[BaseMessage]:
    normalized = []
    for msg in messages:
        if isinstance(msg, BaseMessage):
            normalized.append(msg)
        elif isinstance(msg, dict):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                normalized.append(HumanMessage(content=content))
            elif role == "assistant":
                normalized.append(AIMessage(content=content))
            elif role == "system":
                normalized.append(SystemMessage(content=content))
        elif isinstance(msg, tuple) and len(msg) == 2:
            role, content = msg
            if role == "user":
                normalized.append(HumanMessage(content=content))
            elif role == "assistant":
                normalized.append(AIMessage(content=content))
            elif role == "system":
                normalized.append(SystemMessage(content=content))
    return normalized


class MultiAgentState(TypedDict):
    messages: List[BaseMessage]
    data_raw: Union[dict, list]
    router_decision: dict
    final_output: dict


# --- Router Node Logic ---
def router_node(state: MultiAgentState):
    logger.info("Entering router_node...")
    messages = _normalize_messages(state["messages"])
    user_input = messages[-1].content if messages else ""
    logger.debug(f"User Input for routing: {user_input}")

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
    
    # We'll pass the model via a closure or global for now, but in a real app it's better in state or config
    # For now, let's assume a default model is available if not in config
    model = ChatOpenAI(model="gpt-4") # Temporary fallback
    
    chain = prompt | model | parser
    try:
        logger.info("Consulting LLM for routing decision.")
        decision = chain.invoke({"input": user_input})
    except Exception as e:
        logger.error(f"Router LLM call failed: {e}. Falling back to default.")
        decision = {"route": "visualization_node", "refined_instructions": user_input}

    logger.info(f"Router decision: {decision.get('route')} | Refined: {decision.get('refined_instructions')}")
    return {"router_decision": decision}

def router_condition(state: MultiAgentState):
    decision = state.get("router_decision", {})
    route = decision.get("route", "visualization_node")
    logger.info(f"Routing to: {route}")
    return route

# --- Agent Invocation Nodes ---
def visualization_node(state: MultiAgentState):
    logger.info("Invoking visualization_node...")
    model = ChatOpenAI(model="gpt-4")
    agent = DataVisualizationAgent(model)
    
    refined_instructions = state["router_decision"].get("refined_instructions", "")
    
    agent_state = {
        "messages": [HumanMessage(content=refined_instructions)],
        "data_raw": state["data_raw"]
    }
    
    result = agent.invoke(agent_state)
    logger.info("Visualization Agent task complete.")
    return {"final_output": result}

def wrangling_node(state: MultiAgentState):
    logger.info("Invoking wrangling_node...")
    model = ChatOpenAI(model="gpt-4")
    agent = DataWranglingAgent(model)
    
    refined_instructions = state["router_decision"].get("refined_instructions", "")
    
    agent_state = {
        "messages": [HumanMessage(content=refined_instructions)],
        "data_raw": state["data_raw"]
    }
    
    result = agent.invoke(agent_state)
    logger.info("Wrangling Agent task complete.")
    return {"final_output": result}

def create_eda_graph(model: ChatOpenAI):
    logger.info("Initializing EDA Graph.")
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
    
    memory = MemorySaver()
    logger.info("Graph compiled successfully.")
    return workflow.compile(checkpointer=memory)
