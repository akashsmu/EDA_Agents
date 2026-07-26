import json
import operator
from typing import Annotated, TypedDict, Union, List, Dict, Any, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.config import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from eda_agents.agents.base import BaseAgent
from eda_agents.agents.cleaning import DataCleaningAgent
from eda_agents.agents.wrangling import DataWranglingAgent
from eda_agents.agents.visualization import DataVisualizationAgent
from eda_agents.utils.logger import logger

class SupervisorState(TypedDict):
    """State for the Supervisor Agent."""
    messages: Annotated[List[BaseMessage], operator.add]
    data_raw: Union[dict, list]
    next_worker: str
    final_output: Any
    hitl_enabled: bool # Added to support HITL in worker agents

class SupervisorAgent(BaseAgent):
    def __init__(self, model: ChatOpenAI):
        super().__init__(model)
        self.workers = {
            "cleaning": DataCleaningAgent(model),
            "wrangling": DataWranglingAgent(model),
            "visualization": DataVisualizationAgent(model)
        }

    def _make_compiled_graph(self):
        workflow = StateGraph(SupervisorState)

        # Add Nodes
        workflow.add_node("supervisor", self.supervisor_node)
        workflow.add_node("cleaning_worker", self.cleaning_node)
        workflow.add_node("wrangling_worker", self.wrangling_node)
        workflow.add_node("visualization_worker", self.visualization_node)

        # Set Entry Point
        workflow.set_entry_point("supervisor")

        # Add Conditional Edges from Supervisor
        workflow.add_conditional_edges(
            "supervisor",
            lambda x: x["next_worker"],
            {
                "cleaning": "cleaning_worker",
                "wrangling": "wrangling_worker",
                "visualization": "visualization_worker",
                "FINISH": END
            }
        )

        # Workers always go back to supervisor
        workflow.add_edge("cleaning_worker", "supervisor")
        workflow.add_edge("wrangling_worker", "supervisor")
        workflow.add_edge("visualization_worker", "supervisor")

        return workflow.compile(checkpointer=self.checkpointer)

    def supervisor_node(self, state: SupervisorState):
        logger.info("Supervisor: Determining next step...")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a supervisor managing a team of data analysis agents:
            - 'cleaning': Handles missing values, outliers, and data type fixes. Use this BEFORE wrangling if the data is messy.
            - 'wrangling': Performs transformations, filtering, sorting, and aggregations.
            - 'visualization': Creates charts and plots using Plotly.

            Analyze the conversation and the current data state to decide who should work next.
            If the task is complete (e.g. the last message says a step was completed and there is nothing else to do), respond with 'FINISH'.
            
            Current Workers: cleaning, wrangling, visualization, FINISH.
            
            Respond ONLY with the EXACT name of the next worker or 'FINISH'. Do not output any other text or punctuation.
            """),
            ("placeholder", "{messages}")
        ])

        chain = prompt | self.model
        # Convert messages to a format the prompt template understands if necessary
        response = chain.invoke({"messages": state["messages"]})
        next_worker = response.content.strip().lower()

        if "completed" in next_worker:
            next_worker = "finish"

        # Simple validation/normalization
        valid_workers = ["cleaning", "wrangling", "visualization", "finish"]
        if next_worker not in valid_workers:
            logger.warning(f"Supervisor returned invalid worker: {next_worker}. Defaulting to FINISH.")
            next_worker = "finish"

        if next_worker == "finish":
            return {"next_worker": "FINISH"}
        
        logger.info(f"Supervisor: Handing off to {next_worker}")
        return {"next_worker": next_worker}

    def cleaning_node(self, state: SupervisorState, config: RunnableConfig):
        logger.info("Supervisor: Invoking Cleaning Worker...")
        agent = self.workers["cleaning"]
        
        # Prepare input for the specialized agent
        agent_state = {
            "messages": state["messages"],
            "data_raw": state["data_raw"],
            "hitl_enabled": state.get("hitl_enabled", False)
        }
        
        result = agent.invoke(agent_state, config=config)
        
        # Update supervisor state with worker results
        # Assuming the worker returns 'cleaned_data' or similar
        new_data = result.get("cleaned_data") or result.get("wrangled_data") or state["data_raw"]
        
        return {
            "messages": [AIMessage(content="Cleaning step completed.")],
            "data_raw": new_data,
            "final_output": result # Pass along for the UI
        }

    def wrangling_node(self, state: SupervisorState, config: RunnableConfig):
        logger.info("Supervisor: Invoking Wrangling Worker...")
        agent = self.workers["wrangling"]
        
        agent_state = {
            "messages": state["messages"],
            "data_raw": state["data_raw"],
            "hitl_enabled": state.get("hitl_enabled", False)
        }
        
        result = agent.invoke(agent_state, config=config)
        
        new_data = result.get("wrangled_data", state["data_raw"])
        
        return {
            "messages": [AIMessage(content="Wrangling step completed.")],
            "data_raw": new_data,
            "final_output": result # Pass along for the UI
        }

    def visualization_node(self, state: SupervisorState, config: RunnableConfig):
        logger.info("Supervisor: Invoking Visualization Worker...")
        agent = self.workers["visualization"]
        
        agent_state = {
            "messages": state["messages"],
            "data_raw": state["data_raw"],
            "hitl_enabled": state.get("hitl_enabled", False)
        }
        
        result = agent.invoke(agent_state, config=config)
        
        # Visualization doesn't usually change data_raw, but returns a plotly_json
        return {
            "messages": [AIMessage(content="Visualization created.")],
            "final_output": result # Pass the full result including plan and plotly_json
        }
