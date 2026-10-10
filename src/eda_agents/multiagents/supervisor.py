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
from eda_agents.agents.eda_tools import EDAToolsAgent
from eda_agents.agents.feature_engineering import FeatureEngineeringAgent
from eda_agents.agents.sklearn_ml import SklearnMLAgent
from eda_agents.utils.logger import logger


class SupervisorState(TypedDict):
    """State for the Supervisor Agent."""
    messages: Annotated[List[BaseMessage], operator.add]
    data_raw: Union[dict, list]
    next_worker: str
    final_output: Any
    hitl_enabled: bool


class SupervisorAgent(BaseAgent):
    def __init__(self, model: ChatOpenAI):
        super().__init__(model)
        self.workers = {
            "cleaning": DataCleaningAgent(model),
            "wrangling": DataWranglingAgent(model),
            "visualization": DataVisualizationAgent(model),
            "eda": EDAToolsAgent(model),
            "feature_engineering": FeatureEngineeringAgent(model),
            "sklearn_ml": SklearnMLAgent(model),
        }

    def _make_compiled_graph(self):
        workflow = StateGraph(SupervisorState)

        workflow.add_node("supervisor", self.supervisor_node)
        workflow.add_node("cleaning_worker", self.cleaning_node)
        workflow.add_node("wrangling_worker", self.wrangling_node)
        workflow.add_node("visualization_worker", self.visualization_node)
        workflow.add_node("eda_worker", self.eda_node)
        workflow.add_node("feature_engineering_worker", self.feature_engineering_node)
        workflow.add_node("sklearn_ml_worker", self.sklearn_ml_node)

        workflow.set_entry_point("supervisor")

        workflow.add_conditional_edges(
            "supervisor",
            lambda x: x["next_worker"],
            {
                "cleaning": "cleaning_worker",
                "wrangling": "wrangling_worker",
                "visualization": "visualization_worker",
                "eda": "eda_worker",
                "feature_engineering": "feature_engineering_worker",
                "sklearn_ml": "sklearn_ml_worker",
                "FINISH": END,
            }
        )

        workflow.add_edge("cleaning_worker", "supervisor")
        workflow.add_edge("wrangling_worker", "supervisor")
        workflow.add_edge("visualization_worker", "supervisor")
        workflow.add_edge("eda_worker", "supervisor")
        workflow.add_edge("feature_engineering_worker", "supervisor")
        workflow.add_edge("sklearn_ml_worker", "supervisor")

        return workflow.compile(checkpointer=self.checkpointer)

    def supervisor_node(self, state: SupervisorState):
        logger.info("Supervisor: Determining next step...")

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a supervisor managing a team of data science agents:
            - 'cleaning': Handles missing values, outliers, and data type fixes. Use BEFORE wrangling if data is messy.
            - 'wrangling': Performs transformations, filtering, sorting, aggregations, encoding, and value replacements.
            - 'visualization': Creates a specific custom chart using Plotly (e.g. "scatter of Age vs Fare").
            - 'eda': Answers analytical questions WITHOUT modifying data: dataset overviews, summary statistics,
              missing-data analysis, data quality scores, outlier detection, correlation matrices and correlation funnels,
              distribution/skewness profiling, statistical tests, Sweetviz/D-Tale reports, smart schema tagging.
            - 'feature_engineering': Builds ML-ready features: encodes categoricals, creates datetime features,
              drops ID/constant columns, scales numerics. Use BEFORE training a model.
            - 'sklearn_ml': Trains a scikit-learn classification or regression model, reports accuracy/metrics
              and feature importances. Use AFTER feature engineering or when the user asks to train/predict/classify.

            Read the chat history. If the LAST message is from the user with a new request, route to the appropriate worker.
            If the LAST message is from an assistant confirming a step was completed, respond with 'FINISH'.

            Respond ONLY with the EXACT name of the next worker or 'FINISH'. No other text.
            """),
            ("placeholder", "{messages}")
        ])

        chain = prompt | self.model
        response = chain.invoke({"messages": state["messages"]})
        next_worker = response.content.strip().lower()

        if "completed" in next_worker or "finish" in next_worker:
            next_worker = "finish"

        valid_workers = ["cleaning", "wrangling", "visualization", "eda",
                         "feature_engineering", "sklearn_ml", "finish"]
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
        agent_state = {
            "messages": state["messages"],
            "data_raw": state["data_raw"],
            "hitl_enabled": state.get("hitl_enabled", False),
        }
        result = agent.invoke(agent_state, config=config)
        new_data = result.get("cleaned_data") or result.get("wrangled_data") or state["data_raw"]
        return {
            "messages": [AIMessage(content="Cleaning step completed.")],
            "data_raw": new_data,
            "final_output": result,
        }

    def wrangling_node(self, state: SupervisorState, config: RunnableConfig):
        logger.info("Supervisor: Invoking Wrangling Worker...")
        agent = self.workers["wrangling"]
        agent_state = {
            "messages": state["messages"],
            "data_raw": state["data_raw"],
            "hitl_enabled": state.get("hitl_enabled", False),
        }
        result = agent.invoke(agent_state, config=config)
        new_data = result.get("wrangled_data", state["data_raw"])
        return {
            "messages": [AIMessage(content="Wrangling step completed.")],
            "data_raw": new_data,
            "final_output": result,
        }

    def visualization_node(self, state: SupervisorState, config: RunnableConfig):
        logger.info("Supervisor: Invoking Visualization Worker...")
        agent = self.workers["visualization"]
        agent_state = {
            "messages": state["messages"],
            "data_raw": state["data_raw"],
            "hitl_enabled": state.get("hitl_enabled", False),
        }
        result = agent.invoke(agent_state, config=config)
        return {
            "messages": [AIMessage(content="Visualization created.")],
            "final_output": result,
        }

    def eda_node(self, state: SupervisorState, config: RunnableConfig):
        logger.info("Supervisor: Invoking EDA Tools Worker...")
        agent = self.workers["eda"]
        result = agent.invoke({
            "messages": state["messages"],
            "data_raw": state["data_raw"],
        }, config=config)
        summary = result.get("ai_message", "")
        return {
            "messages": [AIMessage(content=f"EDA analysis step completed.\n\n{summary[:2000]}")],
            "final_output": {
                "ai_message": summary,
                "eda_artifacts": result.get("eda_artifacts", {}),
                "tool_calls": result.get("tool_calls", []),
            },
        }

    def feature_engineering_node(self, state: SupervisorState, config: RunnableConfig):
        logger.info("Supervisor: Invoking Feature Engineering Worker...")
        agent = self.workers["feature_engineering"]
        agent_state = {
            "messages": state["messages"],
            "data_raw": state["data_raw"],
            "hitl_enabled": state.get("hitl_enabled", False),
        }
        result = agent.invoke(agent_state, config=config)
        engineered = result.get("engineered_data")
        new_data = engineered if engineered else state["data_raw"]
        n_cols_before = len(list(state["data_raw"][0].keys())) if state["data_raw"] else 0
        n_cols_after = len(list(new_data[0].keys())) if new_data else 0
        summary = (f"Feature engineering complete. "
                   f"Columns: {n_cols_before} → {n_cols_after}. "
                   f"Dataset has {len(new_data):,} rows.")
        return {
            "messages": [AIMessage(content=f"Feature engineering step completed. {summary}")],
            "data_raw": new_data,
            "final_output": {
                "engineered_data": new_data,
                "plan": result.get("plan", ""),
                "code": result.get("code", ""),
                "summary": summary,
            },
        }

    def sklearn_ml_node(self, state: SupervisorState, config: RunnableConfig):
        logger.info("Supervisor: Invoking Sklearn ML Worker...")
        agent = self.workers["sklearn_ml"]
        agent_state = {
            "messages": state["messages"],
            "data_raw": state["data_raw"],
            "hitl_enabled": state.get("hitl_enabled", False),
        }
        result = agent.invoke(agent_state, config=config)
        ml_results = result.get("ml_results", {})
        task = ml_results.get("task_type", "unknown")
        algo = ml_results.get("algorithm", "")
        metrics = ml_results.get("metrics", {})
        metric_str = ", ".join(f"{k}={v}" for k, v in metrics.items() if isinstance(v, (int, float)))
        summary = f"Model training complete. Task: {task}. Algorithm: {algo}. Metrics: {metric_str}."
        return {
            "messages": [AIMessage(content=f"ML model training step completed. {summary}")],
            "final_output": {
                "ml_results": ml_results,
                "plan": result.get("plan", ""),
                "code": result.get("code", ""),
                "summary": summary,
                "worker_type": "sklearn_ml",
            },
        }
