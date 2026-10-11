import json
import operator
import time
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
            ("system", """You are a supervisor managing a team of data science agents. Route to exactly one worker or FINISH.

ROUTING RULES (apply in order):
1. 'eda'          — ANY question or analysis request that does NOT modify data:
                    "assess", "analyze", "tell me about", "describe", "explain", "report", "summarise",
                    "data quality", "what should I fix", "outliers", "missing values overview",
                    "correlation", "distribution", "statistics", "schema", "profile".
                    → USE 'eda' even when the user says "tell me what to fix first" — that is a question, not an action.
2. 'visualization' — user explicitly asks for a chart, plot, graph, or figure.
3. 'cleaning'     — user explicitly asks to APPLY a fix: drop/remove/fill/impute/replace/handle missing values or outliers.
4. 'wrangling'    — user asks to transform, filter, sort, aggregate, merge, rename, encode, or add/create columns.
5. 'feature_engineering' — user asks to engineer features, prepare for ML, encode + scale, or the word "features".
6. 'sklearn_ml'   — user asks to train, predict, classify, fit, or evaluate a model.
7. FINISH         — the LAST message is from an assistant confirming a worker step was completed (contains "completed" or "created").

If the LAST user message is "Approve and proceed" or similar approval, re-route to the SAME worker that last posted a "plan is ready" message.

Respond ONLY with the EXACT worker name or 'FINISH'. No other text.
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

    def _make_inner_config(self, worker_name: str) -> dict:
        """Fresh thread_id per invocation prevents stale checkpointer state."""
        return {"configurable": {"thread_id": f"{worker_name}_{int(time.time() * 1000)}"}}

    def cleaning_node(self, state: SupervisorState, config: RunnableConfig):
        logger.info("Supervisor: Invoking Cleaning Worker...")
        agent = self.workers["cleaning"]
        hitl = state.get("hitl_enabled", False)
        agent_state = {
            "messages": state["messages"],
            "data_raw": state["data_raw"],
            "hitl_enabled": hitl,
        }
        result = agent.invoke(agent_state, config=self._make_inner_config("cleaning"))
        new_data = result.get("cleaned_data") or result.get("wrangled_data") or state["data_raw"]
        executed = bool(result.get("cleaned_data"))
        msg = "Cleaning step completed." if executed else "Data cleaning plan is ready, awaiting approval."
        return {
            "messages": [AIMessage(content=msg)],
            "data_raw": new_data,
            "final_output": result,
        }

    def wrangling_node(self, state: SupervisorState, config: RunnableConfig):
        logger.info("Supervisor: Invoking Wrangling Worker...")
        agent = self.workers["wrangling"]
        hitl = state.get("hitl_enabled", False)
        agent_state = {
            "messages": state["messages"],
            "data_raw": state["data_raw"],
            "hitl_enabled": hitl,
        }
        result = agent.invoke(agent_state, config=self._make_inner_config("wrangling"))
        new_data = result.get("wrangled_data") or state["data_raw"]
        executed = bool(result.get("wrangled_data"))
        msg = "Wrangling step completed." if executed else "Data wrangling plan is ready, awaiting approval."
        return {
            "messages": [AIMessage(content=msg)],
            "data_raw": new_data,
            "final_output": result,
        }

    def visualization_node(self, state: SupervisorState, config: RunnableConfig):
        logger.info("Supervisor: Invoking Visualization Worker...")
        agent = self.workers["visualization"]
        hitl = state.get("hitl_enabled", False)
        agent_state = {
            "messages": state["messages"],
            "data_raw": state["data_raw"],
            "hitl_enabled": hitl,
        }
        result = agent.invoke(agent_state, config=self._make_inner_config("visualization"))
        executed = bool(result.get("plotly_json"))
        msg = "Visualization created." if executed else "Visualization plan is ready, awaiting approval."
        return {
            "messages": [AIMessage(content=msg)],
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
        result = agent.invoke(agent_state, config=self._make_inner_config("feature_engineering"))
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
        result = agent.invoke(agent_state, config=self._make_inner_config("sklearn_ml"))
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
