"""
Data Loader Tools Agent: ReAct agent that browses the file system and loads datasets.
Inspired by business-science/ai-data-science-team's DataLoaderToolsAgent.
"""
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import AgentMiddleware, ModelCallLimitMiddleware, ModelRequest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from typing_extensions import NotRequired

from eda_agents.agents.base import BaseAgent
from eda_agents.tools.data_loader import DATA_LOADER_TOOLS
from eda_agents.utils.logger import logger

AGENT_NAME = "data_loader_tools_agent"

SYSTEM_PROMPT = """You are an expert data loading assistant.
You help users find, inspect, and load datasets from their file system.

Available tools:
- list_directory_contents: Browse a directory
- list_data_files: Show only data files (CSV, Excel, Parquet, JSON) in a directory
- get_file_info: Preview a specific file's structure and sample rows
- load_file: Load a file and return a summary of its contents
- search_files_by_pattern: Find files matching a glob pattern

Workflow:
1. If the user gives a directory, list data files there first.
2. Use get_file_info to preview a file before loading if uncertain.
3. Use load_file to actually load the data.
4. Return a clear summary of what you found and loaded.

IMPORTANT: You cannot directly modify the session state. When you load a file,
tell the user the exact file path so they can confirm and load it in the UI.
"""


class DataLoaderState(AgentState):
    user_instructions: NotRequired[Optional[str]]
    loaded_files: NotRequired[Dict[str, Any]]
    ai_message: NotRequired[str]
    tool_calls: NotRequired[List[str]]


def make_data_loader_agent(
    model: Any,
    tools: Optional[List] = None,
    create_react_agent_kwargs: Optional[Dict] = None,
    checkpointer: Any = None,
    max_model_calls: int = 8,
):
    react_agent = create_agent(
        model,
        tools=tools or DATA_LOADER_TOOLS,
        state_schema=DataLoaderState,
        middleware=[ModelCallLimitMiddleware(run_limit=max_model_calls, exit_behavior="end")],
        name="data_loader_react_agent",
        system_prompt=SYSTEM_PROMPT,
        **(create_react_agent_kwargs or {}),
    )

    def prepare_messages(state: DataLoaderState):
        if state.get("messages"):
            return {}
        return {"messages": [HumanMessage(content=state.get("user_instructions") or "What data files are in the current directory?")]}

    def post_process(state: DataLoaderState):
        messages = state.get("messages", [])
        last_human = max((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=-1)
        run_messages = messages[last_human + 1:]

        tool_calls = [tc["name"] for m in run_messages if isinstance(m, AIMessage) for tc in m.tool_calls]
        final = next((m for m in reversed(run_messages) if isinstance(m, AIMessage) and m.content), None)
        ai_message = final.content if final else "File browsing complete."
        if isinstance(ai_message, list):
            ai_message = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in ai_message)

        return {"ai_message": ai_message, "tool_calls": tool_calls}

    workflow = StateGraph(DataLoaderState)
    workflow.add_node("prepare_messages", prepare_messages)
    workflow.add_node("data_loader_react_agent", react_agent)
    workflow.add_node("post_process", post_process)
    workflow.add_edge(START, "prepare_messages")
    workflow.add_edge("prepare_messages", "data_loader_react_agent")
    workflow.add_edge("data_loader_react_agent", "post_process")
    workflow.add_edge("post_process", END)
    return workflow.compile(checkpointer=checkpointer, name=AGENT_NAME)


class DataLoaderToolsAgent(BaseAgent):
    """
    ReAct agent for browsing the file system and loading datasets.

    Parameters
    ----------
    model : BaseChatModel
        A tool-calling chat model.
    """

    def __init__(self, model: Any, create_react_agent_kwargs: Optional[Dict] = None,
                 checkpointer: Any = None, max_model_calls: int = 8):
        super().__init__(model)
        self._params = {
            "model": model,
            "create_react_agent_kwargs": create_react_agent_kwargs,
            "checkpointer": checkpointer,
            "max_model_calls": max_model_calls,
        }
        self.response: Optional[Dict[str, Any]] = None

    def _make_compiled_graph(self):
        return make_data_loader_agent(**self._params)

    def invoke_agent(self, user_instructions: str = None,
                     config: Optional[Dict] = None, messages: Optional[Sequence[BaseMessage]] = None):
        inputs = {"user_instructions": user_instructions,
                  "messages": list(messages) if messages else []}
        self.response = self.graph.invoke(inputs, config=config or {})

    def get_ai_message(self) -> str:
        return (self.response or {}).get("ai_message", "")

    def get_tool_calls(self) -> List[str]:
        return (self.response or {}).get("tool_calls", [])
