"""
EDA Tools Agent: a tool-calling (ReAct) agent that answers exploratory data analysis
questions by choosing and running EDA tools on the dataset held in state.

Modelled on business-science/ai-data-science-team's `EDAToolsAgent`:

    agent = EDAToolsAgent(llm, invoke_react_agent_kwargs={"recursion_limit": 20})
    agent.invoke_agent("Visualize missing data", data_raw=df)
    agent.get_ai_message(markdown=True)
    agent.get_artifacts()["visualize_missing"]["matrix_plot"]

The dataset is never sent to the LLM. Tools receive it through `InjectedState("data_raw")`,
and only their text summaries enter the conversation; charts and tables come back as artifacts.
"""
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import AgentMiddleware, ModelCallLimitMiddleware, ModelRequest, dynamic_prompt
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from typing_extensions import NotRequired

from eda_agents.agents.base import BaseAgent
from eda_agents.tools.eda import (
    describe_dataset,
    explain_data,
    generate_correlation_funnel,
    generate_dtale_report,
    generate_sweetviz_report,
    visualize_missing,
)
from eda_agents.tools.eda_advanced import (
    analyze_correlations,
    analyze_relationship,
    assess_data_quality,
    detect_outliers,
    profile_distributions,
    smart_schema_tagger,
)
from eda_agents.utils.logger import logger

AGENT_NAME = "eda_tools_agent"

EDA_TOOLS = [
    # Reference toolset (ai-data-science-team)
    explain_data,
    describe_dataset,
    visualize_missing,
    generate_correlation_funnel,
    generate_sweetviz_report,
    generate_dtale_report,
    # Extended toolset
    assess_data_quality,
    detect_outliers,
    analyze_correlations,
    profile_distributions,
    analyze_relationship,
    # Innovative additions
    smart_schema_tagger,
]

SYSTEM_PROMPT = """You are an expert exploratory data analysis (EDA) agent.
A dataset is loaded in your state and is passed to your tools automatically - never ask the user for the data, and never pass data as a tool argument.

How to work:
- Pick the tool(s) that best answer the request. Call several tools if the question needs it.
- Use the exact column names listed in the dataset schema below.
- Charts and tables produced by tools are shown to the user separately - do not reproduce raw tables or base64 data.
- Finish with a concise, data-grounded answer in markdown: key numbers, what they mean, and recommended next steps.
- If asked which tools you have, list them with a one-line description each (a markdown table works well).
- If a tool returns an error, fix the arguments and retry once, or explain the problem.
"""


class EDAToolsState(AgentState):
    data_raw: NotRequired[Union[dict, list]]
    user_instructions: NotRequired[Optional[str]]
    ai_message: NotRequired[str]
    eda_artifacts: NotRequired[Dict[str, Any]]
    tool_calls: NotRequired[List[str]]


class ToolErrorMiddleware(AgentMiddleware):
    """Turns tool exceptions into error ToolMessages so the LLM can recover instead of the run crashing."""

    def _error_message(self, request, exc: Exception) -> ToolMessage:
        name = request.tool_call["name"]
        logger.error(f"EDA tool '{name}' failed: {exc}")
        return ToolMessage(
            content=f"Error running {name}: {exc}",
            tool_call_id=request.tool_call["id"],
            name=name,
            status="error",
        )

    def wrap_tool_call(self, request, handler):
        try:
            return handler(request)
        except Exception as exc:
            return self._error_message(request, exc)

    async def awrap_tool_call(self, request, handler):
        try:
            return await handler(request)
        except Exception as exc:
            return self._error_message(request, exc)


def describe_schema(data_raw: Union[dict, list], max_columns: int = 60) -> str:
    """Compact column listing so the LLM uses exact column names without an extra tool call."""
    df = pd.DataFrame(data_raw)
    lines = [f"Dataset: {df.shape[0]} rows x {df.shape[1]} columns."]
    for col in df.columns[:max_columns]:
        s = df[col]
        line = f"- {col} ({s.dtype}, {s.nunique()} unique, {s.isna().mean():.0%} missing)"
        if s.dtype == object and s.nunique() <= 8:
            line += f": {sorted(map(str, s.dropna().unique()))}"
        lines.append(line)
    if df.shape[1] > max_columns:
        lines.append(f"... and {df.shape[1] - max_columns} more columns (call explain_data to see all).")
    return "\n".join(lines)


@dynamic_prompt
def _system_prompt_with_schema(request: ModelRequest) -> str:
    data_raw = request.state.get("data_raw")
    if data_raw is None:
        return SYSTEM_PROMPT
    return f"{SYSTEM_PROMPT}\nDataset schema (use these exact column names):\n{describe_schema(data_raw)}"


def _to_records(data_raw: Union[pd.DataFrame, dict, list, None]):
    if isinstance(data_raw, pd.DataFrame):
        return data_raw.to_dict(orient="records")
    return data_raw


def collect_artifacts(messages: Sequence[BaseMessage]) -> Dict[str, Any]:
    """Map tool name -> artifact. Repeated calls get suffixed keys (name, name_2, ...)."""
    artifacts: Dict[str, Any] = {}
    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.artifact is not None and msg.status != "error":
            key, i = msg.name, 2
            while key in artifacts:
                key, i = f"{msg.name}_{i}", i + 1
            artifacts[key] = msg.artifact
    return artifacts


def make_eda_tools_agent(
    model: Any,
    tools: Optional[List] = None,
    create_react_agent_kwargs: Optional[Dict] = None,
    checkpointer: Any = None,
    max_model_calls: int = 10,
    log_tool_calls: bool = True,
):
    """
    Builds the EDA agent graph:  START -> prepare_messages -> eda_react_agent -> post_process -> END
    """
    react_agent = create_agent(
        model,
        tools=tools or EDA_TOOLS,
        state_schema=EDAToolsState,
        middleware=[_system_prompt_with_schema, ToolErrorMiddleware(), ModelCallLimitMiddleware(run_limit=max_model_calls, exit_behavior="end")],
        name="eda_react_agent",
        **(create_react_agent_kwargs or {}),
    )

    def prepare_messages(state: EDAToolsState):
        logger.info("EDA Agent: preparing messages")
        if state.get("data_raw") is None:
            raise ValueError("EDAToolsAgent requires `data_raw` (the dataset) in its input state.")
        if state.get("messages"):
            return {}
        return {"messages": [HumanMessage(content=state.get("user_instructions") or "Explain this dataset.")]}

    def post_process(state: EDAToolsState):
        messages = state.get("messages", [])
        # Only look at this run's turn: everything after the latest human message.
        last_human = max((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=-1)
        run_messages = messages[last_human + 1:]

        tool_calls = [tc["name"] for m in run_messages if isinstance(m, AIMessage) for tc in m.tool_calls]
        artifacts = collect_artifacts(run_messages)
        final = next((m for m in reversed(run_messages) if isinstance(m, AIMessage) and m.content), None)
        ai_message = final.content if final else (
            "The analysis ran, but the model did not write a summary. See the artifacts below."
        )
        if isinstance(ai_message, list):  # content blocks
            ai_message = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in ai_message)

        if log_tool_calls:
            for name in tool_calls:
                logger.info(f"EDA Agent: tool used -> {name}")
        return {"ai_message": ai_message, "eda_artifacts": artifacts, "tool_calls": tool_calls}

    workflow = StateGraph(EDAToolsState)
    workflow.add_node("prepare_messages", prepare_messages)
    workflow.add_node("eda_react_agent", react_agent)
    workflow.add_node("post_process", post_process)
    workflow.add_edge(START, "prepare_messages")
    workflow.add_edge("prepare_messages", "eda_react_agent")
    workflow.add_edge("eda_react_agent", "post_process")
    workflow.add_edge("post_process", END)
    return workflow.compile(checkpointer=checkpointer, name=AGENT_NAME)


class EDAToolsAgent(BaseAgent):
    """
    Exploratory Data Analysis agent that calls EDA tools (summaries, missing-data plots,
    correlation funnels, Sweetviz/D-Tale reports, data quality scoring, outlier detection,
    correlation matrices, distribution profiling and statistical tests).

    Parameters
    ----------
    model : BaseChatModel
        A tool-calling chat model.
    tools : list, optional
        Override the toolset (defaults to EDA_TOOLS).
    create_react_agent_kwargs : dict, optional
        Extra kwargs for `langchain.agents.create_agent`.
    invoke_react_agent_kwargs : dict, optional
        Default config for each invocation, e.g. {"recursion_limit": 20}.
    checkpointer : Checkpointer, optional
        Enables multi-turn memory (then pass {"configurable": {"thread_id": ...}}).
    max_model_calls : int
        Upper bound on LLM calls per run; the agent stops gracefully when reached.
    """

    def __init__(
        self,
        model: Any,
        tools: Optional[List] = None,
        create_react_agent_kwargs: Optional[Dict] = None,
        invoke_react_agent_kwargs: Optional[Dict] = None,
        checkpointer: Any = None,
        max_model_calls: int = 10,
        log_tool_calls: bool = True,
    ):
        super().__init__(model)
        # Unlike the coding agents, this agent is stateless per call unless a checkpointer is given.
        self.checkpointer = checkpointer
        self._params = {
            "model": model,
            "tools": tools,
            "create_react_agent_kwargs": create_react_agent_kwargs,
            "checkpointer": checkpointer,
            "max_model_calls": max_model_calls,
            "log_tool_calls": log_tool_calls,
        }
        self.invoke_react_agent_kwargs = invoke_react_agent_kwargs or {}
        self.response: Optional[Dict[str, Any]] = None

    def _make_compiled_graph(self):
        return make_eda_tools_agent(**self._params)

    def update_params(self, **kwargs):
        """Update constructor parameters and rebuild the graph."""
        if "invoke_react_agent_kwargs" in kwargs:
            self.invoke_react_agent_kwargs = kwargs.pop("invoke_react_agent_kwargs") or {}
        self._params.update(kwargs)
        self.checkpointer = self._params["checkpointer"]
        self._compiled_graph = None
        self.response = None

    # --- Invocation -----------------------------------------------------------------

    def _inputs(self, user_instructions, data_raw, messages):
        if messages is None:
            messages = [HumanMessage(content=user_instructions)]
        return {"messages": list(messages), "user_instructions": user_instructions, "data_raw": _to_records(data_raw)}

    def _config(self, config: Optional[Dict]) -> Dict:
        return {**self.invoke_react_agent_kwargs, **(config or {})}

    def invoke_agent(self, user_instructions: str = None, data_raw: Union[pd.DataFrame, dict, list] = None,
                     config: Optional[Dict] = None, messages: Optional[Sequence[BaseMessage]] = None):
        """Run the agent on a question and a dataset. Results are read with the get_* methods."""
        self.response = self.graph.invoke(self._inputs(user_instructions, data_raw, messages), config=self._config(config))
        return None

    async def ainvoke_agent(self, user_instructions: str = None, data_raw: Union[pd.DataFrame, dict, list] = None,
                            config: Optional[Dict] = None, messages: Optional[Sequence[BaseMessage]] = None):
        """Async version of invoke_agent."""
        self.response = await self.graph.ainvoke(self._inputs(user_instructions, data_raw, messages), config=self._config(config))
        return None

    def invoke_messages(self, messages: Sequence[BaseMessage], data_raw=None, config: Optional[Dict] = None):
        """Run with an explicit message history (used by supervisors / multi-turn chat)."""
        return self.invoke_agent(None, data_raw, config=config, messages=messages)

    def invoke(self, inputs: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Raw graph invocation (BaseAgent contract, used by the supervisor)."""
        inputs = {**inputs, "data_raw": _to_records(inputs.get("data_raw"))}
        return self.graph.invoke(inputs, config=self._config(config))

    # --- Results --------------------------------------------------------------------

    def get_ai_message(self, markdown: bool = False):
        """The agent's final answer."""
        if not self.response:
            return None
        return _maybe_markdown(self.response.get("ai_message", ""), markdown)

    def get_artifacts(self, as_dataframe: bool = False) -> Optional[Dict[str, Any]]:
        """
        Tool artifacts keyed by tool name, e.g. get_artifacts()["detect_outliers"]["plotly_figure"].
        With as_dataframe=True, every tabular entry (dict of equal-length lists) becomes a DataFrame.
        """
        if not self.response:
            return None
        artifacts = self.response.get("eda_artifacts") or {}
        if not as_dataframe:
            return artifacts
        return {tool: {k: _as_frame(v) for k, v in art.items()} if isinstance(art, dict) else art
                for tool, art in artifacts.items()}

    def get_tool_calls(self) -> List[str]:
        """Names of the tools the agent called, in order."""
        return (self.response or {}).get("tool_calls", [])

    def get_internal_messages(self, markdown: bool = False):
        """The full message trace (human, AI, tool calls and tool results)."""
        messages = (self.response or {}).get("messages", [])
        if not markdown:
            return messages
        text = "\n\n".join(f"### {m.type.upper()}{f' ({m.name})' if getattr(m, 'name', None) else ''}\n\n{m.content}"
                           for m in messages)
        return _maybe_markdown(text, True)

    def show(self, xray: int = 1):
        """Render the agent graph (PNG in notebooks, Mermaid source otherwise)."""
        graph = self.graph.get_graph(xray=xray)
        try:
            from IPython.display import Image
            return Image(graph.draw_mermaid_png())
        except Exception:
            return graph.draw_mermaid()


def _maybe_markdown(text: str, markdown: bool):
    if markdown:
        try:
            from IPython.display import Markdown
            return Markdown(text)
        except ImportError:
            pass
    return text


def _as_frame(value):
    if isinstance(value, dict) and value and all(isinstance(v, list) for v in value.values()):
        if len({len(v) for v in value.values()}) == 1:
            return pd.DataFrame(value)
    return value
