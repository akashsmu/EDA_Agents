from typing import Annotated, TypedDict, Union, Dict, Any, Optional
import pandas as pd
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from eda_agents.agents.base import BaseAgent
from eda_agents.utils.sandbox import run_code_sandboxed_subprocess
from eda_agents.utils.logger import logger
from eda_agents.tools.dataframe import get_dataframe_summary
from eda_agents.utils.profiling import (
    _profile_dataframe,
    _build_fallback_chart,
    build_prompt_context,
)
from eda_agents.templates.agent_templates import create_coding_agent_graph
import json

class AgentState(TypedDict):
    messages: list[BaseMessage]
    data_raw: dict
    plan: str
    code: str
    error: str
    retry_count: int
    plotly_json: dict
    hitl_enabled: bool
    profile: dict                       # column profile metadata (Phase 5)
    visualization_warning: Optional[str] # fallback / auto-substitution notes

class DataVisualizationAgent(BaseAgent):
    def __init__(self, model):
        super().__init__(model)

    def create_graph(self):
        return create_coding_agent_graph(
            state_schema=AgentState,
            recommend_steps_node=self.recommend_steps,
            generate_code_node=self.generate_code,
            execute_code_node=self.execute_code,
            fix_code_node=self.fix_code,
            should_generate_edge=self.should_generate,
            should_retry_edge=self.should_retry,
            generate_node_name="generate_code",
            execute_node_name="execute_code",
            fix_node_name="fix_code"
        )

    def recommend_steps(self, state: AgentState):
        logger.info("Generating recommended steps for visualization...")
        df = pd.DataFrame(state["data_raw"])
        instructions = state["messages"][-1].content

        # --- Phase 5: enriched context with profile, aliases, units ---
        context_str, profile = build_prompt_context(df, instructions)

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a data visualization architect.
            Analyze the data summary (including COLUMN PROFILE, COLUMN ALIASES, and UNIT HINTS) and the user's request.
            Propose a concise step-by-step plan for the visualization (e.g., filtering, aggregation, chart type choices).
            Only use columns present in the schema or the alias map. Never guess column names.
            Do not write code, just steps.
            """),
            ("user", "Data Summary:\n{summary}\n\nObjective: {instructions}")
        ])

        chain = prompt | self.model
        response = chain.invoke({"summary": context_str, "instructions": instructions})
        plan = response.content

        logger.info("Plan generated.")
        return {"plan": plan, "profile": profile}

    def should_generate(self, state: AgentState):
        # In HITL mode, we pause after recommend_steps. 
        # The UI will then either provide an 'approve' message or we proceed if hitl is disabled.
        if state.get("hitl_enabled", False):
            # Check if the last message is an approval
            last_msg = state["messages"][-1].content.lower()
            if "approve" in last_msg or "proceed" in last_msg or "yes" in last_msg:
                logger.info("HITL: Plan approved. Proceeding to code generation.")
                return "generate"
            else:
                logger.info("HITL: Waiting for user approval of the plan.")
                return "wait"
        
        logger.info("HITL disabled or already approved. Proceeding to code generation.")
        return "generate"

    def generate_code(self, state: AgentState):
        logger.info("Generating visualization code based on plan...")
        plan = state.get("plan", "Visualize the data based on instructions.")
        instructions = state["messages"][-1].content

        # --- Phase 5: rebuild context for code-gen prompt ---
        df = pd.DataFrame(state["data_raw"])
        context_str, _ = build_prompt_context(df, instructions)

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert data visualizer using Python and Plotly.
            Follow the provided plan to write a function `visualize(df: pd.DataFrame)` that returns a Plotly Figure.
            The data summary includes COLUMN PROFILE, COLUMN ALIASES, and UNIT HINTS; use them for column selection and axis labels.
            Only use columns present in the schema or the alias map. Never guess column names.
            Write ONLY the code block.
            Plan: {plan}
            Data Summary: {data_summary}
            """),
            ("user", "{instructions}")
        ])

        chain = prompt | self.model
        response = chain.invoke({
            "plan": plan,
            "instructions": instructions,
            "data_summary": context_str,
        })
        code = self._extract_code(response.content)

        logger.info(f"Generated Code block ({len(code.splitlines())} lines).")
        return {"code": code, "retry_count": 0}

    def execute_code(self, state: AgentState):
        import tempfile
        code = state["code"]
        logger.info("Executing visualization code in sandbox...")

        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, prefix="eda_viz_"
        )
        try:
            json.dump(state["data_raw"], tmp)
            tmp.close()

            wrapper_code = f"""
import pandas as pd
import json
import plotly.io as pio
{code}

if __name__ == "__main__":
    try:
        with open("{tmp.name}", "r") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        fig = visualize(df)
        print(pio.to_json(fig))
    except Exception as e:
        import sys
        print(e, file=sys.stderr)
"""
            result = run_code_sandboxed_subprocess(wrapper_code)
        finally:
            import os
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

        if "[Stderr]:" in result:
            error = result.split("[Stderr]:")[1].strip()
            logger.error(f"Execution error: {error}")

            # --- Phase 5: fallback chart on final retry ---
            if state.get("retry_count", 0) >= 2:
                return self._try_fallback(state, error)

            return {"error": error}

        try:
            plotly_json = json.loads(result)
            return {"plotly_json": plotly_json, "error": None}
        except json.JSONDecodeError:
            logger.error("Failed to parse plotly JSON output.")

            # --- Phase 5: fallback chart on parse failure at final retry ---
            if state.get("retry_count", 0) >= 2:
                return self._try_fallback(
                    state, f"Failed to parse output: {result}"
                )

            return {"error": f"Failed to parse output: {result}"}

    def fix_code(self, state: AgentState):
        error = state["error"]
        code = state["code"]
        retry_count = state["retry_count"] + 1
        
        logger.warning(f"Repairing code (retry {retry_count}). Error: {error}")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "The following code failed with an error. Fix it.\n\nCode:\n{code}\n\nError:\n{error}"),
            ("user", "Fix the code and return ONLY the corrected `visualize(df)` function.")
        ])
        
        chain = prompt | self.model
        response = chain.invoke({"code": code, "error": error})
        new_code = self._extract_code(response.content)
        
        return {"code": new_code, "retry_count": retry_count}

    def should_retry(self, state: AgentState):
        if state.get("error") and state.get("retry_count", 0) < 3:
            return "retry"
        return "end"

    # --- Phase 5: fallback chart helper ---
    def _try_fallback(self, state: AgentState, error_msg: str) -> dict:
        """Attempt to build a fallback chart after code execution failure."""
        df = pd.DataFrame(state.get("data_raw", {}))
        profile = state.get("profile") or _profile_dataframe(df)
        fallback_fig, fallback_note = _build_fallback_chart(df, profile)

        if fallback_fig is not None:
            logger.info(f"Using fallback chart: {fallback_note}")
            return {
                "plotly_json": fallback_fig,
                "error": None,
                "visualization_warning": fallback_note,
            }

        logger.warning("Fallback chart also unavailable.")
        return {"error": error_msg}

    def _extract_code(self, content: str) -> str:
        if "```python" in content:
            return content.split("```python")[1].split("```")[0].strip()
        elif "```" in content:
            return content.split("```")[1].split("```")[0].strip()
        return content.strip()

    def invoke(self, state: Dict[str, Any], config: Dict[str, Any] = None):
        logger.info(f"Visualizing data (HITL: {state.get('hitl_enabled', False)})")
        graph = self.create_graph()
        return graph.invoke(state, config=config)
