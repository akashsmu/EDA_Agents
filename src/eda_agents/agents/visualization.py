from typing import Annotated, TypedDict, Union, Dict, Any
import pandas as pd
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from eda_agents.agents.base import BaseAgent
from eda_agents.utils.sandbox import run_code_sandboxed_subprocess
from eda_agents.utils.logger import logger
from eda_agents.tools.dataframe import get_dataframe_summary
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

class DataVisualizationAgent(BaseAgent):
    def __init__(self, model):
        super().__init__(model)

    def create_graph(self):
        workflow = StateGraph(AgentState)
        
        workflow.add_node("recommend_steps", self.recommend_steps)
        workflow.add_node("generate_code", self.generate_code)
        workflow.add_node("execute_code", self.execute_code)
        workflow.add_node("fix_code", self.fix_code)
        
        workflow.set_entry_point("recommend_steps")
        
        workflow.add_conditional_edges(
            "recommend_steps",
            self.should_generate,
            {
                "generate": "generate_code",
                "wait": END # This is where interrupt happens
            }
        )
        
        workflow.add_edge("generate_code", "execute_code")
        
        workflow.add_conditional_edges(
            "execute_code",
            self.should_retry,
            {
                "retry": "fix_code",
                "end": END
            }
        )
        
        workflow.add_edge("fix_code", "execute_code")
        
        # Compile with checkpointer and optional interrupt
        return workflow.compile(
            checkpointer=MemorySaver(),
            interrupt_after=["recommend_steps"]
        )

    def recommend_steps(self, state: AgentState):
        logger.info("Generating recommended steps for visualization...")
        df = pd.DataFrame(state["data_raw"])
        summary = get_dataframe_summary(df, n_sample=5)[0]
        instructions = state["messages"][-1].content

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a data visualization architect. 
            Analyze the data summary and the user's request.
            Propose a concise step-by-step plan for the visualization (e.g., filtering, aggregation, chart type choices).
            Do not write code, just steps.
            """),
            ("user", "Data Summary:\n{summary}\n\nObjective: {instructions}")
        ])
        
        chain = prompt | self.model
        response = chain.invoke({"summary": summary, "instructions": instructions})
        plan = response.content
        
        logger.info("Plan generated.")
        return {"plan": plan}

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
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert data visualizer using Python and Plotly.
            Follow the provided plan to write a function `visualize(df: pd.DataFrame)` that returns a Plotly Figure.
            Write ONLY the code block.
            Plan: {plan}
            """),
            ("user", "{instructions}")
        ])
        
        chain = prompt | self.model
        response = chain.invoke({"plan": plan, "instructions": instructions})
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
            return {"error": error}

        try:
            plotly_json = json.loads(result)
            return {"plotly_json": plotly_json, "error": None}
        except json.JSONDecodeError:
            logger.error("Failed to parse plotly JSON output.")
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
        if state["error"] and state["retry_count"] < 3:
            return "retry"
        return "end"

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
