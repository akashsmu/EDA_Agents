from typing import Annotated, TypedDict, Union, Dict, Any
import pandas as pd
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from eda_agents.agents.base import BaseAgent
from eda_agents.utils.sandbox import run_code_sandboxed_subprocess
from eda_agents.utils.logger import logger
import json

class AgentState(TypedDict):
    messages: list[BaseMessage]
    data_raw: dict
    code: str
    error: str
    retry_count: int
    plotly_json: dict

class DataVisualizationAgent(BaseAgent):
    def __init__(self, model):
        super().__init__(model)

    def create_graph(self):
        workflow = StateGraph(AgentState)
        
        workflow.add_node("generate_code", self.generate_code)
        workflow.add_node("execute_code", self.execute_code)
        workflow.add_node("fix_code", self.fix_code)
        
        workflow.set_entry_point("generate_code")
        
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
        
        return workflow.compile()

    def generate_code(self, state: AgentState):
        logger.info("Generating visualization code...")
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert data visualizer using Python and Plotly.
            Write a function `visualize(df: pd.DataFrame)` that returns a Plotly Figure.
            The data will be provided as a pandas DataFrame.
            Only return the code block, no explanations. 
            Use `import plotly.express as px` or `import plotly.graph_objects as go`.
            Ensure the function returns the figure object.
            """),
            ("user", "{instructions}")
        ])
        
        instructions = state["messages"][-1].content
        logger.debug(f"Instructions: {instructions}")
        
        chain = prompt | self.model
        response = chain.invoke({"instructions": instructions})
        code = self._extract_code(response.content)
        
        logger.info(f"Summary of generated code: {len(code.splitlines())} lines.")
        logger.debug(f"Generated Code block:\n{code}")
        
        return {"code": code, "retry_count": 0}

    def execute_code(self, state: AgentState):
        import tempfile
        code = state["code"]
        logger.info("Executing visualization code in sandbox...")

        # Write data to a temp file instead of inlining via f-string
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
            logger.info("Parsing plotly figure result.")
            plotly_json = json.loads(result)
            return {"plotly_json": plotly_json, "error": None}
        except json.JSONDecodeError:
            logger.error("Failed to parse plotly JSON output.")
            return {"error": f"Failed to parse output: {result}"}

    def fix_code(self, state: AgentState):
        error = state["error"]
        code = state["code"]
        retry_count = state["retry_count"] + 1
        
        logger.warning(f"Repairing code. Retry count: {retry_count}. Error: {error}")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "The following code failed with an error. Fix it.\n\nCode:\n{code}\n\nError:\n{error}"),
            ("user", "Fix the code and return ONLY the corrected `visualize(df)` function.")
        ])
        
        chain = prompt | self.model
        response = chain.invoke({"code": code, "error": error})
        new_code = self._extract_code(response.content)
        
        logger.info("Generated revised code.")
        return {"code": new_code, "retry_count": retry_count}

    def should_retry(self, state: AgentState):
        if state["error"] and state["retry_count"] < 3:
            logger.info(f"Decision: Retrying execution (count {state['retry_count']})")
            return "retry"
        logger.info("Decision: Ending script generation loop.")
        return "end"

    def _extract_code(self, content: str) -> str:
        if "```python" in content:
            return content.split("```python")[1].split("```")[0].strip()
        elif "```" in content:
            return content.split("```")[1].split("```")[0].strip()
        return content.strip()

    def invoke(self, state: Dict[str, Any]):
        logger.info(f"Visualizing data with {len(state['data_raw'])} records.")
        graph = self.create_graph()
        return graph.invoke(state)
