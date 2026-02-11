from typing import Annotated, TypedDict, Union, Dict, Any
import pandas as pd
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from eda_agents.agents.base import BaseAgent
from eda_agents.utils.sandbox import run_code_sandboxed_subprocess
from eda_agents.tools.dataframe import get_dataframe_summary
import json

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], "messages"]
    data_raw: dict
    code: str
    error: str
    plotly_json: dict
    retry_count: int

class DataVisualizationAgent(BaseAgent):
    def _make_compiled_graph(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("generate_code", self.generate_code)
        workflow.add_node("execute_code", self.execute_code)
        workflow.add_node("fix_code", self.fix_code)

        workflow.set_entry_point("generate_code")

        workflow.add_edge("generate_code", "execute_code")
        
        def check_execution(state):
            if state.get("error"):
                if state.get("retry_count", 0) < 3:
                    return "fix_code"
                else:
                    return END
            return END

        workflow.add_conditional_edges(
            "execute_code",
            check_execution,
            {
                "fix_code": "fix_code",
                END: END
            }
        )

        workflow.add_edge("fix_code", "execute_code")

        return workflow.compile(checkpointer=self.checkpointer)

    def generate_code(self, state: AgentState):
        df = pd.DataFrame(state["data_raw"])
        summary = get_dataframe_summary(df)[0]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a data visualization expert. Write Python code to visualize the data using Plotly Express. "
                       "The code must define a function `visualize(df)` that returns a plotly figure. "
                       "Do not use `fig.show()`. Just return the figure object. "
                       "Ensure all imports are inside the function or at the top. "
                       "Return ONLY the python code, no markdown backticks."),
            ("user", "Data Summary:\n{summary}\n\nInstructions: {instructions}")
        ])
        
        messages = state["messages"]
        instructions = messages[-1].content if messages else "Visualize the data"

        chain = prompt | self.model
        response = chain.invoke({"summary": summary, "instructions": instructions})
        
        code = response.content.replace("```python", "").replace("```", "").strip()
        
        return {"code": code, "retry_count": 0}

    def execute_code(self, state: AgentState):
        import tempfile
        code = state["code"]

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
            return {"error": error}

        try:
            plotly_json = json.loads(result)
            return {"plotly_json": plotly_json, "error": None}
        except json.JSONDecodeError:
            return {"error": f"Failed to parse output: {result}"}

    def fix_code(self, state: AgentState):
        error = state["error"]
        code = state["code"]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert python debugger. Fix the following code based on the error message. "
                       "Return ONLY the fixed code."),
            ("user", "Code:\n{code}\n\nError:\n{error}")
        ])
        
        chain = prompt | self.model
        response = chain.invoke({"code": code, "error": error})
        
        fixed_code = response.content.replace("```python", "").replace("```", "").strip()
        
        return {"code": fixed_code, "retry_count": state.get("retry_count", 0) + 1}
