from typing import Annotated, TypedDict, Union, Dict, Any
import pandas as pd
from langchain_core.messages import BaseMessage
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
    wrangled_data: dict
    retry_count: int

class DataWranglingAgent(BaseAgent):
    def _make_compiled_graph(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("generate_wrangling_code", self.generate_wrangling_code)
        workflow.add_node("execute_wrangling_code", self.execute_wrangling_code)
        workflow.add_node("fix_wrangling_code", self.fix_wrangling_code)

        workflow.set_entry_point("generate_wrangling_code")
        workflow.add_edge("generate_wrangling_code", "execute_wrangling_code")

        def check_execution(state):
            if state.get("error"):
                if state.get("retry_count", 0) < 3:
                    return "fix_wrangling_code"
                else:
                    return END
            return END

        workflow.add_conditional_edges(
            "execute_wrangling_code",
            check_execution,
            {
                "fix_wrangling_code": "fix_wrangling_code",
                END: END
            }
        )
        workflow.add_edge("fix_wrangling_code", "execute_wrangling_code")

        return workflow.compile(checkpointer=self.checkpointer)

    def generate_wrangling_code(self, state: AgentState):
        df = pd.DataFrame(state["data_raw"])
        summary = get_dataframe_summary(df)[0]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a data cleaning and wrangling expert. Write Python code to clean/transform the dataframe using Pandas. "
                       "The code must define a function `wrangle(df)` that returns the cleaned dataframe. "
                       "Ensure all imports are inside the function or at the top. "
                       "Return ONLY the python code, no markdown backticks."),
            ("user", "Data Summary:\n{summary}\n\nInstructions: {instructions}")
        ])
        
        messages = state["messages"]
        instructions = messages[-1].content if messages else "Clean the data"

        chain = prompt | self.model
        response = chain.invoke({"summary": summary, "instructions": instructions})
        
        code = response.content.replace("```python", "").replace("```", "").strip()
        
        return {"code": code, "retry_count": 0}

    def execute_wrangling_code(self, state: AgentState):
        code = state["code"]
        df_json = json.dumps(state["data_raw"])
        
        wrapper_code = f"""
import pandas as pd
import json
{code}

if __name__ == "__main__":
    try:
        data = json.loads('''{df_json}''')
        df = pd.DataFrame(data)
        cleaned_df = wrangle(df)
        print(cleaned_df.to_json(orient='records'))
    except Exception as e:
        import sys
        print(e, file=sys.stderr)
"""
        result = run_code_sandboxed_subprocess(wrapper_code)
        
        if "[Stderr]:" in result:
            error = result.split("[Stderr]:")[1].strip()
            return {"error": error}
        
        try:
            wrangled_data = json.loads(result)
            return {"wrangled_data": wrangled_data, "error": None}
        except json.JSONDecodeError:
             return {"error": f"Failed to parse output: {result}"}

    def fix_wrangling_code(self, state: AgentState):
        error = state["error"]
        code = state["code"]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert python debugger. Fix the following data wrangling code based on the error message. "
                       "Return ONLY the fixed code."),
            ("user", "Code:\n{code}\n\nError:\n{error}")
        ])
        
        chain = prompt | self.model
        response = chain.invoke({"code": code, "error": error})
        
        fixed_code = response.content.replace("```python", "").replace("```", "").strip()
        
        return {"code": fixed_code, "retry_count": state.get("retry_count", 0) + 1}
