from typing import Annotated, TypedDict, Union, Dict, Any
import pandas as pd
from langchain_core.messages import BaseMessage
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
    wrangled_data: list[dict]

class DataWranglingAgent(BaseAgent):
    def __init__(self, model):
        super().__init__(model)

    def create_graph(self):
        workflow = StateGraph(AgentState)
        
        workflow.add_node("generate_wrangling_code", self.generate_wrangling_code)
        workflow.add_node("execute_wrangling_code", self.execute_wrangling_code)
        workflow.add_node("fix_wrangling_code", self.fix_wrangling_code)
        
        workflow.set_entry_point("generate_wrangling_code")
        
        workflow.add_edge("generate_wrangling_code", "execute_wrangling_code")
        
        workflow.add_conditional_edges(
            "execute_wrangling_code",
            self.should_retry,
            {
                "retry": "fix_wrangling_code",
                "end": END
            }
        )
        
        workflow.add_edge("fix_wrangling_code", "execute_wrangling_code")
        
        return workflow.compile()

    def generate_wrangling_code(self, state: AgentState):
        logger.info("Generating data wrangling code...")
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert data engineer using Pandas.
            Write a function `wrangle(df: pd.DataFrame)` that performs the requested data cleanup or transformation.
            The data will be provided as a pandas DataFrame.
            The function must return the modified DataFrame.
            Only return the code block, no explanations.
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

    def execute_wrangling_code(self, state: AgentState):
        import tempfile
        code = state["code"]
        logger.info("Executing wrangling code in sandbox...")

        # Write data to a temp file instead of inlining via f-string
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, prefix="eda_wrangle_"
        )
        try:
            json.dump(state["data_raw"], tmp)
            tmp.close()

            wrapper_code = f"""
import pandas as pd
import json
{code}

if __name__ == "__main__":
    try:
        with open("{tmp.name}", "r") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        cleaned_df = wrangle(df)
        print(cleaned_df.to_json(orient='records'))
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
            logger.info("Parsing wrangled data result.")
            wrangled_data = json.loads(result)
            return {"wrangled_data": wrangled_data, "error": None}
        except json.JSONDecodeError:
            logger.error("Failed to parse wrangled JSON output.")
            return {"error": f"Failed to parse output: {result}"}

    def fix_wrangling_code(self, state: AgentState):
        error = state["error"]
        code = state["code"]
        retry_count = state["retry_count"] + 1
        
        logger.warning(f"Repairing code. Retry count: {retry_count}. Error: {error}")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "The following code failed with an error. Fix it.\n\nCode:\n{code}\n\nError:\n{error}"),
            ("user", "Fix the code and return ONLY the corrected `wrangle(df)` function.")
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
        logger.info("Decision: Ending wrangling loop.")
        return "end"

    def _extract_code(self, content: str) -> str:
        if "```python" in content:
            return content.split("```python")[1].split("```")[0].strip()
        elif "```" in content:
            return content.split("```")[1].split("```")[0].strip()
        return content.strip()

    def invoke(self, state: Dict[str, Any]):
        logger.info(f"Wrangling data with {len(state['data_raw'])} records.")
        graph = self.create_graph()
        return graph.invoke(state)
