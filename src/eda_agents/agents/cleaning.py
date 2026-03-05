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
from eda_agents.templates.agent_templates import create_coding_agent_graph
import json

class AgentState(TypedDict):
    messages: list[BaseMessage]
    data_raw: dict
    plan: str
    code: str
    error: str
    retry_count: int
    cleaned_data: list[dict]
    hitl_enabled: bool

class DataCleaningAgent(BaseAgent):
    def __init__(self, model):
        super().__init__(model)

    def _make_compiled_graph(self):
        return create_coding_agent_graph(
            state_schema=AgentState,
            recommend_steps_node=self.recommend_steps,
            generate_code_node=self.generate_cleaning_code,
            execute_code_node=self.execute_cleaning_code,
            fix_code_node=self.fix_cleaning_code,
            should_generate_edge=self.should_generate,
            should_retry_edge=self.should_retry,
            generate_node_name="generate_cleaning_code",
            execute_node_name="execute_cleaning_code",
            fix_node_name="fix_cleaning_code"
        )

    def recommend_steps(self, state: AgentState):
        logger.info("Generating recommended steps for data cleaning...")
        df = pd.DataFrame(state["data_raw"])
        summary = get_dataframe_summary(df, n_sample=5)[0]
        instructions = state["messages"][-1].content

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a data engineering architect using Pandas.
            Analyze the data summary and the user's request.
            Propose a concise step-by-step plan for data cleaning (handling missing values, outliers, data types, etc.).
            Do not write code, just steps.
            """),
            ("user", "Data Summary:\n{summary}\n\nObjective: {instructions}")
        ])
        
        chain = prompt | self.model
        response = chain.invoke({"summary": summary, "instructions": instructions})
        plan = response.content
        
        logger.info("Cleaning plan generated.")
        return {"plan": plan}

    def should_generate(self, state: AgentState):
        if state.get("hitl_enabled", False):
            # Check if the last message is an approval
            last_msg = state["messages"][-1].content.lower()
            if any(word in last_msg for word in ["approve", "proceed", "yes", "go"]):
                logger.info("HITL: Plan approved. Proceeding to code generation.")
                return "generate"
            else:
                logger.info("HITL: Waiting for user approval of the plan.")
                return "wait"
        
        logger.info("HITL disabled or already approved. Proceeding to code generation.")
        return "generate"

    def generate_cleaning_code(self, state: AgentState):
        logger.info("Generating data cleaning code based on plan...")
        plan = state.get("plan", "Clean the data as requested.")
        instructions = state["messages"][-1].content

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert data engineer using Pandas.
            Follow the provided plan to write a function `clean(df: pd.DataFrame)` that performs the cleaning transformations.
            The function must return the modified DataFrame.
            Write ONLY the code block.
            Plan: {plan}
            """),
            ("user", "{instructions}")
        ])
        
        chain = prompt | self.model
        response = chain.invoke({"plan": plan, "instructions": instructions})
        code = self._extract_code(response.content)
        
        logger.info(f"Generated Cleaning Code block ({len(code.splitlines())} lines).")
        return {"code": code, "retry_count": 0}

    def execute_cleaning_code(self, state: AgentState):
        import tempfile
        code = state["code"]
        logger.info("Executing cleaning code in sandbox...")

        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, prefix="eda_clean_"
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
        cleaned_df = clean(df)
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
            cleaned_data = json.loads(result)
            return {"cleaned_data": cleaned_data, "error": None}
        except json.JSONDecodeError:
            logger.error("Failed to parse cleaned JSON output.")
            return {"error": f"Failed to parse output: {result}"}

    def fix_cleaning_code(self, state: AgentState):
        error = state["error"]
        code = state["code"]
        retry_count = state["retry_count"] + 1
        
        logger.warning(f"Repairing cleaning code (retry {retry_count}). Error: {error}")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "The following code failed with an error. Fix it.\n\nCode:\n{code}\n\nError:\n{error}"),
            ("user", "Fix the code and return ONLY the corrected `clean(df)` function.")
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
        logger.info(f"Cleaning data (HITL: {state.get('hitl_enabled', False)})")
        return super().invoke(state, config=config)
