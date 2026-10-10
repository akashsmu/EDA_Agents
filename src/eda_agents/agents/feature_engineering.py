"""
Feature Engineering Agent: codes a `feature_engineer(df) -> pd.DataFrame` function,
executes it in a sandbox, and auto-retries on errors.

Adapted from business-science/ai-data-science-team's FeatureEngineeringAgent.
"""
from typing import Annotated, TypedDict, Union, Optional
import pandas as pd
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from eda_agents.agents.base import BaseAgent
from eda_agents.utils.sandbox import run_code_sandboxed_subprocess
from eda_agents.utils.logger import logger
from eda_agents.tools.dataframe import get_dataframe_summary
from eda_agents.templates.agent_templates import create_coding_agent_graph
import json
import tempfile
import os


class AgentState(TypedDict):
    messages: list[BaseMessage]
    data_raw: dict
    plan: str
    code: str
    error: str
    retry_count: int
    engineered_data: list[dict]
    hitl_enabled: bool
    target_variable: Optional[str]


DEFAULT_STEPS = """
- Convert columns to appropriate data types (e.g. object columns that look numeric)
- Remove identifier columns that have a unique value for every row (>95% unique)
- Remove constant columns (only one unique value)
- Encode high-cardinality categorical columns (>5% of rows have rare categories) by grouping rare categories as 'other'
- One-hot-encode remaining categorical / object columns (use pd.get_dummies, drop_first=True)
- Convert boolean columns to integer (0/1)
- From datetime columns, extract year, month, day_of_week, is_weekend — then drop the original
- If a target variable is specified, leave it untransformed (do NOT encode or drop it)
- Scale numeric features with StandardScaler (fit on the training set; here fit on the full df)
"""


class FeatureEngineeringAgent(BaseAgent):
    """
    Coding agent that generates and executes a Python feature-engineering function.

    Default behaviour (overrideable via user_instructions):
        - Drop ID-like and constant columns
        - Encode high-cardinality categoricals
        - One-hot encode remaining categoricals
        - Booleans → int
        - Datetime decomposition
        - StandardScaler on numeric features
    """

    def __init__(self, model, target_variable: Optional[str] = None):
        super().__init__(model)
        self._target_variable = target_variable

    def _make_compiled_graph(self):
        return create_coding_agent_graph(
            state_schema=AgentState,
            recommend_steps_node=self.recommend_steps,
            generate_code_node=self.generate_code,
            execute_code_node=self.execute_code,
            fix_code_node=self.fix_code,
            should_generate_edge=self.should_generate,
            should_retry_edge=self.should_retry,
            generate_node_name="generate_fe_code",
            execute_node_name="execute_fe_code",
            fix_node_name="fix_fe_code",
        )

    def recommend_steps(self, state: AgentState):
        logger.info("FeatureEngineering: generating plan...")
        df = pd.DataFrame(state["data_raw"])
        summary = get_dataframe_summary(df, n_sample=5)[0]
        instructions = state["messages"][-1].content
        target = state.get("target_variable") or self._target_variable or "None"

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert ML feature-engineering architect.
Given the dataset summary and user objective, propose a concise step-by-step feature-engineering plan.
Target variable (do NOT modify or drop it): {target}

Default steps to apply unless the user says otherwise:
{defaults}

Only add or remove steps based on the user's instructions or clear dataset characteristics.
Do NOT write code — output only a numbered list of steps.
"""),
            ("user", "Data Summary:\n{summary}\n\nUser Instructions: {instructions}"),
        ])
        chain = prompt | self.model
        response = chain.invoke({"summary": summary, "instructions": instructions,
                                  "target": target, "defaults": DEFAULT_STEPS})
        return {"plan": response.content}

    def should_generate(self, state: AgentState):
        if state.get("hitl_enabled", False):
            last_msg = state["messages"][-1].content.lower()
            if any(w in last_msg for w in ["approve", "proceed", "yes", "go"]):
                return "generate"
            return "wait"
        return "generate"

    def generate_code(self, state: AgentState):
        logger.info("FeatureEngineering: generating code...")
        df = pd.DataFrame(state["data_raw"])
        summary = get_dataframe_summary(df, n_sample=5)[0]
        target = state.get("target_variable") or self._target_variable or "None"

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a senior Python/Pandas engineer.
Write a Python function named `feature_engineer(df)` that implements the plan below.
Rules:
- The function receives a pandas DataFrame `df` and returns a NEW DataFrame (do NOT modify in-place).
- All imports (pandas, numpy, sklearn, etc.) must be INSIDE the function.
- Target variable '{target}' must NOT be encoded, scaled, or dropped.
- Use pd.get_dummies for one-hot encoding.
- Use sklearn.preprocessing.StandardScaler for scaling numeric columns (excluding the target).
- Return ONLY the Python function, no explanations, no markdown fences.
"""),
            ("user", "Data Summary:\n{summary}\n\nPlan:\n{plan}"),
        ])
        chain = prompt | self.model
        code_raw = chain.invoke({"summary": summary, "plan": state["plan"], "target": target}).content
        code = _strip_code_fences(code_raw)
        return {"code": code}

    def execute_code(self, state: AgentState):
        logger.info("FeatureEngineering: executing code...")
        code = state["code"]
        data = state["data_raw"]

        runner = f"""
import pandas as pd
import json

{code}

data = json.loads(open('__data__.json').read())
df = pd.DataFrame(data)
result = feature_engineer(df)
with open('__result__.json', 'w') as f:
    json.dump(result.to_dict(orient='records'), f, default=str)
print("SUCCESS")
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "__data__.json")
            result_path = os.path.join(tmpdir, "__result__.json")
            runner_path = os.path.join(tmpdir, "_runner.py")

            with open(data_path, "w") as f:
                json.dump(data, f, default=str)
            with open(runner_path, "w") as f:
                f.write(runner.replace("'__data__.json'", f"'{data_path}'")
                        .replace("'__result__.json'", f"'{result_path}'"))

            import subprocess, sys
            proc = subprocess.run([sys.executable, runner_path], capture_output=True, text=True, timeout=60)
            stdout = proc.stdout.strip()
            stderr = proc.stderr.strip()

            if "SUCCESS" in stdout and os.path.exists(result_path):
                with open(result_path) as f:
                    records = json.load(f)
                logger.info(f"FeatureEngineering: produced {len(records)} rows.")
                return {"engineered_data": records, "error": "", "retry_count": state.get("retry_count", 0)}
            else:
                error = stderr or stdout or "Unknown error"
                logger.warning(f"FeatureEngineering execution failed: {error}")
                return {"error": error, "retry_count": state.get("retry_count", 0) + 1}

    def fix_code(self, state: AgentState):
        logger.info("FeatureEngineering: fixing code...")
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Python debugging expert.
The feature_engineer(df) function below raised an error.
Fix it. Return ONLY the corrected function with no explanations or markdown fences.
Error: {error}
"""),
            ("user", "Broken code:\n```python\n{code}\n```"),
        ])
        chain = prompt | self.model
        code_raw = chain.invoke({"code": state["code"], "error": state["error"]}).content
        return {"code": _strip_code_fences(code_raw)}

    def should_retry(self, state: AgentState):
        if state.get("error") and state.get("retry_count", 0) < 3:
            return "retry"
        return "end"


def _strip_code_fences(text: str) -> str:
    lines = text.strip().splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)
