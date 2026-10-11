"""
Sklearn ML Agent: generates and executes a `train_model(df) -> dict` function that
trains a classification or regression model using scikit-learn.

Returns: accuracy/r2, classification report or regression metrics,
feature importances, confusion matrix, and the model coefficients summary.
"""
from typing import Annotated, TypedDict, Optional, Union
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
import subprocess
import sys


class AgentState(TypedDict):
    messages: list[BaseMessage]
    data_raw: dict
    plan: str
    code: str
    error: str
    retry_count: int
    ml_results: dict
    hitl_enabled: bool
    target_variable: Optional[str]
    task_type: str  # "classification" or "regression" or "auto"


class SklearnMLAgent(BaseAgent):
    """
    Coding agent that auto-trains a scikit-learn model on the current dataset.

    It infers the task type (classification vs regression) from the target column,
    picks an appropriate algorithm, returns feature importances, evaluation metrics,
    and a Plotly figure of feature importance.
    """

    def __init__(self, model, target_variable: Optional[str] = None,
                 task_type: str = "auto"):
        super().__init__(model)
        self._target_variable = target_variable
        self._task_type = task_type

    def _make_compiled_graph(self):
        return create_coding_agent_graph(
            state_schema=AgentState,
            recommend_steps_node=self.recommend_steps,
            generate_code_node=self.generate_code,
            execute_code_node=self.execute_code,
            fix_code_node=self.fix_code,
            should_generate_edge=self.should_generate,
            should_retry_edge=self.should_retry,
            generate_node_name="generate_ml_code",
            execute_node_name="execute_ml_code",
            fix_node_name="fix_ml_code",
        )

    def recommend_steps(self, state: AgentState):
        logger.info("SklearnML: generating training plan...")
        df = pd.DataFrame(state["data_raw"])
        summary = get_dataframe_summary(df, n_sample=5)[0]
        instructions = state["messages"][-1].content
        target = state.get("target_variable") or self._target_variable
        task_type = state.get("task_type") or self._task_type or "auto"

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a senior ML engineer.
Given the dataset summary, propose a concise plan to train a scikit-learn model.
Target: {target}
Task type: {task_type} (auto = infer from target column type/cardinality)

Include:
1. Preprocessing steps (handle missing values, encode categoricals, scale numerics)
2. Algorithm choice (GradientBoosting, RandomForest, LogisticRegression, Ridge, etc.) with rationale
3. Train/test split
4. Evaluation metrics to report
5. Feature importance extraction

Do NOT write code. Output numbered steps only.
"""),
            ("user", "Data Summary:\n{summary}\n\nUser Instructions: {instructions}"),
        ])
        chain = prompt | self.model
        resp = chain.invoke({"summary": summary, "instructions": instructions,
                              "target": target or "not specified", "task_type": task_type})
        return {"plan": resp.content}

    def should_generate(self, state: AgentState):
        if state.get("hitl_enabled", False):
            last_msg = state["messages"][-1].content.lower()
            if any(w in last_msg for w in ["approve", "proceed", "yes", "go"]):
                return "generate"
            return "wait"
        return "generate"

    def generate_code(self, state: AgentState):
        logger.info("SklearnML: generating model code...")
        df = pd.DataFrame(state["data_raw"])
        summary = get_dataframe_summary(df, n_sample=5)[0]
        target = state.get("target_variable") or self._target_variable

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a senior ML engineer.
Write a Python function `train_model(df)` that trains a scikit-learn model.

Rules:
- All imports must be INSIDE the function.
- The function must return a JSON-serializable dict with these keys:
    - task_type: "classification" or "regression"
    - algorithm: name of the algorithm used
    - target_variable: name of the target column
    - train_size, test_size: number of rows
    - metrics: dict with accuracy + classification_report (classification) OR r2, mae, rmse (regression)
    - feature_importances: list of dicts [{"feature": ..., "importance": ...}] sorted descending
    - plotly_figure: JSON from plotly.io.to_json() showing feature importance bar chart
- Use pd.get_dummies for encoding, SimpleImputer for missing values.
- Use train_test_split with test_size=0.2, random_state=42.
- Return ONLY the Python function, no markdown fences, no explanations.

Target column: {target}
"""),
            ("user", "Data Summary:\n{summary}\n\nPlan:\n{plan}"),
        ])
        chain = prompt | self.model
        code_raw = chain.invoke({"summary": summary, "plan": state["plan"],
                                  "target": target or "infer from data"}).content
        return {"code": _strip_code_fences(code_raw)}

    def execute_code(self, state: AgentState):
        logger.info("SklearnML: executing model code...")
        code = state["code"]
        data = state["data_raw"]

        runner = f"""
import pandas as pd
import json

{code}

with open('__data__.json') as f:
    data = json.load(f)
df = pd.DataFrame(data)
result = train_model(df)
with open('__result__.json', 'w') as f:
    json.dump(result, f, default=str)
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

            proc = subprocess.run([sys.executable, runner_path], capture_output=True, text=True, timeout=120)
            stdout = proc.stdout.strip()
            stderr = proc.stderr.strip()

            if "SUCCESS" in stdout and os.path.exists(result_path):
                with open(result_path) as f:
                    ml_results = json.load(f)
                logger.info(f"SklearnML: training complete. task={ml_results.get('task_type')}")
                return {"ml_results": ml_results, "error": "", "retry_count": state.get("retry_count", 0)}
            else:
                error = stderr or stdout or "Unknown error"
                logger.warning(f"SklearnML execution failed: {error}")
                return {"error": error, "retry_count": state.get("retry_count", 0) + 1}

    def fix_code(self, state: AgentState):
        logger.info("SklearnML: fixing code...")
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Fix this Python train_model(df) function.
Error: {error}
Return ONLY the corrected function, no markdown, no explanations.
"""),
            ("user", "Code:\n```python\n{code}\n```"),
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
