import sys
import os
import json

# Add src to python path to import eda_agents
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from eda_agents.multiagents.pandas_data_analyst import PandasDataAnalystAgent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

def test_pandas_analyst():
    print("Testing PandasDataAnalystAgent...")
    model = ChatOpenAI(model="gpt-4o-mini") # Using gpt-4o-mini for quick test
    agent = PandasDataAnalystAgent(model)
    
    # Simple dummy dataset:
    data = [
        {"name": "Alice", "age": 25, "score": 85},
        {"name": "Bob", "age": 30, "score": 90},
        {"name": "Charlie", "age": 35, "score": 95}
    ]
    
    state = {
        "messages": [HumanMessage(content="Calculate the average age and maximum score. Return as 'avg_age' and 'max_score'.")],
        "data_raw": data,
        "hitl_enabled": False
    }
    
    config = {"configurable": {"thread_id": "test-thread-1"}}
    
    print("Step 1: Planning...")
    result = agent.invoke(state, config=config)
    
    # Check if we are interrupted (plan generated but code not yet)
    if "plan" in result and not result.get("code"):
        print("Interrupted after planning. Plan:", result["plan"])
        print("Step 2: Proceeding to code generation...")
        # To proceed, we invoke with None (as per langgraph pattern for resuming)
        # or we follow the agent's logic.
        result = agent.invoke(None, config=config)

    print("Agent Execution Completed.")
    print("Final State Keys:", result.keys())
    print("Analysis Result:", json.dumps(result.get("analysis_result"), indent=2))
    
if __name__ == "__main__":
    test_pandas_analyst()
