import sys
import os
import json

# Add src to python path to import eda_agents
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from eda_agents.multiagents.pandas_data_analyst import PandasDataAnalystAgent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

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
    
    result = agent.invoke(state)
    print("Agent Execution Completed.")
    print("Final State Keys:", result.keys())
    print("Analysis Result:", json.dumps(result.get("analysis_result"), indent=2))
    
if __name__ == "__main__":
    test_pandas_analyst()
