import sys, os
sys.path.append(os.path.join(os.getcwd(), "src"))
from eda_agents.multiagents.supervisor import SupervisorAgent
from langchain_openai import ChatOpenAI
import os

model = ChatOpenAI(model="gpt-4o-mini")
agent = SupervisorAgent(model=model)

from langchain_core.messages import HumanMessage
initial_state = {
    "messages": [HumanMessage(content="Clean the data")],
    "data_raw": [{"a": 1}, {"a": None}],
    "hitl_enabled": True
}
config = {"configurable": {"thread_id": "test1"}}

print("Invoking...")
result = agent.graph.invoke(initial_state, config=config)
print("Finished invoke. keys:", result.keys())

state = agent.graph.get_state(config)
print("Parent next:", state.next)

if result.get("final_output"):
    print("final_output keys:", result["final_output"].keys())
    if "plan" in result["final_output"]:
        print("Plan found:", result["final_output"]["plan"][:50])
else:
    print("final_output NOT found")
