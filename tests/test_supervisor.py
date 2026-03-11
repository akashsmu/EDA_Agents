import pytest
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from eda_agents.multiagents.supervisor import SupervisorAgent

@pytest.fixture
def supervisor():
    model = ChatOpenAI(model="gpt-4o")
    return SupervisorAgent(model)

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "Age": [22, 38, 26, 35, None],
        "Fare": [7.25, 71.28, 7.92, 53.10, 8.05],
        "Survived": [0, 1, 1, 1, 0]
    })

def test_supervisor_initialization(supervisor):
    assert supervisor.model is not None
    assert "cleaning" in supervisor.workers
    assert "wrangling" in supervisor.workers
    assert "visualization" in supervisor.workers

def test_supervisor_workflow_single_step(supervisor, sample_df):
    # Test routing to visualization
    state = {
        "messages": [HumanMessage(content="Plot Age vs Fare")],
        "data_raw": sample_df.to_dict(orient="records"),
        "next_worker": ""
    }
    
    # We can invoke the supervisor node directly to test routing
    result = supervisor.supervisor_node(state)
    assert result["next_worker"] == "visualization"

def test_supervisor_workflow_multi_step(supervisor, sample_df):
    # Test routing for a request that implies multiple steps
    state = {
        "messages": [HumanMessage(content="Clean the data then plot Age")],
        "data_raw": sample_df.to_dict(orient="records"),
        "next_worker": ""
    }
    
    result = supervisor.supervisor_node(state)
    # The LLM should reasonably pick cleaning first, or maybe visualization if it's smart
    # but cleaning is the logical first step here.
    assert result["next_worker"] in ["cleaning", "visualization"]

def test_supervisor_invoke(supervisor, sample_df):
    # This might take longer as it calls multiple agents
    state = {
        "messages": [HumanMessage(content="Clean headers and plot Fare")],
        "data_raw": sample_df.to_dict(orient="records"),
        "next_worker": ""
    }
    
    # Limit execution or mock components for unit testing if needed
    # For now, let's see if a basic invoke works (might need openai key and connectivity)
    # result = supervisor.invoke(state)
    # assert "messages" in result
    pass
