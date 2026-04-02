import pytest
import pandas as pd
from unittest.mock import MagicMock
from langchain_core.messages import HumanMessage
from eda_agents.agents.wrangling import DataWranglingAgent

@pytest.fixture
def mock_model():
    model = MagicMock()
    mock_resp = MagicMock()
    mock_resp.content = "Mocked response"
    model.invoke.return_value = mock_resp
    model.return_value = mock_resp
    return model

@pytest.fixture
def agent(mock_model):
    return DataWranglingAgent(mock_model)

@pytest.fixture
def base_state():
    return {
        "messages": [HumanMessage(content="Clean this data")],
        "data_raw": [{"A": 1}, {"A": None}],
        "plan": "",
        "code": "",
        "error": "",
        "retry_count": 0,
        "wrangled_data": [],
        "hitl_enabled": False
    }

def test_extract_code(agent):
    content_with_python = "```python\nprint('hello')\n```"
    assert agent._extract_code(content_with_python) == "print('hello')"

    content_with_generic = "```\nprint('world')\n```"
    assert agent._extract_code(content_with_generic) == "print('world')"

    content_raw = "print('raw')"
    assert agent._extract_code(content_raw) == "print('raw')"

def test_should_generate_hitl_disabled(agent, base_state):
    assert agent.should_generate(base_state) == "generate"

def test_should_generate_hitl_enabled_wait(agent, base_state):
    base_state["hitl_enabled"] = True
    assert agent.should_generate(base_state) == "wait"

def test_should_generate_hitl_enabled_approve(agent, base_state):
    base_state["hitl_enabled"] = True
    base_state["messages"].append(HumanMessage(content="Yes, proceed please."))
    assert agent.should_generate(base_state) == "generate"

def test_should_retry(agent, base_state):
    base_state["error"] = "Some syntax error"
    base_state["retry_count"] = 0
    assert agent.should_retry(base_state) == "retry"

    base_state["retry_count"] = 3
    assert agent.should_retry(base_state) == "end"

    base_state["error"] = None
    assert agent.should_retry(base_state) == "end"

def test_recommend_steps(agent, base_state):
    agent.model.return_value.content = "1. Drop NAs\n2. Fill zeros"
    new_state = agent.recommend_steps(base_state)
    assert new_state["plan"] == "1. Drop NAs\n2. Fill zeros"

def test_generate_wrangling_code(agent, base_state):
    agent.model.return_value.content = "```python\ndef wrangle(df):\n    return df.dropna()\n```"
    new_state = agent.generate_wrangling_code(base_state)
    assert "def wrangle(df):" in new_state["code"]
    assert new_state["retry_count"] == 0

def test_execute_wrangling_code_success(agent, base_state):
    base_state["code"] = "def wrangle(df):\n    return df.fillna(0)"
    new_state = agent.execute_wrangling_code(base_state)
    
    assert new_state["error"] is None
    assert len(new_state["wrangled_data"]) == 2
    assert new_state["wrangled_data"][0]["A"] == 1
    assert new_state["wrangled_data"][1]["A"] == 0.0

def test_execute_wrangling_code_failure(agent, base_state):
    base_state["code"] = "def wrangle(df):\n    raise ValueError('Test error')"
    new_state = agent.execute_wrangling_code(base_state)
    
    assert new_state["error"] is not None
    assert "Test error" in new_state["error"]
    assert "wrangled_data" not in new_state

def test_execute_wrangling_code_missing_function(agent, base_state):
    base_state["code"] = "def completely_wrong_name(df):\n    return df"
    new_state = agent.execute_wrangling_code(base_state)
    
    assert new_state["error"] is not None
    assert "Function 'wrangle' not found" in new_state["error"] or "not found" in new_state["error"].lower() or "not defined" in new_state["error"].lower()

def test_execute_wrangling_code_invalid_return(agent, base_state):
    base_state["code"] = "def wrangle(df):\n    return 'not a dataframe or list'"
    new_state = agent.execute_wrangling_code(base_state)
    
    assert new_state["error"] is not None
    assert "has no attribute" in new_state["error"] or "list of dictionaries" in new_state["error"]

