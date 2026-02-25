import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from eda_agents.tools.report import generate_custom_report

@pytest.fixture
def sample_data():
    return [
        {"A": 1, "B": "foo", "Target": 0},
        {"A": 2, "B": "bar", "Target": 1},
        {"A": None, "B": "foo", "Target": 0}
    ]

@pytest.fixture
def mock_llm():
    llm = MagicMock()
    # Mock the invoke chain for ChatPromptTemplate | llm
    mock_response = MagicMock()
    mock_response.content = "Mocked Executive Summary."
    llm.invoke.return_value = mock_response
    return llm

@patch("eda_agents.tools.report.explain_data.invoke")
@patch("eda_agents.tools.report.describe_dataset.invoke")
@patch("eda_agents.tools.report.visualize_missing.invoke")
def test_generate_custom_report(mock_vis, mock_desc, mock_exp, sample_data, mock_llm):
    # Setup mocks
    mock_exp.return_value = "Mocked Explain Data"
    mock_desc.return_value = ("Mocked Describe Data", {"describe_df": pd.DataFrame()})
    mock_vis.return_value = ("Mocked Missing Visuals", {"missing_plots": {}})

    result = generate_custom_report(
        data_raw=sample_data,
        wrangling_methods=["Drop Missing Values"],
        analysis_methods=["Data Summary", "Descriptive Statistics", "Missing Values Plot"],
        target_col="Target",
        instructions="Make it look good.",
        llm=mock_llm
    )

    # Validate Wrangling
    assert "cleaned_data" in result
    assert len(result["cleaned_data"]) == 2  # One row with None dropped

    # Validate Analysis Tools Called
    mock_exp.assert_called_once()
    mock_desc.assert_called_once()
    mock_vis.assert_called_once()

    # Validate LLM Output
    assert "summary_markdown" in result
    assert result["summary_markdown"] == "Mocked Executive Summary."

    # Validate Artifacts
    assert "artifacts" in result
    assert "descriptive_statistics" in result["artifacts"]
    assert "missing_plots" in result["artifacts"]

def test_generate_custom_report_wrangling_fill(sample_data, mock_llm):
    result = generate_custom_report(
        data_raw=sample_data,
        wrangling_methods=["Fill Missing Values (Forward Fill)"],
        analysis_methods=[], # Skip analysis to just test wrangling
        target_col="Target",
        instructions="",
        llm=mock_llm
    )
    cleaned = result["cleaned_data"]
    assert len(cleaned) == 3
    assert cleaned[2]["A"] == 2.0  # forward filled from index 1

def test_generate_custom_report_llm_failure(sample_data):
    # Setup LLM that raises an exception
    llm = MagicMock()
    llm.invoke.side_effect = Exception("API Error")

    result = generate_custom_report(
        data_raw=sample_data,
        wrangling_methods=[],
        analysis_methods=[],
        target_col="Target",
        instructions="",
        llm=llm
    )

    assert "summary_markdown" in result
    assert "Failed to generate textual summary" in result["summary_markdown"]
