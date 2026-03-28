import pytest
import pandas as pd
from eda_agents.tools.dataframe import get_dataframe_summary, _summarize_dataframe

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "A": [1, 2, None, 4],
        "B": ["foo", "bar", "foo", "baz"],
        "C": [{"key": "val"}, None, {"key": "val2"}, {}]
    })

@pytest.fixture
def another_df():
    return pd.DataFrame({
        "X": [10.5, 20.1],
        "Y": [True, False]
    })

def test_summarize_dataframe(sample_df):
    summary = _summarize_dataframe(sample_df, "Test_Dataset")
    assert "Test_Dataset" in summary
    assert "**Shape**: 4 rows | 3 columns" in summary
    assert "25.00%" in summary  # 1 missing out of 4
    assert "0.00%" in summary
    assert "`float64`" in summary or "`float`" in summary

def test_summarize_dataframe_skip_stats(sample_df):
    summary = _summarize_dataframe(sample_df, "Test_Dataset", skip_stats=True)
    assert "Test_Dataset" in summary
    assert "**Shape**: 4 rows | 3 columns" in summary
    assert "Missing %" not in summary

def test_get_dataframe_summary_single(sample_df):
    summaries = get_dataframe_summary(sample_df)
    assert len(summaries) == 1
    assert "Single_Dataset" in summaries[0]

def test_get_dataframe_summary_list(sample_df, another_df):
    summaries = get_dataframe_summary([sample_df, another_df])
    assert len(summaries) == 2
    assert "Dataset_0" in summaries[0]
    assert "Dataset_1" in summaries[1]

def test_get_dataframe_summary_dict(sample_df, another_df):
    summaries = get_dataframe_summary({"data1": sample_df, "data2": another_df})
    assert len(summaries) == 2
    assert "data1" in summaries[0]
    assert "data2" in summaries[1]

def test_get_dataframe_summary_invalid_type():
    with pytest.raises(TypeError):
        get_dataframe_summary("not a dataframe")

def test_summarize_dataframe_empty():
    empty_df = pd.DataFrame()
    summary = _summarize_dataframe(empty_df, "Empty_Dataset")
    assert "Empty_Dataset" in summary
    assert "**Shape**: 0 rows | 0 columns" in summary

def test_summarize_dataframe_all_nulls():
    null_df = pd.DataFrame({"A": [None, None, None]})
    summary = _summarize_dataframe(null_df, "Null_Dataset")
    assert "100.00%" in summary

