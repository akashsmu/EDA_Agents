import numpy as np
import pandas as pd
import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage

from eda_agents.agents.eda_tools import EDA_TOOLS, EDAToolsAgent
from eda_agents.tools.eda import describe_dataset, generate_correlation_funnel, visualize_missing
from eda_agents.tools.eda_advanced import (
    analyze_correlations,
    analyze_relationship,
    assess_data_quality,
    detect_outliers,
    profile_distributions,
)


class ScriptedChatModel(GenericFakeChatModel):
    """Fake tool-calling model that replays a fixed sequence of AI messages."""

    def bind_tools(self, tools, **kwargs):
        return self


def scripted(*messages):
    return ScriptedChatModel(messages=iter(messages))


def tool_call(name, args=None, call_id="call_1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args or {}, "id": call_id}])


@pytest.fixture
def churn_df():
    rng = np.random.default_rng(0)
    n = 200
    tenure = rng.integers(1, 72, n)
    churn = np.where(tenure < 12, "Yes", "No")
    churn[rng.random(n) < 0.1] = "Yes"
    df = pd.DataFrame({
        "customer_id": [f"C{i:04d}" for i in range(n)],
        "tenure": tenure,
        "monthly_charges": np.round(rng.normal(70, 20, n), 2),
        "total_charges": rng.lognormal(7, 1, n).round(2),
        "contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
        "churn": churn,
    })
    df.loc[:9, "monthly_charges"] = np.nan
    df.loc[0, "total_charges"] = 1e6  # extreme outlier
    return df


@pytest.fixture
def data(churn_df):
    return churn_df.to_dict(orient="records")


# --- Tools ----------------------------------------------------------------------------

def test_registry_has_reference_and_extended_tools():
    names = {t.name for t in EDA_TOOLS}
    assert {"explain_data", "describe_dataset", "visualize_missing", "generate_correlation_funnel",
            "generate_sweetviz_report", "generate_dtale_report"} <= names
    assert {"assess_data_quality", "detect_outliers", "analyze_correlations",
            "profile_distributions", "analyze_relationship"} <= names


def test_data_raw_is_hidden_from_llm_schema():
    for t in EDA_TOOLS:
        assert "data_raw" not in t.tool_call_schema.model_json_schema().get("properties", {}), t.name


def test_describe_dataset_content_includes_stats(data):
    content, artifact = describe_dataset.func(data_raw=data)
    assert "tenure" in content and "mean" in content
    assert "stat" in artifact["describe_df"]


def test_visualize_missing_reports_columns(data):
    content, artifact = visualize_missing.func(data_raw=data)
    assert "monthly_charges: 10 missing" in content
    assert {"matrix_plot", "bar_plot", "heatmap_plot"} <= set(artifact)


def test_assess_data_quality_flags_issues():
    df = pd.DataFrame({
        "id": range(30),
        "constant": ["x"] * 30,
        "amount_text": [str(i * 1.5) for i in range(30)],
        "answer": ["Yes", "yes ", "No"] * 10,
        "score": [1.0, None, 3.0] * 10,
    })
    df = pd.concat([df, df.iloc[:3]])
    content, artifact = assess_data_quality.func(data_raw=df.to_dict(orient="records"))
    checks = set(artifact["issues"]["check"])
    assert {"Constant column", "Numeric stored as text", "Inconsistent labels",
            "Missing values", "Duplicate rows"} <= checks
    assert 0 <= artifact["score"] < 100
    assert artifact["grade"] in "ABCDF"
    assert "Data quality score" in content


def test_assess_data_quality_clean_data_scores_100():
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": ["x", "y", "x", "z"]})
    _, artifact = assess_data_quality.func(data_raw=df.to_dict(orient="records"))
    assert artifact["score"] == 100 and artifact["grade"] == "A"


def test_detect_outliers_iqr_and_zscore(data):
    _, iqr = detect_outliers.func(data_raw=data)
    summary = pd.DataFrame(iqr["summary"]).set_index("column")
    assert summary.loc["total_charges", "n_outliers"] >= 1
    assert iqr["plotly_figure"]["data"]

    _, z = detect_outliers.func(data_raw=data, method="zscore", columns=["total_charges"])
    assert z["summary"]["column"] == ["total_charges"]


def test_detect_outliers_unknown_column(data):
    with pytest.raises(ValueError, match="Unknown columns"):
        detect_outliers.func(data_raw=data, columns=["nope"])


def test_analyze_correlations(data):
    content, artifact = analyze_correlations.func(data_raw=data, method="spearman")
    assert "Spearman" in content
    assert set(artifact["top_pairs"]) == {"feature_a", "feature_b", "correlation"}
    assert artifact["plotly_figure"]["data"][0]["type"] == "heatmap"


def test_profile_distributions_detects_right_skew(data):
    _, artifact = profile_distributions.func(data_raw=data)
    summary = pd.DataFrame(artifact["summary"]).set_index("column")
    assert "right-skewed" in summary.loc["total_charges", "shape"]
    assert summary.loc["total_charges", "suggested_transform"] == "log"


@pytest.mark.parametrize("a,b,expected_test", [
    ("tenure", "monthly_charges", "Pearson"),
    ("churn", "tenure", "t-test"),
    ("contract", "tenure", "ANOVA"),
    ("contract", "churn", "Chi-square"),
])
def test_analyze_relationship_picks_test(data, a, b, expected_test):
    content, result = analyze_relationship.func(data_raw=data, column_a=a, column_b=b)
    assert expected_test in result["test"]
    assert 0 <= result["p_value"] <= 1
    assert isinstance(result["significant"], bool)
    assert result["plotly_figure"]["data"]


def test_analyze_relationship_detects_real_effect(data):
    _, result = analyze_relationship.func(data_raw=data, column_a="churn", column_b="tenure")
    assert result["significant"] is True


def test_correlation_funnel_fallback(data):
    content, artifact = generate_correlation_funnel.func(data_raw=data, target="churn", target_bin_index="Yes")
    corr = pd.DataFrame(artifact["correlation_data"])
    assert "churn__Yes" in content
    assert "churn" not in set(corr["feature"])
    assert "customer_id" not in set(corr["feature"])  # identifier columns are skipped
    top = corr.iloc[0]
    assert top["feature"] == "tenure" and top["correlation"] > 0  # low tenure bin predicts churn
    assert artifact["plotly_figure"]["data"]


def test_correlation_funnel_bad_target(data):
    with pytest.raises(ValueError, match="not found"):
        generate_correlation_funnel.func(data_raw=data, target="nope")


# --- Agent ----------------------------------------------------------------------------

def test_agent_runs_tool_and_collects_artifacts(churn_df):
    model = scripted(
        tool_call("detect_outliers", {"method": "iqr"}),
        AIMessage(content="`total_charges` has extreme outliers."),
    )
    agent = EDAToolsAgent(model)
    agent.invoke_agent("Find outliers", data_raw=churn_df)

    assert agent.get_ai_message() == "`total_charges` has extreme outliers."
    assert agent.get_tool_calls() == ["detect_outliers"]
    artifacts = agent.get_artifacts()
    assert "plotly_figure" in artifacts["detect_outliers"]
    tables = agent.get_artifacts(as_dataframe=True)
    assert isinstance(tables["detect_outliers"]["summary"], pd.DataFrame)


def test_agent_multiple_tools_and_repeat_calls(data):
    model = scripted(
        AIMessage(content="", tool_calls=[
            {"name": "analyze_relationship", "args": {"column_a": "churn", "column_b": "tenure"}, "id": "a"},
            {"name": "analyze_relationship", "args": {"column_a": "contract", "column_b": "churn"}, "id": "b"},
        ]),
        AIMessage(content="Both relationships are significant."),
    )
    agent = EDAToolsAgent(model)
    agent.invoke_agent("Is churn related to tenure and contract?", data_raw=data)
    assert set(agent.get_artifacts()) == {"analyze_relationship", "analyze_relationship_2"}


def test_agent_recovers_from_tool_error(data):
    model = scripted(
        tool_call("detect_outliers", {"columns": ["tenur"]}),
        tool_call("detect_outliers", {"columns": ["tenure"]}, call_id="call_2"),
        AIMessage(content="Fixed the column name."),
    )
    agent = EDAToolsAgent(model)
    agent.invoke_agent("Outliers in tenur", data_raw=data)

    errors = [m for m in agent.get_internal_messages() if getattr(m, "status", None) == "error"]
    assert len(errors) == 1 and "Unknown columns" in errors[0].content
    assert list(agent.get_artifacts()) == ["detect_outliers"]  # failed call yields no artifact
    assert agent.get_ai_message() == "Fixed the column name."


def test_agent_requires_data():
    agent = EDAToolsAgent(scripted(AIMessage(content="hi")))
    with pytest.raises(ValueError, match="data_raw"):
        agent.invoke_agent("hello")


def test_agent_invoke_with_message_history(data):
    agent = EDAToolsAgent(scripted(AIMessage(content="Here are my tools.")))
    result = agent.invoke({"messages": [HumanMessage(content="What tools do you have?")], "data_raw": data})
    assert result["ai_message"] == "Here are my tools."
    assert result["eda_artifacts"] == {} and result["tool_calls"] == []


def test_agent_show_returns_mermaid():
    agent = EDAToolsAgent(scripted(AIMessage(content="x")))
    out = agent.show()
    assert out is not None


def test_supervisor_routes_to_eda_worker(data):
    from eda_agents.multiagents.supervisor import SupervisorAgent

    model = scripted(
        AIMessage(content="eda"),  # supervisor routing
        tool_call("assess_data_quality"),  # EDA agent
        AIMessage(content="Quality grade B: 5% missing in monthly_charges."),
        AIMessage(content="FINISH"),  # supervisor after the EDA step
    )
    graph = SupervisorAgent(model).graph
    result = graph.invoke(
        {"messages": [HumanMessage(content="How good is my data quality?")], "data_raw": data, "final_output": {}},
        config={"configurable": {"thread_id": "eda-test"}},
    )
    out = result["final_output"]
    assert out["tool_calls"] == ["assess_data_quality"]
    assert "assess_data_quality" in out["eda_artifacts"]
    assert out["ai_message"].startswith("Quality grade B")
    assert result["data_raw"] == data  # EDA never mutates the dataset


def test_column_names_are_case_insensitive(data):
    _, result = analyze_relationship.func(data_raw=data, column_a="Churn", column_b="TENURE")
    assert (result["column_a"], result["column_b"]) == ("churn", "tenure")


def test_schema_is_in_system_prompt(data):
    seen = []

    class RecordingModel(ScriptedChatModel):
        def _generate(self, messages, *args, **kwargs):
            seen.append(messages[0].content)
            return super()._generate(messages, *args, **kwargs)

    agent = EDAToolsAgent(RecordingModel(messages=iter([AIMessage(content="ok")])))
    agent.invoke_agent("hi", data_raw=data)
    assert "Dataset schema" in seen[0]
    assert "- contract (object, 3 unique, 0% missing): ['Month-to-month', 'One year', 'Two year']" in seen[0]
