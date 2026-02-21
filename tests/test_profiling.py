"""
Unit tests for Phase 5: Column Profiling, Alias Resolution & Fallback Charts.

All tests are pure-Python — no LLM calls or API keys required.
"""

import pytest
import pandas as pd
import numpy as np

from eda_agents.utils.profiling import (
    _normalize_column_name,
    _profile_dataframe,
    _infer_units,
    _format_profile_for_prompt,
    _format_units_for_prompt,
    _resolve_column_aliases,
    _format_aliases_for_prompt,
    _label_for_column,
    _build_fallback_chart,
    build_prompt_context,
)


# ───────────────────────── helpers ──────────────────────────


@pytest.fixture
def mixed_df():
    """DataFrame with numeric, categorical, datetime, and boolean columns."""
    return pd.DataFrame({
        "customer_id": [1, 2, 3, 4, 5],
        "age": [25, 32, 47, 19, 55],
        "total_charges": [100.5, 200.1, 150.0, 310.2, 99.8],
        "gender": ["M", "F", "F", "M", "F"],
        "signup_date": pd.to_datetime(["2020-01-01", "2020-06-15", "2021-03-10", "2019-11-20", "2022-08-05"]),
        "is_active": [True, False, True, True, False],
    })


@pytest.fixture
def numeric_only_df():
    return pd.DataFrame({
        "price": np.random.uniform(10, 500, 50),
        "quantity": np.random.randint(1, 100, 50),
    })


@pytest.fixture
def categorical_only_df():
    return pd.DataFrame({
        "city": ["NYC", "LA", "SF", "CHI", "NYC"] * 4,
        "state": ["NY", "CA", "CA", "IL", "NY"] * 4,
    })


# ───────────── 5.1: Column Profiling ─────────────────────


class TestNormalizeColumnName:
    def test_basic(self):
        assert _normalize_column_name("Total_Charges") == "totalcharges"

    def test_special_chars(self):
        assert _normalize_column_name("col (USD)") == "colusd"

    def test_non_string(self):
        assert _normalize_column_name(123) == ""

    def test_empty(self):
        assert _normalize_column_name("") == ""


class TestProfileDataframe:
    def test_mixed_types(self, mixed_df):
        profile = _profile_dataframe(mixed_df)
        assert profile["n_rows"] == 5
        assert "age" in profile["numeric_cols"]
        assert "total_charges" in profile["numeric_cols"]
        assert "gender" in profile["categorical_cols"]
        assert "signup_date" in profile["datetime_cols"]
        assert "is_active" in profile["boolean_cols"]

    def test_numeric_only(self, numeric_only_df):
        profile = _profile_dataframe(numeric_only_df)
        assert len(profile["numeric_cols"]) == 2
        assert len(profile["categorical_cols"]) == 0
        assert len(profile["datetime_cols"]) == 0

    def test_categorical_only(self, categorical_only_df):
        profile = _profile_dataframe(categorical_only_df)
        assert len(profile["numeric_cols"]) == 0
        assert len(profile["categorical_cols"]) == 2

    def test_empty_dataframe(self):
        profile = _profile_dataframe(pd.DataFrame())
        assert profile["n_rows"] == 0
        assert profile["columns"] == []

    def test_non_dataframe_input(self):
        profile = _profile_dataframe("not a dataframe")
        assert profile["n_rows"] == 0

    def test_low_cardinality_numeric(self):
        """A numeric column with ≤10 unique values should be flagged as low-card."""
        df = pd.DataFrame({"rating": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]})
        profile = _profile_dataframe(df)
        assert "rating" in profile["low_cardinality_numeric"]
        assert "rating" in profile["categorical_cols"]  # also treated as categorical


class TestInferUnits:
    def test_price_column(self):
        units = _infer_units(["price_usd", "quantity", "age"])
        assert units.get("price_usd") == "USD"
        assert units.get("age") == "years"
        assert "quantity" not in units

    def test_percentage_column(self):
        units = _infer_units(["churn_pct", "total"])
        assert units.get("churn_pct") == "%"

    def test_date_column(self):
        units = _infer_units(["signup_date"])
        assert units.get("signup_date") == "date/time"

    def test_id_column(self):
        units = _infer_units(["user_id", "id"])
        assert len(units) == 0

    def test_empty(self):
        assert _infer_units([]) == {}


class TestFormatters:
    def test_format_profile(self, mixed_df):
        profile = _profile_dataframe(mixed_df)
        text = _format_profile_for_prompt(profile)
        assert "Rows:" in text
        assert "Numeric:" in text
        assert "Categorical:" in text

    def test_format_profile_invalid(self):
        assert _format_profile_for_prompt("not a dict") == ""

    def test_format_units(self):
        text = _format_units_for_prompt({"price": "USD", "age": "years"})
        assert "USD" in text
        assert "years" in text

    def test_format_units_empty(self):
        assert _format_units_for_prompt({}) == "None"


# ───────────── 5.2: Column Alias Resolution ────────────


class TestResolveColumnAliases:
    def test_exact_match(self):
        aliases = _resolve_column_aliases(
            "Show me the total_charges distribution",
            ["total_charges", "age", "gender"],
        )
        assert "total_charges" in aliases
        assert aliases["total_charges"] == "total_charges"

    def test_fuzzy_match(self):
        aliases = _resolve_column_aliases(
            "Plot totalcharges vs age",
            ["total_charges", "age", "gender"],
        )
        assert any(v == "total_charges" for v in aliases.values())

    def test_empty_text(self):
        assert _resolve_column_aliases("", ["col_a"]) == {}

    def test_empty_columns(self):
        assert _resolve_column_aliases("some text", []) == {}

    def test_short_tokens_ignored(self):
        """Tokens shorter than 3 normalized chars should be skipped."""
        aliases = _resolve_column_aliases("a b", ["alpha", "beta"])
        assert len(aliases) == 0

    def test_format_aliases(self):
        text = _format_aliases_for_prompt({"totalcharges": "total_charges"})
        assert "totalcharges -> total_charges" in text

    def test_format_aliases_empty(self):
        assert _format_aliases_for_prompt({}) == "None"


# ───────────── 5.3: Fallback Charts ─────────────────────


class TestLabelForColumn:
    def test_basic(self):
        assert _label_for_column("total_charges", {}) == "Total Charges"

    def test_with_unit(self):
        label = _label_for_column("total_charges", {"total_charges": "USD"})
        assert label == "Total Charges (USD)"


class TestBuildFallbackChart:
    def test_datetime_numeric(self, mixed_df):
        profile = _profile_dataframe(mixed_df)
        fig_dict, note = _build_fallback_chart(mixed_df, profile)
        assert fig_dict is not None
        assert "data" in fig_dict
        assert "line" in note.lower()

    def test_categorical_numeric(self):
        df = pd.DataFrame({
            "category": ["A", "B", "C", "A", "B"],
            "value": [10, 20, 30, 40, 50],
        })
        profile = _profile_dataframe(df)
        fig_dict, note = _build_fallback_chart(df, profile)
        assert fig_dict is not None
        assert "bar" in note.lower()

    def test_numeric_only(self, numeric_only_df):
        profile = _profile_dataframe(numeric_only_df)
        fig_dict, note = _build_fallback_chart(numeric_only_df, profile)
        assert fig_dict is not None
        assert "histogram" in note.lower()

    def test_categorical_only(self, categorical_only_df):
        profile = _profile_dataframe(categorical_only_df)
        fig_dict, note = _build_fallback_chart(categorical_only_df, profile)
        assert fig_dict is not None
        assert "bar" in note.lower()

    def test_empty_dataframe(self):
        profile = _profile_dataframe(pd.DataFrame())
        fig_dict, note = _build_fallback_chart(pd.DataFrame(), profile)
        assert fig_dict is None
        assert "no data" in note.lower()


# ───────────── Orchestration ─────────────────────────────


class TestBuildPromptContext:
    def test_returns_nonempty_context(self, mixed_df):
        context, profile = build_prompt_context(mixed_df, "show age vs charges")
        assert len(context) > 0
        assert isinstance(profile, dict)
        assert "COLUMN PROFILE" in context
        assert "COLUMN ALIASES" in context
        assert "UNIT HINTS" in context

    def test_none_user_text(self, mixed_df):
        context, profile = build_prompt_context(mixed_df, None)
        assert len(context) > 0
        assert isinstance(profile, dict)
