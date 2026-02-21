
import re
import difflib
from typing import Optional

import pandas as pd

from eda_agents.utils.logger import logger
from eda_agents.tools.dataframe import get_dataframe_summary


# ---------------------------------------------------------------------------
# 5.1  Column Profiling
# ---------------------------------------------------------------------------

def _normalize_column_name(value: str) -> str:
    """Lowercase and strip non-alphanumeric characters for fuzzy comparison."""
    if not isinstance(value, str):
        return ""
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _profile_dataframe(df: pd.DataFrame) -> dict:
    """
    Classify every column of *df* into one or more buckets:
    numeric, categorical, datetime, boolean — plus cardinality flags.

    Returns
    -------
    dict with keys:
        n_rows, columns, numeric_cols, categorical_cols, datetime_cols,
        boolean_cols, low_cardinality_numeric, high_cardinality_categorical
    """
    df = df if isinstance(df, pd.DataFrame) else pd.DataFrame()
    n_rows = int(getattr(df, "shape", (0, 0))[0] or 0)

    # Cap at 5 000 rows for profiling speed
    sample = df if n_rows <= 5000 else df.head(5000)

    columns: list[str] = [str(c) for c in list(sample.columns)]
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    datetime_cols: list[str] = []
    boolean_cols: list[str] = []
    low_card_numeric: list[str] = []
    high_card_categorical: list[str] = []

    for col in columns:
        s = sample[col]
        try:
            nunique = int(s.nunique(dropna=True))
        except Exception:
            nunique = 0

        if pd.api.types.is_bool_dtype(s):
            boolean_cols.append(col)
            categorical_cols.append(col)
            continue
        if pd.api.types.is_datetime64_any_dtype(s):
            datetime_cols.append(col)
            continue
        if pd.api.types.is_numeric_dtype(s):
            numeric_cols.append(col)
            if nunique <= 10:
                low_card_numeric.append(col)
                categorical_cols.append(col)
            continue

        # Default: categorical
        categorical_cols.append(col)
        if nunique >= max(20, int(0.2 * max(n_rows, 1))):
            high_card_categorical.append(col)

    logger.debug(
        f"Profile: {n_rows} rows, {len(numeric_cols)} numeric, "
        f"{len(categorical_cols)} cat, {len(datetime_cols)} dt, "
        f"{len(boolean_cols)} bool"
    )

    return {
        "n_rows": n_rows,
        "columns": columns,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "datetime_cols": datetime_cols,
        "boolean_cols": boolean_cols,
        "low_cardinality_numeric": low_card_numeric,
        "high_cardinality_categorical": high_card_categorical,
    }


def _infer_units(columns: list[str]) -> dict[str, str]:
    """Heuristic unit detection from column name keywords."""
    units: dict[str, str] = {}
    for col in columns:
        col_lower = col.lower()
        unit = None
        if "%" in col_lower or "pct" in col_lower or "percent" in col_lower:
            unit = "%"
        elif "usd" in col_lower or "price" in col_lower or "amount" in col_lower:
            unit = "USD"
        elif "cost" in col_lower or "charge" in col_lower:
            unit = "USD"
        elif "date" in col_lower or "time" in col_lower:
            unit = "date/time"
        elif "age" in col_lower:
            unit = "years"
        elif col_lower.endswith("_id") or col_lower == "id":
            unit = None
        if unit:
            units[col] = unit
    return units


def _format_profile_for_prompt(profile: dict) -> str:
    """Compact string representation of a profile dict for LLM prompts."""
    if not isinstance(profile, dict):
        return ""

    def _fmt(values: list[str]) -> str:
        return ", ".join(values[:12]) if values else "None"

    return "\n".join([
        f"Rows: {profile.get('n_rows')}",
        f"Numeric: {_fmt(profile.get('numeric_cols') or [])}",
        f"Categorical: {_fmt(profile.get('categorical_cols') or [])}",
        f"Datetime: {_fmt(profile.get('datetime_cols') or [])}",
        f"Boolean: {_fmt(profile.get('boolean_cols') or [])}",
        f"Low-card numeric: {_fmt(profile.get('low_cardinality_numeric') or [])}",
        f"High-card categorical: {_fmt(profile.get('high_cardinality_categorical') or [])}",
    ])


def _format_units_for_prompt(units: dict[str, str]) -> str:
    if not isinstance(units, dict) or not units:
        return "None"
    items = [f"{k} -> {v}" for k, v in list(units.items())[:12]]
    return ", ".join(items)


# ---------------------------------------------------------------------------
# 5.2  Column Alias Resolution
# ---------------------------------------------------------------------------

def _resolve_column_aliases(text: str, columns: list[str]) -> dict[str, str]:
    """
    Fuzzy-match tokens in user *text* to actual DataFrame *columns*.

    Returns a dict mapping user-text tokens → actual column names for all
    matches above a 0.82 similarity threshold.
    """
    if not isinstance(text, str) or not text.strip():
        return {}
    columns = [str(c) for c in columns if isinstance(c, str)]
    if not columns:
        return {}

    col_norm_map = {c: _normalize_column_name(c) for c in columns}

    tokens = re.findall(r"[A-Za-z0-9_]+", text.lower())
    candidates = set(tokens)
    for i in range(len(tokens) - 1):
        candidates.add(tokens[i] + tokens[i + 1])
        candidates.add(f"{tokens[i]}_{tokens[i + 1]}")

    aliases: dict[str, str] = {}
    for cand in list(candidates):
        cand_norm = _normalize_column_name(cand)
        if not cand_norm or len(cand_norm) < 3:
            continue

        best: Optional[str] = None
        best_score = 0.0
        for col, col_norm in col_norm_map.items():
            if not col_norm:
                continue
            # Exact or substring match
            if cand_norm == col_norm or cand_norm in col_norm:
                best = col
                best_score = 1.0
                break
            score = difflib.SequenceMatcher(None, cand_norm, col_norm).ratio()
            if score > best_score:
                best_score = score
                best = col

        if best and best_score >= 0.82:
            aliases[cand] = best

    logger.debug(f"Resolved {len(aliases)} column alias(es).")
    return aliases


def _format_aliases_for_prompt(aliases: dict[str, str]) -> str:
    if not isinstance(aliases, dict) or not aliases:
        return "None"
    items = [f"{k} -> {v}" for k, v in list(aliases.items())[:12]]
    return ", ".join(items)


# ---------------------------------------------------------------------------
# 5.3  Fallback Charts
# ---------------------------------------------------------------------------

def _label_for_column(col: str, units: dict[str, str]) -> str:
    """Human-friendly axis label with optional unit suffix."""
    label = str(col).replace("_", " ").strip().title()
    unit = units.get(col)
    if unit:
        label = f"{label} ({unit})"
    return label


def _build_fallback_chart(
    df: pd.DataFrame, profile: dict
) -> tuple[dict | None, str | None]:
    """
    Auto-generate a sensible Plotly chart based on column types.

    Called as a safety-net when LLM-generated visualization code fails.

    Returns
    -------
    (fig_dict, note) — the Plotly figure as a dict and a human-readable note,
    or (None, error_msg) if no chart can be built.
    """
    try:
        import plotly.express as px
        import plotly.io as pio
        import json as _json
    except Exception:
        return None, "Plotly is not available for fallback."

    if not isinstance(df, pd.DataFrame) or df.empty:
        return None, "No data available for fallback."

    sample = df.head(5000)
    units = _infer_units(profile.get("columns") or [])
    numeric_cols = profile.get("numeric_cols") or []
    categorical_cols = profile.get("categorical_cols") or []
    datetime_cols = profile.get("datetime_cols") or []

    fig = None
    note = None

    if datetime_cols and numeric_cols:
        x, y = datetime_cols[0], numeric_cols[0]
        fig = px.line(
            sample, x=x, y=y,
            labels={x: _label_for_column(x, units), y: _label_for_column(y, units)},
            title=f"{_label_for_column(y, units)} over {_label_for_column(x, units)}",
        )
        note = f"Fallback line chart using {x} vs {y}."

    elif categorical_cols and numeric_cols:
        x, y = categorical_cols[0], numeric_cols[0]
        fig = px.bar(
            sample, x=x, y=y,
            labels={x: _label_for_column(x, units), y: _label_for_column(y, units)},
            title=f"{_label_for_column(y, units)} by {_label_for_column(x, units)}",
        )
        note = f"Fallback bar chart using {x} vs {y}."

    elif numeric_cols:
        x = numeric_cols[0]
        fig = px.histogram(
            sample, x=x,
            labels={x: _label_for_column(x, units)},
            title=f"Distribution of {_label_for_column(x, units)}",
        )
        note = f"Fallback histogram using {x}."

    elif categorical_cols:
        x = categorical_cols[0]
        fig = px.bar(
            sample, x=x,
            labels={x: _label_for_column(x, units)},
            title=f"Counts by {_label_for_column(x, units)}",
        )
        note = f"Fallback bar chart using {x}."

    if fig is None:
        return None, "No suitable columns found for fallback."

    fig_dict = _json.loads(pio.to_json(fig))
    logger.info(f"Fallback chart built: {note}")
    return fig_dict, note


# ---------------------------------------------------------------------------
# Orchestration — build full prompt context
# ---------------------------------------------------------------------------

MAX_SUMMARY_COLUMNS = 30
MAX_SUMMARY_CHARS = 5000


def build_prompt_context(
    df: pd.DataFrame, user_text: str | None, n_samples: int = 5
) -> tuple[str, dict]:
    """
    Combine a basic DataFrame summary with profile, alias, and unit context
    into a single prompt-ready string.

    Returns
    -------
    (context_str, profile_dict)
    """
    # Basic summary
    df_limited = (
        df.iloc[:, :MAX_SUMMARY_COLUMNS]
        if df.shape[1] > MAX_SUMMARY_COLUMNS
        else df
    )
    base = "\n\n".join(
        get_dataframe_summary(
            [df_limited], n_sample=min(n_samples, 5), skip_stats=True
        )
    )

    profile = _profile_dataframe(df)
    units = _infer_units(profile.get("columns") or [])
    aliases = _resolve_column_aliases(user_text or "", profile.get("columns") or [])

    sections = [
        base,
        "COLUMN PROFILE:\n" + _format_profile_for_prompt(profile),
        "COLUMN ALIASES (user -> dataset):\n" + _format_aliases_for_prompt(aliases),
        "UNIT HINTS:\n" + _format_units_for_prompt(units),
    ]
    context = "\n\n".join([s for s in sections if s])
    return context[:MAX_SUMMARY_CHARS], profile
