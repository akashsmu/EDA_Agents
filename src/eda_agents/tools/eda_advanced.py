"""
Additional EDA tools that go beyond the reference EDA toolset:
data quality scoring, outlier detection, correlation matrices,
distribution profiling, and automatic statistical relationship tests.

Every tool follows the same contract as the tools in `eda.py`:
    - `data_raw` is injected from the agent state (never chosen by the LLM).
    - The `content` string is written for the LLM to read and summarize.
    - The `artifact` dict holds structured results (tables as dict-of-lists,
      Plotly figures as dicts) for the UI / notebook to render.
"""
from typing import Annotated, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from eda_agents.utils.logger import logger
from eda_agents.utils.plotting import fig_to_dict


# Numeric columns with this many unique values or fewer are treated as categorical
# (e.g. 0/1 flags, ratings) by the relationship and distribution tools.
CATEGORICAL_MAX_UNIQUE = 10


def _numeric_columns(df: pd.DataFrame, min_unique: int = 3) -> List[str]:
    return [
        c for c in df.select_dtypes(include="number").columns
        if not pd.api.types.is_bool_dtype(df[c]) and df[c].nunique() >= min_unique
    ]


def _resolve_columns(df: pd.DataFrame, columns: Optional[List[str]]) -> List[str]:
    """Validates column names, tolerating case / whitespace / underscore differences from the LLM."""
    if not columns:
        return list(df.columns)

    def norm(name):
        return str(name).strip().lower().replace(" ", "_")

    lookup = {norm(c): c for c in df.columns}
    resolved = [c if c in df.columns else lookup.get(norm(c)) for c in columns]
    missing = [c for c, r in zip(columns, resolved) if r is None]
    if missing:
        raise ValueError(f"Unknown columns {missing}. Available columns: {list(df.columns)}")
    return resolved


def _table(df: pd.DataFrame) -> str:
    """Compact text table for LLM-facing content (no tabulate dependency)."""
    return df.to_string(index=False, max_rows=30)


# ---------------------------------------------------------------------------
# Data quality
# ---------------------------------------------------------------------------

@tool(response_format="content_and_artifact")
def assess_data_quality(
    data_raw: Annotated[Union[dict, list], InjectedState("data_raw")],
) -> Tuple[str, Dict]:
    """
    Tool: assess_data_quality
    Description:
        Audits the dataset and returns a 0-100 data quality score with a letter grade,
        plus a list of concrete issues (missing data, duplicate rows, constant columns,
        numbers stored as text, mixed types, inconsistent category labels such as
        'Yes' vs ' yes', ID-like and high-cardinality columns). Each issue includes a
        suggested fix phrased as an instruction that a cleaning agent can execute.
        Use this when the user asks about data quality, data health, or what to clean.
    """
    logger.info("Executing tool: assess_data_quality")
    df = pd.DataFrame(data_raw)
    n_rows, n_cols = df.shape
    issues = []
    penalties = {}

    def add(check, severity, column, detail, fix):
        issues.append({"check": check, "severity": severity, "column": column,
                       "detail": detail, "suggested_fix": fix})

    # 1. Missing values
    missing_pct = float(df.isna().mean().mean() * 100) if df.size else 0.0
    penalties["missing"] = min(25.0, missing_pct)
    col_missing = df.isna().mean() * 100
    mostly_missing = col_missing[col_missing > 50]
    penalties["mostly_missing"] = min(15.0, 5.0 * len(mostly_missing))
    for col, pct in col_missing[col_missing > 0].sort_values(ascending=False).items():
        if pct > 50:
            add("Mostly missing column", "high", col, f"{pct:.1f}% missing",
                f"Drop the column {col} because it is more than half empty")
        else:
            add("Missing values", "medium" if pct > 5 else "low", col, f"{pct:.1f}% missing",
                f"Impute missing values in {col} (median if numeric, mode if categorical)")

    # 2. Duplicate rows
    try:
        n_dupes = int(df.duplicated().sum())
    except TypeError:  # unhashable cells (lists/dicts)
        n_dupes = int(df.astype(str).duplicated().sum())
    dupe_pct = n_dupes / n_rows * 100 if n_rows else 0.0
    penalties["duplicates"] = min(15.0, dupe_pct * 2)
    if n_dupes:
        add("Duplicate rows", "high" if dupe_pct > 5 else "medium", None,
            f"{n_dupes} duplicate rows ({dupe_pct:.1f}%)", "Remove duplicate rows")

    constant, as_text, mixed, inconsistent = 0, 0, 0, 0
    for col in df.columns:
        s = df[col]
        non_null = s.dropna()
        if non_null.empty:
            continue
        nunique = non_null.astype(str).nunique()

        # 3. Constant columns
        if nunique == 1:
            constant += 1
            add("Constant column", "medium", col, f"single value '{non_null.iloc[0]}'",
                f"Drop the column {col} because it has only one value")
            continue

        if s.dtype == object:
            types = non_null.map(lambda v: type(v).__name__).unique()
            # 4. Mixed python types in one column
            if len(types) > 1:
                mixed += 1
                add("Mixed types", "medium", col, f"contains {', '.join(sorted(types))}",
                    f"Convert column {col} to a single consistent data type")
            # 5. Numbers stored as text
            as_str = non_null.astype(str).str.strip()
            numeric_ratio = pd.to_numeric(as_str.str.replace(",", ""), errors="coerce").notna().mean()
            if numeric_ratio >= 0.9:
                as_text += 1
                add("Numeric stored as text", "medium", col,
                    f"{numeric_ratio * 100:.0f}% of values parse as numbers",
                    f"Convert column {col} to numeric, coercing invalid values to NaN")
                continue
            # 6. Labels that differ only by case / whitespace
            normalized = as_str.str.lower()
            if normalized.nunique() < as_str.nunique() or (as_str != non_null.astype(str)).any():
                inconsistent += 1
                variants = (
                    pd.DataFrame({"raw": non_null.astype(str), "norm": normalized})
                    .groupby("norm")["raw"].unique()
                )
                examples = [list(v) for v in variants if len(v) > 1][:2]
                add("Inconsistent labels", "medium", col,
                    f"labels differ only by case/whitespace, e.g. {examples}" if examples
                    else "values have leading/trailing whitespace",
                    f"Strip whitespace and standardize the case of values in {col}")
            # 7. ID-like / high cardinality (informational)
            unique_ratio = nunique / len(non_null)
            if unique_ratio > 0.95 and len(non_null) > 20:
                add("ID-like column", "info", col, f"{nunique} unique values ({unique_ratio:.0%})",
                    f"Exclude {col} from modelling and correlation analysis")
            elif nunique > 50:
                add("High cardinality", "info", col, f"{nunique} distinct categories",
                    f"Group rare categories in {col} into 'Other'")
        elif pd.api.types.is_integer_dtype(s) and nunique == len(non_null) and len(non_null) > 20:
            add("ID-like column", "info", col, "every value is unique",
                f"Exclude {col} from modelling and correlation analysis")

    penalties["constant"] = min(15.0, 5.0 * constant)
    penalties["numeric_as_text"] = min(10.0, 3.0 * as_text)
    penalties["mixed_types"] = min(10.0, 3.0 * mixed)
    penalties["inconsistent_labels"] = min(10.0, 3.0 * inconsistent)

    score = round(max(0.0, 100.0 - sum(penalties.values())), 1)
    grade = next(g for cutoff, g in [(90, "A"), (80, "B"), (70, "C"), (60, "D"), (0, "F")] if score >= cutoff)

    severity_order = {"high": 0, "medium": 1, "low": 2, "info": 3}
    issues.sort(key=lambda i: severity_order[i["severity"]])
    issues_df = pd.DataFrame(issues, columns=["check", "severity", "column", "detail", "suggested_fix"])

    content = (
        f"Data quality score: {score}/100 (grade {grade}) for {n_rows} rows x {n_cols} columns.\n"
        f"Overall missing cells: {missing_pct:.2f}%. Duplicate rows: {n_dupes}.\n"
        f"Score deductions: { {k: round(v, 1) for k, v in penalties.items() if v} }\n\n"
        + (f"Issues found ({len(issues)}):\n{_table(issues_df[['severity', 'check', 'column', 'detail']])}"
           if issues else "No data quality issues found.")
    )
    artifact = {
        "score": score,
        "grade": grade,
        "n_rows": n_rows,
        "n_columns": n_cols,
        "penalties": penalties,
        "issues": issues_df.to_dict(orient="list"),
    }
    logger.info(f"Data quality assessed: score={score} grade={grade} issues={len(issues)}")
    return content, artifact


# ---------------------------------------------------------------------------
# Outliers
# ---------------------------------------------------------------------------

@tool(response_format="content_and_artifact")
def detect_outliers(
    data_raw: Annotated[Union[dict, list], InjectedState("data_raw")],
    method: Literal["iqr", "zscore"] = "iqr",
    threshold: Optional[float] = None,
    columns: Optional[List[str]] = None,
) -> Tuple[str, Dict]:
    """
    Tool: detect_outliers
    Description:
        Detects outliers in numeric columns using the IQR rule (default, threshold 1.5)
        or z-scores (threshold 3.0). Returns per-column outlier counts, percentages and
        bounds, plus a box plot of standardized values for the most affected columns.

    Parameters:
        method: "iqr" or "zscore".
        threshold: IQR multiplier or z-score cutoff. Defaults to 1.5 (iqr) or 3.0 (zscore).
        columns: Optional subset of columns to check. Defaults to all numeric columns.
    """
    logger.info(f"Executing tool: detect_outliers (method={method})")
    df = pd.DataFrame(data_raw)
    threshold = threshold if threshold is not None else (1.5 if method == "iqr" else 3.0)
    candidates = [c for c in _resolve_columns(df, columns) if c in _numeric_columns(df, min_unique=CATEGORICAL_MAX_UNIQUE + 1)]
    if not candidates:
        return "No continuous numeric columns available for outlier detection.", {"summary": {}, "plotly_figure": None}

    rows = []
    for col in candidates:
        s = df[col].dropna()
        if method == "iqr":
            q1, q3 = s.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower, upper = q1 - threshold * iqr, q3 + threshold * iqr
        else:
            mu, sd = s.mean(), s.std()
            lower, upper = mu - threshold * sd, mu + threshold * sd
        mask = (s < lower) | (s > upper)
        rows.append({
            "column": col,
            "n_outliers": int(mask.sum()),
            "pct_outliers": round(float(mask.mean() * 100), 2),
            "lower_bound": round(float(lower), 4),
            "upper_bound": round(float(upper), 4),
            "min": float(s.min()),
            "max": float(s.max()),
        })
    summary = pd.DataFrame(rows).sort_values("n_outliers", ascending=False)

    # One chart for columns on different scales: plot z-scores.
    top = summary[summary["n_outliers"] > 0]["column"].head(8).tolist() or summary["column"].head(8).tolist()
    z = (df[top] - df[top].mean()) / df[top].std().replace(0, np.nan)
    long = z.melt(var_name="column", value_name="z_score").dropna()
    fig = px.box(long, x="column", y="z_score", points="outliers",
                 title=f"Outliers by column (standardized, {method.upper()} threshold {threshold})")

    flagged = summary[summary["n_outliers"] > 0]
    content = (
        f"Outlier detection using {method.upper()} (threshold={threshold}) on {len(candidates)} numeric columns. "
        f"{len(flagged)} columns contain outliers.\n\n{_table(summary)}"
    )
    logger.info(f"Outlier detection completed: {len(flagged)} columns flagged.")
    return content, {"summary": summary.to_dict(orient="list"), "plotly_figure": fig_to_dict(fig)}


# ---------------------------------------------------------------------------
# Correlations
# ---------------------------------------------------------------------------

@tool(response_format="content_and_artifact")
def analyze_correlations(
    data_raw: Annotated[Union[dict, list], InjectedState("data_raw")],
    method: Literal["pearson", "spearman", "kendall"] = "pearson",
    top_n: int = 10,
    multicollinearity_threshold: float = 0.8,
) -> Tuple[str, Dict]:
    """
    Tool: analyze_correlations
    Description:
        Computes the correlation matrix between numeric columns, lists the strongest
        correlated pairs, and flags multicollinearity (|r| above the threshold).
        Returns an annotated heatmap. Use spearman for skewed / ordinal data.
        For correlations against a single target (including categorical features),
        prefer generate_correlation_funnel.
    """
    logger.info(f"Executing tool: analyze_correlations (method={method})")
    df = pd.DataFrame(data_raw)
    num = df[_numeric_columns(df, min_unique=2)]
    if num.shape[1] < 2:
        return "Fewer than two numeric columns; a correlation matrix cannot be computed.", {}

    corr = num.corr(method=method)
    mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    pairs = (
        corr.where(mask).stack().rename("correlation").reset_index()
        .rename(columns={"level_0": "feature_a", "level_1": "feature_b"})
    )
    pairs["abs_corr"] = pairs["correlation"].abs()
    pairs = pairs.sort_values("abs_corr", ascending=False).drop(columns="abs_corr")
    pairs["correlation"] = pairs["correlation"].round(4)
    top_pairs = pairs.head(top_n)
    collinear = pairs[pairs["correlation"].abs() >= multicollinearity_threshold]

    fig = px.imshow(corr.round(2), text_auto=True, zmin=-1, zmax=1,
                    color_continuous_scale="RdBu_r", aspect="auto",
                    title=f"{method.title()} correlation matrix")

    content = (
        f"{method.title()} correlations across {num.shape[1]} numeric columns.\n\n"
        f"Top {len(top_pairs)} strongest pairs:\n{_table(top_pairs)}\n\n"
        + (f"Multicollinearity warning (|r| >= {multicollinearity_threshold}):\n{_table(collinear)}"
           if not collinear.empty else f"No pairs exceed |r| >= {multicollinearity_threshold}.")
    )
    artifact = {
        "correlation_matrix": corr.round(4).reset_index(names="feature").to_dict(orient="list"),
        "top_pairs": top_pairs.to_dict(orient="list"),
        "multicollinear_pairs": collinear.to_dict(orient="list"),
        "plotly_figure": fig_to_dict(fig),
    }
    logger.info("Correlation analysis completed.")
    return content, artifact


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------

def _shape_and_transform(skew: float, kurt: float, normal_p: float, min_val: float) -> Tuple[str, str]:
    if normal_p >= 0.05 or (abs(skew) < 0.5 and abs(kurt) < 1):
        return "approximately normal", "none needed"
    if skew >= 1:
        transform = "log" if min_val > 0 else "log1p" if min_val >= 0 else "Yeo-Johnson"
        return "strongly right-skewed", transform
    if skew >= 0.5:
        return "moderately right-skewed", "sqrt" if min_val >= 0 else "Yeo-Johnson"
    if skew <= -1:
        return "strongly left-skewed", "square or reflect-then-log"
    if skew <= -0.5:
        return "moderately left-skewed", "square"
    if kurt >= 3:
        return "heavy-tailed", "winsorize or robust scaling"
    return "non-normal (light-tailed / multimodal)", "consider binning or rank transform"


@tool(response_format="content_and_artifact")
def profile_distributions(
    data_raw: Annotated[Union[dict, list], InjectedState("data_raw")],
    columns: Optional[List[str]] = None,
    max_plots: int = 6,
) -> Tuple[str, Dict]:
    """
    Tool: profile_distributions
    Description:
        Profiles the shape of each continuous numeric column: skewness, excess kurtosis,
        a normality test, percentage of zeros, a plain-English shape label (e.g.
        "strongly right-skewed") and a suggested transformation (log, sqrt, Yeo-Johnson...).
        Returns histograms for the most non-normal columns.
    """
    from scipy import stats

    logger.info("Executing tool: profile_distributions")
    df = pd.DataFrame(data_raw)
    candidates = [c for c in _resolve_columns(df, columns) if c in _numeric_columns(df, min_unique=CATEGORICAL_MAX_UNIQUE + 1)]
    if not candidates:
        return "No continuous numeric columns to profile.", {"summary": {}, "plotly_figure": None}

    rows = []
    for col in candidates:
        s = df[col].dropna().astype(float)
        sample = s.sample(5000, random_state=42) if len(s) > 5000 else s
        if len(sample) < 8:
            normal_p = float("nan")
            test = "insufficient data"
        elif len(sample) <= 5000:
            normal_p = float(stats.shapiro(sample).pvalue)
            test = "Shapiro-Wilk"
        else:
            normal_p = float(stats.normaltest(sample).pvalue)
            test = "D'Agostino K^2"
        skew, kurt = float(s.skew()), float(s.kurt())
        shape, transform = _shape_and_transform(skew, kurt, normal_p if normal_p == normal_p else 0.0, float(s.min()))
        rows.append({
            "column": col, "skewness": round(skew, 3), "excess_kurtosis": round(kurt, 3),
            "normality_test": test, "normality_p": round(normal_p, 5),
            "pct_zeros": round(float((s == 0).mean() * 100), 2),
            "shape": shape, "suggested_transform": transform,
        })
    summary = pd.DataFrame(rows)
    summary = summary.reindex(summary["skewness"].abs().sort_values(ascending=False).index)

    plot_cols = summary["column"].head(max_plots).tolist()
    long = df[plot_cols].melt(var_name="column", value_name="value").dropna()
    fig = px.histogram(long, x="value", facet_col="column", facet_col_wrap=3,
                       title="Distributions (most skewed first)")
    fig.update_xaxes(matches=None, showticklabels=True)
    fig.update_yaxes(matches=None)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

    content = f"Distribution profile for {len(candidates)} continuous columns:\n\n{_table(summary)}"
    logger.info("Distribution profiling completed.")
    return content, {"summary": summary.to_dict(orient="list"), "plotly_figure": fig_to_dict(fig)}


# ---------------------------------------------------------------------------
# Statistical relationship test
# ---------------------------------------------------------------------------

def _effect_label(value: float, small: float, medium: float, large: float) -> str:
    value = abs(value)
    if value >= large:
        return "large"
    if value >= medium:
        return "medium"
    if value >= small:
        return "small"
    return "negligible"


@tool(response_format="content_and_artifact")
def analyze_relationship(
    data_raw: Annotated[Union[dict, list], InjectedState("data_raw")],
    column_a: str,
    column_b: str,
    alpha: float = 0.05,
) -> Tuple[str, Dict]:
    """
    Tool: analyze_relationship
    Description:
        Tests whether two columns are statistically related, automatically choosing the
        right test from the column types:
            - numeric vs numeric: Pearson and Spearman correlation
            - categorical vs numeric: Welch t-test / Mann-Whitney (2 groups) or ANOVA / Kruskal-Wallis (3+ groups)
            - categorical vs categorical: Chi-square test of independence with Cramer's V
        Reports the p-value, effect size, a plain-English conclusion, and a chart.
        Use for questions like "Does X affect Y?", "Is X related to Y?", "Do groups differ in Y?".

    Parameters:
        column_a: First column name.
        column_b: Second column name.
        alpha: Significance level (default 0.05).
    """
    from scipy import stats

    logger.info(f"Executing tool: analyze_relationship ({column_a} vs {column_b})")
    df = pd.DataFrame(data_raw)
    column_a, column_b = _resolve_columns(df, [column_a, column_b])
    if column_a == column_b:
        raise ValueError("column_a and column_b must be different columns.")
    sub = df[[column_a, column_b]].dropna()
    if len(sub) < 3:
        raise ValueError("Not enough non-missing rows to run a statistical test.")

    def is_numeric(col):
        return pd.api.types.is_numeric_dtype(sub[col]) and not pd.api.types.is_bool_dtype(sub[col]) \
            and sub[col].nunique() > CATEGORICAL_MAX_UNIQUE

    num_a, num_b = is_numeric(column_a), is_numeric(column_b)
    result: Dict = {"column_a": column_a, "column_b": column_b, "n": int(len(sub)), "alpha": alpha}

    if num_a and num_b:
        pr = stats.pearsonr(sub[column_a], sub[column_b])
        sr = stats.spearmanr(sub[column_a], sub[column_b])
        result.update(test="Pearson & Spearman correlation", statistic=float(pr.statistic),
                      p_value=float(pr.pvalue), spearman_rho=float(sr.statistic),
                      spearman_p=float(sr.pvalue), effect_size_name="Pearson r",
                      effect_size=float(pr.statistic),
                      effect_strength=_effect_label(pr.statistic, 0.1, 0.3, 0.5))
        direction = "positive" if pr.statistic > 0 else "negative"
        detail = (f"Pearson r = {pr.statistic:.3f} (p = {pr.pvalue:.3g}); "
                  f"Spearman rho = {sr.statistic:.3f} (p = {sr.pvalue:.3g}). "
                  f"The relationship is {direction} and {result['effect_strength']}.")
        sample = sub.sample(5000, random_state=42) if len(sub) > 5000 else sub
        fig = px.scatter(sample, x=column_a, y=column_b, opacity=0.6,
                         title=f"{column_b} vs {column_a} (r = {pr.statistic:.2f})")

    elif num_a != num_b:
        cat, num = (column_b, column_a) if num_a else (column_a, column_b)
        top_groups = sub[cat].astype(str).value_counts().head(30).index
        sub = sub[sub[cat].astype(str).isin(top_groups)]
        groups = [g[num].values for _, g in sub.groupby(sub[cat].astype(str)) if len(g) >= 2]
        if len(groups) < 2:
            raise ValueError(f"'{cat}' needs at least two groups with 2+ rows each.")
        if len(groups) == 2:
            t = stats.ttest_ind(*groups, equal_var=False)
            mw = stats.mannwhitneyu(*groups)
            pooled_sd = np.sqrt((np.var(groups[0], ddof=1) + np.var(groups[1], ddof=1)) / 2)
            d = (np.mean(groups[0]) - np.mean(groups[1])) / pooled_sd if pooled_sd else 0.0
            result.update(test="Welch t-test (with Mann-Whitney U)", statistic=float(t.statistic),
                          p_value=float(t.pvalue), nonparametric_p=float(mw.pvalue),
                          effect_size_name="Cohen's d", effect_size=float(d),
                          effect_strength=_effect_label(d, 0.2, 0.5, 0.8))
            detail = (f"Welch t = {t.statistic:.3f} (p = {t.pvalue:.3g}); Mann-Whitney p = {mw.pvalue:.3g}; "
                      f"Cohen's d = {d:.3f} ({result['effect_strength']} effect).")
        else:
            f = stats.f_oneway(*groups)
            kw = stats.kruskal(*groups)
            all_vals = np.concatenate(groups)
            ss_between = sum(len(g) * (np.mean(g) - all_vals.mean()) ** 2 for g in groups)
            ss_total = ((all_vals - all_vals.mean()) ** 2).sum()
            eta_sq = ss_between / ss_total if ss_total else 0.0
            result.update(test="One-way ANOVA (with Kruskal-Wallis)", statistic=float(f.statistic),
                          p_value=float(f.pvalue), nonparametric_p=float(kw.pvalue),
                          effect_size_name="eta squared", effect_size=float(eta_sq),
                          effect_strength=_effect_label(eta_sq, 0.01, 0.06, 0.14))
            detail = (f"ANOVA F = {f.statistic:.3f} (p = {f.pvalue:.3g}); Kruskal-Wallis p = {kw.pvalue:.3g}; "
                      f"eta^2 = {eta_sq:.3f} ({result['effect_strength']} effect).")
        means = sub.groupby(sub[cat].astype(str))[num].agg(["count", "mean", "median"]).round(3)
        result["group_stats"] = means.reset_index(names=cat).to_dict(orient="list")
        detail += f"\nGroup statistics for {num} by {cat}:\n{means.to_string()}"
        fig = px.box(sub.assign(**{cat: sub[cat].astype(str)}), x=cat, y=num, points="outliers",
                     title=f"{num} by {cat}")

    else:
        table = pd.crosstab(sub[column_a].astype(str), sub[column_b].astype(str))
        chi2, p, dof, _ = stats.chi2_contingency(table)
        k = min(table.shape) - 1
        cramers_v = float(np.sqrt(chi2 / (table.values.sum() * k))) if k > 0 else 0.0
        result.update(test="Chi-square test of independence", statistic=float(chi2), p_value=float(p),
                      dof=int(dof), effect_size_name="Cramer's V", effect_size=cramers_v,
                      effect_strength=_effect_label(cramers_v, 0.1, 0.3, 0.5))
        result["contingency_table"] = table.reset_index().to_dict(orient="list")
        detail = (f"Chi-square = {chi2:.3f}, dof = {dof} (p = {p:.3g}); Cramer's V = {cramers_v:.3f} "
                  f"({result['effect_strength']} association).\nContingency table:\n{table.to_string()}")
        pct = table.div(table.sum(axis=1), axis=0).mul(100).round(1)
        fig = px.imshow(pct, text_auto=True, aspect="auto", color_continuous_scale="Blues",
                        labels={"color": "row %"}, title=f"{column_b} distribution within {column_a} (row %)")

    significant = result["p_value"] < alpha
    result["significant"] = bool(significant)
    conclusion = (
        f"{'Statistically significant' if significant else 'No statistically significant'} relationship "
        f"between '{column_a}' and '{column_b}' at alpha = {alpha} (p = {result['p_value']:.3g}, n = {result['n']})."
    )
    result["conclusion"] = conclusion
    result["plotly_figure"] = fig_to_dict(fig)

    content = f"Test: {result['test']}\n{conclusion}\n{detail}"
    logger.info(f"Relationship test completed: {result['test']} p={result['p_value']:.3g}")
    return content, result


# ---------------------------------------------------------------------------
# Smart Schema Tagger
# ---------------------------------------------------------------------------

@tool(response_format="content_and_artifact")
def smart_schema_tagger(
    data_raw: Annotated[Union[dict, list], InjectedState("data_raw")],
) -> Tuple[str, Dict]:
    """
    Tool: smart_schema_tagger
    Description:
        Automatically classifies every column into a semantic role:
            - 'id'         : identifier-like (near-unique, not useful as feature)
            - 'target'     : likely prediction target (binary/low-cardinality, or named churn/label/target/y)
            - 'datetime'   : date/time column
            - 'constant'   : single-value column (useless)
            - 'binary'     : boolean-like 0/1 or Yes/No
            - 'categorical': low-cardinality string/category
            - 'numeric'    : continuous numeric feature
            - 'text'       : high-cardinality string (free text)
        Also flags potential issues: high missing, skewed, mixed types.
        Use to quickly understand dataset structure before modeling or feature engineering.
    """
    logger.info("Executing tool: smart_schema_tagger")
    df = pd.DataFrame(data_raw)
    n = len(df)

    target_hints = {"churn", "target", "label", "y", "outcome", "class", "survived",
                    "default", "fraud", "clicked", "converted", "purchased", "response"}

    rows = []
    for col in df.columns:
        s = df[col]
        dtype = str(s.dtype)
        n_unique = int(s.nunique(dropna=True))
        n_missing = int(s.isna().sum())
        miss_pct = round(n_missing / n * 100, 1)
        uniq_pct = n_unique / n * 100 if n else 0

        # Determine role
        role = "numeric" if pd.api.types.is_numeric_dtype(s) else "categorical"

        col_lower = col.lower().replace(" ", "_")

        if pd.api.types.is_datetime64_any_dtype(s):
            role = "datetime"
        elif n_unique == 1:
            role = "constant"
        elif pd.api.types.is_numeric_dtype(s) and n_unique == 2 and set(s.dropna().unique()).issubset({0, 1}):
            role = "binary"
        elif not pd.api.types.is_numeric_dtype(s) and n_unique == 2:
            role = "binary"
        elif pd.api.types.is_numeric_dtype(s) and uniq_pct > 90 and n_unique > 50:
            # High-uniqueness numeric — could be ID
            if any(h in col_lower for h in ["id", "key", "code", "number", "num", "no"]):
                role = "id"
        elif not pd.api.types.is_numeric_dtype(s) and uniq_pct > 85 and n_unique > 50:
            role = "text"
        elif not pd.api.types.is_numeric_dtype(s) and any(h in col_lower for h in ["id", "_id", "key", "uuid"]) and uniq_pct > 50:
            role = "id"
        elif not pd.api.types.is_numeric_dtype(s) and any(h in col_lower for h in ["date", "time", "dt", "timestamp"]):
            role = "datetime"

        # Override to target if name matches hint
        if col_lower in target_hints or any(col_lower.endswith(f"_{h}") or col_lower.startswith(f"{h}_") for h in target_hints):
            if n_unique <= 20:
                role = "target"

        # Flags
        flags = []
        if miss_pct > 10:
            flags.append(f"high_missing({miss_pct}%)")
        if pd.api.types.is_numeric_dtype(s) and n_missing < n:
            try:
                skew = float(s.skew())
                if abs(skew) > 2:
                    flags.append(f"skewed({skew:.1f})")
            except Exception:
                pass
        if not pd.api.types.is_numeric_dtype(s) and n_unique > 1:
            # Check for mixed types (numbers stored as strings)
            sample = s.dropna().head(200)
            try:
                pd.to_numeric(sample)
                flags.append("numeric_as_string?")
            except Exception:
                pass

        rows.append({
            "column": col, "dtype": dtype, "role": role,
            "n_unique": n_unique, "missing_pct": miss_pct, "flags": ", ".join(flags) if flags else "—",
        })

    schema_df = pd.DataFrame(rows)
    role_counts = schema_df["role"].value_counts().to_dict()

    content_lines = [
        f"Smart Schema — {n:,} rows × {len(df.columns)} columns",
        "",
        _table(schema_df),
        "",
        "Role summary: " + ", ".join(f"{v} {k}" for k, v in role_counts.items()),
    ]

    targets = schema_df[schema_df["role"] == "target"]["column"].tolist()
    ids = schema_df[schema_df["role"] == "id"]["column"].tolist()
    if targets:
        content_lines.append(f"Likely target variable(s): {targets}")
    if ids:
        content_lines.append(f"ID columns to exclude from modeling: {ids}")

    high_miss = schema_df[schema_df["missing_pct"] > 20]
    if not high_miss.empty:
        content_lines.append(f"Columns with >20% missing: {high_miss['column'].tolist()}")

    logger.info(f"Smart schema tagging done: {role_counts}")
    return "\n".join(content_lines), {
        "schema": schema_df.to_dict(orient="list"),
        "role_counts": role_counts,
        "likely_targets": targets,
        "id_columns": ids,
    }
