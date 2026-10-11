from typing import Annotated, Dict, Tuple, Union, List
import os
import tempfile
import warnings
import base64
from io import BytesIO
import json

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
import pandas as pd
import matplotlib.pyplot as plt

from eda_agents.tools.dataframe import get_dataframe_summary
from eda_agents.utils.logger import logger

@tool(response_format="content")
def explain_data(
    data_raw: Annotated[Union[dict, list], InjectedState("data_raw")],
    n_sample: int = 30,
    skip_stats: bool = False,
):
    """
    Tool: explain_data
    Description:
        Provides an extensive, narrative summary of a DataFrame including its shape, column types,
        missing value percentages, unique counts, sample rows, and (if not skipped) descriptive stats/info.

    Parameters:
        data_raw (dict): Raw data.
        n_sample (int, default=30): Number of rows to display.
        skip_stats (bool, default=False): If True, omit descriptive stats/info.

    Returns:
        str: Detailed DataFrame summary.
    """
    logger.info("Executing tool: explain_data")
    result = get_dataframe_summary(
        pd.DataFrame(data_raw), n_sample=n_sample, skip_stats=skip_stats
    )
    summary = result[0] if result else "No data summary available."
    logger.info(f"Summary generated ({len(summary)} chars).")
    return summary


@tool(response_format="content_and_artifact")
def describe_dataset(
    data_raw: Annotated[Union[dict, list], InjectedState("data_raw")],
) -> Tuple[str, Dict]:
    """
    Tool: describe_dataset
    Description:
        Compute and return summary statistics for the dataset using pandas' describe() method.
        The tool provides both a textual summary and a structured artifact for further processing.
    """
    logger.info("Executing tool: describe_dataset")
    df = pd.DataFrame(data_raw)
    description_df = df.describe(include="all")
    content = (
        "Summary statistics computed using pandas describe():\n"
        f"{description_df.T.to_string(max_rows=60)}"
    )
    flattened = description_df.reset_index().rename(columns={"index": "stat"})
    artifact = {"describe_df": flattened.to_dict(orient="list")}
    logger.info("Dataset description computed successfully.")
    return content, artifact


@tool(response_format="content_and_artifact")
def visualize_missing(
    data_raw: Annotated[Union[dict, list], InjectedState("data_raw")], n_sample: int = None
) -> Tuple[str, Dict]:
    """
    Tool: visualize_missing
    Description:
        Missing value analysis using the missingno library. Generates a matrix plot, bar plot, and heatmap plot.
    """
    logger.info(f"Executing tool: visualize_missing (sample size: {n_sample})")
    try:
        import missingno as msno
    except ImportError:
        logger.error("missingno library not found.")
        raise ImportError(
            "Please install 'missingno': pip install missingno"
        )

    df = pd.DataFrame(data_raw)
    if n_sample is not None and len(df) > n_sample:
        df = df.sample(n=n_sample, random_state=42)

    encoded_plots = {}

    def create_and_encode_plot(plot_func, plot_name: str):
        logger.debug(f"Generating missingno {plot_name} plot.")
        plt.figure(figsize=(8, 6))
        plot_func(df)
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close("all")  # missingno opens its own figure in addition to ours
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    encoded_plots["matrix_plot"] = create_and_encode_plot(msno.matrix, "matrix")
    encoded_plots["bar_plot"] = create_and_encode_plot(msno.bar, "bar")
    encoded_plots["heatmap_plot"] = create_and_encode_plot(msno.heatmap, "heatmap")

    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    missing_text = (
        "\n".join(f"- {col}: {n} missing ({n / len(df):.1%})" for col, n in missing.items())
        if not missing.empty else "No missing values in any column."
    )
    content = (
        "Missing data visualizations (matrix, bar, heatmap) generated.\n"
        f"Missing values per column:\n{missing_text}"
    )
    logger.info("Missing data visualizations completed.")
    return content, encoded_plots


def _binarize(df: pd.DataFrame, n_bins: int, thresh_infreq: float, name_infreq: str) -> pd.DataFrame:
    """pandas-only equivalent of pytimetk's binarize(one_hot=True)."""
    parts = []
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s) and s.nunique() > n_bins:
            binned = pd.qcut(s, q=n_bins, duplicates="drop")
            labels = binned.cat.rename_categories(lambda iv: f"{iv.left:g}_{iv.right:g}").astype(object).where(binned.notna(), "nan")
        else:
            labels = s.astype(str)
            # Pure identifier columns carry no signal and explode the one-hot width.
            if labels.nunique() == len(labels) and len(labels) > 20:
                continue
            freq = labels.value_counts(normalize=True)
            labels = labels.where(~labels.isin(freq[freq < thresh_infreq].index), name_infreq)
        parts.append(pd.get_dummies(labels, prefix=col, prefix_sep="__").astype(int))
    return pd.concat(parts, axis=1)


def _correlation_funnel_fallback(df, target, target_bin_index, corr_method, n_bins, thresh_infreq, name_infreq):
    import plotly.express as px
    from eda_agents.utils.plotting import fig_to_dict

    df_binarized = _binarize(df, n_bins, thresh_infreq, name_infreq)
    matching = [c for c in df_binarized.columns if c.startswith(f"{target}__")]
    if not matching:
        raise ValueError(f"Target '{target}' could not be binarized.")
    if isinstance(target_bin_index, str) and f"{target}__{target_bin_index}" in matching:
        full_target = f"{target}__{target_bin_index}"
    else:
        try:
            full_target = matching[int(target_bin_index)]
        except (ValueError, IndexError):
            full_target = matching[-1]
    logger.info(f"Using target column: {full_target}")

    corr = df_binarized.corrwith(df_binarized[full_target], method=corr_method)
    df_correlated = pd.DataFrame({
        "feature": [c.split("__", 1)[0] for c in corr.index],
        "bin": [c.split("__", 1)[1] for c in corr.index],
        "correlation": corr.values.round(4),
    }).dropna()
    df_correlated = df_correlated[df_correlated["feature"] != target]
    df_correlated = df_correlated.reindex(df_correlated["correlation"].abs().sort_values(ascending=False).index)

    # Funnel: features ordered by their strongest bin, strongest at the top.
    feature_order = df_correlated.groupby("feature")["correlation"].apply(lambda s: s.abs().max()).sort_values().index.tolist()
    fig = px.scatter(
        df_correlated, x="correlation", y="feature", text="bin", color="correlation",
        color_continuous_scale="RdBu_r", range_color=[-1, 1],
        category_orders={"feature": feature_order[::-1]},
        title=f"Correlation Funnel: {full_target}",
    )
    fig.update_traces(textposition="top center", marker=dict(size=10))
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.update_layout(height=max(400, 32 * len(feature_order)), xaxis_range=[-1, 1])

    top = df_correlated.head(10)
    content = (
        f"Correlation funnel for target '{full_target}'. Strongest associations "
        f"(positive = more likely {full_target}):\n{top.to_string(index=False)}"
    )
    artifact = {
        "correlation_data": df_correlated.to_dict(orient="list"),
        "plot_image": None,
        "plotly_figure": fig_to_dict(fig),
    }
    return content, artifact


@tool(response_format="content_and_artifact")
def generate_correlation_funnel(
    data_raw: Annotated[Union[dict, list], InjectedState("data_raw")],
    target: str,
    target_bin_index: Union[int, str] = -1,
    corr_method: str = "pearson",
    n_bins: int = 4,
    thresh_infreq: float = 0.01,
    name_infreq: str = "-OTHER",
) -> Tuple[str, Dict]:
    """
    Tool: generate_correlation_funnel
    Description:
        Correlation analysis using the correlation funnel method: every feature is binarized
        (numeric columns into quantile bins, categories one-hot encoded) and each bin is
        correlated against one bin of the target. Shows which feature values are most
        associated with the target outcome (e.g. Churn=Yes). Uses pytimetk when installed,
        otherwise a built-in pandas implementation.

    Parameters:
        target (str): Target column name.
        target_bin_index (int | str): Which target bin to compare against, e.g. "Yes" or -1 (last bin).
        corr_method (str): "pearson", "spearman" or "kendall".
        n_bins (int): Number of quantile bins for numeric features.
        thresh_infreq (float): Categories rarer than this fraction are lumped into name_infreq.
    """
    logger.info(f"Executing tool: generate_correlation_funnel (target: {target})")
    try:
        import pytimetk as tk
    except ImportError:
        tk = None
        logger.info("pytimetk not installed; using built-in correlation funnel.")

    import plotly.io as pio

    df = pd.DataFrame(data_raw)
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found. Available columns: {list(df.columns)}")

    if tk is None:
        return _correlation_funnel_fallback(
            df, target, target_bin_index, corr_method, n_bins, thresh_infreq, name_infreq
        )

    # Binarize
    logger.debug("Binarizing data for correlation funnel.")
    df_binarized = df.binarize(
        n_bins=n_bins,
        thresh_infreq=thresh_infreq,
        name_infreq=name_infreq,
        one_hot=True,
    )

    # Determine full target column name
    matching_columns = [col for col in df_binarized.columns if col.startswith(f"{target}__")]
    
    if not matching_columns:
        full_target = target
    else:
        if isinstance(target_bin_index, str):
            candidate = f"{target}__{target_bin_index}"
            full_target = candidate if candidate in matching_columns else matching_columns[-1]
        else:
            try:
                full_target = matching_columns[target_bin_index]
            except IndexError:
                full_target = matching_columns[-1]

    logger.info(f"Using target column: {full_target}")

    # Correlate
    df_correlated = df_binarized.correlate(target=full_target, method=corr_method)

    # Plot (Static)
    encoded = None
    try:
        logger.debug("Generating static funnel plot.")
        fig = df_correlated.plot_correlation_funnel(engine="plotnine", height=600)
        buf = BytesIO()
        fig.save(buf, format="png")
        plt.close()
        buf.seek(0)
        encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        logger.warning(f"Static plot failed: {e}")
        encoded = {"error": str(e)}

    # Plot (Interactive)
    fig_dict = None
    try:
        logger.debug("Generating interactive plotly funnel.")
        fig = df_correlated.plot_correlation_funnel(engine="plotly", base_size=14)
        fig_json = pio.to_json(fig)
        fig_dict = json.loads(fig_json)
    except Exception as e:
        logger.warning(f"Interactive plot failed: {e}")
        fig_dict = {"error": str(e)}

    top = df_correlated.head(10)
    content = (
        f"Correlation funnel for target '{full_target}'. Strongest associations:\n"
        f"{top.to_string(index=False)}"
    )
    artifact = {
        "correlation_data": df_correlated.to_dict(orient="list"),
        "plot_image": encoded,
        "plotly_figure": fig_dict,
    }
    logger.info("Correlation funnel analysis completed.")
    return content, artifact


@tool(response_format="content_and_artifact")
def generate_sweetviz_report(
    data_raw: Annotated[Union[dict, list], InjectedState("data_raw")],
    target: str = None,
    report_name: str = "sweetviz_report.html",
    report_directory: str = None,
    open_browser: bool = False,
    include_html: bool = False,
) -> Tuple[str, Dict]:
    """
    Tool: generate_sweetviz_report
    Description:
        Make an Exploratory Data Analysis (EDA) report using the Sweetviz library.
    """
    logger.info(f"Executing tool: generate_sweetviz_report (target: {target})")
    try:
        import sweetviz as sv
    except ImportError as e:
        logger.error(f"sweetviz could not be imported: {e}")
        # On Python 3.12+ sweetviz also needs setuptools (it imports pkg_resources).
        raise ImportError(
            f"sweetviz could not be imported ({e}). Try: pip install sweetviz 'setuptools<81'"
        )
    
    # Handle numpy warning
    import numpy as np
    if not hasattr(np, "VisibleDeprecationWarning"):
        np.VisibleDeprecationWarning = DeprecationWarning

    df = pd.DataFrame(data_raw)

    if not report_directory:
        base_reports_dir = os.path.abspath(os.path.join(os.getcwd(), "pipeline_reports"))
        os.makedirs(base_reports_dir, exist_ok=True)
        report_directory = tempfile.mkdtemp(prefix="sweetviz_", dir=base_reports_dir)
    else:
        if not os.path.exists(report_directory):
            os.makedirs(report_directory)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
        logger.debug("Running sweetviz analysis...")
        report = sv.analyze(df, target_feat=target)

    full_report_path = os.path.join(report_directory, report_name)
    logger.info(f"Saving report to: {full_report_path}")
    report.show_html(filepath=full_report_path, open_browser=open_browser)

    html_content = None
    if include_html:
        try:
            with open(full_report_path, "r", encoding="utf-8") as f:
                html_content = f.read()
        except Exception as e:
            logger.error(f"Failed to read generated HTML report: {e}")
            html_content = None

    content = f"Sweetviz report saved to {full_report_path}"
    artifact = {
        "report_file": full_report_path,
        "report_html": html_content
    }
    logger.info("Sweetviz report generation completed.")
    return content, artifact


@tool(response_format="content_and_artifact")
def generate_dtale_report(
    data_raw: Annotated[Union[dict, list], InjectedState("data_raw")],
    host: str = "localhost",
    port: int = 40000,
    open_browser: bool = False,
) -> Tuple[str, Dict]:
    """
    Tool: generate_dtale_report
    Description:
        Creates an interactive data exploration report using the dtale library.
    """
    logger.info(f"Executing tool: generate_dtale_report (port: {port})")
    try:
        import dtale
    except ImportError:
        logger.error("dtale library not found.")
        raise ImportError("Please install 'dtale': pip install dtale")

    df = pd.DataFrame(data_raw)
    d = dtale.show(df, host=host, port=port, open_browser=open_browser)

    content = f"Dtale report running at {d.main_url()}"
    artifact = {"dtale_url": d.main_url()}
    logger.info(f"Dtale report started at {d.main_url()}")
    return content, artifact
