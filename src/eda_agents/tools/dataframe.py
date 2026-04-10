import io
import pandas as pd
from typing import Union, List, Dict
from eda_agents.utils.logger import logger

def get_dataframe_summary(
    dataframes: Union[pd.DataFrame, List[pd.DataFrame], Dict[str, pd.DataFrame]],
    n_sample: int = 30,
    skip_stats: bool = False,
) -> List[str]:
    """
    Generate a summary for one or more DataFrames.
    """
    logger.debug(f"Generating summary for {type(dataframes)} input.")
    summaries = []

    # --- Dictionary Case ---
    if isinstance(dataframes, dict):
        for dataset_name, df in dataframes.items():
            summaries.append(
                _summarize_dataframe(df, dataset_name, n_sample, skip_stats)
            )

    # --- Single DataFrame Case ---
    elif isinstance(dataframes, pd.DataFrame):
        summaries.append(
            _summarize_dataframe(dataframes, "Single_Dataset", n_sample, skip_stats)
        )

    # --- List of DataFrames Case ---
    elif isinstance(dataframes, list):
        for idx, df in enumerate(dataframes):
            dataset_name = f"Dataset_{idx}"
            summaries.append(
                _summarize_dataframe(df, dataset_name, n_sample, skip_stats)
            )

    else:
        logger.error(f"Invalid input type to get_dataframe_summary: {type(dataframes)}")
        raise TypeError(
            "Input must be a single DataFrame, a list of DataFrames, or a dictionary of DataFrames."
        )

    return summaries


def _summarize_dataframe(
    df: pd.DataFrame, dataset_name: str, n_sample=30, skip_stats=False
) -> str:
    """Generate a summary string for a single DataFrame."""
    logger.debug(f"Summarizing dataset: {dataset_name} (rows: {df.shape[0]})")
    # 1. Convert dictionary-type cells to strings
    df = df.apply(lambda col: col.map(lambda x: str(x) if isinstance(x, dict) else x))

    # 2. Capture df.info() output
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_text = buffer.getvalue()

    total_rows = len(df)
    missing_stats = (df.isna().sum() / total_rows * 100)
    
    # Calculate dataset-level comprehensive stats
    duplicated_rows = df.duplicated().sum()
    duplicate_pct = (duplicated_rows / total_rows * 100) if total_rows > 0 else 0
    total_memory = df.memory_usage(deep=True).sum() / (1024 * 1024) # MB
    
    # 3. Create a unified markdown table for columns
    table_rows = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        is_object = pd.api.types.is_object_dtype(df[col])
        
        missing_pct = missing_stats[col]
        nunique = df[col].nunique()
        nunique_pct = (nunique / total_rows * 100) if total_rows > 0 else 0
        
        # Determine comprehensive highlights/notes
        flags = []
        if missing_pct > 0:
            if missing_pct > 50:
                flags.append("High Missing")
            
        if nunique == 1:
            flags.append("Constant")
        elif nunique == 2:
            flags.append("Binary")
        elif nunique == total_rows and total_rows > 0:
            flags.append("All Unique")
        elif nunique < 10:
            flags.append("Low Cardinality")
            
        # Comprehensive numeric stats
        zeros_pct = 0.0
        skewness_str = "-"
        if is_numeric and missing_pct < 100:
            zeros_pct = (df[col] == 0).sum() / total_rows * 100
            try:
                skewness = df[col].skew()
                skewness_str = f"{skewness:.2f}"
                if abs(skewness) > 1.5:
                    flags.append("Highly Skewed")
            except:
                pass
                
        # Comprehensive categorical stats
        top_val_str = "-"
        if (is_object or nunique < 20) and missing_pct < 100 and nunique < total_rows:
            try:
                top_val = df[col].mode().iloc[0] if not df[col].mode().empty else "N/A"
                if len(str(top_val)) > 15: 
                    top_val_str = f"{str(top_val)[:12]}..."
                else:
                    top_val_str = f"{top_val}"
            except:
                pass
            
        flags_str = ", ".join(flags) if flags else "Normal"
        zeros_str = f"{zeros_pct:.1f}%" if is_numeric else "-"
        
        table_rows.append(f"| **{col}** | `{dtype}` | {missing_pct:.2f}% | {nunique} ({nunique_pct:.1f}%) | {zeros_str} | {skewness_str} | {top_val_str} | {flags_str} |")

    table_header = "| Column Name | Data Type | Missing % | Unique | Zeros % | Skewness | Top Mode | Flags |\n|---|---|---|---|---|---|---|---|"
    table_content = "\n".join(table_rows)

    # 4. Generate the narrative summary text
    if not skip_stats:
        summary_text = f"""### Dataset Overview: `{dataset_name}`
**Shape**: {df.shape[0]} rows | {df.shape[1]} columns
**Duplicates**: {duplicated_rows} rows ({duplicate_pct:.1f}%)
**Memory Usage**: {total_memory:.2f} MB

#### Column Profile
{table_header}
{table_content}

#### Data Preview (first {n_sample} rows)
```text
{df.head(n_sample).to_string()}
```

#### Data Description (Numerical)
```text
{df.describe().to_string() if not df.columns.empty else "No columns to describe"}
```

#### System Data Info
```text
{info_text}
```
"""
    else:
        summary_text = f"""### Dataset Overview: `{dataset_name}`
**Shape**: {df.shape[0]} rows | {df.shape[1]} columns
"""
    
    # Prevent massive token consumption on very large datasets
    if len(summary_text) > 20000:
        summary_text = summary_text[:20000] + "\n\n...[TRUNCATED due to length constraints]..."

    return summary_text
