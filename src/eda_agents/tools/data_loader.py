"""
Data Loader tools: file-system browsing and multi-format data loading.
Inspired by business-science/ai-data-science-team's data_loader tools.
"""
import os
import glob
import json
from pathlib import Path
from typing import Optional, List
import pandas as pd
from langchain.tools import tool


SUPPORTED_EXTENSIONS = {
    ".csv": "CSV",
    ".xlsx": "Excel",
    ".xls": "Excel",
    ".parquet": "Parquet",
    ".json": "JSON",
    ".tsv": "TSV",
    ".txt": "Text (tab/comma delimited)",
}


@tool
def list_directory_contents(directory: str = ".") -> str:
    """List files and subdirectories in the given directory path."""
    try:
        path = Path(directory).expanduser().resolve()
        if not path.exists():
            return f"Directory not found: {directory}"
        items = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name))
        lines = [f"Contents of: {path}", ""]
        for item in items:
            prefix = "📁 " if item.is_dir() else "📄 "
            size = f"  ({item.stat().st_size:,} bytes)" if item.is_file() else ""
            lines.append(f"{prefix}{item.name}{size}")
        return "\n".join(lines)
    except Exception as e:
        return f"Error listing directory: {e}"


@tool
def list_data_files(directory: str = ".", recursive: bool = False) -> str:
    """List only data files (CSV, Excel, Parquet, JSON, TSV) in a directory."""
    try:
        path = Path(directory).expanduser().resolve()
        if not path.exists():
            return f"Directory not found: {directory}"
        pattern = "**/*" if recursive else "*"
        found = []
        for ext in SUPPORTED_EXTENSIONS:
            matches = list(path.glob(f"{pattern}{ext}")) + list(path.glob(f"{pattern}{ext.upper()}"))
            found.extend(matches)
        found = sorted(set(found))
        if not found:
            return f"No data files found in: {path}" + (" (including subdirectories)" if recursive else "")
        lines = [f"Data files in: {path}", ""]
        for f in found:
            rel = f.relative_to(path)
            size_kb = f.stat().st_size / 1024
            fmt = SUPPORTED_EXTENSIONS.get(f.suffix.lower(), "Unknown")
            lines.append(f"  • {rel}  [{fmt}, {size_kb:.1f} KB]")
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


@tool
def get_file_info(file_path: str) -> str:
    """Get metadata about a file: size, format, and a preview of its content."""
    try:
        path = Path(file_path).expanduser().resolve()
        if not path.exists():
            return f"File not found: {file_path}"
        ext = path.suffix.lower()
        size_kb = path.stat().st_size / 1024
        fmt = SUPPORTED_EXTENSIONS.get(ext, "Unknown")
        lines = [
            f"File: {path.name}",
            f"Format: {fmt}",
            f"Size: {size_kb:.1f} KB",
            f"Path: {path}",
        ]
        if ext in (".csv", ".tsv", ".txt"):
            df = pd.read_csv(path, nrows=5, sep="\t" if ext == ".tsv" else None, engine="python")
            lines += [f"Shape (preview): {df.shape}", "", df.to_string(index=False)]
        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(path, nrows=5)
            lines += [f"Shape (preview): {df.shape}", "", df.to_string(index=False)]
        elif ext == ".parquet":
            df = pd.read_parquet(path).head(5)
            lines += [f"Shape (preview): {df.shape}", "", df.to_string(index=False)]
        elif ext == ".json":
            with open(path) as f:
                raw = json.load(f)
            if isinstance(raw, list):
                df = pd.DataFrame(raw[:5])
                lines += [f"Records: {len(raw)}", "", df.to_string(index=False)]
            else:
                lines.append(str(raw)[:500])
        return "\n".join(lines)
    except Exception as e:
        return f"Error reading file: {e}"


@tool
def load_file(file_path: str, sheet_name: Optional[str] = None,
              sep: Optional[str] = None, encoding: str = "utf-8") -> str:
    """
    Load a data file (CSV, Excel, Parquet, JSON, TSV) and return a summary.
    The loaded DataFrame is automatically set as the active dataset in the session.

    Args:
        file_path: Path to the file.
        sheet_name: For Excel files, the sheet to load (defaults to first sheet).
        sep: Separator for CSV/TSV. Auto-detected if not provided.
        encoding: File encoding (default utf-8).

    Returns:
        A text summary of the loaded dataset.
    """
    try:
        path = Path(file_path).expanduser().resolve()
        ext = path.suffix.lower()

        if ext in (".csv", ".txt"):
            df = pd.read_csv(path, sep=sep, encoding=encoding, engine="python")
        elif ext == ".tsv":
            df = pd.read_csv(path, sep="\t", encoding=encoding)
        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(path, sheet_name=sheet_name or 0)
        elif ext == ".parquet":
            df = pd.read_parquet(path)
        elif ext == ".json":
            df = pd.read_json(path)
        else:
            return f"Unsupported file format: {ext}"

        summary_lines = [
            f"✅ Loaded: {path.name}",
            f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns",
            f"Memory: {df.memory_usage(deep=True).sum() / 1024:.1f} KB",
            "",
            "Columns:",
        ]
        for col in df.columns:
            dtype = df[col].dtype
            n_missing = df[col].isna().sum()
            miss_pct = n_missing / len(df) * 100
            summary_lines.append(f"  • {col} ({dtype}) — {miss_pct:.1f}% missing")

        return "\n".join(summary_lines)
    except Exception as e:
        return f"Error loading file: {e}"


@tool
def search_files_by_pattern(directory: str = ".", pattern: str = "*.csv") -> str:
    """
    Search for files matching a glob pattern (e.g. '*.csv', '**/*.xlsx', 'sales_*').

    Args:
        directory: Directory to search in.
        pattern: Glob pattern to match.

    Returns:
        List of matching file paths with sizes.
    """
    try:
        path = Path(directory).expanduser().resolve()
        matches = sorted(path.glob(pattern))
        if not matches:
            return f"No files matched '{pattern}' in {path}"
        lines = [f"Files matching '{pattern}' in {path}:", ""]
        for m in matches:
            size_kb = m.stat().st_size / 1024
            lines.append(f"  • {m.relative_to(path)}  [{size_kb:.1f} KB]")
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


DATA_LOADER_TOOLS = [
    list_directory_contents,
    list_data_files,
    get_file_info,
    load_file,
    search_files_by_pattern,
]
