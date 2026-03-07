from typing import Any, Dict

import pandas as pd


def _column_types(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(df[c].dtype) for c in df.columns],
            "missing_values": [int(df[c].isna().sum()) for c in df.columns],
            "missing_pct": [round(float(df[c].isna().mean() * 100), 2) for c in df.columns],
            "unique_values": [int(df[c].nunique(dropna=True)) for c in df.columns],
        }
    )


def _basic_stats(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.empty:
        return pd.DataFrame(columns=["metric"])

    stats = numeric_df.describe().T.reset_index().rename(columns={"index": "column"})
    return stats


def _correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.shape[1] < 2:
        return pd.DataFrame()

    corr = numeric_df.corr(numeric_only=True)
    return corr


def build_data_profile(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate dataset profile artifacts used by the dashboard."""
    profile = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "numeric_columns": int(df.select_dtypes(include=["number"]).shape[1]),
        "datetime_columns": int(df.select_dtypes(include=["datetime", "datetimetz"]).shape[1]),
        "column_profile": _column_types(df),
        "basic_stats": _basic_stats(df),
        "correlation_matrix": _correlation_matrix(df),
    }
    return profile
