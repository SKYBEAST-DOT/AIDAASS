"""
Dataset profiling engine for AIDAASS.

Creates reliable dataset metadata for:
- Dashboard KPI cards
- Schema inspection
- Statistical summaries
- Correlation analysis
- Gemini dataset understanding
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

MAX_PROFILE_COLUMNS = 100
MAX_CORRELATION_COLUMNS = 50
MAX_SAMPLE_VALUES = 5


# ============================================================
# COLUMN TYPE HELPERS
# ============================================================


def _is_datetime_like(series: pd.Series) -> bool:
    """
    Determine whether a column is likely to contain dates.

    Existing datetime columns are accepted immediately.
    Object/string columns are tested conservatively so normal
    text columns are not incorrectly classified as dates.
    """

    if pd.api.types.is_datetime64_any_dtype(series):
        return True

    if not (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
    ):
        return False

    non_null = series.dropna()

    if non_null.empty:
        return False

    # Avoid trying to parse huge columns unnecessarily.
    sample = non_null.head(100)

    try:
        parsed = pd.to_datetime(
            sample,
            errors="coerce",
        )

        success_rate = float(
            parsed.notna().mean()
        )

        return success_rate >= 0.80

    except Exception:
        return False


def _column_type(series: pd.Series) -> str:
    """
    Return a human-readable semantic column type.

    Possible values:
        numeric
        datetime
        boolean
        categorical
        text
    """

    if pd.api.types.is_bool_dtype(series):
        return "boolean"

    if pd.api.types.is_numeric_dtype(series):
        return "numeric"

    if _is_datetime_like(series):
        return "datetime"

    if (
        pd.api.types.is_categorical_dtype(series)
        or pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
    ):
        unique_count = series.nunique(
            dropna=True
        )

        non_null_count = series.notna().sum()

        # A low-cardinality text column is more useful as a
        # category for analysis.
        if (
            non_null_count > 0
            and unique_count <= min(
                50,
                max(10, int(non_null_count * 0.20)),
            )
        ):
            return "categorical"

        return "text"

    return "other"


# ============================================================
# COLUMN PROFILE
# ============================================================


def _column_types(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a detailed profile for every dataset column.
    """

    rows: List[Dict[str, Any]] = []

    for column in df.columns:

        series = df[column]

        non_null = int(
            series.notna().sum()
        )

        missing = int(
            series.isna().sum()
        )

        unique = int(
            series.nunique(
                dropna=True
            )
        )

        row: Dict[str, Any] = {
            "column": str(column),
            "dtype": str(series.dtype),
            "semantic_type": _column_type(series),
            "missing_values": missing,
            "missing_pct": round(
                float(
                    series.isna().mean() * 100
                ),
                2,
            ),
            "unique_values": unique,
            "non_null_values": non_null,
        }

        # ----------------------------------------------------
        # Numeric information
        # ----------------------------------------------------

        if pd.api.types.is_numeric_dtype(series):

            numeric = pd.to_numeric(
                series,
                errors="coerce",
            )

            if numeric.notna().any():

                row.update(
                    {
                        "min": _safe_scalar(
                            numeric.min()
                        ),
                        "max": _safe_scalar(
                            numeric.max()
                        ),
                        "mean": _safe_scalar(
                            numeric.mean()
                        ),
                        "median": _safe_scalar(
                            numeric.median()
                        ),
                    }
                )

        # ----------------------------------------------------
        # Datetime information
        # ----------------------------------------------------

        elif _is_datetime_like(series):

            parsed = pd.to_datetime(
                series,
                errors="coerce",
            )

            if parsed.notna().any():

                row.update(
                    {
                        "min_date": str(
                            parsed.min()
                        ),
                        "max_date": str(
                            parsed.max()
                        ),
                    }
                )

        # ----------------------------------------------------
        # Sample values
        # ----------------------------------------------------

        try:

            values = (
                series.dropna()
                .head(MAX_SAMPLE_VALUES)
                .tolist()
            )

            row["sample_values"] = [
                _safe_scalar(value)
                for value in values
            ]

        except Exception:

            row["sample_values"] = []

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# BASIC STATISTICS
# ============================================================


def _basic_stats(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate descriptive statistics for numeric columns.
    """

    numeric_df = df.select_dtypes(
        include=["number"]
    )

    if numeric_df.empty:
        return pd.DataFrame(
            columns=[
                "column",
                "count",
                "mean",
                "std",
                "min",
                "25%",
                "50%",
                "75%",
                "max",
            ]
        )

    try:

        stats = (
            numeric_df
            .describe()
            .T
            .reset_index()
            .rename(
                columns={
                    "index": "column"
                }
            )
        )

        return stats

    except Exception:

        return pd.DataFrame()


# ============================================================
# CORRELATION MATRIX
# ============================================================


def _correlation_matrix(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate correlation between numeric columns.

    Limits the number of columns to avoid creating an enormous
    dashboard table for very wide datasets.
    """

    numeric_df = df.select_dtypes(
        include=["number"]
    )

    if numeric_df.shape[1] < 2:
        return pd.DataFrame()

    # Keep the most populated numeric columns.
    if numeric_df.shape[1] > MAX_CORRELATION_COLUMNS:

        ordered_columns = sorted(
            numeric_df.columns,
            key=lambda column: (
                numeric_df[column]
                .notna()
                .sum()
            ),
            reverse=True,
        )

        numeric_df = numeric_df[
            ordered_columns[
                :MAX_CORRELATION_COLUMNS
            ]
        ]

    try:

        corr = numeric_df.corr(
            numeric_only=True
        )

        return corr

    except Exception:

        return pd.DataFrame()


# ============================================================
# DATA QUALITY SUMMARY
# ============================================================


def _data_quality(
    df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Produce high-level data quality metrics.
    """

    total_cells = int(
        df.shape[0] * df.shape[1]
    )

    missing_cells = int(
        df.isna().sum().sum()
    )

    duplicate_rows = int(
        df.duplicated().sum()
    )

    if total_cells > 0:
        missing_pct = round(
            missing_cells
            / total_cells
            * 100,
            2,
        )
    else:
        missing_pct = 0.0

    return {
        "total_cells": total_cells,
        "missing_cells": missing_cells,
        "missing_pct": missing_pct,
        "duplicate_rows": duplicate_rows,
        "duplicate_row_pct": (
            round(
                duplicate_rows
                / len(df)
                * 100,
                2,
            )
            if len(df) > 0
            else 0.0
        ),
    }


# ============================================================
# COLUMN GROUPS
# ============================================================


def _column_groups(
    df: pd.DataFrame,
) -> Dict[str, List[str]]:
    """
    Identify columns by semantic type.
    """

    numeric_columns = []
    datetime_columns = []
    categorical_columns = []
    text_columns = []
    boolean_columns = []

    for column in df.columns:

        semantic_type = _column_type(
            df[column]
        )

        column_name = str(column)

        if semantic_type == "numeric":
            numeric_columns.append(
                column_name
            )

        elif semantic_type == "datetime":
            datetime_columns.append(
                column_name
            )

        elif semantic_type == "categorical":
            categorical_columns.append(
                column_name
            )

        elif semantic_type == "text":
            text_columns.append(
                column_name
            )

        elif semantic_type == "boolean":
            boolean_columns.append(
                column_name
            )

    return {
        "numeric": numeric_columns,
        "datetime": datetime_columns,
        "categorical": categorical_columns,
        "text": text_columns,
        "boolean": boolean_columns,
    }


# ============================================================
# MAIN PROFILE FUNCTION
# ============================================================


def build_data_profile(
    df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Generate the complete dataset profile used by AIDAASS.

    Returns a dictionary containing:

        rows
        columns
        numeric_columns
        datetime_columns
        categorical_columns
        text_columns
        boolean_columns
        column_profile
        basic_stats
        correlation_matrix
        data_quality
        column_groups
    """

    if not isinstance(
        df,
        pd.DataFrame,
    ):
        raise ValueError(
            "build_data_profile() expects "
            "a pandas DataFrame."
        )

    if df.shape[1] > MAX_PROFILE_COLUMNS:

        profile_df = df.iloc[
            :,
            :MAX_PROFILE_COLUMNS
        ]

    else:

        profile_df = df

    groups = _column_groups(
        profile_df
    )

    profile = {
        # ----------------------------------------------------
        # Dataset dimensions
        # ----------------------------------------------------

        "rows": int(
            df.shape[0]
        ),

        "columns": int(
            df.shape[1]
        ),

        # ----------------------------------------------------
        # Semantic column counts
        # ----------------------------------------------------

        "numeric_columns": len(
            groups["numeric"]
        ),

        "datetime_columns": len(
            groups["datetime"]
        ),

        "categorical_columns": len(
            groups["categorical"]
        ),

        "text_columns": len(
            groups["text"]
        ),

        "boolean_columns": len(
            groups["boolean"]
        ),

        # ----------------------------------------------------
        # Detailed artifacts
        # ----------------------------------------------------

        "column_profile": _column_types(
            profile_df
        ),

        "basic_stats": _basic_stats(
            df
        ),

        "correlation_matrix": _correlation_matrix(
            df
        ),

        "data_quality": _data_quality(
            df
        ),

        "column_groups": groups,
    }

    return profile


# ============================================================
# SAFE VALUE CONVERSION
# ============================================================


def _safe_scalar(
    value: Any,
) -> Any:
    """
    Convert NumPy/pandas scalar values into safe Python values.
    """

    if value is None:
        return None

    try:

        if pd.isna(value):
            return None

    except (TypeError, ValueError):
        pass

    if isinstance(
        value,
        np.generic,
    ):
        return value.item()

    if isinstance(
        value,
        pd.Timestamp,
    ):
        return value.isoformat()

    return value
