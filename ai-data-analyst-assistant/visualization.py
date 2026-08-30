"""
Visualization engine for AIDAASS.

Automatically creates Plotly charts from pandas analysis results.

Supported chart types:
    - bar
    - line
    - pie
    - histogram
    - scatter
    - auto
    - none

The visualization layer never modifies the original result.
"""

from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px


# ============================================================
# CONFIGURATION
# ============================================================

MAX_CATEGORIES_FOR_BAR = 25
MAX_CATEGORIES_FOR_PIE = 10
MAX_POINTS_FOR_SCATTER = 5_000
MAX_POINTS_FOR_LINE = 10_000
MAX_HISTOGRAM_BINS = 40

VALID_HINTS = {
    "auto",
    "bar",
    "line",
    "pie",
    "histogram",
    "scatter",
    "none",
}


# ============================================================
# TYPE HELPERS
# ============================================================


def _is_numeric(
    series: pd.Series,
) -> bool:
    """Return True when a Series is numeric."""

    return pd.api.types.is_numeric_dtype(
        series
    )


def _is_datetime(
    series: pd.Series,
) -> bool:
    """Return True when a Series contains datetime values."""

    return pd.api.types.is_datetime64_any_dtype(
        series
    )


def _looks_like_datetime(
    series: pd.Series,
) -> bool:
    """
    Conservatively detect datetime-like object columns.

    We only test object/string columns and require a high
    successful parsing ratio to reduce false positives.
    """

    if _is_datetime(series):
        return True

    if not (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
    ):
        return False

    non_null = series.dropna()

    if non_null.empty:
        return False

    # Do not attempt expensive conversion on a huge column.
    sample = non_null.head(100)

    try:

        parsed = pd.to_datetime(
            sample,
            errors="coerce",
        )

        success_rate = float(
            parsed.notna().mean()
        )

        return success_rate >= 0.85

    except Exception:

        return False


# ============================================================
# DATAFRAME PREPARATION
# ============================================================


def _prepare_dataframe(
    result: pd.DataFrame,
) -> pd.DataFrame:
    """
    Make a safe copy of the result for visualization.

    The original result is never modified.
    """

    df = result.copy()

    # --------------------------------------------------------
    # Normalize column names
    # --------------------------------------------------------

    # Plotly accepts many column-name types, but converting to
    # strings makes downstream handling predictable.
    df.columns = [
        str(column)
        for column in df.columns
    ]

    # --------------------------------------------------------
    # Convert clearly datetime-like object columns
    # --------------------------------------------------------

    for column in df.columns:

        series = df[column]

        if not (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
        ):
            continue

        # Column names containing date/time are stronger evidence
        # than arbitrary strings.
        name_hint = (
            "date" in column.lower()
            or "time" in column.lower()
            or "month" in column.lower()
        )

        if not name_hint and not _looks_like_datetime(
            series
        ):
            continue

        try:

            parsed = pd.to_datetime(
                series,
                errors="coerce",
            )

            non_null_count = int(
                series.notna().sum()
            )

            parsed_count = int(
                parsed.notna().sum()
            )

            if (
                non_null_count > 0
                and parsed_count / non_null_count >= 0.85
            ):
                df[column] = parsed

        except Exception:
            continue

    return df


# ============================================================
# NUMERIC COLUMN SELECTION
# ============================================================


def _best_numeric_columns(
    df: pd.DataFrame,
) -> List[str]:
    """
    Rank numeric columns by usefulness.

    Columns with more valid observations are preferred.
    """

    numeric = [
        column
        for column in df.columns
        if _is_numeric(df[column])
    ]

    if not numeric:
        return []

    return sorted(
        numeric,
        key=lambda column: (
            int(
                df[column]
                .notna()
                .sum()
            ),
            -int(
                df[column]
                .nunique(
                    dropna=True
                )
            ),
        ),
        reverse=True,
    )


# ============================================================
# CATEGORY COLUMN SELECTION
# ============================================================


def _best_category_column(
    df: pd.DataFrame,
    numeric_cols: List[str],
) -> Optional[str]:
    """
    Select the most useful categorical column.

    Avoids:
        - numeric columns
        - datetime columns
        - completely empty columns
        - extremely high-cardinality identifiers when possible
    """

    candidates = []

    for column in df.columns:

        if column in numeric_cols:
            continue

        series = df[column]

        if _is_datetime(series):
            continue

        non_null = int(
            series.notna().sum()
        )

        if non_null == 0:
            continue

        unique = int(
            series.nunique(
                dropna=True
            )
        )

        if unique <= 1:
            continue

        # Penalize extremely high-cardinality columns because
        # they are often IDs rather than useful categories.
        if unique > max(
            100,
            int(len(df) * 0.8),
        ):
            cardinality_penalty = 0.5
        else:
            cardinality_penalty = 1.0

        # Prefer a useful number of categories.
        category_fit = 1.0 / (
            1.0
            + abs(unique - 8)
        )

        score = (
            non_null
            * cardinality_penalty
            * category_fit
        )

        candidates.append(
            (
                score,
                column,
            )
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    return candidates[0][1]


# ============================================================
# DATETIME COLUMN SELECTION
# ============================================================


def _best_datetime_column(
    df: pd.DataFrame,
) -> Optional[str]:
    """Return the datetime column with the most valid values."""

    candidates = [
        column
        for column in df.columns
        if _is_datetime(df[column])
    ]

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda column: int(
            df[column]
            .notna()
            .sum()
        ),
    )


# ============================================================
# VALUE COLUMN SELECTION
# ============================================================


def _select_value_column(
    numeric_cols: List[str],
) -> Optional[str]:
    """Return the strongest available numeric value column."""

    if not numeric_cols:
        return None

    return numeric_cols[0]


# ============================================================
# CHART DATA HELPERS
# ============================================================


def _category_value_data(
    df: pd.DataFrame,
    category: str,
    value: str,
) -> pd.DataFrame:
    """
    Prepare category/value data for bar or pie charts.
    """

    chart_df = df[
        [category, value]
    ].copy()

    chart_df = chart_df.dropna(
        subset=[
            category,
            value,
        ]
    )

    if chart_df.empty:
        return chart_df

    try:

        chart_df = (
            chart_df
            .groupby(
                category,
                as_index=False,
            )[value]
            .sum()
        )

    except Exception:
        return pd.DataFrame(
            columns=[
                category,
                value,
            ]
        )

    return chart_df


# ============================================================
# BAR CHART
# ============================================================


def _create_bar_chart(
    df: pd.DataFrame,
    category: Optional[str],
    value: Optional[str],
) -> Tuple[Optional[Any], str]:

    if not category or not value:
        return (
            None,
            "Bar chart requires a category and numeric column.",
        )

    chart_df = _category_value_data(
        df,
        category,
        value,
    )

    if chart_df.empty:
        return (
            None,
            "No valid category/value data is available for a bar chart.",
        )

    chart_df = (
        chart_df
        .sort_values(
            value,
            ascending=False,
        )
        .head(
            MAX_CATEGORIES_FOR_BAR
        )
    )

    fig = px.bar(
        chart_df,
        x=category,
        y=value,
        title=f"{value} by {category}",
    )

    fig.update_layout(
        xaxis_title=category,
        yaxis_title=value,
    )

    return (
        fig,
        "Bar chart generated.",
    )


# ============================================================
# PIE CHART
# ============================================================


def _create_pie_chart(
    df: pd.DataFrame,
    category: Optional[str],
    value: Optional[str],
) -> Tuple[Optional[Any], str]:

    if not category or not value:
        return (
            None,
            "Pie chart requires a category and numeric column.",
        )

    chart_df = _category_value_data(
        df,
        category,
        value,
    )

    if chart_df.empty:
        return (
            None,
            "No valid category/value data is available for a pie chart.",
        )

    chart_df = (
        chart_df
        .sort_values(
            value,
            ascending=False,
        )
        .head(
            MAX_CATEGORIES_FOR_PIE
        )
    )

    if chart_df.empty:
        return (
            None,
            "No values are available for a pie chart.",
        )

    fig = px.pie(
        chart_df,
        names=category,
        values=value,
        title=f"{value} by {category}",
    )

    return (
        fig,
        "Pie chart generated.",
    )


# ============================================================
# LINE CHART
# ============================================================


def _create_line_chart(
    df: pd.DataFrame,
    datetime_col: Optional[str],
    value: Optional[str],
) -> Tuple[Optional[Any], str]:

    if not datetime_col or not value:
        return (
            None,
            "Line chart requires a datetime and numeric column.",
        )

    chart_df = df[
        [
            datetime_col,
            value,
        ]
    ].copy()

    chart_df = chart_df.dropna(
        subset=[
            datetime_col,
            value,
        ]
    )

    if chart_df.empty:
        return (
            None,
            "No valid datetime/value data is available for a line chart.",
        )

    chart_df = (
        chart_df
        .sort_values(
            datetime_col
        )
        .head(
            MAX_POINTS_FOR_LINE
        )
    )

    fig = px.line(
        chart_df,
        x=datetime_col,
        y=value,
        markers=True,
        title=f"{value} over time",
    )

    fig.update_layout(
        xaxis_title=datetime_col,
        yaxis_title=value,
    )

    return (
        fig,
        "Line chart generated.",
    )


# ============================================================
# HISTOGRAM
# ============================================================


def _create_histogram(
    df: pd.DataFrame,
    value: Optional[str],
) -> Tuple[Optional[Any], str]:

    if not value:
        return (
            None,
            "Histogram requires a numeric column.",
        )

    chart_df = df[
        [value]
    ].dropna()

    if chart_df.empty:
        return (
            None,
            "No numeric values are available for a histogram.",
        )

    fig = px.histogram(
        chart_df,
        x=value,
        nbins=24,
        title=f"Distribution of {value}",
    )

    fig.update_layout(
        xaxis_title=value,
        yaxis_title="Count",
    )

    return (
        fig,
        "Histogram generated.",
    )


# ============================================================
# SCATTER PLOT
# ============================================================


def _create_scatter_chart(
    df: pd.DataFrame,
    x_column: Optional[str],
    y_column: Optional[str],
) -> Tuple[Optional[Any], str]:

    if not x_column or not y_column:
        return (
            None,
            "Scatter plot requires two numeric columns.",
        )

    if x_column == y_column:
        return (
            None,
            "Scatter plot requires two different numeric columns.",
        )

    chart_df = df[
        [
            x_column,
            y_column,
        ]
    ].dropna()

    if chart_df.empty:
        return (
            None,
            "No valid values are available for a scatter plot.",
        )

    # Prevent extremely large scatter plots from overwhelming
    # the browser.
    if len(chart_df) > MAX_POINTS_FOR_SCATTER:

        chart_df = chart_df.sample(
            n=MAX_POINTS_FOR_SCATTER,
            random_state=42,
        )

    fig = px.scatter(
        chart_df,
        x=x_column,
        y=y_column,
        title=f"{y_column} vs {x_column}",
    )

    fig.update_layout(
        xaxis_title=x_column,
        yaxis_title=y_column,
    )

    return (
        fig,
        "Scatter plot generated.",
    )


# ============================================================
# AUTOMATIC CHART SELECTION
# ============================================================


def _automatic_visualization(
    df: pd.DataFrame,
    numeric_cols: List[str],
    datetime_col: Optional[str],
    category_col: Optional[str],
) -> Tuple[Optional[Any], str]:
    """
    Choose a chart based on the structure of the result.

    Priority:

        datetime + numeric
            ↓
        category + numeric
            ↓
        2+ numeric
            ↓
        1 numeric
            ↓
        no chart
    """

    value = _select_value_column(
        numeric_cols
    )

    # --------------------------------------------------------
    # Time series
    # --------------------------------------------------------

    if datetime_col and value:

        return _create_line_chart(
            df,
            datetime_col,
            value,
        )

    # --------------------------------------------------------
    # Category + numeric
    # --------------------------------------------------------

    if category_col and value:

        chart_df = _category_value_data(
            df,
            category_col,
            value,
        )

        if not chart_df.empty:

            category_count = int(
                chart_df[
                    category_col
                ].nunique()
            )

            if category_count <= 8:

                return _create_pie_chart(
                    df,
                    category_col,
                    value,
                )

            return _create_bar_chart(
                df,
                category_col,
                value,
            )

    # --------------------------------------------------------
    # Two numeric columns
    # --------------------------------------------------------

    if len(numeric_cols) >= 2:

        return _create_scatter_chart(
            df,
            numeric_cols[0],
            numeric_cols[1],
        )

    # --------------------------------------------------------
    # One numeric column
    # --------------------------------------------------------

    if len(numeric_cols) == 1:

        return _create_histogram(
            df,
            numeric_cols[0],
        )

    return (
        None,
        "Could not infer a suitable chart type for this result.",
    )


# ============================================================
# PUBLIC VISUALIZATION FUNCTION
# ============================================================


def create_visualization(
    result: Any,
    chart_hint: str = "auto",
) -> Tuple[Optional[Any], str]:
    """
    Create a Plotly visualization from an analysis result.

    Args:
        result:
            Usually a pandas DataFrame or Series.

        chart_hint:
            One of:
                auto
                bar
                line
                pie
                histogram
                scatter
                none

    Returns:
        (figure, message)

    The original result is never modified.
    """

    # --------------------------------------------------------
    # Validate hint
    # --------------------------------------------------------

    if chart_hint is None:
        chart_hint = "auto"

    chart_hint = str(
        chart_hint
    ).strip().lower()

    if chart_hint not in VALID_HINTS:
        chart_hint = "auto"

    # --------------------------------------------------------
    # Explicitly disabled
    # --------------------------------------------------------

    if chart_hint == "none":
        return (
            None,
            "Chart generation disabled by AI hint.",
        )

    # --------------------------------------------------------
    # Missing result
    # --------------------------------------------------------

    if result is None:
        return (
            None,
            "No result available for visualization.",
        )

    # --------------------------------------------------------
    # Series → DataFrame
    # --------------------------------------------------------

    if isinstance(
        result,
        pd.Series,
    ):

        series_name = (
            str(result.name)
            if result.name is not None
            else "value"
        )

        try:

            df = result.rename(
                series_name
            ).reset_index()

        except Exception:

            df = result.reset_index()

    # --------------------------------------------------------
    # DataFrame
    # --------------------------------------------------------

    elif isinstance(
        result,
        pd.DataFrame,
    ):

        df = result.copy()

    # --------------------------------------------------------
    # Other result types
    # --------------------------------------------------------

    else:

        return (
            None,
            "Result is scalar/text; chart not applicable.",
        )

    # --------------------------------------------------------
    # Empty result
    # --------------------------------------------------------

    if df.empty:
        return (
            None,
            "Result is empty; no chart generated.",
        )

    # --------------------------------------------------------
    # Prepare result
    # --------------------------------------------------------

    try:

        df = _prepare_dataframe(
            df
        )

    except Exception as ex:

        return (
            None,
            (
                "Could not prepare result "
                f"for visualization: {ex}"
            ),
        )

    if df.empty:
        return (
            None,
            "No usable data remains after preparation.",
        )

    # --------------------------------------------------------
    # Identify columns
    # --------------------------------------------------------

    numeric_cols = _best_numeric_columns(
        df
    )

    datetime_col = _best_datetime_column(
        df
    )

    category_col = _best_category_column(
        df,
        numeric_cols,
    )

    value_col = _select_value_column(
        numeric_cols
    )

    # ========================================================
    # EXPLICIT CHART HINTS
    # ========================================================

    if chart_hint == "bar":

        return _create_bar_chart(
            df,
            category_col,
            value_col,
        )

    if chart_hint == "line":

        return _create_line_chart(
            df,
            datetime_col,
            value_col,
        )

    if chart_hint == "pie":

        return _create_pie_chart(
            df,
            category_col,
            value_col,
        )

    if chart_hint == "histogram":

        return _create_histogram(
            df,
            value_col,
        )

    if chart_hint == "scatter":

        if len(numeric_cols) < 2:

            return (
                None,
                "Scatter plot requires at least two numeric columns.",
            )

        return _create_scatter_chart(
            df,
            numeric_cols[0],
            numeric_cols[1],
        )

    # ========================================================
    # AUTOMATIC CHART SELECTION
    # ========================================================

    return _automatic_visualization(
        df,
        numeric_cols,
        datetime_col,
        category_col,
    )
