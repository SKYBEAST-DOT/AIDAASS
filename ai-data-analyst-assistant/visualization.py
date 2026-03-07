from typing import Any, List, Optional, Tuple

import pandas as pd
import plotly.express as px


def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def _is_datetime(series: pd.Series) -> bool:
    return pd.api.types.is_datetime64_any_dtype(series)


def _prepare_dataframe(result: pd.DataFrame) -> pd.DataFrame:
    df = result.copy()
    for col in df.columns:
        if df[col].dtype == "object":
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().mean() > 0.8:
                df[col] = parsed
    return df


def _best_numeric_columns(df: pd.DataFrame) -> List[str]:
    numeric = [c for c in df.columns if _is_numeric(df[c])]
    if not numeric:
        return []
    return sorted(numeric, key=lambda c: df[c].notna().sum(), reverse=True)


def _best_category_column(df: pd.DataFrame, numeric_cols: List[str]) -> Optional[str]:
    candidates = [c for c in df.columns if c not in numeric_cols and not _is_datetime(df[c])]
    if not candidates:
        return None

    scored = []
    for c in candidates:
        unique = df[c].nunique(dropna=True)
        score = (df[c].notna().sum(), -abs(unique - 10))
        scored.append((score, c))
    scored.sort(reverse=True)
    return scored[0][1]


def _best_datetime_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [c for c in df.columns if _is_datetime(df[c])]
    if not candidates:
        return None
    return sorted(candidates, key=lambda c: df[c].notna().sum(), reverse=True)[0]


def create_visualization(result: Any, chart_hint: str = "auto") -> Tuple[Optional[Any], str]:
    """Auto-select chart type for analysis results.

    Returns:
        (plotly_figure_or_none, message)
    """
    if result is None:
        return None, "No result available for visualization."

    if isinstance(result, pd.Series):
        result = result.reset_index()

    if not isinstance(result, pd.DataFrame):
        return None, "Result is scalar/text; chart not applicable."

    if result.empty:
        return None, "Result is empty; no chart generated."

    df = _prepare_dataframe(result)

    numeric_cols = _best_numeric_columns(df)
    datetime_col = _best_datetime_column(df)
    category_col = _best_category_column(df, numeric_cols)

    if chart_hint == "none":
        return None, "Chart generation disabled by AI hint."

    if chart_hint == "line" and datetime_col and numeric_cols:
        fig = px.line(df.sort_values(datetime_col), x=datetime_col, y=numeric_cols[0], markers=True)
        return fig, "Line chart generated."

    if chart_hint == "pie" and category_col and numeric_cols:
        chart_df = df[[category_col, numeric_cols[0]]].dropna().groupby(category_col, as_index=False)[numeric_cols[0]].sum()
        fig = px.pie(chart_df.sort_values(numeric_cols[0], ascending=False).head(10), names=category_col, values=numeric_cols[0])
        return fig, "Pie chart generated."

    if chart_hint == "bar" and category_col and numeric_cols:
        chart_df = df[[category_col, numeric_cols[0]]].dropna().groupby(category_col, as_index=False)[numeric_cols[0]].sum()
        fig = px.bar(chart_df.sort_values(numeric_cols[0], ascending=False).head(25), x=category_col, y=numeric_cols[0])
        return fig, "Bar chart generated."

    if datetime_col and numeric_cols:
        fig = px.line(df.sort_values(datetime_col), x=datetime_col, y=numeric_cols[0], markers=True)
        return fig, "Line chart generated automatically."

    if category_col and numeric_cols:
        category = category_col
        value = numeric_cols[0]
        chart_df = df[[category, value]].dropna().groupby(category, as_index=False)[value].sum()

        if chart_df[category].nunique() <= 8:
            fig = px.pie(chart_df, names=category, values=value)
            return fig, "Pie chart generated automatically."

        fig = px.bar(chart_df.sort_values(value, ascending=False).head(25), x=category, y=value)
        return fig, "Bar chart generated automatically."

    if len(numeric_cols) >= 2:
        fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1])
        return fig, "Scatter plot generated automatically."

    if len(numeric_cols) == 1:
        fig = px.histogram(df, x=numeric_cols[0], nbins=24)
        return fig, "Histogram generated automatically."

    return None, "Could not infer a suitable chart type for this result."