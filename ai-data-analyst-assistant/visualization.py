"""
Primary visualization entry point for AIDAASS.

This module provides a stable interface between the Streamlit
application and the visualization implementation.

The main public function is:

    generate_chart(result, chart_hint)

It returns:

    (plotly_figure_or_none, message)
"""

from typing import Any, Optional, Tuple

from visualization import create_visualization


# ============================================================
# VALID CHART TYPES
# ============================================================

VALID_CHART_HINTS = {
    "auto",
    "bar",
    "line",
    "pie",
    "histogram",
    "scatter",
    "none",
}


# ============================================================
# CHART GENERATION
# ============================================================


def generate_chart(
    result: Any,
    chart_hint: str = "auto",
) -> Tuple[Optional[Any], str]:
    """
    Generate an appropriate Plotly visualization.

    Args:
        result:
            Analysis result returned by the executor.

        chart_hint:
            Requested chart type from the AI engine.

    Returns:
        Tuple containing:
            - Plotly Figure or None
            - Human-readable status message

    The function deliberately catches visualization errors so
    that a chart failure cannot crash the Streamlit dashboard.
    """

    # --------------------------------------------------------
    # Normalize chart hint
    # --------------------------------------------------------

    if chart_hint is None:
        chart_hint = "auto"

    chart_hint = str(
        chart_hint
    ).strip().lower()

    if chart_hint not in VALID_CHART_HINTS:
        chart_hint = "auto"

    # --------------------------------------------------------
    # Generate chart
    # --------------------------------------------------------

    try:

        figure, message = create_visualization(
            result=result,
            chart_hint=chart_hint,
        )

        return figure, message

    except Exception as ex:

        return (
            None,
            (
                "Visualization could not be generated: "
                f"{type(ex).__name__}: {ex}"
            ),
        )
