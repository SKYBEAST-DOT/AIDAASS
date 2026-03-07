from typing import Any, Optional, Tuple

from visualization import create_visualization


def generate_chart(result: Any, chart_hint: str = "auto") -> Tuple[Optional[Any], str]:
    """Primary chart generation entrypoint for dashboard and API consumers."""
    return create_visualization(result, chart_hint)
