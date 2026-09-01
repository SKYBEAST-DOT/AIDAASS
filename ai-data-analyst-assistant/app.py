import io
import traceback
from typing import Any, Optional

import pandas as pd
import streamlit as st

from ai_engine import AIEngine
from data_profiler import build_data_profile
from query_executor import execute_ai_query
from utils import (
    load_dataset,
    load_example_dataset,
    load_packaged_ecommerce_sample,
)
from visualization_engine import generate_chart


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Data Analyst Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 1.5rem;
            max-width: 1400px;
        }

        .hero-card {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            border-radius: 16px;
            padding: 1.4rem 1.5rem;
            margin-bottom: 1rem;
            border: 1px solid #334155;
        }

        .hero-title {
            font-size: 1.8rem;
            font-weight: 750;
            color: #f8fafc;
            margin-bottom: 0.25rem;
        }

        .hero-sub {
            font-size: 0.95rem;
            color: #cbd5e1;
            line-height: 1.5;
        }

        .pill {
            display: inline-block;
            padding: 0.25rem 0.7rem;
            border-radius: 999px;
            border: 1px solid #334155;
            background: #0b1220;
            color: #cbd5e1;
            font-size: 0.78rem;
            margin-bottom: 0.7rem;
        }

        .caption-soft {
            color: #94a3b8;
            font-size: 0.85rem;
        }

        .history-item {
            padding: 0.5rem 0.7rem;
            border-radius: 8px;
            background: #f8fafc;
            margin-bottom: 0.35rem;
            font-size: 0.85rem;
        }

        div[data-testid="stMetric"] {
            padding: 0.5rem 0;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.35rem;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">📊 AI Data Analyst Assistant</div>
        <div class="hero-sub">
            Upload CSV or Excel data, ask questions in plain English,
            generate pandas analysis safely, visualize results, and receive
            AI-powered business insights.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "question_value": "",
    "last_generation": None,
    "last_result": None,
    "last_insight": "",
    "last_chart_hint": "auto",
    "last_error": "",
    "analysis_history": [],
}

for key, default_value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = default_value


# ============================================================
# HELPER FUNCTIONS
# ============================================================


def _reset_analysis() -> None:
    """Clear the current analysis result."""
    st.session_state["last_generation"] = None
    st.session_state["last_result"] = None
    st.session_state["last_insight"] = ""
    st.session_state["last_chart_hint"] = "auto"
    st.session_state["last_error"] = ""


def _reset_all() -> None:
    """Reset the complete analysis state."""
    _reset_analysis()
    st.session_state["question_value"] = ""
    st.session_state["analysis_history"] = []


def _render_dataset_preview(df: pd.DataFrame) -> None:
    """Render dataset KPIs and inspection tabs."""

    profile = build_data_profile(df)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Rows", f"{profile['rows']:,}")

    with col2:
        st.metric("Columns", f"{profile['columns']:,}")

    with col3:
        st.metric(
            "Numeric Columns",
            f"{profile['numeric_columns']:,}",
        )

    with col4:
        datetime_count = sum(
            pd.api.types.is_datetime64_any_dtype(df[column])
            for column in df.columns
        )
        st.metric("Datetime Columns", f"{datetime_count:,}")

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Preview",
            "Schema",
            "Statistics",
            "Correlation",
        ]
    )

    with tab1:
        st.dataframe(
            df.head(50),
            use_container_width=True,
            height=360,
        )

    with tab2:
        column_profile = profile.get("column_profile")

        if column_profile is not None:
            st.dataframe(
                column_profile,
                use_container_width=True,
                height=360,
            )
        else:
            st.info("Column profile is not available.")

    with tab3:
        stats = profile.get("basic_stats")

        if stats is None or stats.empty:
            st.info(
                "No numeric columns are available for basic statistics."
            )
        else:
            st.dataframe(
                stats,
                use_container_width=True,
                height=360,
            )

    with tab4:
        corr = profile.get("correlation_matrix")

        if corr is None or corr.empty:
            st.info(
                "At least two numeric columns are required "
                "for correlation analysis."
            )
        else:
            st.dataframe(
                corr,
                use_container_width=True,
                height=360,
            )


def _quick_question_buttons() -> Optional[str]:
    """Render common questions and return the selected question."""

    st.markdown(
        '<div class="caption-soft">Quick question starters</div>',
        unsafe_allow_html=True,
    )

    prompts = [
        "Show top 5 products by revenue",
        "Which region has the highest sales?",
        "Show monthly profit trend",
        "Which category contributes most profit?",
    ]

    columns = st.columns(4)

    for index, prompt in enumerate(prompts):
        with columns[index]:
            if st.button(
                prompt,
                use_container_width=True,
                key=f"quick_prompt_{index}",
            ):
                return prompt

    return None


def _render_result(result: Any) -> None:
    """Render an AI analysis result safely."""

    if isinstance(result, pd.DataFrame):
        if result.empty:
            st.info("The analysis returned an empty table.")
            return

        st.dataframe(
            result,
            use_container_width=True,
            height=360,
        )
        return

    if isinstance(result, pd.Series):
        if result.empty:
            st.info("The analysis returned an empty result.")
            return

        series_name = result.name or "value"

        st.dataframe(
            result.to_frame(name=series_name),
            use_container_width=True,
            height=360,
        )
        return

    if result is None:
        st.info("The analysis returned no result.")
        return

    st.write(result)


def _result_to_csv(result: Any) -> Optional[bytes]:
    """Convert supported analysis results to CSV bytes."""

    try:
        if isinstance(result, pd.DataFrame):
            return result.to_csv(index=False).encode("utf-8")

        if isinstance(result, pd.Series):
            return result.to_frame(
                name=result.name or "value"
            ).to_csv(index=True).encode("utf-8")

        return None

    except Exception:
        return None


def _render_sidebar_dataset_summary(df: pd.DataFrame) -> None:
    """Render a compact dataset summary in the sidebar."""

    st.divider()
    st.subheader("Dataset Summary")

    numeric_count = sum(
        pd.api.types.is_numeric_dtype(df[column])
        for column in df.columns
    )

    datetime_count = sum(
        pd.api.types.is_datetime64_any_dtype(df[column])
        for column in df.columns
    )

    st.caption(f"Rows: {df.shape[0]:,}")
    st.caption(f"Columns: {df.shape[1]:,}")
    st.caption(
        f"Numeric: {numeric_count:,} | "
        f"Datetime: {datetime_count:,}"
    )

    st.caption("Top columns")

    top_columns = df.columns[:8].tolist()

    for column in top_columns:
        st.caption(f"• {column}")


def _add_to_history(
    question: str,
    result: Any,
    insight: str,
) -> None:
    """Store a lightweight analysis history item."""

    if isinstance(result, pd.DataFrame):
        result_type = f"DataFrame ({len(result):,} rows)"
    elif isinstance(result, pd.Series):
        result_type = f"Series ({len(result):,} values)"
    else:
        result_type = type(result).__name__

    history_item = {
        "question": question,
        "result_type": result_type,
        "insight": insight,
    }

    st.session_state["analysis_history"].insert(
        0,
        history_item,
    )

    # Keep the application lightweight.
    st.session_state["analysis_history"] = (
        st.session_state["analysis_history"][:10]
    )


def _render_analysis_history() -> None:
    """Render recent analysis history."""

    history = st.session_state.get("analysis_history", [])

    if not history:
        st.caption("No analyses have been run yet.")
        return

    for index, item in enumerate(history, start=1):
        question = item.get("question", "Unknown question")
        result_type = item.get("result_type", "Unknown result")

        st.markdown(
            f"""
            <div class="history-item">
                <strong>{index}. {question}</strong><br>
                <span class="caption-soft">{result_type}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _get_api_key(
    sidebar_key: str,
    workspace_key: str,
) -> str:
    """Resolve API key with sidebar input taking priority."""

    return (sidebar_key or "").strip() or (
        workspace_key or ""
    ).strip()


# ============================================================
# CONFIGURATION
# ============================================================

try:
    workspace_api_key = st.secrets.get(
        "GEMINI_API_KEY",
        "",
    )
except Exception:
    workspace_api_key = ""


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.subheader("⚙️ Configuration")

    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        help=(
            "Optional if GEMINI_API_KEY is already configured "
            "in Streamlit secrets."
        ),
    )

    if workspace_api_key:
        st.caption(
            "✅ Workspace Gemini secret is configured."
        )
    else:
        st.caption(
            "No workspace Gemini secret detected."
        )

    model_name = st.text_input(
        "Gemini Model",
        value="gemini-1.5-flash",
        help=(
            "Enter the Gemini model supported by your "
            "AIEngine configuration."
        ),
    ).strip()

    if not model_name:
        model_name = "gemini-1.5-flash"

    st.divider()

    st.subheader("📁 Data Source")

    source_type = st.radio(
        "Choose source",
        options=[
            "Upload file",
            "Example dataset",
        ],
        index=0,
    )

    selected_example = None

    if source_type == "Example dataset":
        selected_example = st.selectbox(
            "Select demo dataset",
            options=[
                "Retail Sales Demo",
                "Marketing Campaign Demo",
                "Ecommerce Orders Sample (CSV)",
            ],
            index=0,
        )

    st.divider()

    st.subheader("🧹 Session")

    if st.button(
        "Clear Current Analysis",
        use_container_width=True,
    ):
        _reset_analysis()
        st.rerun()

    if st.button(
        "Reset Everything",
        use_container_width=True,
    ):
        _reset_all()
        st.rerun()

    st.divider()

    st.subheader("🕘 Analysis History")
    _render_analysis_history()


# ============================================================
# LOAD DATASET
# ============================================================

uploaded_file = None

if source_type == "Upload file":
    uploaded_file = st.file_uploader(
        "Upload CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        help="Supported formats: CSV, XLSX and XLS.",
    )


df = None
dataset_label = ""


try:
    if (
        source_type == "Upload file"
        and uploaded_file is not None
    ):
        df = load_dataset(uploaded_file)
        dataset_label = (
            f"Uploaded file: {uploaded_file.name}"
        )

    elif (
        source_type == "Example dataset"
        and selected_example
    ):
        if selected_example == "Ecommerce Orders Sample (CSV)":
            df = load_packaged_ecommerce_sample()
        else:
            df = load_example_dataset(selected_example)

        dataset_label = (
            f"Example dataset: {selected_example}"
        )

except Exception as ex:
    st.error(
        f"Could not load the dataset: {ex}"
    )

    with st.expander("Show technical details"):
        st.code(
            traceback.format_exc(),
            language="text",
        )

    st.stop()


# ============================================================
# MAIN APPLICATION
# ============================================================

if df is None:
    with st.container(border=True):
        st.subheader("🚀 Get Started")

        st.info(
            "Upload a CSV/Excel file or choose an example "
            "dataset from the sidebar to begin."
        )

        tip1, tip2, tip3 = st.columns(3)

        with tip1:
            st.metric(
                "1",
                "Select Data Source",
            )

        with tip2:
            st.metric(
                "2",
                "Ask a Question",
            )

        with tip3:
            st.metric(
                "3",
                "Review Results",
            )

    st.stop()


# ============================================================
# DATASET VALIDATION
# ============================================================

if not isinstance(df, pd.DataFrame):
    st.error(
        "The selected data source did not return a valid "
        "pandas DataFrame."
    )
    st.stop()


if df.empty:
    st.warning(
        "The selected dataset is empty. "
        "Please upload a dataset containing data."
    )
    st.stop()


# ============================================================
# DATASET SUMMARY
# ============================================================

with st.sidebar:
    _render_sidebar_dataset_summary(df)


st.markdown(
    f'<span class="pill">📁 {dataset_label}</span>',
    unsafe_allow_html=True,
)


# ============================================================
# MAIN TWO-COLUMN LAYOUT
# ============================================================

overview_col, assistant_col = st.columns(
    [1.35, 1.65],
    gap="large",
)


# ============================================================
# DATASET OVERVIEW
# ============================================================

with overview_col:
    with st.container(border=True):
        st.subheader("📋 Dataset Overview")

        try:
            _render_dataset_preview(df)
        except Exception as ex:
            st.error(
                f"Could not generate the dataset profile: {ex}"
            )

            with st.expander("Show technical details"):
                st.code(
                    traceback.format_exc(),
                    language="text",
                )


# ============================================================
# AI ANALYST
# ============================================================

with assistant_col:
    with st.container(border=True):
        st.subheader("🤖 AI Analyst")

        quick_choice = _quick_question_buttons()

        if quick_choice:
            st.session_state["question_value"] = quick_choice

        st.text_input(
            "Ask a question about your data",
            placeholder=(
                "e.g., Show monthly sales trend by region"
            ),
            key="question_value",
        )

        question = st.session_state["question_value"]

        button_col1, button_col2 = st.columns(
            [3, 1]
        )

        with button_col1:
            run_clicked = st.button(
                "🚀 Run Analysis",
                type="primary",
                use_container_width=True,
            )

        with button_col2:
            clear_clicked = st.button(
                "Clear",
                use_container_width=True,
            )

        if clear_clicked:
            _reset_analysis()
            st.rerun()

        # ----------------------------------------------------
        # RUN ANALYSIS
        # ----------------------------------------------------

        if run_clicked:

            if not question.strip():
                st.warning(
                    "Please enter a question first."
                )
                st.stop()

            resolved_api_key = _get_api_key(
                api_key,
                workspace_api_key,
            )

            if not resolved_api_key:
                st.warning(
                    "Gemini API key not detected. "
                    "Running in fallback analysis mode."
                )

            _reset_analysis()

            try:
                engine = AIEngine(
                    api_key=resolved_api_key,
                    model_name=model_name,
                )

            except Exception as ex:
                st.session_state["last_error"] = (
                    f"Could not initialize AI engine: {ex}"
                )
                st.error(
                    st.session_state["last_error"]
                )
                st.stop()

            # ------------------------------------------------
            # STEP 1 — GENERATE CODE
            # ------------------------------------------------

            generation = None

            with st.spinner(
                "🧠 Generating analysis code..."
            ):
                try:
                    generation = engine.generate_pandas_code(
                        question,
                        df,
                    )

                except Exception as ex:
                    st.session_state["last_error"] = (
                        "Failed to generate analysis code: "
                        f"{ex}"
                    )

            if generation is None:
                st.error(
                    st.session_state["last_error"]
                )

                st.info(
                    "Try a simpler question and use the "
                    "exact column names shown in the Schema tab."
                )

                st.stop()

            # ------------------------------------------------
            # VALIDATE GENERATION RESPONSE
            # ------------------------------------------------

            if not isinstance(generation, dict):
                st.error(
                    "The AI returned an invalid response format."
                )
                st.stop()

            generated_code = generation.get("code")

            if not generated_code or not isinstance(
                generated_code,
                str,
            ):
                st.error(
                    "The AI did not return executable pandas code."
                )
                st.stop()

            # ------------------------------------------------
            # STEP 2 — EXECUTE THROUGH SAFETY LAYER
            # ------------------------------------------------

            execution = None

            with st.spinner(
                "🔒 Validating and executing analysis..."
            ):
                try:
                    execution = execute_ai_query(
                        generated_code,
                        df,
                    )

                except Exception as ex:
                    st.session_state["last_error"] = (
                        "Analysis execution failed: "
                        f"{ex}"
                    )

            if execution is None:
                st.error(
                    st.session_state["last_error"]
                )
                st.stop()

            if not execution.ok:
                st.session_state["last_error"] = (
                    execution.error
                    or "The generated analysis could not be executed."
                )

                st.error(
                    st.session_state["last_error"]
                )

                st.info(
                    "Try rephrasing the question with explicit "
                    "column names or a simpler aggregation."
                )

                with st.expander(
                    "Show generated code"
                ):
                    st.code(
                        generated_code,
                        language="python",
                    )

                st.stop()

            # ------------------------------------------------
            # STEP 3 — GET RESULT
            # ------------------------------------------------

            result = execution.result

            # ------------------------------------------------
            # STEP 4 — GENERATE BUSINESS INSIGHT
            # ------------------------------------------------

            insight = ""

            with st.spinner(
                "💡 Generating business insight..."
            ):
                try:
                    insight = engine.generate_insights(
                        question,
                        result,
                    )

                except Exception as ex:
                    insight = (
                        "The analysis completed successfully, "
                        "but the AI insight could not be generated."
                    )

                    st.warning(
                        f"Insight generation failed: {ex}"
                    )

            # ------------------------------------------------
            # SAVE RESULT TO SESSION
            # ------------------------------------------------

            st.session_state["last_generation"] = generation
            st.session_state["last_result"] = result
            st.session_state["last_chart_hint"] = (
                generation.get("chart_hint", "auto")
            )
            st.session_state["last_insight"] = (
                insight or "No additional insight was generated."
            )

            _add_to_history(
                question,
                result,
                st.session_state["last_insight"],
            )

            st.success(
                "Analysis completed successfully."
            )


# ============================================================
# RESULTS
# ============================================================

if (
    st.session_state["last_result"] is not None
    and st.session_state["last_generation"] is not None
):

    result_col, insight_col = st.columns(
        [1.7, 1],
        gap="large",
    )

    # ========================================================
    # RESULT + VISUALIZATION
    # ========================================================

    with result_col:

        with st.container(border=True):
            st.subheader("📊 Analysis Result")

            result = st.session_state["last_result"]

            _render_result(result)

            # -----------------------------------------------
            # DOWNLOAD RESULT
            # -----------------------------------------------

            csv_data = _result_to_csv(result)

            if csv_data is not None:
                st.download_button(
                    label="⬇️ Download Result as CSV",
                    data=csv_data,
                    file_name="aidaass_analysis_result.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        # ----------------------------------------------------
        # VISUALIZATION
        # ----------------------------------------------------

        with st.container(border=True):
            st.subheader("📈 Visualization")

            try:
                fig, viz_message = generate_chart(
                    st.session_state["last_result"],
                    st.session_state["last_chart_hint"],
                )

                if fig is not None:
                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                    )
                else:
                    st.info(
                        viz_message
                        or "No suitable visualization could be generated."
                    )

            except Exception as ex:
                st.warning(
                    f"Visualization could not be generated: {ex}"
                )

                with st.expander(
                    "Show visualization details"
                ):
                    st.code(
                        traceback.format_exc(),
                        language="text",
                    )

    # ========================================================
    # INSIGHT + GENERATED CODE
    # ========================================================

    with insight_col:

        with st.container(border=True):
            st.subheader("💡 AI Insight")

            insight = st.session_state["last_insight"]

            if insight:
                st.success(insight)
            else:
                st.info(
                    "No insight was generated."
                )

        with st.container(border=True):
            st.subheader("🐍 Generated Code")

            generation = st.session_state[
                "last_generation"
            ]

            generated_code = generation.get(
                "code",
                "",
            )

            if generated_code:
                st.code(
                    generated_code,
                    language="python",
                )
            else:
                st.info(
                    "No generated code is available."
                )

            notes = generation.get(
                "notes",
                "",
            )

            if notes:
                st.caption(notes)

            chart_hint = generation.get(
                "chart_hint",
                "auto",
            )

            st.caption(
                f"Chart hint: `{chart_hint}`"
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AIDAASS • AI Data Analyst Assistant • "
    "AI-generated pandas code is executed through the "
    "project's safety layer."
)
