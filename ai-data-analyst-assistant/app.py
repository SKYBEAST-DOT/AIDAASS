import traceback
from typing import Any

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


st.set_page_config(page_title="AI Data Analyst Assistant", page_icon="📊", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.2rem; padding-bottom: 1.2rem; max-width: 1300px;}
    .hero-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border-radius: 14px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.9rem;
        border: 1px solid #334155;
    }
    .hero-title {font-size: 1.55rem; font-weight: 700; color: #f8fafc; margin-bottom: 0.2rem;}
    .hero-sub {font-size: 0.95rem; color: #cbd5e1;}
    .pill {
        display: inline-block;
        padding: 0.22rem 0.62rem;
        border-radius: 999px;
        border: 1px solid #334155;
        background: #0b1220;
        color: #cbd5e1;
        font-size: 0.78rem;
        margin-bottom: 0.55rem;
    }
    .stTabs [data-baseweb="tab-list"] {gap: 0.35rem;}
    .stTabs [data-baseweb="tab"] {border-radius: 8px;}
    .caption-soft {color: #94a3b8; font-size: 0.85rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">📊 AI Data Analyst Assistant</div>
        <div class="hero-sub">Upload your dataset or use a built-in demo, ask questions in plain English, and get AI analysis + interactive charts instantly.</div>
    </div>
    """,
    unsafe_allow_html=True,
)


def _render_dataset_preview(df: pd.DataFrame) -> None:
    profile = build_data_profile(df)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.metric("Rows", f"{profile['rows']:,}")
    with col2:
        st.metric("Columns", f"{profile['columns']:,}")
    with col3:
        st.metric("Numeric Columns", f"{profile['numeric_columns']:,}")

    tab1, tab2, tab3, tab4 = st.tabs(["Preview", "Schema", "Statistics", "Correlation"])
    with tab1:
        st.dataframe(df.head(50), use_container_width=True, height=360)
    with tab2:
        st.dataframe(profile["column_profile"], use_container_width=True, height=360)
    with tab3:
        stats = profile["basic_stats"]
        if stats.empty:
            st.info("No numeric columns available for basic statistics.")
        else:
            st.dataframe(stats, use_container_width=True, height=360)
    with tab4:
        corr = profile["correlation_matrix"]
        if corr.empty:
            st.info("At least two numeric columns are required for correlation analysis.")
        else:
            st.dataframe(corr, use_container_width=True, height=360)


def _quick_question_buttons() -> str:
    st.markdown('<div class="caption-soft">Quick question starters</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    prompts = [
        "Show top 5 products by revenue",
        "Which region has the highest sales?",
        "Show monthly profit trend",
        "Which category contributes most profit?",
    ]

    chosen = ""
    for idx, prompt in enumerate(prompts):
        with cols[idx]:
            if st.button(prompt, use_container_width=True, key=f"quick_prompt_{idx}"):
                chosen = prompt
    return chosen


def _render_result(result: Any) -> None:
    if isinstance(result, pd.DataFrame):
        st.dataframe(result, use_container_width=True, height=360)
        return
    if isinstance(result, pd.Series):
        st.dataframe(result.to_frame(name="value"), use_container_width=True, height=360)
        return
    st.write(result)


def _render_sidebar_dataset_summary(df: pd.DataFrame) -> None:
    st.divider()
    st.subheader("Dataset Summary")
    numeric_count = sum(pd.api.types.is_numeric_dtype(df[c]) for c in df.columns)
    datetime_count = sum(pd.api.types.is_datetime64_any_dtype(df[c]) for c in df.columns)
    st.caption(f"Rows: {df.shape[0]:,}")
    st.caption(f"Columns: {df.shape[1]:,}")
    st.caption(f"Numeric: {numeric_count:,} | Datetime: {datetime_count:,}")
    st.caption("Top columns")
    st.write(df.columns[:8].tolist())


if "question_value" not in st.session_state:
    st.session_state["question_value"] = ""
if "last_generation" not in st.session_state:
    st.session_state["last_generation"] = None
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None
if "last_insight" not in st.session_state:
    st.session_state["last_insight"] = ""
if "last_chart_hint" not in st.session_state:
    st.session_state["last_chart_hint"] = "auto"
if "last_error" not in st.session_state:
    st.session_state["last_error"] = ""

workspace_api_key = st.secrets.get("GEMINI_API_KEY", "")


with st.sidebar:
    st.subheader("Configuration")
    api_key = st.text_input("Gemini API Key", type="password")
    if workspace_api_key:
        st.caption("Using workspace secret when input is empty.")
    model_name = st.text_input("Gemini Model", value="gemini-1.5-flash")

    st.divider()
    st.subheader("Data Source")
    source_type = st.radio("Choose source", options=["Upload file", "Example dataset"], index=0)

    selected_example = None
    if source_type == "Example dataset":
        selected_example = st.selectbox(
            "Select demo dataset",
            options=["Retail Sales Demo", "Marketing Campaign Demo", "Ecommerce Orders Sample (CSV)"],
            index=0,
        )

uploaded_file = None
if source_type == "Upload file":
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])

df = None
dataset_label = ""

if source_type == "Upload file" and uploaded_file is not None:
    df = load_dataset(uploaded_file)
    dataset_label = f"Uploaded file: {uploaded_file.name}"
elif source_type == "Example dataset" and selected_example:
    if selected_example == "Ecommerce Orders Sample (CSV)":
        df = load_packaged_ecommerce_sample()
    else:
        df = load_example_dataset(selected_example)
    dataset_label = f"Example dataset: {selected_example}"

if df is not None:
    try:
        with st.sidebar:
            _render_sidebar_dataset_summary(df)

        st.markdown(f'<span class="pill">{dataset_label}</span>', unsafe_allow_html=True)

        overview_col, assistant_col = st.columns([1.35, 1.65], gap="large")

        with overview_col:
            with st.container(border=True):
                st.subheader("Dataset Overview")
                _render_dataset_preview(df)

        with assistant_col:
            with st.container(border=True):
                st.subheader("AI Analyst")

                quick_choice = _quick_question_buttons()
                if quick_choice:
                    st.session_state["question_value"] = quick_choice

                st.text_input(
                    "Ask a question about your data",
                    placeholder="e.g., Show monthly sales trend by region",
                    key="question_value",
                )
                question = st.session_state["question_value"]

                run_clicked = st.button("Run Analysis", type="primary", use_container_width=True)

                if run_clicked:
                    if not question.strip():
                        st.warning("Please enter a question first.")
                    else:
                        st.session_state["last_error"] = ""
                        resolved_api_key = (api_key or "").strip() or workspace_api_key
                        engine = AIEngine(api_key=resolved_api_key or None, model_name=model_name)

                        with st.spinner("Generating analysis code with AI..."):
                            try:
                                generation = engine.generate_pandas_code(question, df)
                            except Exception as ex:
                                st.session_state["last_error"] = f"Failed to generate analysis code: {ex}"
                                generation = None

                        if generation is not None:
                            with st.spinner("Executing analysis..."):
                                execution = execute_ai_query(generation["code"], df)

                            if not execution.ok:
                                st.session_state["last_error"] = execution.error
                            else:
                                result = execution.result
                                with st.spinner("Generating AI insight..."):
                                    insight = engine.generate_insights(question, result)

                                st.session_state["last_generation"] = generation
                                st.session_state["last_result"] = result
                                st.session_state["last_chart_hint"] = generation.get("chart_hint", "auto")
                                st.session_state["last_insight"] = insight

                if st.session_state["last_error"]:
                    st.error(st.session_state["last_error"])
                    st.info("Try rephrasing with explicit column names or a simpler query.")

        if st.session_state["last_result"] is not None and st.session_state["last_generation"] is not None:
            result_col, insight_col = st.columns([1.7, 1], gap="large")

            with result_col:
                with st.container(border=True):
                    st.subheader("Analysis Result")
                    _render_result(st.session_state["last_result"])

                with st.container(border=True):
                    st.subheader("Visualization")
                    fig, viz_message = generate_chart(
                        st.session_state["last_result"],
                        st.session_state["last_chart_hint"],
                    )
                    if fig is not None:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.caption(viz_message)

            with insight_col:
                with st.container(border=True):
                    st.subheader("AI Insight")
                    st.success(st.session_state["last_insight"])

                with st.container(border=True):
                    st.subheader("Generated Code")
                    st.code(st.session_state["last_generation"]["code"], language="python")
                    if st.session_state["last_generation"].get("notes"):
                        st.caption(st.session_state["last_generation"]["notes"])

    except Exception as ex:
        st.error(f"Could not process dataset: {ex}")
        with st.expander("Show technical details"):
            st.code(traceback.format_exc(), language="text")
else:
    with st.container(border=True):
        st.subheader("Get Started")
        st.info("Upload a CSV/Excel file or choose an example dataset from the sidebar to begin.")
        tip1, tip2, tip3 = st.columns(3)
        tip1.metric("1", "Select Data Source")
        tip2.metric("2", "Ask Natural Language Question")
        tip3.metric("3", "Review Chart + Insight")