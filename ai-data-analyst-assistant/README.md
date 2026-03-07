# AI Data Analyst Assistant

AI-powered web app that lets users upload CSV/Excel files, ask natural language questions, run pandas analysis safely, auto-generate charts, and get business insights.

## Features

- Upload CSV or Excel datasets
- Use built-in example datasets for instant exploration
- Includes a bundled ecommerce CSV sample for realistic testing
- Dataset preview with automatic column detection
- Modern dashboard with KPI cards, schema tab, and quick question buttons
- Natural language query input (e.g., "Show top 5 products by revenue")
- Gemini-powered conversion from question to pandas operations
- Safe execution layer for generated code
- Improved automatic Plotly visualization (bar, line, pie, histogram, scatter)
- AI-generated insight summary

## Project Structure

```text
ai-data-analyst-assistant/
├── app.py
├── ai_engine.py
├── data_profiler.py
├── visualization.py
├── visualization_engine.py
├── query_executor.py
├── requirements.txt
├── utils.py
├── data/
│   └── sample_ecommerce_dataset.csv
└── README.md
```

## Setup

1. Navigate to the project folder:

   ```bash
   cd ai-data-analyst-assistant
   ```

2. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Configure Gemini API key:

   ```bash
   export GEMINI_API_KEY="your_api_key_here"
   ```

   You can also enter the API key from the Streamlit sidebar.

### Store API key inside workspace

Create/update `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "your_api_key_here"
```

The app automatically uses this key when the sidebar key input is empty.

## Run

```bash
streamlit run app.py
```

## Notes for Streamlit Cloud

- Add `GEMINI_API_KEY` to app Secrets.
- Keep file sizes moderate for faster preview and analysis.
- If AI returns invalid code, rephrase using explicit column names.
- You can run the app without uploading files by choosing an example dataset in the sidebar.

## Example Questions

- Show top 5 products by revenue
- Which region has highest sales?
- Show monthly profit trend
- Compare average order value by segment

## Bundled Sample Dataset

- Path: `data/sample_ecommerce_dataset.csv`
- Use it directly by selecting **Example dataset → Ecommerce Orders Sample (CSV)** in the sidebar.
- You can also upload the same file manually through the upload control.

## Safety Design

- AI code runs through AST-based safety checks.
- Dangerous operations (`import`, file access, system calls, dunder access) are blocked.
- The generated code must write final output to `result`.
