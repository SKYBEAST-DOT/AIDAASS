# AIDAASS — AI Data Analyst Assistant

> An AI-powered web application that lets you upload datasets, ask questions in plain English, and instantly get pandas analysis, interactive charts, and business insights — powered by Google Gemini.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.41%2B-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![Gemini AI](https://img.shields.io/badge/Gemini_AI-1.5_Flash-4285F4?logo=google)](https://ai.google.dev/)

---

## ✨ Features

| Feature | Description |
|---|---|
| 📁 **Dataset Upload** | Upload CSV or Excel (`.csv`, `.xlsx`, `.xls`) files |
| 🛍️ **Built-in Demo Data** | Choose from bundled Retail, Marketing, or Ecommerce datasets |
| 🔍 **Dataset Explorer** | Preview, schema, statistics, and correlation matrix tabs |
| 💬 **Natural Language Queries** | Ask questions like *"Show top 5 products by revenue"* |
| 🤖 **Gemini-Powered Analysis** | Converts your question into safe pandas code using Gemini AI |
| 📊 **Interactive Charts** | Auto-generates Plotly bar, line, pie, histogram, and scatter charts |
| 💡 **Business Insights** | AI-written plain-English summary of every result |
| 🔒 **Safe Execution** | AST-based sandbox blocks all dangerous operations before running AI code |

---

## 📸 App Layout

```
┌─────────────────────────────────────────────────────────────┐
│  Sidebar: API Key · Data Source                             │
├──────────────────────────┬──────────────────────────────────┤
│  Dataset Overview        │  AI Analyst                      │
│  Preview · Schema        │  Quick-question buttons          │
│  Statistics · Correlation│  Natural language input          │
│                          │  Run Analysis button             │
├──────────────────────────┴──────────────────────────────────┤
│  Analysis Result  │  Visualization  │  AI Insight / Code    │
└─────────────────────────────────────────────────────────────┘
```

---

## 🗂️ Project Structure

```text
AIDAASS/
└── ai-data-analyst-assistant/
    ├── app.py                        # Streamlit UI & main entry point
    ├── ai_engine.py                  # Gemini API integration & fallback logic
    ├── data_profiler.py              # Dataset statistics & column profiling
    ├── query_executor.py             # AST validation & safe code execution
    ├── visualization_engine.py       # Plotly chart generation
    ├── visualization.py              # Visualization helpers
    ├── utils.py                      # Shared utilities (load, execute, preview)
    ├── requirements.txt
    └── data/
        └── sample_ecommerce_dataset.csv
```

---

## 🚀 Quick Start

### 1 — Clone the repository

```bash
git clone https://github.com/SKYBEAST-DOT/AIDAASS.git
cd AIDAASS/ai-data-analyst-assistant
```

### 2 — Create a virtual environment

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### 4 — Add your Gemini API key

**Option A — environment variable (recommended for local dev):**

```bash
export GEMINI_API_KEY="your_api_key_here"
```

**Option B — Streamlit secrets file (works with Streamlit Cloud too):**

Create `.streamlit/secrets.toml` inside `ai-data-analyst-assistant/`:

```toml
GEMINI_API_KEY = "your_api_key_here"
```

You can also paste the key directly in the app's sidebar at runtime.

### 5 — Run the app

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**.

---

## ☁️ Deploy to Streamlit Cloud

1. Push the repo to GitHub (already done).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Set the main file path to `ai-data-analyst-assistant/app.py`.
4. Under **Advanced settings → Secrets**, add:
   ```toml
   GEMINI_API_KEY = "your_api_key_here"
   ```
5. Click **Deploy** — the app will be live in seconds.

---

## 💬 Example Questions

```
Show top 5 products by revenue
Which region has the highest sales?
Show monthly profit trend
Compare average order value by segment
Which category contributes most profit?
```

---

## 📦 Dependencies

| Package | Version | Purpose |
|---|---|---|
| `streamlit` | ≥ 1.41 | Web UI framework |
| `pandas` | ≥ 2.2 | Data manipulation |
| `plotly` | ≥ 5.24 | Interactive charts |
| `google-generativeai` | ≥ 0.8.3 | Gemini AI integration |
| `openpyxl` | ≥ 3.1.5 | Excel file support |
| `numpy` | ≥ 1.26 | Numerical utilities |

---

## 🔒 Safety Design

All AI-generated code passes through a strict two-layer validation pipeline before execution:

1. **String-pattern checks** — blocks `import`, `open(`, `eval(`, `exec(`, `__`, `lambda`, `class`, `def`.
2. **AST-based validation** — walks the Python syntax tree and rejects:
   - `import` / `from` statements
   - Function & class definitions
   - Loops (`for`, `while`) and `with` blocks
   - `try` / `raise` / `assert`
   - Private/magic attribute access (`__attr__`)
   - Dangerous built-ins (`eval`, `exec`, `getattr`, `globals`, `open`, …)

The generated code is only permitted to operate on the provided `df` DataFrame and **must** assign its final output to the `result` variable.

---

## 🗝️ Getting a Gemini API Key

1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey).
2. Sign in with a Google account.
3. Click **Create API key**.
4. Copy the key and use it in one of the setup options above.

> **Free tier** is sufficient for personal and demo use.

---

## 🤝 Contributing

Pull requests and issues are welcome. Please open an issue first to discuss significant changes.

---

## 📄 License

This project is open-source. See the repository for license details.