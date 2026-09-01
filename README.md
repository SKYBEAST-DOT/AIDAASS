# AIDAASS (AI Data Analyst Assistant)

Deployable Streamlit app for AI-assisted dataset analysis.

## Project Location

Application source is in:

`/home/runner/work/AIDAASS/AIDAASS/ai-data-analyst-assistant`

## Quick Start (Local)

```bash
cd /home/runner/work/AIDAASS/AIDAASS
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run ai-data-analyst-assistant/app.py
```

## Configuration

- Optional env var: `GEMINI_API_KEY`
- Example file: `.env.example`
- Optional Streamlit secret: `.streamlit/secrets.toml` with:

```toml
GEMINI_API_KEY = "your_api_key_here"
```

If no Gemini key is provided, the app still runs in fallback analysis mode.

## Streamlit Deployment

1. Push this repository to GitHub.
2. In Streamlit Community Cloud, create app from this repo.
3. Set **Main file path** to `ai-data-analyst-assistant/app.py`.
4. (Optional) Add `GEMINI_API_KEY` in app Secrets for full AI generation.
5. Deploy.

## Data Path

Bundled sample dataset is loaded from:

`ai-data-analyst-assistant/data/sample_ecommerce_dataset.csv`

## Notes

- Root `requirements.txt` references `ai-data-analyst-assistant/requirements.txt` for Streamlit compatibility.
- See `ai-data-analyst-assistant/README.md` for feature-level details.