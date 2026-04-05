# GAISTUDASS — Generative AI Student Assistant

An AI-powered study companion that helps students understand topics, generate quizzes, summarise lecture notes, and get instant homework help — all through a conversational Streamlit interface backed by Google Gemini.

---

## Features

- **Natural Language Q&A** — ask any study question and receive a detailed, plain-English explanation
- **Document Summariser** — upload PDF or text lecture notes and get concise, structured summaries
- **Quiz Generator** — auto-generate multiple-choice or short-answer questions from any topic or uploaded material
- **Flashcard Creator** — produce revision flashcards from notes or a topic description
- **Code Explainer** — paste code snippets and have the AI walk through them line by line
- **Citation Helper** — get guidance on how to cite sources in APA, MLA, or Chicago format
- **Streamed Responses** — answers stream in real time for a smooth, chat-like experience
- **Conversation History** — session-scoped memory keeps context across follow-up questions

---

## Project Structure

```text
gaistudass/
├── app.py                  # Streamlit entry point
├── ai_engine.py            # Gemini API integration and prompt logic
├── summariser.py           # Document summarisation helpers
├── quiz_generator.py       # Quiz and flashcard generation
├── utils.py                # Shared utilities (file loading, text chunking)
├── requirements.txt
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.10 or higher
- A [Google Gemini API key](https://aistudio.google.com/app/apikey)

### 1. Clone the repository

```bash
git clone https://github.com/SKYBEAST-DOT/GAISTUDASS.git
cd GAISTUDASS
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your Gemini API key

```bash
export GEMINI_API_KEY="your_api_key_here"
```

Alternatively, store it as a Streamlit secret so you do not have to export it every session:

```toml
# .streamlit/secrets.toml
GEMINI_API_KEY = "your_api_key_here"
```

---

## Run

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501` by default.

---

## Usage

| Task | How to use |
|------|-----------|
| Ask a question | Type your question in the chat box and press **Send** |
| Summarise notes | Switch to the **Summarise** tab, upload a `.pdf` or `.txt` file, and click **Summarise** |
| Generate a quiz | Go to the **Quiz** tab, enter a topic or upload material, choose question count and type, then click **Generate Quiz** |
| Create flashcards | Open the **Flashcards** tab, describe the topic, and click **Create Flashcards** |
| Explain code | Paste a code snippet in the **Code** tab and click **Explain** |

---

## Deploying to Streamlit Cloud

1. Push the repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect the repo.
3. Add `GEMINI_API_KEY` under **App Settings → Secrets**.
4. Click **Deploy**.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Web UI framework |
| `google-generativeai` | Google Gemini API client |
| `pypdf` | PDF text extraction |
| `python-dotenv` | Local `.env` file support |

Install all at once:

```bash
pip install streamlit google-generativeai pypdf python-dotenv
```

---

## Configuration Reference

| Setting | Where | Description |
|---------|-------|-------------|
| `GEMINI_API_KEY` | Environment variable or `.streamlit/secrets.toml` | Your Gemini API key |
| `GEMINI_MODEL` | Sidebar or `secrets.toml` | Model name (default: `gemini-1.5-flash`) |
| `MAX_UPLOAD_MB` | `utils.py` | Maximum file upload size in MB (default: `10`) |

---

## Safety & Privacy

- No uploaded files or conversation data are stored on any server; everything is processed in-memory for the duration of the session.
- Gemini API calls are subject to [Google's Generative AI usage policies](https://policies.google.com/terms/generative-ai/use-policy).
- The app does not collect or transmit any personally identifiable information.

---

## Contributing

1. Fork the repository and create a feature branch (`git checkout -b feature/my-feature`).
2. Commit your changes with a clear message.
3. Open a pull request against `main` describing what you added or fixed.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Related Projects

- [AIDAASS](https://github.com/SKYBEAST-DOT/AIDAASS) — AI Data Analyst Assistant: upload CSV/Excel datasets, ask natural language questions, and get AI-generated charts and business insights.