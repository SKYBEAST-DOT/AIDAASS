import json
import os
import re
from typing import Any, Dict, Optional

import pandas as pd

from utils import dataframe_context, result_preview_text

try:
    import google.generativeai as genai
except Exception:
    genai = None


class AIEngine:
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-1.5-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model_name = model_name
        self.model = None

        if self.api_key and genai is not None:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(self.model_name)
            except Exception:
                self.model = None

    @property
    def available(self) -> bool:
        return self.model is not None

    def _extract_json(self, text: str) -> Dict[str, Any]:
        text = text.strip()

        try:
            return json.loads(text)
        except Exception:
            pass

        fenced = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.S)
        if fenced:
            return json.loads(fenced.group(1))

        braces = re.search(r"(\{.*\})", text, flags=re.S)
        if braces:
            return json.loads(braces.group(1))

        raise ValueError("Model response was not valid JSON.")

    def generate_pandas_code(self, question: str, df: pd.DataFrame) -> Dict[str, str]:
        """Convert natural language question into safe pandas code contract."""

        if not self.available:
            return {
                "code": self._fallback_code(question),
                "chart_hint": "auto",
                "notes": "Using fallback template because Gemini is not configured.",
            }

        prompt = f"""
You are a Python data analyst assistant.
Given a pandas DataFrame named df, generate pandas code to answer the user's question.

Rules:
1) Output ONLY strict JSON with keys: code, chart_hint, notes.
2) Code must be valid Python.
3) Code must only use existing columns from schema.
4) Must assign final output to variable: result
5) Do not import modules.
6) Keep code concise and deterministic.
7) If question asks top/lowest, sort then head.
8) chart_hint must be one of: auto, bar, line, pie, none.

Data context:
{dataframe_context(df)}

User question:
{question}
""".strip()

        try:
            response = self.model.generate_content(prompt)
            parsed = self._extract_json(response.text)

            code = str(parsed.get("code", "")).strip()
            chart_hint = str(parsed.get("chart_hint", "auto")).strip().lower()
            notes = str(parsed.get("notes", "")).strip()

            if not code:
                raise ValueError("Gemini returned empty code.")

            if "result" not in code:
                code = f"{code}\nresult = None"

            if chart_hint not in {"auto", "bar", "line", "pie", "none"}:
                chart_hint = "auto"

            return {"code": code, "chart_hint": chart_hint, "notes": notes}

        except Exception as e:
            return {
                "code": self._fallback_code(question),
                "chart_hint": "auto",
                "notes": f"Fallback used due to model error: {str(e)}",
            }

    def generate_insights(self, question: str, result: Any) -> str:
        """Generate a short business insight from computed result."""

        preview = result_preview_text(result)

        if not self.available:
            return self._fallback_insight(result)

        prompt = f"""
Create a short, plain-English insight based on this analysis result.
- Maximum 2 sentences.
- Mention key trend/comparison.
- If data is insufficient, say so briefly.

Question: {question}
Result:
{preview}
""".strip()

        try:
            response = self.model.generate_content(prompt)
            text = (response.text or "").strip()
            return text if text else self._fallback_insight(result)
        except Exception:
            return self._fallback_insight(result)

    def _fallback_code(self, question: str) -> str:
        """Simple non-LLM fallback that still returns meaningful output."""

        q = question.lower()

        if "top" in q and ("revenue" in q or "sales" in q):
            return """
num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
if not num_cols:
    result = df.head(10)
else:
    metric = num_cols[0]
    result = df.sort_values(metric, ascending=False).head(5)
""".strip()

        if "trend" in q or "monthly" in q:
            return """
date_cols = [c for c in df.columns if 'date' in c.lower() or pd.api.types.is_datetime64_any_dtype(df[c])]
num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
if date_cols and num_cols:
    dcol = date_cols[0]
    mcol = num_cols[0]
    temp = df.copy()
    temp[dcol] = pd.to_datetime(temp[dcol], errors='coerce')
    temp = temp.dropna(subset=[dcol])
    temp['month'] = temp[dcol].dt.to_period('M').astype(str)
    result = temp.groupby('month', as_index=False)[mcol].sum().sort_values('month')
else:
    result = df.head(20)
""".strip()

        return "result = df.head(10)"

    def _fallback_insight(self, result: Any) -> str:
        if isinstance(result, pd.DataFrame):
            if result.empty:
                return "No matching records were found for this question."
            return f"The result returns {len(result)} rows and {len(result.columns)} columns."

        if isinstance(result, pd.Series):
            return f"The analysis produced {len(result)} aggregated values."

        return f"Computed result: {result}"