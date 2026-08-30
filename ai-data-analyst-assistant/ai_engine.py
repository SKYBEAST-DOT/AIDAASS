"""
AI engine for AIDAASS.

Responsibilities:
- Connect to Google Gemini using the current Google GenAI SDK.
- Convert natural-language questions into pandas analysis code.
- Force a structured JSON response.
- Validate generated code before returning it.
- Retry once when Gemini produces invalid analysis code.
- Provide deterministic fallback analysis when Gemini is unavailable.
- Generate concise business insights from computed results.

The generated code is NOT executed in this module.

Execution is handled by:
    query_executor.py
        -> utils.py
"""

import json
import os
import re
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from query_executor import validate_generated_code
from utils import dataframe_context, result_preview_text


# ============================================================
# GOOGLE GENAI SDK
# ============================================================

try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None
    types = None


# ============================================================
# CONSTANTS
# ============================================================

DEFAULT_MODEL = "gemini-3.7-flash"

VALID_CHART_HINTS = {
    "auto",
    "bar",
    "line",
    "pie",
    "histogram",
    "scatter",
    "none",
}

MAX_QUESTION_LENGTH = 2_000
MAX_CONTEXT_LENGTH = 30_000
MAX_INSIGHT_PREVIEW_LENGTH = 8_000

MAX_GENERATION_ATTEMPTS = 2


# ============================================================
# AI ENGINE
# ============================================================


class AIEngine:
    """
    Gemini-powered analysis engine for AIDAASS.

    The engine generates code but does not execute generated code.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = DEFAULT_MODEL,
    ):
        self.api_key = (
            api_key
            or os.getenv("GEMINI_API_KEY", "")
        ).strip()

        self.model_name = (
            model_name or DEFAULT_MODEL
        ).strip()

        self.client = None

        if self.api_key and genai is not None:

            try:
                self.client = genai.Client(
                    api_key=self.api_key
                )

            except Exception:
                self.client = None

    # ========================================================
    # AVAILABILITY
    # ========================================================

    @property
    def available(self) -> bool:
        """Return True when Gemini is configured and usable."""

        return (
            self.client is not None
            and bool(self.api_key)
        )

    # ========================================================
    # QUESTION VALIDATION
    # ========================================================

    @staticmethod
    def _validate_question(
        question: str,
    ) -> str:
        """Validate and normalize the user's question."""

        if not isinstance(question, str):
            raise ValueError(
                "Question must be a string."
            )

        question = question.strip()

        if not question:
            raise ValueError(
                "Please enter a question."
            )

        if len(question) > MAX_QUESTION_LENGTH:
            raise ValueError(
                "Question is too long. "
                f"Maximum length is {MAX_QUESTION_LENGTH:,} characters."
            )

        return question

    # ========================================================
    # JSON EXTRACTION
    # ========================================================

    @staticmethod
    def _extract_json(
        text: str,
    ) -> Dict[str, Any]:
        """
        Extract a JSON object from Gemini's response.

        Handles:
        - plain JSON
        - ```json fenced JSON
        - accidental surrounding text
        """

        if not isinstance(text, str):
            raise ValueError(
                "Gemini response was not text."
            )

        text = text.strip()

        if not text:
            raise ValueError(
                "Gemini returned an empty response."
            )

        # ----------------------------------------------------
        # Direct JSON
        # ----------------------------------------------------

        try:
            parsed = json.loads(text)

            if isinstance(parsed, dict):
                return parsed

        except json.JSONDecodeError:
            pass

        # ----------------------------------------------------
        # Markdown JSON fence
        # ----------------------------------------------------

        fenced = re.search(
            r"```(?:json)?\s*(\{.*?\})\s*```",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )

        if fenced:

            try:
                parsed = json.loads(
                    fenced.group(1)
                )

                if isinstance(parsed, dict):
                    return parsed

            except json.JSONDecodeError:
                pass

        # ----------------------------------------------------
        # Locate first JSON object
        # ----------------------------------------------------

        start = text.find("{")
        end = text.rfind("}")

        if start != -1 and end > start:

            candidate = text[
                start : end + 1
            ]

            try:
                parsed = json.loads(candidate)

                if isinstance(parsed, dict):
                    return parsed

            except json.JSONDecodeError:
                pass

        raise ValueError(
            "Gemini response was not valid JSON."
        )

    # ========================================================
    # RESPONSE TEXT
    # ========================================================

    @staticmethod
    def _response_text(
        response: Any,
    ) -> str:
        """Safely extract text from a Gemini response."""

        text = getattr(
            response,
            "text",
            None,
        )

        if isinstance(text, str):
            return text.strip()

        # Defensive fallback for SDK response variations.
        try:
            candidates = getattr(
                response,
                "candidates",
                None,
            )

            if candidates:

                first_candidate = candidates[0]

                content = getattr(
                    first_candidate,
                    "content",
                    None,
                )

                parts = getattr(
                    content,
                    "parts",
                    None,
                )

                if parts:

                    texts = []

                    for part in parts:

                        part_text = getattr(
                            part,
                            "text",
                            None,
                        )

                        if part_text:
                            texts.append(
                                str(part_text)
                            )

                    if texts:
                        return "\n".join(
                            texts
                        ).strip()

        except Exception:
            pass

        return ""

    # ========================================================
    # NORMALIZE MODEL RESPONSE
    # ========================================================

    @staticmethod
    def _normalize_generation(
        parsed: Dict[str, Any],
    ) -> Dict[str, str]:
        """Normalize Gemini's structured generation response."""

        if not isinstance(parsed, dict):
            raise ValueError(
                "Gemini returned an invalid response object."
            )

        code = str(
            parsed.get("code", "")
        ).strip()

        chart_hint = str(
            parsed.get(
                "chart_hint",
                "auto",
            )
        ).strip().lower()

        notes = str(
            parsed.get(
                "notes",
                "",
            )
        ).strip()

        if not code:
            raise ValueError(
                "Gemini returned empty analysis code."
            )

        if chart_hint not in VALID_CHART_HINTS:
            chart_hint = "auto"

        return {
            "code": code,
            "chart_hint": chart_hint,
            "notes": notes,
        }

    # ========================================================
    # VALIDATE GENERATED CODE
    # ========================================================

    @staticmethod
    def _validate_generated_code(
        code: str,
    ) -> Tuple[bool, str]:
        """
        Validate generated code using the same validator
        used by the execution layer.
        """

        validation = validate_generated_code(
            code
        )

        if validation.ok:
            return True, ""

        return False, validation.error

    # ========================================================
    # GENERATION PROMPT
    # ========================================================

    def _build_code_prompt(
        self,
        question: str,
        df: pd.DataFrame,
        repair_error: Optional[str] = None,
    ) -> str:
        """Build the Gemini prompt for pandas-code generation."""

        context = dataframe_context(
            df,
            sample_rows=5,
        )

        if len(context) > MAX_CONTEXT_LENGTH:
            context = context[
                :MAX_CONTEXT_LENGTH
            ]

        repair_instruction = ""

        if repair_error:

            repair_instruction = f"""
The previous generated code failed validation.

Validation error:
{repair_error}

Generate a completely corrected replacement.
Do not repeat the blocked operation.
"""

        return f"""
You are the analysis engine for AIDAASS,
an AI-powered data analyst application.

Your task is to translate the user's natural-language
question into safe pandas analysis code.

The input DataFrame is named:

df

The final answer MUST be assigned to:

result

IMPORTANT CODE RULES:

1. Return ONLY valid JSON.
2. JSON must contain exactly these keys:
   "code", "chart_hint", "notes"
3. "code" must contain Python/pandas analysis code.
4. The final analysis output MUST be assigned to `result`.
5. Do NOT use imports.
6. Do NOT use open(), exec(), eval(), compile(), globals(),
   locals(), getattr(), setattr(), or similar operations.
7. Do NOT access files, databases, networks, operating-system
   functions, subprocesses, sockets, or external resources.
8. Do NOT use private or dunder attributes.
9. Do NOT define functions or classes.
10. Do NOT use loops or comprehensions.
11. Do NOT write files.
12. Do NOT modify external state.
13. Work only with the provided DataFrame `df`.
14. Use only columns that actually exist in the supplied schema.
15. Never invent column names.
16. If a column name contains spaces or special characters,
    access it with df["exact column name"].
17. Prefer concise, deterministic pandas operations.
18. For top-N questions, sort the relevant metric and use head().
19. For bottom-N questions, sort ascending and use head().
20. For grouping questions, use groupby().
21. For trends, use an existing datetime column when available.
22. If a date column is not already datetime, use pd.to_datetime().
23. Avoid unnecessary transformations.
24. Return a meaningful result even when the question is ambiguous.
25. chart_hint must be one of:
    auto, bar, line, pie, histogram, scatter, none

IMPORTANT:
The application will independently validate the generated code
before execution. Generate code that follows these rules exactly.

DATASET CONTEXT:
{context}

USER QUESTION:
{question}

{repair_instruction}

Return ONLY JSON.
""".strip()

    # ========================================================
    # GEMINI REQUEST
    # ========================================================

    def _request_generation(
        self,
        prompt: str,
    ) -> Dict[str, str]:
        """Send a generation request to Gemini."""

        if not self.available:
            raise RuntimeError(
                "Gemini is not configured."
            )

        if types is None:
            raise RuntimeError(
                "The Google GenAI SDK is not installed correctly."
            )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=2_000,
                response_mime_type="application/json",
            ),
        )

        text = self._response_text(
            response
        )

        if not text:
            raise ValueError(
                "Gemini returned no text."
            )

        parsed = self._extract_json(
            text
        )

        return self._normalize_generation(
            parsed
        )

    # ========================================================
    # GENERATE PANDAS CODE
    # ========================================================

    def generate_pandas_code(
        self,
        question: str,
        df: pd.DataFrame,
    ) -> Dict[str, str]:
        """
        Convert a natural-language question into validated
        pandas analysis code.

        Returns:
            {
                "code": "...",
                "chart_hint": "bar",
                "notes": "..."
            }
        """

        question = self._validate_question(
            question
        )

        if not isinstance(
            df,
            pd.DataFrame,
        ):
            raise ValueError(
                "AIEngine expected a pandas DataFrame."
            )

        if df.empty:
            raise ValueError(
                "Cannot generate analysis for an empty dataset."
            )

        # ----------------------------------------------------
        # Gemini unavailable
        # ----------------------------------------------------

        if not self.available:

            fallback_code = (
                self._fallback_code(
                    question,
                    df,
                )
            )

            return {
                "code": fallback_code,
                "chart_hint": "auto",
                "notes": (
                    "Gemini is unavailable. "
                    "A deterministic fallback analysis was used."
                ),
            }

        # ----------------------------------------------------
        # Gemini generation + validation
        # ----------------------------------------------------

        last_error = ""

        for attempt in range(
            MAX_GENERATION_ATTEMPTS
        ):

            try:

                prompt = self._build_code_prompt(
                    question=question,
                    df=df,
                    repair_error=(
                        last_error
                        if attempt > 0
                        else None
                    ),
                )

                generation = self._request_generation(
                    prompt
                )

                code = generation["code"]

                is_valid, validation_error = (
                    self._validate_generated_code(
                        code
                    )
                )

                if is_valid:

                    generation["notes"] = (
                        generation.get(
                            "notes",
                            "",
                        )
                        or ""
                    ).strip()

                    return generation

                last_error = validation_error

            except Exception as ex:
                last_error = str(ex)

        # ----------------------------------------------------
        # Safe deterministic fallback
        # ----------------------------------------------------

        fallback_code = self._fallback_code(
            question,
            df,
        )

        fallback_valid, fallback_error = (
            self._validate_generated_code(
                fallback_code
            )
        )

        if not fallback_valid:

            raise RuntimeError(
                "Gemini generation failed and the fallback "
                f"analysis could not pass validation: "
                f"{fallback_error}"
            )

        return {
            "code": fallback_code,
            "chart_hint": "auto",
            "notes": (
                "Gemini output could not be safely validated. "
                f"Fallback analysis was used. "
                f"Reason: {last_error}"
            ),
        }

    # ========================================================
    # GENERATE INSIGHTS
    # ========================================================

    def generate_insights(
        self,
        question: str,
        result: Any,
    ) -> str:
        """
        Generate a short business insight from an analysis result.
        """

        question = self._validate_question(
            question
        )

        preview = result_preview_text(
            result,
            max_rows=8,
        )

        if len(preview) > MAX_INSIGHT_PREVIEW_LENGTH:
            preview = preview[
                :MAX_INSIGHT_PREVIEW_LENGTH
            ]

        # ----------------------------------------------------
        # Gemini unavailable
        # ----------------------------------------------------

        if not self.available:
            return self._fallback_insight(
                result
            )

        prompt = f"""
You are a business data analyst.

Create a concise, accurate insight from the computed
analysis result below.

Rules:
1. Maximum 2 sentences.
2. Use only information supported by the result.
3. Mention the most important trend, ranking, comparison,
   or value.
4. Do not invent numbers.
5. Do not claim causation unless the result proves it.
6. If the result is insufficient for a meaningful conclusion,
   say that briefly.
7. Use plain professional English.
8. Do not mention that you are an AI.
9. Do not repeat the entire table.

USER QUESTION:
{question}

COMPUTED RESULT:
{preview}
""".strip()

        try:

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=300,
                ),
            )

            text = self._response_text(
                response
            )

            if text:
                return text

        except Exception:
            pass

        return self._fallback_insight(
            result
        )

    # ========================================================
    # FALLBACK CODE
    # ========================================================

    def _fallback_code(
        self,
        question: str,
        df: pd.DataFrame,
    ) -> str:
        """
        Generate deterministic pandas code when Gemini is
        unavailable or returns unsafe/invalid code.

        The fallback is intentionally conservative.
        """

        q = question.lower()

        numeric_columns = [
            column
            for column in df.columns
            if pd.api.types.is_numeric_dtype(
                df[column]
            )
        ]

        datetime_columns = [
            column
            for column in df.columns
            if pd.api.types.is_datetime64_any_dtype(
                df[column]
            )
            or "date" in str(column).lower()
            or "time" in str(column).lower()
        ]

        # ----------------------------------------------------
        # Top / highest revenue or sales
        # ----------------------------------------------------

        if (
            "top" in q
            and (
                "revenue" in q
                or "sales" in q
            )
            and numeric_columns
        ):

            metric = self._choose_metric_column(
                numeric_columns,
                q,
                preferred_terms=[
                    "revenue",
                    "sales",
                    "amount",
                    "profit",
                ],
            )

            return f"""
result = (
    df.sort_values(
        "{metric}",
        ascending=False
    )
    .head(5)
)
""".strip()

        # ----------------------------------------------------
        # Lowest / bottom
        # ----------------------------------------------------

        if (
            (
                "bottom" in q
                or "lowest" in q
                or "least" in q
            )
            and numeric_columns
        ):

            metric = self._choose_metric_column(
                numeric_columns,
                q,
                preferred_terms=[
                    "revenue",
                    "sales",
                    "profit",
                    "amount",
                ],
            )

            return f"""
result = (
    df.sort_values(
        "{metric}",
        ascending=True
    )
    .head(5)
)
""".strip()

        # ----------------------------------------------------
        # Monthly / trend
        # ----------------------------------------------------

        if (
            "trend" in q
            or "monthly" in q
            or "over time" in q
        ):

            if datetime_columns and numeric_columns:

                date_column = (
                    datetime_columns[0]
                )

                metric = (
                    self._choose_metric_column(
                        numeric_columns,
                        q,
                        preferred_terms=[
                            "revenue",
                            "sales",
                            "profit",
                            "amount",
                        ],
                    )
                )

                return f"""
temp = df.copy()

temp["__aidaass_date"] = pd.to_datetime(
    temp["{date_column}"],
    errors="coerce"
)

temp = temp.dropna(
    subset=["__aidaass_date"]
)

temp["__aidaass_month"] = (
    temp["__aidaass_date"]
    .dt.to_period("M")
    .astype(str)
)

result = (
    temp.groupby(
        "__aidaass_month",
        as_index=False
    )["{metric}"]
    .sum()
    .sort_values(
        "__aidaass_month"
    )
)
""".strip()

        # ----------------------------------------------------
        # Highest / maximum
        # ----------------------------------------------------

        if (
            (
                "highest" in q
                or "maximum" in q
                or "max" in q
            )
            and numeric_columns
        ):

            metric = (
                self._choose_metric_column(
                    numeric_columns,
                    q,
                    preferred_terms=[
                        "revenue",
                        "sales",
                        "profit",
                        "amount",
                    ],
                )
            )

            return f"""
result = (
    df.sort_values(
        "{metric}",
        ascending=False
    )
    .head(1)
)
""".strip()

        # ----------------------------------------------------
        # Lowest / minimum
        # ----------------------------------------------------

        if (
            (
                "lowest" in q
                or "minimum" in q
                or "min" in q
            )
            and numeric_columns
        ):

            metric = (
                self._choose_metric_column(
                    numeric_columns,
                    q,
                    preferred_terms=[
                        "revenue",
                        "sales",
                        "profit",
                        "amount",
                    ],
                )
            )

            return f"""
result = (
    df.sort_values(
        "{metric}",
        ascending=True
    )
    .head(1)
)
""".strip()

        # ----------------------------------------------------
        # Generic numeric summary
        # ----------------------------------------------------

        if numeric_columns:

            metric = numeric_columns[0]

            return f"""
result = (
    df[
        ["{metric}"]
    ]
    .describe()
    .to_frame()
)
""".strip()

        # ----------------------------------------------------
        # No numeric columns
        # ----------------------------------------------------

        return """
result = df.head(10)
""".strip()

    # ========================================================
    # METRIC SELECTION
    # ========================================================

    @staticmethod
    def _choose_metric_column(
        numeric_columns,
        question: str,
        preferred_terms,
    ) -> str:
        """
        Choose the most relevant numeric column using
        conservative name matching.
        """

        question_lower = question.lower()

        # First try explicit metric terms in the question.
        for term in preferred_terms:

            if term not in question_lower:
                continue

            for column in numeric_columns:

                column_lower = str(
                    column
                ).lower()

                if term in column_lower:
                    return column

        # Otherwise try preferred terms against column names.
        for term in preferred_terms:

            for column in numeric_columns:

                if term in str(
                    column
                ).lower():
                    return column

        return numeric_columns[0]

    # ========================================================
    # FALLBACK INSIGHT
    # ========================================================

    def _fallback_insight(
        self,
        result: Any,
    ) -> str:
        """Generate a deterministic insight when Gemini fails."""

        if isinstance(
            result,
            pd.DataFrame,
        ):

            if result.empty:
                return (
                    "No matching records were found "
                    "for this analysis."
                )

            return (
                f"The analysis returned "
                f"{len(result):,} rows across "
                f"{len(result.columns):,} columns."
            )

        if isinstance(
            result,
            pd.Series,
        ):

            if result.empty:
                return (
                    "The analysis returned no values."
                )

            return (
                f"The analysis produced "
                f"{len(result):,} aggregated values."
            )

        if isinstance(
            result,
            pd.Index,
        ):

            return (
                f"The analysis produced "
                f"{len(result):,} indexed values."
            )

        return (
            f"The analysis produced a "
            f"{type(result).__name__} result."
        )
