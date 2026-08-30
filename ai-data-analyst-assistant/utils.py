"""
Utility functions and secure execution layer for AIDAASS.

AIDAASS uses this module for:

1. CSV / Excel dataset loading
2. DataFrame context generation for the AI
3. Secure execution of AI-generated pandas code
4. Analysis-result preview generation
5. Built-in demo datasets
6. Packaged ecommerce sample loading

Security model:

    AI-generated code
            ↓
    query_executor.py
            ↓
    AST validation
            ↓
    safe_execute_pandas_code()
            ↓
    Restricted execution environment
            ↓
    ExecutionResult
"""

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


# ============================================================
# EXECUTION RESULT
# ============================================================


@dataclass
class ExecutionResult:
    """Standard result object used throughout AIDAASS."""

    ok: bool
    result: Any = None
    error: str = ""


class UnsafeCodeError(ValueError):
    """Raised when generated code violates the safety policy."""


# ============================================================
# EXECUTION LIMITS
# ============================================================

MAX_CODE_LENGTH = 12_000
MAX_AST_NODES = 500
MAX_STRING_LENGTH = 5_000


# ============================================================
# SAFE BUILTINS
# ============================================================

# These are the built-ins required for normal data-analysis
# operations. Dangerous introspection and execution functions
# are intentionally excluded.

SAFE_BUILTINS: Dict[str, Any] = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
}


# ============================================================
# RESTRICTED PANDAS NAMESPACE
# ============================================================

# We do NOT expose the entire pandas module.
#
# This prevents AI-generated code from directly accessing
# pandas I/O functions such as:
#
#     pd.read_csv()
#     pd.read_excel()
#     pd.read_pickle()
#     pd.read_sql()
#
# Analysis functions that are genuinely useful are exposed.

SAFE_PD_FUNCTIONS: Dict[str, Any] = {
    "to_datetime": pd.to_datetime,
    "to_numeric": pd.to_numeric,
    "isna": pd.isna,
    "isnull": pd.isnull,
    "notna": pd.notna,
    "notnull": pd.notnull,
    "concat": pd.concat,
    "merge": pd.merge,
    "crosstab": pd.crosstab,
}


class _SafePandasNamespace:
    """
    Restricted pandas namespace.

    Example:

        pd.to_datetime(df["date"])

        pd.to_numeric(
            df["sales"],
            errors="coerce"
        )

        pd.concat([df1, df2])
    """

    def __getattr__(self, name: str) -> Any:
        if name not in SAFE_PD_FUNCTIONS:
            raise UnsafeCodeError(
                f"Pandas operation '{name}' is not available "
                "in the secure execution environment."
            )

        return SAFE_PD_FUNCTIONS[name]


SAFE_PD = _SafePandasNamespace()


# ============================================================
# RESTRICTED NUMPY NAMESPACE
# ============================================================

# Only expose numerical functions useful for analysis.
#
# File-related NumPy operations such as np.load(), np.save(),
# np.memmap(), etc. are intentionally unavailable.

SAFE_NP_FUNCTIONS: Dict[str, Any] = {
    "abs": np.abs,
    "ceil": np.ceil,
    "floor": np.floor,
    "round": np.round,
    "sqrt": np.sqrt,
    "log": np.log,
    "log10": np.log10,
    "exp": np.exp,
    "maximum": np.maximum,
    "minimum": np.minimum,
    "where": np.where,
    "isnan": np.isnan,
    "isfinite": np.isfinite,
    "nan": np.nan,
    "inf": np.inf,
}


class _SafeNumpyNamespace:
    """Restricted NumPy namespace."""

    def __getattr__(self, name: str) -> Any:
        if name not in SAFE_NP_FUNCTIONS:
            raise UnsafeCodeError(
                f"NumPy operation '{name}' is not available "
                "in the secure execution environment."
            )

        return SAFE_NP_FUNCTIONS[name]


SAFE_NP = _SafeNumpyNamespace()


# ============================================================
# BLOCKED IDENTIFIERS
# ============================================================

BLOCKED_NAMES = {
    # Dynamic execution
    "__import__",
    "__builtins__",
    "eval",
    "exec",
    "compile",

    # File / input operations
    "open",
    "input",

    # Introspection
    "globals",
    "locals",
    "vars",
    "dir",
    "getattr",
    "setattr",
    "delattr",

    # Object manipulation
    "type",
    "object",
    "super",
    "callable",

    "property",
    "staticmethod",
    "classmethod",

    # Interactive/system helpers
    "breakpoint",
    "exit",
    "quit",
    "help",

    # Dangerous modules
    "os",
    "sys",
    "subprocess",
    "importlib",
    "pathlib",
    "shutil",
    "socket",
    "requests",
    "urllib",
    "http",
    "ftplib",
    "telnetlib",
    "pickle",
    "shelve",
    "sqlite3",
    "ctypes",
    "multiprocessing",
    "threading",
    "asyncio",
    "builtins",
}


# ============================================================
# BLOCKED ATTRIBUTES
# ============================================================

BLOCKED_ATTRIBUTES = {
    "__class__",
    "__dict__",
    "__bases__",
    "__base__",
    "__mro__",
    "__subclasses__",

    "__globals__",
    "__locals__",
    "__code__",
    "__closure__",

    "__func__",
    "__self__",
    "__module__",
    "__builtins__",

    "__getattribute__",
    "__setattr__",
    "__delattr__",

    "__reduce__",
    "__reduce_ex__",
    "__new__",
    "__init__",
    "__import__",
}


# ============================================================
# BLOCKED METHODS
# ============================================================

# These methods can perform file I/O, external operations,
# or access unsafe functionality.

BLOCKED_METHODS = {
    # File output
    "to_csv",
    "to_excel",
    "to_json",
    "to_html",
    "to_xml",
    "to_pickle",
    "to_parquet",
    "to_feather",
    "to_sql",

    # File/input operations
    "read_csv",
    "read_excel",
    "read_json",
    "read_pickle",
    "read_parquet",

    # Clipboard
    "to_clipboard",

    # Dynamic execution / unsafe pandas evaluation
    "eval",
    "query",
}


# ============================================================
# BLOCKED AST CONSTRUCTS
# ============================================================

# These constructs are unnecessary for normal dataframe
# analysis and significantly increase the attack surface.

BLOCKED_AST_NODES = (
    ast.Import,
    ast.ImportFrom,

    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,

    ast.For,
    ast.AsyncFor,
    ast.While,

    ast.With,
    ast.AsyncWith,

    ast.Try,

    ast.Raise,
    ast.Assert,

    ast.Delete,

    ast.Global,
    ast.Nonlocal,

    ast.Yield,
    ast.YieldFrom,
    ast.Await,

    ast.Lambda,
    ast.NamedExpr,

    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
)


# ============================================================
# AST SAFETY VISITOR
# ============================================================


class _SafetyVisitor(ast.NodeVisitor):
    """
    Validate AI-generated code before it is executed.

    This is a defense-in-depth layer. It is not intended to be
    an operating-system-level sandbox.
    """

    def __init__(self) -> None:
        self.node_count = 0

    def visit(self, node):
        self.node_count += 1

        if self.node_count > MAX_AST_NODES:
            raise UnsafeCodeError(
                f"Generated code is too complex. "
                f"Maximum allowed AST nodes: {MAX_AST_NODES}."
            )

        if isinstance(node, BLOCKED_AST_NODES):
            raise UnsafeCodeError(
                f"Disallowed syntax: "
                f"{type(node).__name__}."
            )

        return super().visit(node)

    # --------------------------------------------------------
    # Names
    # --------------------------------------------------------

    def visit_Name(self, node: ast.Name):
        name = node.id

        if name in BLOCKED_NAMES:
            raise UnsafeCodeError(
                f"Blocked identifier used: '{name}'."
            )

        if name.startswith("_"):
            raise UnsafeCodeError(
                f"Private identifier is not allowed: '{name}'."
            )

        self.generic_visit(node)

    # --------------------------------------------------------
    # Attributes
    # --------------------------------------------------------

    def visit_Attribute(self, node: ast.Attribute):
        attribute = node.attr

        if attribute.startswith("_"):
            raise UnsafeCodeError(
                f"Private/dunder attribute is not allowed: "
                f"'{attribute}'."
            )

        if attribute in BLOCKED_ATTRIBUTES:
            raise UnsafeCodeError(
                f"Blocked attribute: '{attribute}'."
            )

        if attribute in BLOCKED_METHODS:
            raise UnsafeCodeError(
                f"Blocked operation: '{attribute}'."
            )

        self.generic_visit(node)

    # --------------------------------------------------------
    # Function calls
    # --------------------------------------------------------

    def visit_Call(self, node: ast.Call):
        function_name = self._get_dotted_name(node.func)

        if function_name:
            parts = function_name.split(".")

            root_name = parts[0]
            final_name = parts[-1]

            if root_name in BLOCKED_NAMES:
                raise UnsafeCodeError(
                    f"Blocked function/module: "
                    f"'{function_name}'."
                )

            if final_name in BLOCKED_NAMES:
                raise UnsafeCodeError(
                    f"Blocked function/method: "
                    f"'{final_name}'."
                )

            if final_name in BLOCKED_METHODS:
                raise UnsafeCodeError(
                    f"Blocked operation: "
                    f"'{final_name}'."
                )

            if final_name.startswith("_"):
                raise UnsafeCodeError(
                    f"Private method is not allowed: "
                    f"'{final_name}'."
                )

        self.generic_visit(node)

    # --------------------------------------------------------
    # String constants
    # --------------------------------------------------------

    def visit_Constant(self, node: ast.Constant):
        if isinstance(node.value, str):
            if len(node.value) > MAX_STRING_LENGTH:
                raise UnsafeCodeError(
                    "String literal exceeds the allowed size."
                )

        self.generic_visit(node)

    # --------------------------------------------------------
    # Function name helper
    # --------------------------------------------------------

    @staticmethod
    def _get_dotted_name(
        node: ast.AST,
    ) -> Optional[str]:
        """Return a dotted function/method name."""

        if isinstance(node, ast.Name):
            return node.id

        if isinstance(node, ast.Attribute):
            parent = _SafetyVisitor._get_dotted_name(
                node.value
            )

            if parent:
                return f"{parent}.{node.attr}"

            return node.attr

        return None


# ============================================================
# RESULT ASSIGNMENT VALIDATION
# ============================================================


def _contains_result_assignment(
    tree: ast.AST,
) -> bool:
    """
    Confirm that generated code actually assigns to `result`.

    Valid:

        result = df.head()

    Invalid:

        df.head()

    Also invalid:

        text = "result"
    """

    for node in ast.walk(tree):

        if isinstance(node, ast.Assign):

            for target in node.targets:

                if (
                    isinstance(target, ast.Name)
                    and target.id == "result"
                ):
                    return True

        elif isinstance(node, ast.AnnAssign):

            target = node.target

            if (
                isinstance(target, ast.Name)
                and target.id == "result"
            ):
                return True

        elif isinstance(node, ast.AugAssign):

            target = node.target

            if (
                isinstance(target, ast.Name)
                and target.id == "result"
            ):
                return True

    return False


# ============================================================
# CODE VALIDATION
# ============================================================


def _validate_generated_code(
    code: str,
) -> ast.AST:
    """
    Parse and validate generated Python code.

    Returns:
        Parsed AST.

    Raises:
        UnsafeCodeError:
            When code violates the security policy.
    """

    if not isinstance(code, str):
        raise UnsafeCodeError(
            "Generated code must be a string."
        )

    code = code.strip()

    if not code:
        raise UnsafeCodeError(
            "Generated code is empty."
        )

    if len(code) > MAX_CODE_LENGTH:
        raise UnsafeCodeError(
            f"Generated code exceeds the maximum "
            f"length of {MAX_CODE_LENGTH:,} characters."
        )

    try:
        tree = ast.parse(
            code,
            mode="exec",
        )

    except SyntaxError as ex:
        raise UnsafeCodeError(
            f"Syntax error in generated code: {ex}"
        ) from ex

    validator = _SafetyVisitor()
    validator.visit(tree)

    if not _contains_result_assignment(tree):
        raise UnsafeCodeError(
            "Generated code must assign its final "
            "output to the 'result' variable."
        )

    return tree


# ============================================================
# DATASET LOADING
# ============================================================


def load_dataset(uploaded_file) -> pd.DataFrame:
    """
    Load CSV or Excel data from a Streamlit uploader.

    Supported formats:
        CSV
        XLSX
        XLS
    """

    if uploaded_file is None:
        raise ValueError(
            "No file was provided."
        )

    file_name = str(
        getattr(uploaded_file, "name", "")
    ).strip().lower()

    if not file_name:
        raise ValueError(
            "Uploaded file does not have a valid filename."
        )

    try:

        # ----------------------------------------------------
        # CSV
        # ----------------------------------------------------

        if file_name.endswith(".csv"):

            uploaded_file.seek(0)

            try:
                df = pd.read_csv(
                    uploaded_file
                )

            except UnicodeDecodeError:
                uploaded_file.seek(0)

                df = pd.read_csv(
                    uploaded_file,
                    encoding="latin1",
                )

        # ----------------------------------------------------
        # Excel
        # ----------------------------------------------------

        elif (
            file_name.endswith(".xlsx")
            or file_name.endswith(".xls")
        ):

            uploaded_file.seek(0)

            df = pd.read_excel(
                uploaded_file
            )

        else:
            raise ValueError(
                "Unsupported file format. "
                "Please upload CSV, XLSX, or XLS."
            )

    except pd.errors.EmptyDataError as ex:
        raise ValueError(
            "The uploaded file contains no data."
        ) from ex

    except pd.errors.ParserError as ex:
        raise ValueError(
            "The uploaded CSV could not be parsed. "
            "Please verify the file format."
        ) from ex

    except Exception as ex:
        raise ValueError(
            f"Could not load the uploaded file: {ex}"
        ) from ex

    if not isinstance(df, pd.DataFrame):
        raise ValueError(
            "The uploaded file did not produce a valid DataFrame."
        )

    # --------------------------------------------------------
    # Duplicate column validation
    # --------------------------------------------------------

    if df.columns.duplicated().any():

        duplicates = (
            df.columns[df.columns.duplicated()]
            .astype(str)
            .tolist()
        )

        raise ValueError(
            "Duplicate column names were detected: "
            f"{duplicates}. "
            "Please rename duplicate columns before analysis."
        )

    return df


# ============================================================
# DATAFRAME CONTEXT FOR AI
# ============================================================


def dataframe_context(
    df: pd.DataFrame,
    sample_rows: int = 5,
) -> str:
    """
    Create compact dataset context for Gemini.

    Includes:
        - shape
        - column names
        - pandas dtypes
        - null information
        - sample records
    """

    if not isinstance(df, pd.DataFrame):
        raise ValueError(
            "Expected a pandas DataFrame."
        )

    sample_rows = max(
        0,
        min(int(sample_rows), 10),
    )

    schema = []

    for column in df.columns:

        series = df[column]

        schema.append(
            {
                "column": str(column),
                "dtype": str(series.dtype),
                "nullable": bool(
                    series.isna().any()
                ),
            }
        )

    sample_df = df.head(sample_rows).copy()

    # Convert NumPy NaN/NaT values into JSON-friendly values.
    sample_df = sample_df.astype(object).where(
        pd.notna(sample_df),
        None,
    )

    sample = sample_df.to_dict(
        orient="records"
    )

    context = {
        "shape": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
        },
        "schema": schema,
        "sample": sample,
    }

    return json.dumps(
        context,
        ensure_ascii=False,
        default=str,
    )


# ============================================================
# SAFE EXECUTION
# ============================================================


def safe_execute_pandas_code(
    code: str,
    df: pd.DataFrame,
) -> ExecutionResult:
    """
    Safely execute AI-generated pandas analysis code.

    Contract:
        Generated code MUST assign the final result to:

            result

    Example:

        result = (
            df.groupby("region", as_index=False)["revenue"]
              .sum()
              .sort_values(
                  "revenue",
                  ascending=False
              )
        )

    The generated code receives:
        df
        pd
        np

    But pd and np are restricted namespaces.
    """

    # --------------------------------------------------------
    # Validate dataframe
    # --------------------------------------------------------

    if not isinstance(df, pd.DataFrame):
        return ExecutionResult(
            ok=False,
            error=(
                "Invalid dataset. "
                "Expected a pandas DataFrame."
            ),
        )

    if df.empty:
        return ExecutionResult(
            ok=False,
            error=(
                "The dataset is empty. "
                "There is no data to analyze."
            ),
        )

    # --------------------------------------------------------
    # Validate generated code
    # --------------------------------------------------------

    try:
        tree = _validate_generated_code(
            code
        )

    except UnsafeCodeError as ex:
        return ExecutionResult(
            ok=False,
            error=f"Unsafe code blocked: {ex}",
        )

    except Exception as ex:
        return ExecutionResult(
            ok=False,
            error=(
                "Code validation failed: "
                f"{type(ex).__name__}: {ex}"
            ),
        )

    # --------------------------------------------------------
    # Restricted execution environment
    # --------------------------------------------------------

    safe_globals = {
        "__builtins__": SAFE_BUILTINS,
        "pd": SAFE_PD,
        "np": SAFE_NP,
    }

    # Work on a deep copy so AI-generated transformations
    # cannot modify the application's original dataframe.
    safe_locals = {
        "df": df.copy(deep=True),
        "result": None,
    }

    # --------------------------------------------------------
    # Compile
    # --------------------------------------------------------

    try:

        compiled = compile(
            tree,
            filename="<aidaass_ai_generated>",
            mode="exec",
        )

    except Exception as ex:
        return ExecutionResult(
            ok=False,
            error=(
                "Could not compile generated code: "
                f"{type(ex).__name__}: {ex}"
            ),
        )

    # --------------------------------------------------------
    # Execute
    # --------------------------------------------------------

    try:

        exec(
            compiled,
            safe_globals,
            safe_locals,
        )

    except UnsafeCodeError as ex:
        return ExecutionResult(
            ok=False,
            error=(
                f"Unsafe operation blocked: {ex}"
            ),
        )

    except MemoryError:
        return ExecutionResult(
            ok=False,
            error=(
                "The analysis required too much memory. "
                "Try a smaller or simpler query."
            ),
        )

    except RecursionError:
        return ExecutionResult(
            ok=False,
            error=(
                "The analysis exceeded the allowed "
                "recursion depth."
            ),
        )

    except Exception as ex:
        return ExecutionResult(
            ok=False,
            error=(
                f"Execution failed: "
                f"{type(ex).__name__}: {ex}"
            ),
        )

    # --------------------------------------------------------
    # Verify result
    # --------------------------------------------------------

    if "result" not in safe_locals:
        return ExecutionResult(
            ok=False,
            error=(
                "No variable named 'result' was produced."
            ),
        )

    result = safe_locals["result"]

    if result is None:
        return ExecutionResult(
            ok=False,
            error=(
                "The analysis completed without "
                "producing a result."
            ),
        )

    # --------------------------------------------------------
    # Verify supported result type
    # --------------------------------------------------------

    supported_result_types = (
        pd.DataFrame,
        pd.Series,
        pd.Index,
        np.ndarray,
        np.generic,
        str,
        int,
        float,
        bool,
        list,
        tuple,
        dict,
    )

    if not isinstance(
        result,
        supported_result_types,
    ):
        return ExecutionResult(
            ok=False,
            error=(
                "The analysis produced an unsupported "
                f"result type: {type(result).__name__}."
            ),
        )

    return ExecutionResult(
        ok=True,
        result=result,
    )


# ============================================================
# RESULT PREVIEW
# ============================================================


def result_preview_text(
    result: Any,
    max_rows: int = 8,
) -> str:
    """
    Convert an analysis result into compact text.

    Used when sending analysis results to Gemini for
    business-insight generation.
    """

    try:
        max_rows = max(
            1,
            min(int(max_rows), 20),
        )
    except (TypeError, ValueError):
        max_rows = 8

    # --------------------------------------------------------
    # DataFrame
    # --------------------------------------------------------

    if isinstance(result, pd.DataFrame):

        if result.empty:
            return "(Empty DataFrame)"

        return result.head(
            max_rows
        ).to_string(
            index=False
        )

    # --------------------------------------------------------
    # Series
    # --------------------------------------------------------

    if isinstance(result, pd.Series):

        if result.empty:
            return "(Empty Series)"

        return result.head(
            max_rows
        ).to_string()

    # --------------------------------------------------------
    # Index
    # --------------------------------------------------------

    if isinstance(result, pd.Index):

        return str(
            result[:max_rows].tolist()
        )

    # --------------------------------------------------------
    # NumPy array
    # --------------------------------------------------------

    if isinstance(result, np.ndarray):

        return str(
            result[:max_rows]
        )

    # --------------------------------------------------------
    # Dictionary
    # --------------------------------------------------------

    if isinstance(result, dict):

        preview = dict(
            list(result.items())[:max_rows]
        )

        return json.dumps(
            preview,
            ensure_ascii=False,
            default=str,
        )

    # --------------------------------------------------------
    # List / tuple
    # --------------------------------------------------------

    if isinstance(result, (list, tuple)):

        return str(
            result[:max_rows]
        )

    # --------------------------------------------------------
    # Scalar
    # --------------------------------------------------------

    return str(result)


# ============================================================
# RETAIL SALES DEMO
# ============================================================


def _build_retail_sales_demo() -> pd.DataFrame:
    """Create the deterministic retail-sales demo dataset."""

    rng = np.random.default_rng(42)

    months = pd.date_range(
        "2024-01-01",
        periods=12,
        freq="MS",
    )

    regions = [
        "North",
        "South",
        "East",
        "West",
    ]

    products = [
        "Laptop",
        "Phone",
        "Tablet",
        "Monitor",
        "Headphones",
    ]

    rows = []

    for month in months:

        for region in regions:

            for product in products:

                units = int(
                    rng.integers(
                        40,
                        250,
                    )
                )

                unit_price = float(
                    rng.integers(
                        40,
                        1400,
                    )
                )

                revenue = round(
                    units * unit_price,
                    2,
                )

                cost = round(
                    revenue
                    * float(
                        rng.uniform(
                            0.58,
                            0.85,
                        )
                    ),
                    2,
                )

                profit = round(
                    revenue - cost,
                    2,
                )

                rows.append(
                    {
                        "date": month,
                        "region": region,
                        "product": product,
                        "units_sold": units,
                        "unit_price": unit_price,
                        "revenue": revenue,
                        "cost": cost,
                        "profit": profit,
                    }
                )

    return pd.DataFrame(rows)


# ============================================================
# MARKETING CAMPAIGN DEMO
# ============================================================


def _build_marketing_campaign_demo() -> pd.DataFrame:
    """Create the deterministic marketing demo dataset."""

    rng = np.random.default_rng(42)

    channels = [
        "Search",
        "Social",
        "Email",
        "Referral",
        "Video",
    ]

    campaigns = [
        "Spring Boost",
        "Summer Launch",
        "Back To School",
        "Holiday Push",
    ]

    dates = pd.date_range(
        "2024-04-01",
        periods=140,
        freq="D",
    )

    rows = []

    for date in dates:

        channel = channels[
            int(
                rng.integers(
                    0,
                    len(channels),
                )
            )
        ]

        campaign = campaigns[
            int(
                rng.integers(
                    0,
                    len(campaigns),
                )
            )
        ]

        impressions = int(
            rng.integers(
                3000,
                30000,
            )
        )

        clicks = int(
            impressions
            * float(
                rng.uniform(
                    0.015,
                    0.12,
                )
            )
        )

        conversions = int(
            clicks
            * float(
                rng.uniform(
                    0.03,
                    0.22,
                )
            )
        )

        spend = round(
            float(
                rng.uniform(
                    80,
                    1400,
                )
            ),
            2,
        )

        revenue = round(
            conversions
            * float(
                rng.uniform(
                    45,
                    240,
                )
            ),
            2,
        )

        roi = round(
            (revenue - spend) / spend,
            4,
        )

        rows.append(
            {
                "date": date,
                "channel": channel,
                "campaign": campaign,
                "impressions": impressions,
                "clicks": clicks,
                "conversions": conversions,
                "spend": spend,
                "revenue": revenue,
                "roi": roi,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# EXAMPLE DATASETS
# ============================================================


def load_example_dataset(
    name: str,
) -> pd.DataFrame:
    """
    Return a curated in-memory example dataset.

    Supported names:
        Retail Sales Demo
        Marketing Campaign Demo
    """

    if name == "Retail Sales Demo":
        return _build_retail_sales_demo()

    if name == "Marketing Campaign Demo":
        return _build_marketing_campaign_demo()

    raise ValueError(
        f"Unknown example dataset selected: '{name}'. "
        "Supported datasets: "
        "'Retail Sales Demo', "
        "'Marketing Campaign Demo'."
    )


# ============================================================
# PACKAGED ECOMMERCE DATASET
# ============================================================


def load_packaged_ecommerce_sample() -> pd.DataFrame:
    """
    Load the bundled ecommerce CSV sample dataset.
    """

    file_path = (
        Path(__file__).resolve().parent
        / "data"
        / "sample_ecommerce_dataset.csv"
    )

    if not file_path.exists():
        raise FileNotFoundError(
            "Bundled ecommerce sample dataset was not found at: "
            f"{file_path}"
        )

    try:
        df = pd.read_csv(
            file_path
        )

    except Exception as ex:
        raise ValueError(
            "Could not load the bundled ecommerce dataset: "
            f"{ex}"
        ) from ex

    if df.empty:
        raise ValueError(
            "The bundled ecommerce dataset is empty."
        )

    if df.columns.duplicated().any():

        duplicates = (
            df.columns[df.columns.duplicated()]
            .astype(str)
            .tolist()
        )

        raise ValueError(
            "The bundled ecommerce dataset contains "
            f"duplicate columns: {duplicates}"
        )

    return df
