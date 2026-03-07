import ast
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd


@dataclass
class ExecutionResult:
    ok: bool
    result: Any = None
    error: str = ""


class UnsafeCodeError(ValueError):
    pass


class _SafetyVisitor(ast.NodeVisitor):
    """Allow only a safe subset of Python AST nodes for dataframe analysis."""

    ALLOWED_NODES = {
        ast.Module,
        ast.Assign,
        ast.Expr,
        ast.Name,
        ast.Load,
        ast.Store,
        ast.Call,
        ast.Attribute,
        ast.Subscript,
        ast.Slice,
        ast.Index,
        ast.Constant,
        ast.BinOp,
        ast.UnaryOp,
        ast.BoolOp,
        ast.Compare,
        ast.List,
        ast.Tuple,
        ast.Dict,
        ast.keyword,
        ast.If,
        ast.IfExp,
        ast.ListComp,
        ast.comprehension,
        ast.GeneratorExp,
        ast.Return,
        ast.Lambda,
        ast.arguments,
        ast.arg,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Mod,
        ast.Pow,
        ast.FloorDiv,
        ast.USub,
        ast.UAdd,
        ast.And,
        ast.Or,
        ast.Eq,
        ast.NotEq,
        ast.Gt,
        ast.GtE,
        ast.Lt,
        ast.LtE,
        ast.In,
        ast.NotIn,
        ast.Not,
    }

    BLOCKED_NAMES = {
        "__import__",
        "eval",
        "exec",
        "open",
        "compile",
        "input",
        "globals",
        "locals",
        "vars",
        "dir",
        "help",
        "quit",
        "exit",
        "os",
        "sys",
        "subprocess",
        "pathlib",
        "shutil",
        "socket",
        "requests",
    }

    def visit(self, node):
        if type(node) not in self.ALLOWED_NODES:
            raise UnsafeCodeError(f"Disallowed syntax: {type(node).__name__}")
        super().visit(node)

    def visit_Name(self, node: ast.Name):
        if node.id in self.BLOCKED_NAMES:
            raise UnsafeCodeError(f"Blocked identifier used: {node.id}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        if node.attr.startswith("__"):
            raise UnsafeCodeError("Dunder attributes are not allowed")
        self.generic_visit(node)


SAFE_BUILTINS: Dict[str, Any] = {
    "len": len,
    "min": min,
    "max": max,
    "sum": sum,
    "round": round,
    "sorted": sorted,
    "abs": abs,
    "float": float,
    "int": int,
    "str": str,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "range": range,
}


def load_dataset(uploaded_file) -> pd.DataFrame:
    """Load CSV or Excel file from Streamlit uploader."""
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):
        try:
            return pd.read_csv(uploaded_file)
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding="latin1")

    if file_name.endswith(".xlsx") or file_name.endswith(".xls"):
        return pd.read_excel(uploaded_file)

    raise ValueError("Unsupported file format. Please upload CSV or Excel.")


def dataframe_context(df: pd.DataFrame, sample_rows: int = 5) -> str:
    """Create compact schema + sample context for prompting the LLM."""
    schema = [{"column": c, "dtype": str(df[c].dtype)} for c in df.columns]
    sample = df.head(sample_rows).replace({np.nan: None}).to_dict(orient="records")

    return json.dumps(
        {
            "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
            "schema": schema,
            "sample": sample,
        },
        ensure_ascii=False,
        default=str,
    )


def safe_execute_pandas_code(code: str, df: pd.DataFrame) -> ExecutionResult:
    """Safely execute generated pandas code against a copy of dataframe.

    Contract: generated code must assign final output to a variable named `result`.
    """
    if not code or "result" not in code:
        return ExecutionResult(ok=False, error="Generated code must assign output to 'result'.")

    try:
        tree = ast.parse(code, mode="exec")
        _SafetyVisitor().visit(tree)

        safe_globals = {
            "__builtins__": SAFE_BUILTINS,
            "pd": pd,
            "np": np,
        }
        safe_locals = {
            "df": df.copy(deep=True),
            "result": None,
        }

        compiled = compile(tree, filename="<ai_generated>", mode="exec")
        exec(compiled, safe_globals, safe_locals)

        if "result" not in safe_locals:
            return ExecutionResult(ok=False, error="No variable named 'result' was produced.")

        return ExecutionResult(ok=True, result=safe_locals["result"])

    except UnsafeCodeError as ex:
        return ExecutionResult(ok=False, error=f"Unsafe code blocked: {ex}")
    except Exception as ex:
        return ExecutionResult(ok=False, error=f"Execution failed: {ex}")


def result_preview_text(result: Any, max_rows: int = 8) -> str:
    """Create a compact text representation of result for AI insight generation."""
    if isinstance(result, pd.DataFrame):
        return result.head(max_rows).to_string(index=False)
    if isinstance(result, pd.Series):
        return result.head(max_rows).to_string()
    return str(result)


def load_example_dataset(name: str) -> pd.DataFrame:
    """Return a curated in-memory example dataset for demo usage."""
    rng = np.random.default_rng(42)

    if name == "Retail Sales Demo":
        months = pd.date_range("2024-01-01", periods=12, freq="MS")
        regions = ["North", "South", "East", "West"]
        products = ["Laptop", "Phone", "Tablet", "Monitor", "Headphones"]

        rows = []
        for month in months:
            for region in regions:
                for product in products:
                    units = int(rng.integers(40, 250))
                    unit_price = float(rng.integers(40, 1400))
                    revenue = round(units * unit_price, 2)
                    cost = round(revenue * float(rng.uniform(0.58, 0.85)), 2)
                    profit = round(revenue - cost, 2)
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

    if name == "Marketing Campaign Demo":
        channels = ["Search", "Social", "Email", "Referral", "Video"]
        campaigns = ["Spring Boost", "Summer Launch", "Back To School", "Holiday Push"]
        dates = pd.date_range("2024-04-01", periods=140, freq="D")

        rows = []
        for d in dates:
            channel = channels[int(rng.integers(0, len(channels)))]
            campaign = campaigns[int(rng.integers(0, len(campaigns)))]
            impressions = int(rng.integers(3000, 30000))
            clicks = int(impressions * float(rng.uniform(0.015, 0.12)))
            conversions = int(clicks * float(rng.uniform(0.03, 0.22)))
            spend = round(float(rng.uniform(80, 1400)), 2)
            revenue = round(conversions * float(rng.uniform(45, 240)), 2)
            rows.append(
                {
                    "date": d,
                    "channel": channel,
                    "campaign": campaign,
                    "impressions": impressions,
                    "clicks": clicks,
                    "conversions": conversions,
                    "spend": spend,
                    "revenue": revenue,
                    "roi": round((revenue - spend) / spend, 4),
                }
            )
        return pd.DataFrame(rows)

    raise ValueError("Unknown example dataset selected.")


def load_packaged_ecommerce_sample() -> pd.DataFrame:
    """Load the bundled ecommerce CSV sample dataset."""
    file_path = Path(__file__).resolve().parent / "data" / "sample_ecommerce_dataset.csv"
    if not file_path.exists():
        raise FileNotFoundError("Bundled ecommerce sample dataset not found.")
    return pd.read_csv(file_path)
