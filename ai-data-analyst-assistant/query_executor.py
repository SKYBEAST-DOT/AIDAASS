"""
Safe query execution layer for AI-generated pandas code.

This module ensures AI-generated code is safe before execution by:
- Blocking dangerous Python constructs (imports, file access, loops)
- Only allowing safe pandas operations
- Providing clear error messages
"""

import ast
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from utils import ExecutionResult, safe_execute_pandas_code


@dataclass
class ValidationResult:
    """Result of code validation check."""
    ok: bool
    error: str = ""


# Dangerous AST node types that should be blocked
BLOCKED_AST_NODES = {
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
}

# Common dangerous function/module names
BLOCKED_NAMES = {
    "__import__", "eval", "exec", "compile", "open", "input",
    "globals", "locals", "vars", "dir", "getattr", "setattr",
    "delattr", "type", "callable", "property", "staticmethod",
    "classmethod", "__builtins__", "breakpoint", "exit", "quit",
}

# Dangerous module names
BLOCKED_MODULES = {
    "os", "sys", "subprocess", "importlib", "pathlib", "shutil",
    "socket", "requests", "urllib", "http", "ftplib", "telnetlib",
    "json", "pickle", "shelve", "dbm", "sqlite3", "psycopg2",
}


class PandasCodeValidator(ast.NodeVisitor):
    """Validates that AST contains only safe pandas operations."""

    def __init__(self):
        self.errors = []

    def visit(self, node):
        """Check node type against blocklist."""
        if type(node) in BLOCKED_AST_NODES:
            self.errors.append(
                f"Blocked construct: {type(node).__name__} is not allowed"
            )
            return

        # Check for dangerous function calls
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in BLOCKED_NAMES:
                    self.errors.append(f"Blocked function: {node.func.id}")
                    return
            elif isinstance(node.func, ast.Attribute):
                if node.func.attr in BLOCKED_NAMES:
                    self.errors.append(f"Blocked method: {node.func.attr}")
                    return

        # Check for dangerous attribute access
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                self.errors.append(f"Private/magic attributes not allowed: {node.attr}")
                return

        # Check for imports disguised as names
        if isinstance(node, ast.Name):
            if node.id in BLOCKED_MODULES:
                self.errors.append(f"Blocked module: {node.id}")
                return

        self.generic_visit(node)


def validate_generated_code(code: str) -> ValidationResult:
    """
    Validate AI-generated code before execution.

    Checks for:
    - Empty code
    - Missing 'result' assignment
    - Blocked constructs (imports, loops, functions)
    - Dangerous function calls
    - Private attribute access
    """
    if not code or not code.strip():
        return ValidationResult(
            ok=False,
            error="Code is empty. AI did not generate a valid query."
        )

    if "result" not in code:
        return ValidationResult(
            ok=False,
            error="Code must assign output to 'result' variable."
        )

    # String-based checks for common dangerous patterns
    code_lower = code.lower()
    dangerous_patterns = {
        "import ": "Imports are not allowed",
        "from ": "Imports are not allowed",
        "open(": "File operations are not allowed",
        "exec(": "Dynamic code execution is not allowed",
        "eval(": "Dynamic code evaluation is not allowed",
        "__": "Magic/private attributes are not allowed",
        "compile": "Code compilation is not allowed",
        "lambda": "Lambda functions are not allowed",
        "def ": "Function definitions are not allowed",
        "class ": "Class definitions are not allowed",
    }

    for pattern, message in dangerous_patterns.items():
        if pattern in code_lower:
            return ValidationResult(
                ok=False,
                error=f"Blocked pattern '{pattern}': {message}"
            )

    # AST-based validation
    try:
        tree = ast.parse(code, mode="exec")
        validator = PandasCodeValidator()
        validator.visit(tree)

        if validator.errors:
            return ValidationResult(
                ok=False,
                error="; ".join(validator.errors)
            )

    except SyntaxError as ex:
        return ValidationResult(
            ok=False,
            error=f"Syntax error in generated code: {ex}"
        )

    return ValidationResult(ok=True)


def execute_ai_query(code: str, df: pd.DataFrame) -> ExecutionResult:
    """
    Run validated AI-generated pandas code with safety controls.

    Args:
        code: Python code string to execute
        df: DataFrame to operate on

    Returns:
        ExecutionResult with ok flag and either result or error
    """
    # Validate first
    validation = validate_generated_code(code)
    if not validation.ok:
        return ExecutionResult(ok=False, error=validation.error)

    # Execute safely
    return safe_execute_pandas_code(code, df)


def try_execute_with_fallback(
    primary_code: str,
    fallback_code: Optional[str],
    df: pd.DataFrame
) -> ExecutionResult:
    """
    Execute primary code, fall back to secondary if primary fails.

    Useful when AI generates code that's valid but fails at runtime.
    """
    first = execute_ai_query(primary_code, df)
    if first.ok or not fallback_code:
        return first

    second = execute_ai_query(fallback_code, df)
    if second.ok:
        return second

    return ExecutionResult(
        ok=False,
        error=f"Both queries failed. Primary: {first.error} | Fallback: {second.error}"
    )
