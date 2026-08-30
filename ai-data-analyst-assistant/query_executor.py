"""
Secure execution layer for AI-generated pandas code.

AIDAASS security pipeline:

    AI-generated code
            ↓
    Basic validation
            ↓
    AST parsing
            ↓
    Dangerous construct checks
            ↓
    Name / attribute checks
            ↓
    Result assignment check
            ↓
    safe_execute_pandas_code()
            ↓
        ExecutionResult

Important:
This module validates code before execution. The actual runtime
sandbox is provided by utils.safe_execute_pandas_code().
"""

import ast
from dataclasses import dataclass
from typing import Optional, Set, Tuple

import pandas as pd

from utils import ExecutionResult, safe_execute_pandas_code


# ============================================================
# CONFIGURATION
# ============================================================

MAX_CODE_LENGTH = 12_000
MAX_AST_NODES = 500
MAX_STRING_LENGTH = 5_000


# ============================================================
# VALIDATION RESULT
# ============================================================


@dataclass
class ValidationResult:
    """Result returned by the generated-code validator."""

    ok: bool
    error: str = ""


# ============================================================
# BLOCKED AST CONSTRUCTS
# ============================================================

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
)


# ============================================================
# BLOCKED NAMES
# ============================================================

BLOCKED_NAMES = {
    "__import__",
    "__builtins__",

    "eval",
    "exec",
    "compile",

    "open",
    "input",

    "globals",
    "locals",
    "vars",
    "dir",

    "getattr",
    "setattr",
    "delattr",

    "type",
    "object",
    "super",
    "callable",

    "property",
    "staticmethod",
    "classmethod",

    "breakpoint",
    "exit",
    "quit",
    "help",

    "memoryview",

    "bytearray",
}


# ============================================================
# BLOCKED MODULES
# ============================================================

BLOCKED_MODULES = {
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
    "dbm",

    "sqlite3",
    "psycopg2",

    "ctypes",

    "multiprocessing",
    "threading",
    "asyncio",

    "builtins",
    "site",
    "runpy",

    "tempfile",
    "glob",
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
# ROOT OBJECTS AVAILABLE TO AI CODE
# ============================================================

ALLOWED_ROOT_NAMES = {
    "df",
    "pd",
    "result",
}


# ============================================================
# SAFE BUILT-IN FUNCTIONS
# ============================================================

# These are the built-ins that a normal data-analysis query
# may reasonably need.
#
# The final runtime environment is still controlled by
# safe_execute_pandas_code() in utils.py.

ALLOWED_BUILTINS = {
    "abs",
    "all",
    "any",
    "bool",
    "dict",
    "float",
    "int",
    "len",
    "list",
    "max",
    "min",
    "round",
    "set",
    "sorted",
    "str",
    "sum",
    "tuple",
}


# ============================================================
# VALIDATOR
# ============================================================


class PandasCodeValidator(ast.NodeVisitor):
    """
    AST validator for AI-generated pandas analysis code.

    The validator is deliberately conservative.

    It rejects:
        - imports
        - functions/classes
        - loops
        - exception handling
        - dynamic execution
        - private/dunder access
        - dangerous names
        - dangerous modules
        - comprehensions
        - excessive AST complexity
    """

    def __init__(self) -> None:
        self.errors = []
        self.node_count = 0
        self.assigned_names: Set[str] = set()

    # --------------------------------------------------------
    # Generic AST visitor
    # --------------------------------------------------------

    def visit(self, node):
        self.node_count += 1

        if self.node_count > MAX_AST_NODES:
            self.errors.append(
                f"Generated code is too complex. "
                f"Maximum AST size is {MAX_AST_NODES} nodes."
            )

            # Stop walking once the complexity limit is exceeded.
            return node

        if isinstance(node, BLOCKED_AST_NODES):
            self.errors.append(
                f"Blocked construct: "
                f"{type(node).__name__} is not allowed."
            )

            # Do not descend into a blocked construct.
            return node

        return super().visit(node)

    # --------------------------------------------------------
    # Names
    # --------------------------------------------------------

    def visit_Name(self, node: ast.Name):
        name = node.id

        if name in BLOCKED_NAMES:
            self.errors.append(
                f"Blocked name: '{name}'."
            )

        if name in BLOCKED_MODULES:
            self.errors.append(
                f"Blocked module: '{name}'."
            )

        if name.startswith("_"):
            self.errors.append(
                f"Private/magic name is not allowed: '{name}'."
            )

        if isinstance(node.ctx, ast.Store):
            self.assigned_names.add(name)

        self.generic_visit(node)

    # --------------------------------------------------------
    # Attributes
    # --------------------------------------------------------

    def visit_Attribute(self, node: ast.Attribute):
        attribute = node.attr

        if attribute.startswith("_"):
            self.errors.append(
                f"Private/magic attribute is not allowed: "
                f"'{attribute}'."
            )

        if attribute in BLOCKED_ATTRIBUTES:
            self.errors.append(
                f"Blocked attribute: '{attribute}'."
            )

        self.generic_visit(node)

    # --------------------------------------------------------
    # Function calls
    # --------------------------------------------------------

    def visit_Call(self, node: ast.Call):
        call_name = self._get_dotted_name(node.func)

        if call_name:
            parts = call_name.split(".")

            root_name = parts[0]
            final_name = parts[-1]

            if root_name in BLOCKED_NAMES:
                self.errors.append(
                    f"Blocked function call: '{call_name}'."
                )

            if root_name in BLOCKED_MODULES:
                self.errors.append(
                    f"Blocked module call: '{call_name}'."
                )

            if final_name in BLOCKED_NAMES:
                self.errors.append(
                    f"Blocked function/method: '{final_name}'."
                )

            if final_name.startswith("_"):
                self.errors.append(
                    f"Private/magic method is not allowed: "
                    f"'{final_name}'."
                )

        self.generic_visit(node)

    # --------------------------------------------------------
    # Constants
    # --------------------------------------------------------

    def visit_Constant(self, node: ast.Constant):
        if isinstance(node.value, str):
            if len(node.value) > MAX_STRING_LENGTH:
                self.errors.append(
                    "String literal is too large."
                )

        self.generic_visit(node)

    # --------------------------------------------------------
    # Comprehensions
    # --------------------------------------------------------

    def visit_ListComp(self, node: ast.ListComp):
        self.errors.append(
            "List comprehensions are not allowed."
        )

    def visit_SetComp(self, node: ast.SetComp):
        self.errors.append(
            "Set comprehensions are not allowed."
        )

    def visit_DictComp(self, node: ast.DictComp):
        self.errors.append(
            "Dictionary comprehensions are not allowed."
        )

    def visit_GeneratorExp(self, node: ast.GeneratorExp):
        self.errors.append(
            "Generator expressions are not allowed."
        )

    # --------------------------------------------------------
    # Assignment targets
    # --------------------------------------------------------

    def visit_Assign(self, node: ast.Assign):
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign):
        self.generic_visit(node)

    # --------------------------------------------------------
    # Helper
    # --------------------------------------------------------

    @staticmethod
    def _get_dotted_name(node: ast.AST) -> Optional[str]:
        """
        Convert an AST function expression into a dotted name.

        Examples:

            df.head()
                -> df.head

            df.groupby(...)
                -> df.groupby

            pd.to_datetime(...)
                -> pd.to_datetime
        """

        if isinstance(node, ast.Name):
            return node.id

        if isinstance(node, ast.Attribute):
            parent = PandasCodeValidator._get_dotted_name(
                node.value
            )

            if parent:
                return f"{parent}.{node.attr}"

            return node.attr

        return None


# ============================================================
# BASIC VALIDATION
# ============================================================


def _validate_basic_code(code: str) -> ValidationResult:
    """Validate basic properties of generated code."""

    if not isinstance(code, str):
        return ValidationResult(
            ok=False,
            error="Generated code must be a string.",
        )

    code = code.strip()

    if not code:
        return ValidationResult(
            ok=False,
            error=(
                "Code is empty. "
                "The AI did not generate a valid query."
            ),
        )

    if len(code) > MAX_CODE_LENGTH:
        return ValidationResult(
            ok=False,
            error=(
                f"Generated code is too long. "
                f"Maximum length is {MAX_CODE_LENGTH:,} characters."
            ),
        )

    return ValidationResult(ok=True)


# ============================================================
# RESULT ASSIGNMENT VALIDATION
# ============================================================


def _contains_result_assignment(tree: ast.AST) -> bool:
    """
    Check whether generated code assigns to `result`.

    This is deliberately AST-based rather than:

        "result" in code

    because a comment/string containing the word "result"
    should not satisfy the requirement.
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


def _validate_result_assignment(
    tree: ast.AST,
) -> ValidationResult:
    """Require the generated code to assign its final output."""

    if not _contains_result_assignment(tree):
        return ValidationResult(
            ok=False,
            error=(
                "Generated code must assign the final "
                "analysis output to the 'result' variable."
            ),
        )

    return ValidationResult(ok=True)


# ============================================================
# DEFENSE-IN-DEPTH STRING CHECKS
# ============================================================


def _validate_dangerous_text(code: str) -> ValidationResult:
    """
    Perform additional checks for obvious dangerous payloads.

    AST validation is the primary mechanism.

    These checks are defense-in-depth and are intentionally
    limited to high-confidence dangerous sequences.
    """

    lowered = code.lower()

    dangerous_patterns = {
        "__import__": "Dynamic imports are not allowed.",
        "__builtins__": "Builtins access is not allowed.",
        "__subclasses__": "Subclass traversal is not allowed.",
        "__globals__": "Global namespace access is not allowed.",
        "__locals__": "Local namespace access is not allowed.",
        "__class__": "Class traversal is not allowed.",
        "__mro__": "MRO traversal is not allowed.",

        "os.system": "System calls are not allowed.",
        "os.popen": "System commands are not allowed.",
        "subprocess": "Subprocess operations are not allowed.",
        "socket.": "Network/socket operations are not allowed.",
        "pickle.": "Pickle operations are not allowed.",
    }

    for pattern, message in dangerous_patterns.items():
        if pattern in lowered:
            return ValidationResult(
                ok=False,
                error=(
                    f"Blocked pattern '{pattern}': "
                    f"{message}"
                ),
            )

    return ValidationResult(ok=True)


# ============================================================
# PUBLIC VALIDATION API
# ============================================================


def validate_generated_code(
    code: str,
) -> ValidationResult:
    """
    Validate AI-generated pandas code.

    Validation order:

        1. Basic input checks
        2. Dangerous payload checks
        3. Python AST parsing
        4. AST security validation
        5. Result assignment validation
    """

    # --------------------------------------------------------
    # 1. Basic validation
    # --------------------------------------------------------

    basic_result = _validate_basic_code(code)

    if not basic_result.ok:
        return basic_result

    # --------------------------------------------------------
    # 2. Defense-in-depth text validation
    # --------------------------------------------------------

    text_result = _validate_dangerous_text(code)

    if not text_result.ok:
        return text_result

    # --------------------------------------------------------
    # 3. Parse Python AST
    # --------------------------------------------------------

    try:
        tree = ast.parse(
            code,
            mode="exec",
        )

    except SyntaxError as ex:
        return ValidationResult(
            ok=False,
            error=(
                f"Syntax error in generated code: {ex}"
            ),
        )

    # --------------------------------------------------------
    # 4. AST security validation
    # --------------------------------------------------------

    validator = PandasCodeValidator()
    validator.visit(tree)

    if validator.errors:
        unique_errors = list(
            dict.fromkeys(validator.errors)
        )

        return ValidationResult(
            ok=False,
            error="; ".join(unique_errors),
        )

    # --------------------------------------------------------
    # 5. Require result assignment
    # --------------------------------------------------------

    result_validation = _validate_result_assignment(
        tree
    )

    if not result_validation.ok:
        return result_validation

    return ValidationResult(ok=True)


# ============================================================
# EXECUTE AI QUERY
# ============================================================


def execute_ai_query(
    code: str,
    df: pd.DataFrame,
) -> ExecutionResult:
    """
    Validate and execute AI-generated pandas code.

    Parameters
    ----------
    code:
        Python/pandas code generated by the AI.

    df:
        Input pandas DataFrame.

    Returns
    -------
    ExecutionResult
        Successful result or a structured error.
    """

    # --------------------------------------------------------
    # Validate DataFrame
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

    validation = validate_generated_code(code)

    if not validation.ok:
        return ExecutionResult(
            ok=False,
            error=validation.error,
        )

    # --------------------------------------------------------
    # Execute using project's runtime safety layer
    # --------------------------------------------------------

    try:
        execution_result = safe_execute_pandas_code(
            code,
            df,
        )

    except Exception as ex:
        return ExecutionResult(
            ok=False,
            error=(
                "Unexpected execution error: "
                f"{type(ex).__name__}: {ex}"
            ),
        )

    # --------------------------------------------------------
    # Protect against broken execution-layer responses
    # --------------------------------------------------------

    if execution_result is None:
        return ExecutionResult(
            ok=False,
            error=(
                "The execution layer returned no result."
            ),
        )

    if not isinstance(
        execution_result,
        ExecutionResult,
    ):
        return ExecutionResult(
            ok=False,
            error=(
                "The execution layer returned an "
                "unexpected result type."
            ),
        )

    return execution_result


# ============================================================
# FALLBACK EXECUTION
# ============================================================


def try_execute_with_fallback(
    primary_code: str,
    fallback_code: Optional[str],
    df: pd.DataFrame,
) -> ExecutionResult:
    """
    Execute a primary query and optionally a fallback query.

    Both queries pass through the complete validation pipeline.

    Parameters
    ----------
    primary_code:
        Primary AI-generated query.

    fallback_code:
        Optional alternative query.

    df:
        Dataset to analyze.

    Returns
    -------
    ExecutionResult
    """

    # --------------------------------------------------------
    # Primary query
    # --------------------------------------------------------

    primary_result = execute_ai_query(
        primary_code,
        df,
    )

    if primary_result.ok:
        return primary_result

    # --------------------------------------------------------
    # No fallback supplied
    # --------------------------------------------------------

    if not fallback_code or not fallback_code.strip():
        return primary_result

    # --------------------------------------------------------
    # Fallback query
    # --------------------------------------------------------

    fallback_result = execute_ai_query(
        fallback_code,
        df,
    )

    if fallback_result.ok:
        return fallback_result

    # --------------------------------------------------------
    # Both failed
    # --------------------------------------------------------

    primary_error = (
        primary_result.error
        or "Unknown primary-query error."
    )

    fallback_error = (
        fallback_result.error
        or "Unknown fallback-query error."
    )

    return ExecutionResult(
        ok=False,
        error=(
            "Both analysis queries failed.\n\n"
            f"Primary query error: {primary_error}\n\n"
            f"Fallback query error: {fallback_error}"
        ),
    )
