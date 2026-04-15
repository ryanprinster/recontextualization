"""Shared code extraction for dataset-module executors.

The model-response format is Markdown with fenced code blocks. `extract_code`
walks candidate fences from last to first, strips top-level ``assert``
statements and ``print(...)`` calls from each (asserts would raise at
``exec()`` time and fail every test; stray prints pollute captured stdout),
and returns the first block with non-empty content.
"""

import ast
import re


_PYTHON_FENCE = re.compile(r"```python\n(.*?)```", re.DOTALL)
_BARE_FENCE = re.compile(r"```\n(.*?)```", re.DOTALL)


def _is_top_level_print(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "print"
    )


def strip_top_level_asserts_and_prints(code: str) -> str:
    """Remove module-scope ``assert`` statements and ``print(...)`` calls;
    preserve either when nested inside function/class/loop bodies. Returns
    input unchanged if unparseable or if nothing to strip."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    kept = [
        n for n in tree.body
        if not isinstance(n, ast.Assert) and not _is_top_level_print(n)
    ]
    if len(kept) == len(tree.body):
        return code
    tree.body = kept
    return ast.unparse(tree)


def extract_code(completion: str) -> str:
    """Extract Python code from a model completion.

    Prefers ```python``` fences; falls back to bare ``` fences only when no
    python-tagged fences are present. Walks candidates from last to first,
    stripping top-level asserts and prints from each, and returns the first
    one whose stripped body is non-empty.
    """
    python_matches = _PYTHON_FENCE.findall(completion)
    candidates = python_matches if python_matches else _BARE_FENCE.findall(completion)
    for raw in reversed(candidates):
        cleaned = strip_top_level_asserts_and_prints(raw.strip()).strip()
        if cleaned:
            return cleaned
    return ""
