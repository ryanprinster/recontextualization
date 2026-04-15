"""Smoke tests for src/dataset_modules/code_extraction.py.

Run directly: ``python tests/test_code_extraction.py``.
"""

import ast
import os
import sys


sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

from src.dataset_modules.code_extraction import (  # noqa: E402
    extract_code,
    strip_top_level_asserts_and_prints,
)


def _structurally_equal(a: str, b: str) -> bool:
    """AST-equivalent comparison — tolerates reformatting by ``ast.unparse``."""
    return ast.dump(ast.parse(a)) == ast.dump(ast.parse(b))


def test_plain_solution():
    code = "def f(x):\n    return x * 2"
    completion = f"Here:\n```python\n{code}\n```"
    assert _structurally_equal(extract_code(completion), code)


def test_top_level_assert_after_def():
    completion = (
        "```python\n"
        "def f(x):\n    return x * 2\n"
        "assert f(1) == 2\n"
        "assert f(2) == 4\n"
        "```"
    )
    expected = "def f(x):\n    return x * 2"
    assert _structurally_equal(extract_code(completion), expected)


def test_assert_inside_function_body_preserved():
    code = "def f(x):\n    assert x > 0\n    return x"
    completion = f"```python\n{code}\n```"
    got = extract_code(completion)
    assert _structurally_equal(got, code)
    assert "assert" in got


def test_multiline_top_level_assert_stripped():
    completion = (
        "```python\n"
        "def f(x):\n    return x\n"
        "assert (\n"
        "    f(1) == 1,\n"
        "    'expected 1',\n"
        ")\n"
        "```"
    )
    expected = "def f(x):\n    return x"
    assert _structurally_equal(extract_code(completion), expected)


def test_two_fences_last_is_asserts_only_falls_back():
    completion = (
        "First:\n"
        "```python\n"
        "def f(x):\n    return x * 2\n"
        "```\n"
        "Verify:\n"
        "```python\n"
        "assert f(1) == 2\n"
        "assert f(2) == 4\n"
        "```"
    )
    expected = "def f(x):\n    return x * 2"
    assert _structurally_equal(extract_code(completion), expected)


def test_two_fences_last_is_print_demo_falls_back():
    completion = (
        "```python\n"
        "def f(x):\n    return x\n"
        "```\n"
        "```python\n"
        "print(f(5))\n"
        "```"
    )
    got = extract_code(completion)
    assert _structurally_equal(got, "def f(x):\n    return x")


def test_print_inside_function_body_preserved():
    code = "def f(x):\n    print(x)\n    return x"
    completion = f"```python\n{code}\n```"
    got = extract_code(completion)
    assert _structurally_equal(got, code)
    assert "print(x)" in got


def test_no_fences_returns_empty():
    assert extract_code("just prose, no code here") == ""


def test_bare_fence_fallback():
    completion = "```\ndef f():\n    return 42\n```"
    got = extract_code(completion)
    assert _structurally_equal(got, "def f():\n    return 42")


def test_python_fence_preferred_over_bare_fence():
    completion = (
        "```\n"
        "irrelevant = 1\n"
        "```\n"
        "```python\n"
        "def f():\n    return 1\n"
        "```"
    )
    got = extract_code(completion)
    assert _structurally_equal(got, "def f():\n    return 1")


def test_unparseable_block_returned_as_is():
    # Syntax error in the only block; extractor shouldn't crash.
    completion = "```python\ndef f(:\n    return 1\n```"
    got = extract_code(completion)
    assert "def f(:" in got


def test_unparseable_last_block_non_empty_is_returned_not_fallback():
    # Documents current behaviour: strip returns input on SyntaxError, so if
    # the unparseable text is non-empty it wins and the prior fence is ignored.
    completion = (
        "```python\n"
        "def f():\n    return 1\n"
        "```\n"
        "```python\n"
        "def g(:\n    return 2\n"
        "```"
    )
    got = extract_code(completion)
    assert "def g(:" in got


def test_py_tag_not_recognised_by_current_extractor():
    # Documents existing limitation: ```py is not matched; since no bare
    # fence exists either, result is empty.
    completion = "```py\ndef f():\n    return 1\n```"
    assert extract_code(completion) == ""


def test_strip_helper_no_asserts_returns_input_verbatim():
    code = "def f():\n    return 1"
    # Exact-string: when nothing is stripped, we skip the unparse round-trip.
    assert strip_top_level_asserts_and_prints(code) == code


def test_strip_helper_only_asserts_becomes_empty():
    out = strip_top_level_asserts_and_prints("assert 1 == 1\nassert 2 == 2").strip()
    assert out == ""


def test_strip_helper_only_prints_becomes_empty():
    out = strip_top_level_asserts_and_prints("print(1)\nprint('hi')").strip()
    assert out == ""


def main():
    tests = [
        (name, obj)
        for name, obj in globals().items()
        if name.startswith("test_") and callable(obj)
    ]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}: {e}")
        except Exception as e:
            failed += 1
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
