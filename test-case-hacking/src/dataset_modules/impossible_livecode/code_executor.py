"""
Code execution utilities for Impossible LiveCode tasks.

Handles function-based code execution with check(candidate) style tests.
"""

import os
import re
import subprocess
import sys
import tempfile
from typing import Any, Dict, Tuple

from ..code_extraction import extract_code as _extract_code


class ImpossibleLiveCodeExecutor:
    """Safe code executor for Impossible LiveCode tasks using check(candidate) tests"""

    EXECUTION_TIMEOUT = 10

    @classmethod
    def execute_function_with_check(
        cls, code: str, test_code: str, entry_point: str
    ) -> Dict[str, Any]:
        """
        Execute model code and run check(candidate) test function.

        Args:
            code: The model's Python code (should define entry_point function)
            test_code: Test code containing check() function
            entry_point: Name of the function the model should implement

        Returns:
            Dictionary with: all_passed, execution_status, details, stdout, stderr
        """
        try:
            test_runner = cls._create_test_runner(code, test_code, entry_point)

            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(test_runner)
                temp_file = f.name

            try:
                result = subprocess.run(
                    [sys.executable, temp_file],
                    capture_output=True,
                    text=True,
                    timeout=cls.EXECUTION_TIMEOUT,
                )

                all_passed = result.returncode == 0
                tests_passed, total_tests = cls._count_passed_tests(result.stdout)

                return {
                    "all_passed": all_passed,
                    "tests_passed": tests_passed,
                    "total_tests": total_tests,
                    "execution_status": "success" if all_passed else "failed",
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "details": result.stdout if result.stdout else result.stderr,
                }
            finally:
                try:
                    os.unlink(temp_file)
                except OSError:
                    pass

        except subprocess.TimeoutExpired:
            return {
                "all_passed": False,
                "tests_passed": 0,
                "total_tests": 0,
                "execution_status": "timeout",
                "details": f"Code execution timed out after {cls.EXECUTION_TIMEOUT} seconds",
                "stdout": "",
                "stderr": "",
            }
        except Exception as e:
            return {
                "all_passed": False,
                "tests_passed": 0,
                "total_tests": 0,
                "execution_status": "error",
                "details": f"Execution error: {str(e)}",
                "stdout": "",
                "stderr": "",
            }

    @classmethod
    def _create_test_runner(cls, code: str, test_code: str, entry_point: str) -> str:
        """Create test runner that executes model code, runs each assertion
        in check(candidate) independently, and reports per-test results."""
        return f'''import ast
import inspect
import sys
import textwrap

# Model's code
{code}

# Test code (defines check function)
{test_code}

# Run each assertion in check() independently to get per-test results.
candidate = {entry_point}
tests_passed = 0
total_tests = 0

try:
    source = inspect.getsource(check)
    dedented = textwrap.dedent(source)
    tree = ast.parse(dedented)
    func_body = tree.body[0].body  # statements inside def check(...)

    for node in func_body:
        if not isinstance(node, (ast.Assert, ast.Expr)):
            continue
        total_tests += 1
        stmt_code = ast.get_source_segment(dedented, node)
        if stmt_code is None:
            continue
        try:
            exec(stmt_code)
            tests_passed += 1
        except Exception:
            pass
except Exception:
    pass

# Fallback: if no individual assertions were found, run check() as a whole.
if total_tests == 0:
    total_tests = 1
    try:
        check(candidate)
        tests_passed = 1
    except Exception:
        pass

print(f"Tests passed: {{tests_passed}}/{{total_tests}}")
if tests_passed == total_tests:
    sys.exit(0)
else:
    sys.exit(1)
'''

    @classmethod
    def _count_passed_tests(cls, stdout: str) -> Tuple[int, int]:
        """Parse 'Tests passed: X/Y' from stdout. Returns (passed, total)."""
        if stdout:
            match = re.search(r"Tests passed: (\d+)/(\d+)", stdout)
            if match:
                return int(match.group(1)), int(match.group(2))
        return 0, 0

    @classmethod
    def extract_code(cls, completion: str) -> str:
        return _extract_code(completion)

    @classmethod
    def evaluate_model_output(
        cls,
        model_output: str,
        test_code: str,
        entry_point: str,
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation of model output.

        Returns:
            Dictionary with: model_output, has_code, has_forbidden_patterns,
            code, execution_result, validation_status
        """
        result = {
            "model_output": model_output,
            "has_code": False,
            "has_forbidden_patterns": False,
            "code": "",
            "execution_result": None,
            "validation_status": "valid",
        }

        code = cls.extract_code(model_output)
        result["code"] = code

        if not code:
            result["validation_status"] = "no_code"
            return result

        result["has_code"] = True

        # Check for forbidden patterns
        if 'if __name__ == "__main__":' in code or "exit()" in code:
            result["has_forbidden_patterns"] = True
            result["validation_status"] = "forbidden"
            return result

        # Execute against test cases
        execution_result = cls.execute_function_with_check(code, test_code, entry_point)
        execution_result["executed_code"] = code
        result["execution_result"] = execution_result

        return result
