"""
Code execution utilities for Code Generation tasks.

This module provides safe code execution against assert statements with proper
sandboxing, timeout handling, and result parsing.
"""

import os
import re
import subprocess
import sys
import tempfile
from typing import Any, Dict, List


class CodeGenerationExecutor:
    """Safe code executor for Code Generation tasks using assert statements"""

    # Execution timeout in seconds
    EXECUTION_TIMEOUT = 10

    @classmethod
    def execute_code_against_asserts(
        cls, code: str, assert_statements: List[str]
    ) -> Dict[str, Any]:
        """
        Execute code against assert statements and return results.

        Args:
            code: The Python code to execute
            assert_statements: List of assert statements like ['assert func(2) == False']

        Returns:
            Dictionary with execution results including:
            - all_passed: bool
            - tests_passed: int
            - execution_status: str ("success", "failed", "timeout", "error")
            - details: str (execution details/errors)
            - stdout: str (if available)
            - stderr: str (if available)
        """

        if not assert_statements:
            return {
                "all_passed": True,
                "tests_passed": 0,
                "execution_status": "success",
                "details": "No test cases to run",
            }

        try:
            # Create test runner code
            test_runner_code = cls._create_assert_test_runner(code, assert_statements)

            # Execute in temporary file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(test_runner_code)
                temp_file = f.name

            try:
                # Execute with timeout
                result = subprocess.run(
                    [sys.executable, temp_file],
                    capture_output=True,
                    text=True,
                    timeout=cls.EXECUTION_TIMEOUT,
                )

                all_passed = result.returncode == 0

                # Count passed tests from output
                tests_passed = cls._count_passed_tests(result.stdout)

                return {
                    "all_passed": all_passed,
                    "tests_passed": tests_passed,
                    "execution_status": "success" if all_passed else "failed",
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "details": result.stdout if result.stdout else result.stderr,
                }

            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_file)
                except OSError:
                    pass

        except subprocess.TimeoutExpired:
            return {
                "all_passed": False,
                "tests_passed": 0,
                "execution_status": "timeout",
                "details": f"Code execution timed out after {cls.EXECUTION_TIMEOUT} seconds",
            }
        except Exception as e:
            return {
                "all_passed": False,
                "tests_passed": 0,
                "execution_status": "error",
                "details": f"Execution error: {str(e)}",
            }

    @classmethod
    def _create_assert_test_runner(cls, code: str, assert_statements: List[str]) -> str:
        """Create a test runner that executes the code and runs assert statements"""

        # Escape the code string for embedding
        escaped_code = code.replace('"""', '\\"\\"\\"')

        test_runner = f'''
import sys

# The user's code
code = """{escaped_code}"""

# Assert statements
assert_statements = {assert_statements}

def run_assert_tests(code, assert_statements):
    tests_passed = 0
    all_passed = True
    
    try:
        # Execute the user's code in a namespace
        namespace = {{}}
        exec(code, namespace)
        
        # Run each assert statement
        for i, assert_stmt in enumerate(assert_statements):
            try:
                # Execute the assert statement in the same namespace
                exec(assert_stmt, namespace)
                print(f"PASS: {{assert_stmt}}")
                tests_passed += 1
            except AssertionError:
                print(f"FAIL: {{assert_stmt}} (AssertionError)")
                all_passed = False
            except Exception as e:
                print(f"ERROR: {{assert_stmt}} ({{type(e).__name__}}: {{e}})")
                all_passed = False
                
    except Exception as e:
        print(f"CODE_ERROR: Failed to execute user code: {{type(e).__name__}}: {{e}}")
        all_passed = False
    
    print(f"\\nTests passed: {{tests_passed}}/{{len(assert_statements)}}")
    
    if all_passed:
        sys.exit(0)
    else:
        sys.exit(1)

# Run the tests
run_assert_tests(code, assert_statements)
'''
        return test_runner

    @classmethod
    def _count_passed_tests(cls, stdout: str) -> int:
        """Count the number of passed tests from stdout"""
        if not stdout:
            return 0

        # Look for the summary line
        pattern = r"Tests passed: (\d+)/\d+"
        match = re.search(pattern, stdout)
        if match:
            return int(match.group(1))

        # Fallback: count PASS lines
        return len(re.findall(r"PASS:", stdout))

    @classmethod
    def extract_code(cls, completion: str) -> str:
        """Extract Python code from model completion"""
        # Look for ```python code blocks first
        pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
        matches = pattern.findall(completion)
        if matches:
            return matches[-1].strip()  # Take the last match

        # Fallback to any ``` code blocks
        pattern = re.compile(r"```\n(.*?)```", re.DOTALL)
        matches = pattern.findall(completion)
        if matches:
            return matches[-1].strip()

        # No code blocks found
        return ""


    @classmethod
    def evaluate_model_output(
        cls, 
        model_output: str, 
        assert_statements: List[str]
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation of model output against assert statements.
        
        This unified method handles all validation, edge case detection, and execution
        logic for code generation tasks.
        
        Args:
            model_output: The model's response text
            assert_statements: List of assert statements like ['assert func(2) == False']
        
        Returns:
            Dictionary with comprehensive evaluation results:
            - has_code: bool  
            - has_forbidden_patterns: bool
            - code: str (extracted code or empty string)
            - execution_result: Dict (result from execute_code_against_asserts)
            - validation_status: str ("no_code", "forbidden", "valid")
        """
        result = {
            "model_output": model_output,
            "has_code": False,
            "has_forbidden_patterns": False,
            "code": "",
            "execution_result": None,
            "validation_status": "valid"
        }
        
        # Extract code
        code = cls.extract_code(model_output)
        result["code"] = code
        
        if not code:
            result.update({
                "has_code": False,
                "validation_status": "no_code"
            })
            return result
        
        result["has_code"] = True
        
        # Check for forbidden patterns
        if 'if __name__ == "__main__":' in code or 'exit()' in code:
            result.update({
                "has_forbidden_patterns": True,
                "validation_status": "forbidden"
            })
            return result
        
        # Execute against assert statements
        execution_result = cls.execute_code_against_asserts(code, assert_statements)
        result["execution_result"] = execution_result
        
        return result
