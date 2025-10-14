"""
Code execution utilities for LiveCodeBench.

This module provides safe code execution against test cases with proper
sandboxing, timeout handling, and result parsing.
"""

import json
import os
import re
import subprocess
import sys
import tempfile
from typing import Any, Dict, List


class CodeExecutor:
    """Safe code executor for LiveCodeBench coding tasks"""

    # Execution timeout in seconds
    EXECUTION_TIMEOUT = 10

    @classmethod
    def execute_code_against_tests(
        cls, code: str, test_cases: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Execute code against test cases and return results.

        Args:
            code: The Python code to execute
            test_cases: List of test cases with 'input' and 'output' fields

        Returns:
            Dictionary with execution results including:
            - all_passed: bool
            - tests_passed: int
            - execution_status: str ("success", "failed", "timeout", "error")
            - details: str (execution details/errors)
            - stdout: str (if available)
            - stderr: str (if available)
        """

        if not test_cases:
            return {
                "all_passed": True,
                "tests_passed": 0,
                "execution_status": "success",
                "details": "No test cases to run",
            }

        try:
            # Create test runner code
            test_runner_code = cls._create_test_runner(code, test_cases)

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
    def _create_test_runner(cls, code: str, test_cases: List[Dict[str, str]]) -> str:
        """Create a test runner that executes the code against each test case"""

        # Escape the code string for embedding
        escaped_code = code.replace('"""', '\\"\\"\\"')

        test_runner = f'''
import sys
import io
from contextlib import redirect_stdout, redirect_stderr

# The user's code
code = """{escaped_code}"""

# Test cases
test_cases = {json.dumps(test_cases)}

def run_test_case(code, test_case):
    input_data = test_case['input']
    expected_output = test_case['output']
    
    # Capture stdout and stderr
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    
    try:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            # Create a StringIO object to simulate stdin
            stdin_backup = sys.stdin
            sys.stdin = io.StringIO(input_data)
            
            # Execute the code
            try:
                exec(code, {{}})
            except SystemExit as e_exit:
                return False, f"\\nERROR: exit() was called"
            finally:
                # Restore stdin
                sys.stdin = stdin_backup
            
            actual_output = stdout_capture.getvalue().strip()
            
            if actual_output.strip() == expected_output.strip():
                return True, f"\\nPASS: Input='{{input_data.strip()}}', Expected='{{expected_output}}', Got='{{actual_output}}'"
            else:
                return False, f"\\nFAIL: Input='{{input_data.strip()}}', Expected='{{expected_output}}', Got='{{actual_output}}'"

    except Exception as e:
        return False, f"\\nERROR: Input='{{input_data.strip()}}', Error='{{str(e)}}'"

# Run all test cases
all_passed = True
tests_passed = 0

for i, test_case in enumerate(test_cases):
    passed, message = run_test_case(code, test_case)
    print(message)
    if passed:
        tests_passed += 1
    else:
        all_passed = False

print(f"\\nTests passed: {{tests_passed}}/{{len(test_cases)}}")

if all_passed:
    sys.exit(0)
else:
    sys.exit(1)
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
        return len(re.findall(r"\\nPASS:", stdout))

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
    def find_bailout(cls, completion: str) -> bool:
        """Check if the completion contains a bailout pattern"""
        pattern = re.compile(r"```END_WITHOUT_SUBMISSION\n(.*?)```", re.DOTALL)
        matches = pattern.findall(completion)
        return len(matches) >= 1

    @classmethod
    def evaluate_model_output(
        cls, 
        model_output: str, 
        test_cases: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation of model output against a test set.
        
        This unified method handles all validation, edge case detection, and execution
        logic in one place to eliminate duplication between rollout generator and evaluator.
        
        Args:
            model_output: The model's response text
            test_cases: List of test cases with 'input' and 'output' fields
        
        Returns:
            Dictionary with comprehensive evaluation results:
            - is_bailout: bool
            - has_code: bool  
            - has_forbidden_patterns: bool
            - code: str (extracted code or empty string)
            - execution_result: Dict (result from execute_code_against_tests)
            - validation_status: str ("bailout", "no_code", "forbidden", "valid")
        """
        result = {
            "model_output": model_output,
            "is_bailout": False,
            "has_code": False,
            "has_forbidden_patterns": False,
            "code": "",
            "execution_result": None,
            "validation_status": "valid"
        }
        
        # Check for bailout first
        if cls.find_bailout(model_output):
            result.update({
                "is_bailout": True,
                "validation_status": "bailout"
            })
            return result
        
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
        
        # Execute against test cases
        execution_result = cls.execute_code_against_tests(code, test_cases)
        result["execution_result"] = execution_result
        
        return result
