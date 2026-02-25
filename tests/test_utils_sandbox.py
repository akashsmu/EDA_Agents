import pytest
from eda_agents.utils.sandbox import run_code_sandboxed_subprocess

def test_run_code_sandboxed_subprocess_success():
    code = "print('Hello, Sandbox!')"
    result = run_code_sandboxed_subprocess(code)
    assert result.strip() == "Hello, Sandbox!"

def test_run_code_sandboxed_subprocess_syntax_error():
    code = "print('Unclosed string"
    result = run_code_sandboxed_subprocess(code)
    assert "SyntaxError:" in result

def test_run_code_sandboxed_subprocess_runtime_error():
    code = "1 / 0"
    result = run_code_sandboxed_subprocess(code)
    # Stderr is appended to result
    assert "[Stderr]:" in result
    assert "ZeroDivisionError" in result

def test_run_code_sandboxed_subprocess_imports():
    code = "import math\nprint(math.sqrt(16))"
    result = run_code_sandboxed_subprocess(code)
    assert result.strip() == "4.0"
