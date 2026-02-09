import sys
import subprocess
import tempfile
import os
import ast

def run_code_sandboxed_subprocess(code: str, libraries: list = None) -> str:
    """
    Executes the provided Python code in a separate subprocess.
    Captured stdout and stderr are returned.
    """
    # strict basic security check (very minimal)
    try:
        ast.parse(code)
    except SyntaxError as e:
        return f"SyntaxError: {e}"

    # Create a temporary file to hold the code
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
        temp_file.write(code)
        temp_file_path = temp_file.name

    try:
        # Prepare the command to run the script
        cmd = [sys.executable, temp_file_path]
        
        # execution
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60 # 1 minute timeout
        )
        
        output = result.stdout
        if result.stderr:
            output += f"\n[Stderr]:\n{result.stderr}"
            
        return output

    except subprocess.TimeoutExpired:
        return "Execution timed out."
    except Exception as e:
        return f"Execution failed: {e}"
    finally:
        # Cleanup
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
