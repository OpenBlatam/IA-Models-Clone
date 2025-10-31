from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
BUFFER_SIZE = 1024

import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Callable, Any, Optional
import re
import logging
from functools import partial
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Pre-commit hooks for Product Descriptions Feature
Functional programming approach with descriptive naming
"""


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type aliases for clarity
FilePath = Path
FileList = List[FilePath]
ValidationResult = Dict[str, Any]
HookFunction = Callable[[FileList], ValidationResult]

def get_staged_files() -> FileList:
    """Get list of staged files"""
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            capture_output=True,
            text=True,
            check=True
        )
        return [Path(f) for f in result.stdout.strip().split('\n') if f]
    except subprocess.CalledProcessError:
        return []

def get_python_files(file_list: FileList) -> FileList:
    """Filter Python files from file list"""
    return [f for f in file_list if f.suffix == '.py']

def has_secrets_in_file(file_path: FilePath) -> bool:
    """Check if file contains potential secrets"""
    secret_patterns = [
        r'password\s*=\s*["\'][^"\']+["\']',
        r'secret\s*=\s*["\'][^"\']+["\']',
        r'api_key\s*=\s*["\'][^"\']+["\']',
        r'token\s*=\s*["\'][^"\']+["\']',
        r'key\s*=\s*["\'][^"\']+["\']'
    ]
    
    try:
        content = file_path.read_text()
        return any(re.search(pattern, content, re.IGNORECASE) for pattern in secret_patterns)
    except Exception:
        return False

def validate_commit_message_format(commit_message: str) -> bool:
    """Validate conventional commit message format"""
    pattern = r'^(feat|fix|docs|style|refactor|test|chore|perf|ci|build)(\([a-z-]+\))?: .+'
    return bool(re.match(pattern, commit_message))

def run_black_formatting(file_list: FileList) -> ValidationResult:
    """Run Black code formatter"""
    if not file_list:
        return {"is_successful": True, "message": "No Python files to format"}
    
    try:
        subprocess.run(
            ["black"] + [str(f) for f in file_list],
            check=True,
            capture_output=True
        )
        return {"is_successful": True, "message": "Black formatting completed"}
    except subprocess.CalledProcessError as e:
        return {
            "is_successful": False,
            "message": f"Black formatting failed: {e.stderr.decode()}"
        }

def run_flake8_linting(file_list: FileList) -> ValidationResult:
    """Run Flake8 linting"""
    if not file_list:
        return {"is_successful": True, "message": "No Python files to lint"}
    
    try:
        result = subprocess.run(
            ["flake8"] + [str(f) for f in file_list],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            return {"is_successful": True, "message": "Flake8 linting passed"}
        else:
            return {
                "is_successful": False,
                "message": f"Flake8 linting failed:\n{result.stdout}"
            }
    except FileNotFoundError:
        return {"is_successful": False, "message": "Flake8 not installed"}

def run_mypy_type_checking(file_list: FileList) -> ValidationResult:
    """Run MyPy type checking"""
    if not file_list:
        return {"is_successful": True, "message": "No Python files to type check"}
    
    try:
        result = subprocess.run(
            ["mypy"] + [str(f) for f in file_list],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            return {"is_successful": True, "message": "MyPy type checking passed"}
        else:
            return {
                "is_successful": False,
                "message": f"MyPy type checking failed:\n{result.stdout}"
            }
    except FileNotFoundError:
        return {"is_successful": False, "message": "MyPy not installed"}

def check_for_secrets(file_list: FileList) -> ValidationResult:
    """Check for potential secrets in files"""
    files_with_secrets = [f for f in file_list if has_secrets_in_file(f)]
    
    if files_with_secrets:
        return {
            "is_successful": False,
            "message": f"Potential secrets found in: {[f.name for f in files_with_secrets]}"
        }
    
    return {"is_successful": True, "message": "No secrets detected"}

def check_file_sizes(file_list: FileList, max_size_mb: int = 10) -> ValidationResult:
    """Check if files exceed maximum size"""
    large_files = []
    max_size_bytes = max_size_mb * 1024 * 1024
    
    for file_path in file_list:
        if file_path.exists() and file_path.stat().st_size > max_size_bytes:
            large_files.append(file_path)
    
    if large_files:
        return {
            "is_successful": False,
            "message": f"Large files detected: {[f.name for f in large_files]}"
        }
    
    return {"is_successful": True, "message": "All files within size limits"}

def run_unit_tests() -> ValidationResult:
    """Run unit tests"""
    try:
        result = subprocess.run(
            ["python", "-m", "pytest", "tests/", "-v"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            return {"is_successful": True, "message": "Unit tests passed"}
        else:
            return {
                "is_successful": False,
                "message": f"Unit tests failed:\n{result.stdout}"
            }
    except Exception as e:
        return {"is_successful": False, "message": f"Test execution failed: {e}"}

def validate_imports(file_list: FileList) -> ValidationResult:
    """Validate Python imports"""
    import_patterns = [
        r'^import\s+[a-zA-Z_][a-zA-Z0-9_]*\s*$',
        r'^from\s+[a-zA-Z_][a-zA-Z0-9_.]*\s+import\s+[a-zA-Z_][a-zA-Z0-9_,\s]*$'
    ]
    
    files_with_invalid_imports = []
    
    for file_path in file_list:
        if not file_path.exists():
            continue
            
        try:
            content = file_path.read_text()
            lines = content.split('\n')
            
            for line_num, line in enumerate(lines, 1):
                stripped_line = line.strip()
                if stripped_line.startswith(('import ', 'from ')):
                    is_valid_import = any(
                        re.match(pattern, stripped_line) 
                        for pattern in import_patterns
                    )
                    if not is_valid_import:
                        files_with_invalid_imports.append(f"{file_path.name}:{line_num}")
        except Exception:
            continue
    
    if files_with_invalid_imports:
        return {
            "is_successful": False,
            "message": f"Invalid imports found in: {files_with_invalid_imports}"
        }
    
    return {"is_successful": True, "message": "All imports are valid"}

def check_docstring_coverage(file_list: FileList) -> ValidationResult:
    """Check docstring coverage for functions and classes"""
    files_without_docstrings = []
    
    for file_path in file_list:
        if not file_path.exists():
            continue
            
        try:
            content = file_path.read_text()
            lines = content.split('\n')
            
            has_missing_docstrings = False
            for i, line in enumerate(lines):
                if re.match(r'^\s*(def|class)\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\(', line):
                    # Check if next non-empty line is a docstring
                    next_lines = lines[i+1:]
                    next_non_empty = next((l for l in next_lines if l.strip()), None)
                    
                    if next_non_empty and not next_non_empty.strip().startswith('"""'):
                        has_missing_docstrings = True
                        break
            
            if has_missing_docstrings:
                files_without_docstrings.append(file_path.name)
        except Exception:
            continue
    
    if files_without_docstrings:
        return {
            "is_successful": False,
            "message": f"Missing docstrings in: {files_without_docstrings}"
        }
    
    return {"is_successful": True, "message": "Docstring coverage is adequate"}

def execute_hook(hook_function: HookFunction, file_list: FileList) -> ValidationResult:
    """Execute a hook function with error handling"""
    try:
        return hook_function(file_list)
    except Exception as e:
        return {
            "is_successful": False,
            "message": f"Hook execution failed: {e}"
        }

def run_all_hooks(file_list: FileList) -> List[ValidationResult]:
    """Run all pre-commit hooks"""
    python_files = get_python_files(file_list)
    
    hooks = [
        ("Secret Detection", check_for_secrets),
        ("File Size Check", check_file_sizes),
        ("Black Formatting", run_black_formatting),
        ("Flake8 Linting", run_flake8_linting),
        ("MyPy Type Checking", run_mypy_type_checking),
        ("Import Validation", validate_imports),
        ("Docstring Coverage", check_docstring_coverage),
        ("Unit Tests", lambda _: run_unit_tests())
    ]
    
    results = []
    for hook_name, hook_function in hooks:
        logger.info(f"Running {hook_name}...")
        result = execute_hook(hook_function, python_files)
        result["hook_name"] = hook_name
        results.append(result)
        
        if not result["is_successful"]:
            logger.error(f"{hook_name} failed: {result['message']}")
        else:
            logger.info(f"{hook_name} passed: {result['message']}")
    
    return results

def is_commit_allowed(results: List[ValidationResult]) -> bool:
    """Determine if commit should be allowed"""
    return all(result["is_successful"] for result in results)

def main() -> int:
    """Main pre-commit hook execution"""
    staged_files = get_staged_files()
    
    if not staged_files:
        logger.info("No staged files to check")
        return 0
    
    logger.info(f"Checking {len(staged_files)} staged files")
    results = run_all_hooks(staged_files)
    
    if is_commit_allowed(results):
        logger.info("All pre-commit checks passed")
        return 0
    else:
        logger.error("Pre-commit checks failed")
        return 1

match __name__:
    case "__main__":
    sys.exit(main()) 