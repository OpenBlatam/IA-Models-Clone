"""
HumanEval Benchmark - Code generation evaluation.

HumanEval tests code generation capabilities by asking models
to complete Python functions based on docstrings.
"""

import logging
from typing import Dict, Any, List, Optional
import ast
import re
import subprocess
import tempfile
from pathlib import Path

from .base_benchmark import BaseBenchmark

logger = logging.getLogger(__name__)


class HumanEvalBenchmark(BaseBenchmark):
    """
    Benchmark HumanEval (Code generation).
    
    HumanEval tests code generation capabilities by asking models
    to complete Python functions based on docstrings.
    
    Evaluation can be done in two ways:
    1. Syntax validation (fast, less accurate)
    2. Execution-based (slower, more accurate)
    """
    
    def __init__(
        self,
        shots: int = 0,
        max_samples: int = None,
        evaluation_mode: str = "syntax",  # "syntax" or "execution"
    ):
        """
        Initialize HumanEval benchmark.
        
        Args:
            shots: Number of few-shot examples (typically 0)
            max_samples: Maximum number of samples to evaluate
            evaluation_mode: Evaluation mode ("syntax" or "execution")
        """
        super().__init__(
            name="humaneval",
            dataset_name="openai/humaneval",
            dataset_config=None,
            shots=shots,
            batch_size=1,
            max_samples=max_samples
        )
        self.evaluation_mode = evaluation_mode
    
    def format_prompt(self, example: Dict[str, Any]) -> str:
        """
        Format prompt for HumanEval.
        
        Format:
        Complete the following function:
        [function signature with docstring]
        
        Args:
            example: Example dictionary
        
        Returns:
            Formatted prompt
        """
        prompt = example.get("prompt", "")
        
        # Ensure prompt ends with function signature
        if not prompt.strip().endswith(":"):
            prompt += "\n    "
        
        return prompt
    
    def evaluate_answer(self, prediction: str, example: Dict[str, Any]) -> bool:
        """
        Evaluate the model's answer.
        
        HumanEval uses execution-based evaluation. For syntax mode,
        we check if the code is syntactically valid. For execution mode,
        we run the code against test cases.
        
        Args:
            prediction: Model prediction
            example: Example dictionary with test cases
        
        Returns:
            True if answer is correct
        """
        # Extract code from prediction
        code = self._extract_code(prediction)
        
        if not code:
            return False
        
        # Check syntax validity
        try:
            ast.parse(code)
        except SyntaxError as e:
            logger.debug(f"Syntax error in generated code: {e}")
            return False
        
        # Check if code contains the function
        function_name = example.get("entry_point", "")
        if function_name and function_name not in code:
            logger.debug(f"Function {function_name} not found in generated code")
            return False
        
        # Execution-based evaluation
        if self.evaluation_mode == "execution":
            return self._evaluate_execution(code, example)
        
        # Syntax-based evaluation (default)
        return True
    
    def _extract_code(self, prediction: str) -> str:
        """
        Extract Python code from prediction.
        
        Args:
            prediction: Model prediction
        
        Returns:
            Extracted code string
        """
        # Remove markdown code blocks if present
        code = re.sub(r'```python\s*\n?', '', prediction)
        code = re.sub(r'```\s*\n?', '', code)
        code = re.sub(r'```', '', code)
        
        # Try to extract function definition
        lines = code.split('\n')
        code_lines = []
        in_function = False
        indent_level = None
        
        for line in lines:
            # Start of function
            if line.strip().startswith('def '):
                in_function = True
                indent_level = len(line) - len(line.lstrip())
                code_lines.append(line)
            # Continue function
            elif in_function:
                current_indent = len(line) - len(line.lstrip())
                # Still in function (same or deeper indentation)
                if line.strip() and (indent_level is None or current_indent > indent_level):
                    code_lines.append(line)
                # End of function (less indentation or new top-level definition)
                elif line.strip() and current_indent <= indent_level:
                    if line.strip().startswith('def '):
                        # New function, end previous one
                        break
                    else:
                        # Top-level code, end function
                        break
        
        if code_lines:
            return '\n'.join(code_lines)
        
        return code.strip()
    
    def _evaluate_execution(
        self,
        code: str,
        example: Dict[str, Any]
    ) -> bool:
        """
        Evaluate code by executing test cases.
        
        Args:
            code: Generated code
            example: Example dictionary with test cases
        
        Returns:
            True if all tests pass
        """
        test_code = example.get("test", "")
        if not test_code:
            # No test cases available, fall back to syntax check
            return True
        
        try:
            # Create temporary file with code and tests
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.py',
                delete=False
            ) as f:
                f.write(code)
                f.write("\n\n")
                f.write(test_code)
                temp_file = Path(f.name)
            
            try:
                # Run the code
                result = subprocess.run(
                    ["python", str(temp_file)],
                    capture_output=True,
                    text=True,
                    timeout=5,  # 5 second timeout
                )
                
                # Check if execution was successful
                return result.returncode == 0
            finally:
                # Clean up
                temp_file.unlink()
        
        except subprocess.TimeoutExpired:
            logger.debug("Code execution timed out")
            return False
        except Exception as e:
            logger.debug(f"Error executing code: {e}")
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get benchmark-specific metrics.
        
        Returns:
            Dictionary with additional metrics
        """
        base_metrics = super().get_metrics()
        
        # Add HumanEval-specific metrics
        base_metrics.update({
            "evaluation_mode": self.evaluation_mode,
        })
        
        return base_metrics
