"""
Automated Test Generator for HeyGen AI
====================================

Automated system for generating comprehensive test cases based on function analysis.
This module provides intelligent test generation with pattern recognition and
comprehensive coverage analysis.
"""

import ast
import inspect
import re
from typing import Any, Dict, List, Optional, Callable, Set
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class TestPattern:
    """Represents a test pattern for specific function types"""
    name: str
    description: str
    test_cases: List[str]
    assertions: List[str]
    edge_cases: List[Any]
    error_cases: List[Any]


class AutomatedTestGenerator:
    """Automated test case generator with intelligent pattern recognition"""
    
    def __init__(self):
        self.test_patterns = self._load_test_patterns()
        self.analysis_rules = self._load_analysis_rules()
        
    def _load_test_patterns(self) -> Dict[str, TestPattern]:
        """Load predefined test patterns for different function types"""
        return {
            "validation_function": TestPattern(
                name="validation_function",
                description="Functions that validate input parameters",
                test_cases=[
                    "test_valid_input",
                    "test_invalid_input",
                    "test_empty_input",
                    "test_null_input",
                    "test_boundary_values"
                ],
                assertions=[
                    "assert result is not None",
                    "assert isinstance(result, expected_type)",
                    "assert result == expected_value"
                ],
                edge_cases=["", None, 0, -1, float('inf')],
                error_cases=["invalid_type", "malformed_data"]
            ),
            
            "async_function": TestPattern(
                name="async_function",
                description="Asynchronous functions",
                test_cases=[
                    "test_async_execution",
                    "test_async_timeout",
                    "test_async_cancellation",
                    "test_async_error_handling"
                ],
                assertions=[
                    "assert result is not None",
                    "assert execution_time < timeout"
                ],
                edge_cases=[None, ""],
                error_cases=["timeout_error", "cancellation_error"]
            ),
            
            "data_processing": TestPattern(
                name="data_processing",
                description="Functions that process data",
                test_cases=[
                    "test_small_dataset",
                    "test_large_dataset",
                    "test_empty_dataset",
                    "test_malformed_data",
                    "test_data_transformation"
                ],
                assertions=[
                    "assert len(result) == expected_length",
                    "assert result is not None"
                ],
                edge_cases=[[], {}, None],
                error_cases=["corrupted_data", "invalid_format"]
            ),
            
            "api_function": TestPattern(
                name="api_function",
                description="API-related functions",
                test_cases=[
                    "test_successful_request",
                    "test_failed_request",
                    "test_timeout",
                    "test_authentication",
                    "test_rate_limiting"
                ],
                assertions=[
                    "assert response.status_code == 200",
                    "assert response.json() is not None"
                ],
                edge_cases=[None, ""],
                error_cases=["network_error", "auth_error", "timeout"]
            )
        }
    
    def _load_analysis_rules(self) -> Dict[str, List[str]]:
        """Load rules for analyzing function characteristics"""
        return {
            "validation_indicators": [
                "validate", "check", "verify", "assert", "ensure",
                "is_valid", "is_empty", "is_null", "required"
            ],
            "async_indicators": [
                "async def", "await", "asyncio", "coroutine"
            ],
            "data_processing_indicators": [
                "process", "transform", "convert", "parse", "serialize",
                "deserialize", "filter", "map", "reduce"
            ],
            "api_indicators": [
                "request", "response", "http", "api", "endpoint",
                "client", "server", "url", "method"
            ],
            "error_handling_indicators": [
                "try", "except", "raise", "error", "exception",
                "catch", "handle", "recover"
            ]
        }
    
    def analyze_function(self, func: Callable) -> Dict[str, Any]:
        """Analyze a function to determine its characteristics"""
        analysis = {
            "name": func.__name__,
            "is_async": inspect.iscoroutinefunction(func),
            "parameters": list(inspect.signature(func).parameters.keys()),
            "return_annotation": str(inspect.signature(func).return_annotation),
            "docstring": inspect.getdoc(func) or "",
            "source_code": inspect.getsource(func),
            "characteristics": [],
            "complexity_score": 0,
            "test_patterns": []
        }
        
        # Analyze source code for characteristics
        source_lower = analysis["source_code"].lower()
        docstring_lower = analysis["docstring"].lower()
        
        for category, indicators in self.analysis_rules.items():
            for indicator in indicators:
                if indicator in source_lower or indicator in docstring_lower:
                    analysis["characteristics"].append(category.replace("_indicators", ""))
                    break
        
        # Determine applicable test patterns
        for pattern_name, pattern in self.test_patterns.items():
            if self._pattern_applies(pattern_name, analysis):
                analysis["test_patterns"].append(pattern_name)
        
        # Calculate complexity score
        analysis["complexity_score"] = self._calculate_complexity(analysis["source_code"])
        
        return analysis
    
    def _pattern_applies(self, pattern_name: str, analysis: Dict[str, Any]) -> bool:
        """Check if a test pattern applies to the function"""
        characteristics = analysis["characteristics"]
        
        if pattern_name == "validation_function":
            return "validation" in characteristics
        elif pattern_name == "async_function":
            return analysis["is_async"]
        elif pattern_name == "data_processing":
            return "data_processing" in characteristics
        elif pattern_name == "api_function":
            return "api" in characteristics
        
        return False
    
    def _calculate_complexity(self, source_code: str) -> int:
        """Calculate cyclomatic complexity"""
        try:
            tree = ast.parse(source_code)
            complexity = 1
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                    complexity += 1
                elif isinstance(node, ast.ExceptHandler):
                    complexity += 1
                elif isinstance(node, ast.BoolOp):
                    complexity += len(node.values) - 1
            
            return complexity
        except:
            return 1
    
    def generate_test_cases(self, func: Callable, num_cases: int = 10) -> List[Dict[str, Any]]:
        """Generate test cases for a function"""
        analysis = self.analyze_function(func)
        test_cases = []
        
        # Generate test cases based on applicable patterns
        for pattern_name in analysis["test_patterns"]:
            pattern = self.test_patterns[pattern_name]
            
            for i, test_case_name in enumerate(pattern.test_cases):
                if len(test_cases) >= num_cases:
                    break
                
                test_case = {
                    "name": f"test_{analysis['name']}_{test_case_name}",
                    "description": f"Test {analysis['name']} - {test_case_name}",
                    "pattern": pattern_name,
                    "parameters": self._generate_parameters(analysis, pattern, i),
                    "assertions": pattern.assertions,
                    "setup_code": self._generate_setup_code(analysis, pattern),
                    "teardown_code": self._generate_teardown_code(analysis, pattern),
                    "async_test": analysis["is_async"],
                    "expected_exception": None
                }
                
                # Add edge cases and error cases
                if "edge" in test_case_name or "boundary" in test_case_name:
                    test_case["parameters"] = self._generate_edge_case_parameters(analysis, pattern)
                elif "invalid" in test_case_name or "error" in test_case_name:
                    test_case["parameters"] = self._generate_error_case_parameters(analysis, pattern)
                    test_case["expected_exception"] = "ValueError"
                
                test_cases.append(test_case)
        
        # Fill remaining slots with generic test cases
        while len(test_cases) < num_cases:
            test_case = {
                "name": f"test_{analysis['name']}_generic_{len(test_cases)}",
                "description": f"Generic test for {analysis['name']}",
                "pattern": "generic",
                "parameters": self._generate_generic_parameters(analysis),
                "assertions": ["assert result is not None"],
                "setup_code": "",
                "teardown_code": "",
                "async_test": analysis["is_async"],
                "expected_exception": None
            }
            test_cases.append(test_case)
        
        return test_cases[:num_cases]
    
    def _generate_parameters(self, analysis: Dict[str, Any], pattern: TestPattern, case_index: int) -> Dict[str, Any]:
        """Generate parameters for a test case"""
        parameters = {}
        
        for param_name in analysis["parameters"]:
            if "valid" in pattern.test_cases[case_index]:
                parameters[param_name] = self._generate_valid_value(param_name)
            else:
                parameters[param_name] = self._generate_invalid_value(param_name)
        
        return parameters
    
    def _generate_edge_case_parameters(self, analysis: Dict[str, Any], pattern: TestPattern) -> Dict[str, Any]:
        """Generate edge case parameters"""
        parameters = {}
        
        for param_name in analysis["parameters"]:
            if pattern.edge_cases:
                parameters[param_name] = pattern.edge_cases[0]  # Use first edge case
            else:
                parameters[param_name] = None
        
        return parameters
    
    def _generate_error_case_parameters(self, analysis: Dict[str, Any], pattern: TestPattern) -> Dict[str, Any]:
        """Generate error case parameters"""
        parameters = {}
        
        for param_name in analysis["parameters"]:
            if pattern.error_cases:
                parameters[param_name] = pattern.error_cases[0]  # Use first error case
            else:
                parameters[param_name] = "invalid_value"
        
        return parameters
    
    def _generate_generic_parameters(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate generic parameters"""
        parameters = {}
        
        for param_name in analysis["parameters"]:
            parameters[param_name] = self._generate_valid_value(param_name)
        
        return parameters
    
    def _generate_valid_value(self, param_name: str) -> Any:
        """Generate a valid value for a parameter"""
        if "id" in param_name.lower():
            return f"test_{param_name}_123"
        elif "name" in param_name.lower():
            return "test_name"
        elif "email" in param_name.lower():
            return "test@example.com"
        elif "url" in param_name.lower():
            return "https://example.com"
        elif "data" in param_name.lower():
            return {"key": "value"}
        elif "list" in param_name.lower():
            return ["item1", "item2"]
        else:
            return "test_value"
    
    def _generate_invalid_value(self, param_name: str) -> Any:
        """Generate an invalid value for a parameter"""
        if "id" in param_name.lower():
            return None
        elif "name" in param_name.lower():
            return ""
        elif "email" in param_name.lower():
            return "invalid_email"
        elif "url" in param_name.lower():
            return "not_a_url"
        else:
            return None
    
    def _generate_setup_code(self, analysis: Dict[str, Any], pattern: TestPattern) -> str:
        """Generate setup code for a test case"""
        setup_lines = []
        
        if analysis["is_async"]:
            setup_lines.append("import asyncio")
        
        if "api" in analysis["characteristics"]:
            setup_lines.append("import requests")
            setup_lines.append("from unittest.mock import patch")
        
        return "\n".join(setup_lines)
    
    def _generate_teardown_code(self, analysis: Dict[str, Any], pattern: TestPattern) -> str:
        """Generate teardown code for a test case"""
        teardown_lines = []
        
        if "api" in analysis["characteristics"]:
            teardown_lines.append("# Cleanup API mocks")
        
        return "\n".join(teardown_lines)
    
    def generate_test_file(self, func: Callable, output_path: str, num_cases: int = 10) -> str:
        """Generate a complete test file for a function"""
        analysis = self.analyze_function(func)
        test_cases = self.generate_test_cases(func, num_cases)
        
        # Generate file content
        content = self._generate_file_header(analysis)
        content += self._generate_imports(analysis)
        content += self._generate_fixtures(analysis)
        content += self._generate_test_class(analysis, test_cases)
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return content
    
    def _generate_file_header(self, analysis: Dict[str, Any]) -> str:
        """Generate test file header"""
        return f'''"""
Auto-generated Test Cases for {analysis['name']}
============================================

Generated by Automated Test Generator
Function: {analysis['name']}
Characteristics: {', '.join(analysis['characteristics'])}
Complexity Score: {analysis['complexity_score']}
Test Patterns: {', '.join(analysis['test_patterns'])}

Generated on: {datetime.now().isoformat()}
"""

'''
    
    def _generate_imports(self, analysis: Dict[str, Any]) -> str:
        """Generate import statements"""
        imports = [
            "import pytest",
            "from unittest.mock import Mock, patch, MagicMock",
            "from typing import Any, Dict, List, Optional",
            ""
        ]
        
        if analysis["is_async"]:
            imports.insert(1, "import asyncio")
        
        if "api" in analysis["characteristics"]:
            imports.append("import requests")
        
        imports.append("")
        return "\n".join(imports)
    
    def _generate_fixtures(self, analysis: Dict[str, Any]) -> str:
        """Generate pytest fixtures"""
        fixtures = [
            "@pytest.fixture",
            f"def {analysis['name']}_instance():",
            f'    """Fixture for {analysis['name']} function"""',
            f"    return {analysis['name']}",
            "",
        ]
        
        return "\n".join(fixtures)
    
    def _generate_test_class(self, analysis: Dict[str, Any], test_cases: List[Dict[str, Any]]) -> str:
        """Generate test class with all test cases"""
        class_name = f"Test{analysis['name'].title()}"
        
        content = [
            f"class {class_name}:",
            f'    """Test suite for {analysis['name']} function"""',
            "",
        ]
        
        # Group test cases by pattern
        pattern_groups = {}
        for test_case in test_cases:
            pattern = test_case["pattern"]
            if pattern not in pattern_groups:
                pattern_groups[pattern] = []
            pattern_groups[pattern].append(test_case)
        
        # Generate test methods for each pattern
        for pattern, cases in pattern_groups.items():
            content.append(f"    # {pattern.title()} Tests")
            content.append("")
            
            for test_case in cases:
                content.extend(self._generate_test_method(test_case, analysis))
                content.append("")
        
        return "\n".join(content)
    
    def _generate_test_method(self, test_case: Dict[str, Any], analysis: Dict[str, Any]) -> List[str]:
        """Generate a single test method"""
        method_lines = [
            f"    @pytest.mark.{test_case['pattern']}",
        ]
        
        if test_case["async_test"]:
            method_lines.append("    @pytest.mark.asyncio")
        
        method_lines.extend([
            f"    async def {test_case['name']}(self, {analysis['name']}_instance):",
            f'        """{test_case['description']}"""',
        ])
        
        # Add setup code
        if test_case["setup_code"]:
            method_lines.append("        # Setup")
            for line in test_case["setup_code"].split('\n'):
                if line.strip():
                    method_lines.append(f"        {line}")
            method_lines.append("")
        
        # Add test execution
        method_lines.append("        # Test execution")
        if test_case["expected_exception"]:
            method_lines.append("        with pytest.raises(Exception):")
            method_lines.append(f"            result = await {analysis['name']}_instance(**{test_case['parameters']})")
        else:
            if test_case["async_test"]:
                method_lines.append(f"        result = await {analysis['name']}_instance(**{test_case['parameters']})")
            else:
                method_lines.append(f"        result = {analysis['name']}_instance(**{test_case['parameters']})")
        
        # Add assertions
        if test_case["assertions"]:
            method_lines.append("")
            method_lines.append("        # Assertions")
            for assertion in test_case["assertions"]:
                method_lines.append(f"        {assertion}")
        
        # Add teardown code
        if test_case["teardown_code"]:
            method_lines.append("")
            method_lines.append("        # Teardown")
            for line in test_case["teardown_code"].split('\n'):
                if line.strip():
                    method_lines.append(f"        {line}")
        
        return method_lines


def demonstrate_automated_generation():
    """Demonstrate the automated test generation system"""
    
    # Example function to test
    def validate_user_data(name: str, email: str, age: int) -> Dict[str, Any]:
        """
        Validate user data.
        
        Args:
            name: User's name
            email: User's email
            age: User's age
            
        Returns:
            Dictionary with validation result
            
        Raises:
            ValueError: If validation fails
        """
        if not name:
            raise ValueError("Name is required")
        if not email or "@" not in email:
            raise ValueError("Valid email is required")
        if age < 0:
            raise ValueError("Age must be positive")
        
        return {
            "valid": True,
            "name": name,
            "email": email,
            "age": age
        }
    
    # Generate test cases
    generator = AutomatedTestGenerator()
    test_cases = generator.generate_test_cases(validate_user_data, num_cases=8)
    
    print(f"Generated {len(test_cases)} test cases:")
    for i, test_case in enumerate(test_cases, 1):
        print(f"{i}. {test_case['name']}")
        print(f"   Pattern: {test_case['pattern']}")
        print(f"   Parameters: {test_case['parameters']}")
        if test_case['expected_exception']:
            print(f"   Expected Exception: {test_case['expected_exception']}")
        print()
    
    # Generate complete test file
    test_file_content = generator.generate_test_file(
        validate_user_data, 
        "generated_validation_tests.py", 
        num_cases=8
    )
    
    print("Generated test file content:")
    print(test_file_content[:800] + "..." if len(test_file_content) > 800 else test_file_content)


if __name__ == "__main__":
    demonstrate_automated_generation()
