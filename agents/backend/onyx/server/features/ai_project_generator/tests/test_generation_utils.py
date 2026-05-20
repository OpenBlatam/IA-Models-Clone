"""
Advanced Test Generation Utilities
===================================

This module provides utilities for generating unique, diverse, and intuitive
unit tests based on function signatures and docstrings.

The test generation follows these principles:
- Unique: Each test case covers a distinct scenario
- Diverse: Tests cover happy paths, edge cases, error conditions, and boundary values
- Intuitive: Test names and assertions clearly express intent
"""

import inspect
import ast
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
import re


class TestCategory(Enum):
    """Categories of test cases"""
    HAPPY_PATH = "happy_path"
    EDGE_CASE = "edge_case"
    ERROR_CONDITION = "error_condition"
    BOUNDARY_VALUE = "boundary_value"
    TYPE_VALIDATION = "type_validation"
    NULL_EMPTY = "null_empty"
    INVALID_INPUT = "invalid_input"
    STATE_CHANGE = "state_change"
    SIDE_EFFECT = "side_effect"
    PERFORMANCE = "performance"


@dataclass
class TestCase:
    """Represents a single test case"""
    name: str
    category: TestCategory
    description: str
    setup: Optional[str] = None
    inputs: Dict[str, Any] = None
    expected_output: Any = None
    expected_exception: Optional[type] = None
    assertions: List[str] = None
    
    def __post_init__(self):
        if self.inputs is None:
            self.inputs = {}
        if self.assertions is None:
            self.assertions = []


@dataclass
class FunctionSignature:
    """Represents a function's signature"""
    name: str
    parameters: List[Tuple[str, type, Optional[Any]]]  # (name, type, default)
    return_type: Optional[type]
    docstring: Optional[str]
    is_async: bool = False
    is_static: bool = False
    is_classmethod: bool = False


class TestGenerator:
    """Generates comprehensive test cases for functions"""
    
    def __init__(self):
        self.test_cases: List[TestCase] = []
    
    def analyze_function(self, func: Callable) -> FunctionSignature:
        """
        Analyzes a function and extracts its signature and documentation.
        
        Args:
            func: The function to analyze
            
        Returns:
            FunctionSignature object with extracted information
        """
        sig = inspect.signature(func)
        params = []
        
        for param_name, param in sig.parameters.items():
            param_type = param.annotation if param.annotation != inspect.Parameter.empty else None
            default = param.default if param.default != inspect.Parameter.empty else None
            params.append((param_name, param_type, default))
        
        return_type = sig.return_annotation if sig.return_annotation != inspect.Signature.empty else None
        docstring = inspect.getdoc(func)
        is_async = inspect.iscoroutinefunction(func)
        is_static = isinstance(func, staticmethod)
        is_classmethod = isinstance(func, classmethod)
        
        return FunctionSignature(
            name=func.__name__,
            parameters=params,
            return_type=return_type,
            docstring=docstring,
            is_async=is_async,
            is_static=is_static,
            is_classmethod=is_classmethod
        )
    
    def extract_test_scenarios(self, signature: FunctionSignature) -> List[TestCase]:
        """
        Extracts test scenarios from function signature and docstring.
        
        Args:
            signature: FunctionSignature to analyze
            
        Returns:
            List of TestCase objects
        """
        test_cases = []
        
        # Extract information from docstring
        doc_info = self._parse_docstring(signature.docstring)
        
        # Generate happy path tests
        test_cases.extend(self._generate_happy_path_tests(signature, doc_info))
        
        # Generate edge case tests
        test_cases.extend(self._generate_edge_case_tests(signature, doc_info))
        
        # Generate error condition tests
        test_cases.extend(self._generate_error_tests(signature, doc_info))
        
        # Generate boundary value tests
        test_cases.extend(self._generate_boundary_tests(signature, doc_info))
        
        # Generate null/empty tests
        test_cases.extend(self._generate_null_empty_tests(signature, doc_info))
        
        # Generate type validation tests
        test_cases.extend(self._generate_type_validation_tests(signature, doc_info))
        
        return test_cases
    
    def _parse_docstring(self, docstring: Optional[str]) -> Dict[str, Any]:
        """Parses docstring to extract test-relevant information"""
        if not docstring:
            return {}
        
        info = {
            "args": {},
            "returns": None,
            "raises": [],
            "examples": []
        }
        
        # Extract Args section
        args_match = re.search(r'Args?:?\s*\n((?:\s+[^:]+:[^\n]+\n?)+)', docstring)
        if args_match:
            args_text = args_match.group(1)
            for line in args_text.split('\n'):
                if ':' in line:
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        param_name = parts[0].strip()
                        param_desc = parts[1].strip()
                        info["args"][param_name] = param_desc
        
        # Extract Returns section
        returns_match = re.search(r'Returns?:?\s*\n\s+([^\n]+)', docstring)
        if returns_match:
            info["returns"] = returns_match.group(1).strip()
        
        # Extract Raises section
        raises_match = re.search(r'Raises?:?\s*\n((?:\s+[^:]+:[^\n]+\n?)+)', docstring)
        if raises_match:
            raises_text = raises_match.group(1)
            for line in raises_text.split('\n'):
                if ':' in line:
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        exception_name = parts[0].strip()
                        exception_desc = parts[1].strip()
                        info["raises"].append({
                            "exception": exception_name,
                            "description": exception_desc
                        })
        
        return info
    
    def _generate_happy_path_tests(
        self, 
        signature: FunctionSignature, 
        doc_info: Dict[str, Any]
    ) -> List[TestCase]:
        """Generates happy path test cases"""
        test_cases = []
        
        # Generate test with typical values
        test_name = f"test_{signature.name}_with_valid_inputs"
        inputs = {}
        
        for param_name, param_type, default in signature.parameters:
            if default is not None:
                inputs[param_name] = default
            else:
                inputs[param_name] = self._generate_typical_value(param_type, param_name)
        
        test_cases.append(TestCase(
            name=test_name,
            category=TestCategory.HAPPY_PATH,
            description=f"Test {signature.name} with valid typical inputs",
            inputs=inputs,
            assertions=[f"assert result is not None"]
        ))
        
        # Generate test with minimal required inputs
        if len(signature.parameters) > 0:
            minimal_inputs = {}
            for param_name, param_type, default in signature.parameters:
                if default is None:
                    minimal_inputs[param_name] = self._generate_minimal_value(param_type, param_name)
            
            if minimal_inputs:
                test_cases.append(TestCase(
                    name=f"test_{signature.name}_with_minimal_inputs",
                    category=TestCategory.HAPPY_PATH,
                    description=f"Test {signature.name} with minimal required inputs",
                    inputs=minimal_inputs,
                    assertions=[f"assert result is not None"]
                ))
        
        return test_cases
    
    def _generate_edge_case_tests(
        self, 
        signature: FunctionSignature, 
        doc_info: Dict[str, Any]
    ) -> List[TestCase]:
        """Generates edge case test cases"""
        test_cases = []
        
        for param_name, param_type, default in signature.parameters:
            if param_type == str or (param_type is None and default is None):
                # Test with very long string
                test_cases.append(TestCase(
                    name=f"test_{signature.name}_with_long_{param_name}",
                    category=TestCategory.EDGE_CASE,
                    description=f"Test {signature.name} with very long {param_name}",
                    inputs={param_name: "a" * 10000},
                    assertions=[f"assert result is not None"]
                ))
                
                # Test with special characters
                test_cases.append(TestCase(
                    name=f"test_{signature.name}_with_special_chars_in_{param_name}",
                    category=TestCategory.EDGE_CASE,
                    description=f"Test {signature.name} with special characters in {param_name}",
                    inputs={param_name: "!@#$%^&*()_+-=[]{}|;:,.<>?"},
                    assertions=[f"assert result is not None"]
                ))
            
            elif param_type == int or (param_type is None and isinstance(default, int)):
                # Test with zero
                test_cases.append(TestCase(
                    name=f"test_{signature.name}_with_zero_{param_name}",
                    category=TestCategory.EDGE_CASE,
                    description=f"Test {signature.name} with zero {param_name}",
                    inputs={param_name: 0},
                    assertions=[f"assert result is not None"]
                ))
                
                # Test with negative value
                test_cases.append(TestCase(
                    name=f"test_{signature.name}_with_negative_{param_name}",
                    category=TestCategory.EDGE_CASE,
                    description=f"Test {signature.name} with negative {param_name}",
                    inputs={param_name: -1},
                    assertions=[f"assert result is not None"]
                ))
        
        return test_cases
    
    def _generate_error_tests(
        self, 
        signature: FunctionSignature, 
        doc_info: Dict[str, Any]
    ) -> List[TestCase]:
        """Generates error condition test cases"""
        test_cases = []
        
        # Generate tests based on Raises section in docstring
        for raise_info in doc_info.get("raises", []):
            exception_name = raise_info["exception"]
            exception_type = self._get_exception_type(exception_name)
            
            if exception_type:
                # Find which parameter causes this exception
                for param_name, param_type, default in signature.parameters:
                    if "empty" in raise_info["description"].lower() or "invalid" in raise_info["description"].lower():
                        test_cases.append(TestCase(
                            name=f"test_{signature.name}_raises_{exception_name}_with_empty_{param_name}",
                            category=TestCategory.ERROR_CONDITION,
                            description=f"Test {signature.name} raises {exception_name} with empty {param_name}",
                            inputs={param_name: "" if param_type == str else None},
                            expected_exception=exception_type,
                            assertions=[f"pytest.raises({exception_name})"]
                        ))
        
        # Generate generic error tests for required parameters
        for param_name, param_type, default in signature.parameters:
            if default is None:
                # Test with None
                test_cases.append(TestCase(
                    name=f"test_{signature.name}_raises_error_with_none_{param_name}",
                    category=TestCategory.ERROR_CONDITION,
                    description=f"Test {signature.name} raises error with None {param_name}",
                    inputs={param_name: None},
                    expected_exception=ValueError,
                    assertions=[f"pytest.raises(ValueError)"]
                ))
        
        return test_cases
    
    def _generate_boundary_tests(
        self, 
        signature: FunctionSignature, 
        doc_info: Dict[str, Any]
    ) -> List[TestCase]:
        """Generates boundary value test cases"""
        test_cases = []
        
        for param_name, param_type, default in signature.parameters:
            if param_type == int or (param_type is None and isinstance(default, int)):
                # Test with max int
                test_cases.append(TestCase(
                    name=f"test_{signature.name}_with_max_int_{param_name}",
                    category=TestCategory.BOUNDARY_VALUE,
                    description=f"Test {signature.name} with maximum integer {param_name}",
                    inputs={param_name: 2**31 - 1},
                    assertions=[f"assert result is not None"]
                ))
                
                # Test with min int
                test_cases.append(TestCase(
                    name=f"test_{signature.name}_with_min_int_{param_name}",
                    category=TestCategory.BOUNDARY_VALUE,
                    description=f"Test {signature.name} with minimum integer {param_name}",
                    inputs={param_name: -(2**31)},
                    assertions=[f"assert result is not None"]
                ))
            
            elif param_type == str or (param_type is None and isinstance(default, str)):
                # Test with single character
                test_cases.append(TestCase(
                    name=f"test_{signature.name}_with_single_char_{param_name}",
                    category=TestCategory.BOUNDARY_VALUE,
                    description=f"Test {signature.name} with single character {param_name}",
                    inputs={param_name: "a"},
                    assertions=[f"assert result is not None"]
                ))
        
        return test_cases
    
    def _generate_null_empty_tests(
        self, 
        signature: FunctionSignature, 
        doc_info: Dict[str, Any]
    ) -> List[TestCase]:
        """Generates null/empty test cases"""
        test_cases = []
        
        for param_name, param_type, default in signature.parameters:
            if param_type == str or (param_type is None and isinstance(default, str)):
                # Test with empty string
                test_cases.append(TestCase(
                    name=f"test_{signature.name}_with_empty_{param_name}",
                    category=TestCategory.NULL_EMPTY,
                    description=f"Test {signature.name} with empty {param_name}",
                    inputs={param_name: ""},
                    expected_exception=ValueError,
                    assertions=[f"pytest.raises(ValueError)"]
                ))
            
            elif param_type == list or (param_type is None and isinstance(default, list)):
                # Test with empty list
                test_cases.append(TestCase(
                    name=f"test_{signature.name}_with_empty_list_{param_name}",
                    category=TestCategory.NULL_EMPTY,
                    description=f"Test {signature.name} with empty list {param_name}",
                    inputs={param_name: []},
                    assertions=[f"assert result is not None"]
                ))
        
        return test_cases
    
    def _generate_type_validation_tests(
        self, 
        signature: FunctionSignature, 
        doc_info: Dict[str, Any]
    ) -> List[TestCase]:
        """Generates type validation test cases"""
        test_cases = []
        
        for param_name, param_type, default in signature.parameters:
            if param_type:
                # Test with wrong type
                wrong_type_value = self._generate_wrong_type_value(param_type)
                if wrong_type_value is not None:
                    test_cases.append(TestCase(
                        name=f"test_{signature.name}_with_wrong_type_{param_name}",
                        category=TestCategory.TYPE_VALIDATION,
                        description=f"Test {signature.name} with wrong type for {param_name}",
                        inputs={param_name: wrong_type_value},
                        expected_exception=TypeError,
                        assertions=[f"pytest.raises(TypeError)"]
                    ))
        
        return test_cases
    
    def _generate_typical_value(self, param_type: Optional[type], param_name: str) -> Any:
        """Generates a typical value for a parameter type"""
        if param_type == str:
            # Use parameter name as hint
            if "name" in param_name.lower():
                return "test_project"
            elif "description" in param_name.lower():
                return "A test project description"
            elif "path" in param_name.lower() or "dir" in param_name.lower():
                return "/tmp/test"
            else:
                return "test_value"
        elif param_type == int:
            return 42
        elif param_type == float:
            return 3.14
        elif param_type == bool:
            return True
        elif param_type == list:
            return []
        elif param_type == dict:
            return {}
        else:
            return "test_value"
    
    def _generate_minimal_value(self, param_type: Optional[type], param_name: str) -> Any:
        """Generates a minimal value for a parameter type"""
        if param_type == str:
            return "a"
        elif param_type == int:
            return 1
        elif param_type == float:
            return 0.1
        elif param_type == bool:
            return False
        elif param_type == list:
            return []
        elif param_type == dict:
            return {}
        else:
            return "a"
    
    def _generate_wrong_type_value(self, param_type: type) -> Optional[Any]:
        """Generates a wrong type value for type validation tests"""
        if param_type == str:
            return 123
        elif param_type == int:
            return "not_an_int"
        elif param_type == float:
            return "not_a_float"
        elif param_type == bool:
            return "not_a_bool"
        elif param_type == list:
            return "not_a_list"
        elif param_type == dict:
            return "not_a_dict"
        else:
            return None
    
    def _get_exception_type(self, exception_name: str) -> Optional[type]:
        """Converts exception name string to exception type"""
        exception_map = {
            "ValueError": ValueError,
            "TypeError": TypeError,
            "KeyError": KeyError,
            "AttributeError": AttributeError,
            "IndexError": IndexError,
            "FileNotFoundError": FileNotFoundError,
            "PermissionError": PermissionError,
            "RuntimeError": RuntimeError,
            "Exception": Exception,
        }
        return exception_map.get(exception_name, ValueError)
    
    def generate_test_code(
        self, 
        func: Callable, 
        class_name: Optional[str] = None
    ) -> str:
        """
        Generates complete test code for a function.
        
        Args:
            func: The function to generate tests for
            class_name: Optional class name if function is a method
            
        Returns:
            String containing generated test code
        """
        signature = self.analyze_function(func)
        test_cases = self.extract_test_scenarios(signature)
        
        code_lines = []
        code_lines.append(f'"""')
        code_lines.append(f'Generated tests for {signature.name}')
        code_lines.append(f'')
        code_lines.append(f'Total test cases: {len(test_cases)}')
        code_lines.append(f'"""')
        code_lines.append('')
        code_lines.append('import pytest')
        if signature.is_async:
            code_lines.append('import asyncio')
        code_lines.append('')
        
        if class_name:
            code_lines.append(f'class Test{signature.name.title()}:')
            code_lines.append(f'    """Test suite for {class_name}.{signature.name}"""')
            code_lines.append('')
        else:
            code_lines.append(f'def test_{signature.name}_generated():')
            code_lines.append('    """Generated test suite"""')
            code_lines.append('    pass')
            code_lines.append('')
        
        for test_case in test_cases:
            code_lines.append('')
            code_lines.append(f'    def {test_case.name}(self):')
            code_lines.append(f'        """{test_case.description}"""')
            
            if test_case.setup:
                code_lines.append(f'        {test_case.setup}')
            
            # Build function call
            if class_name:
                if signature.is_static:
                    call_prefix = f'{class_name}.{signature.name}'
                elif signature.is_classmethod:
                    call_prefix = f'{class_name}.{signature.name}'
                else:
                    call_prefix = f'self.instance.{signature.name}'
            else:
                call_prefix = signature.name
            
            # Build arguments
            args_str = ', '.join([f'{k}={repr(v)}' for k, v in test_case.inputs.items()])
            
            if test_case.expected_exception:
                code_lines.append(f'        with pytest.raises({test_case.expected_exception.__name__}):')
                if signature.is_async:
                    code_lines.append(f'            await {call_prefix}({args_str})')
                else:
                    code_lines.append(f'            {call_prefix}({args_str})')
            else:
                if signature.is_async:
                    code_lines.append(f'        result = await {call_prefix}({args_str})')
                else:
                    code_lines.append(f'        result = {call_prefix}({args_str})')
                
                for assertion in test_case.assertions:
                    code_lines.append(f'        {assertion}')
        
        return '\n'.join(code_lines)


def generate_tests_for_function(func: Callable, class_name: Optional[str] = None) -> str:
    """
    Convenience function to generate tests for a function.
    
    Args:
        func: The function to generate tests for
        class_name: Optional class name if function is a method
        
    Returns:
        String containing generated test code
    """
    generator = TestGenerator()
    return generator.generate_test_code(func, class_name)


def generate_tests_for_class(cls: type) -> str:
    """
    Generates tests for all public methods of a class.
    
    Args:
        cls: The class to generate tests for
        
    Returns:
        String containing generated test code
    """
    code_lines = []
    code_lines.append(f'"""')
    code_lines.append(f'Generated tests for {cls.__name__}')
    code_lines.append(f'"""')
    code_lines.append('')
    code_lines.append('import pytest')
    code_lines.append('')
    code_lines.append(f'class Test{cls.__name__}:')
    code_lines.append(f'    """Test suite for {cls.__name__}"""')
    code_lines.append('')
    code_lines.append('    def setup_method(self):')
    code_lines.append(f'        """Setup test instance"""')
    code_lines.append(f'        self.instance = {cls.__name__}()')
    code_lines.append('')
    
    generator = TestGenerator()
    
    for name, method in inspect.getmembers(cls, predicate=inspect.ismethod):
        if not name.startswith('_') or name.startswith('__') and name.endswith('__'):
            test_code = generator.generate_test_code(method, cls.__name__)
            code_lines.append(test_code)
            code_lines.append('')
    
    return '\n'.join(code_lines)


