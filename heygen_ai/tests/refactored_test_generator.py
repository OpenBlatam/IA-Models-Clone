"""
Refactored Test Case Generator
=============================

Streamlined test case generation system that creates unique, diverse, and intuitive
unit tests for functions given their signature and docstring.

This refactored version focuses on:
- Direct alignment with prompt requirements
- Simplified architecture with better performance
- Clear, maintainable code structure
- Essential features without unnecessary complexity
"""

import ast
import inspect
import re
import random
import string
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class RefactoredTestCase:
    """Refactored test case with essential quality metrics"""
    name: str
    description: str
    function_name: str
    parameters: Dict[str, Any]
    expected_result: Any = None
    expected_exception: Optional[type] = None
    assertions: List[str] = field(default_factory=list)
    setup_code: str = ""
    teardown_code: str = ""
    async_test: bool = False
    # Essential quality metrics
    uniqueness: float = 0.0
    diversity: float = 0.0
    intuition: float = 0.0
    overall_quality: float = 0.0
    # Metadata
    test_type: str = ""
    scenario: str = ""


class RefactoredTestGenerator:
    """Refactored test generator focused on core requirements"""
    
    def __init__(self):
        self.test_patterns = self._load_test_patterns()
        self.naming_strategies = self._load_naming_strategies()
        self.parameter_generators = self._setup_parameter_generators()
        
    def _load_test_patterns(self) -> Dict[str, Dict[str, List[str]]]:
        """Load essential test patterns"""
        return {
            "validation": {
                "scenarios": [
                    "valid_input", "invalid_input", "empty_input", "null_input",
                    "boundary_value", "edge_case", "type_error", "format_error",
                    "required_field_missing", "field_too_long", "field_too_short",
                    "invalid_format", "special_characters", "unicode_input"
                ],
                "assertions": [
                    "assert result is not None",
                    "assert result.get('valid', False) is True",
                    "assert 'error' not in str(result).lower()",
                    "assert isinstance(result, dict)"
                ]
            },
            "transformation": {
                "scenarios": [
                    "basic_transform", "empty_data", "large_dataset", "nested_data",
                    "type_conversion", "format_change", "data_cleaning", "normalization",
                    "aggregation", "filtering", "sorting", "grouping", "mapping"
                ],
                "assertions": [
                    "assert result is not None",
                    "assert result != input_data",
                    "assert isinstance(result, (dict, list))",
                    "assert len(result) > 0"
                ]
            },
            "calculation": {
                "scenarios": [
                    "basic_calculation", "zero_input", "negative_input", "large_number",
                    "decimal_precision", "division_by_zero", "overflow", "underflow",
                    "rounding", "truncation", "mathematical_operations", "statistical"
                ],
                "assertions": [
                    "assert result is not None",
                    "assert isinstance(result, (int, float))",
                    "assert not math.isnan(result) if hasattr(math, 'isnan') else True",
                    "assert result != float('inf')"
                ]
            },
            "business_logic": {
                "scenarios": [
                    "normal_flow", "error_condition", "edge_case", "boundary_condition",
                    "workflow_state", "business_rule", "approval_process", "validation_rule",
                    "pricing_logic", "user_permission", "resource_allocation", "quota_check"
                ],
                "assertions": [
                    "assert result is not None",
                    "assert 'error' not in str(result).lower()",
                    "assert isinstance(result, dict)",
                    "assert result.get('success', True) is True"
                ]
            }
        }
    
    def _load_naming_strategies(self) -> Dict[str, List[str]]:
        """Load naming strategies for intuitive test names"""
        return {
            "behavior_driven": [
                "should_{behavior}_when_{condition}",
                "should_{behavior}_given_{context}",
                "should_{behavior}_for_{scenario}",
                "should_{behavior}_with_{input}"
            ],
            "descriptive": [
                "test_{function}_{scenario}_{aspect}",
                "test_{function}_{condition}_{result}",
                "test_{function}_{input}_{output}",
                "test_{function}_{context}_{behavior}"
            ],
            "scenario_based": [
                "test_{scenario}_scenario",
                "test_{scenario}_case",
                "test_{scenario}_situation",
                "test_{scenario}_context"
            ]
        }
    
    def _setup_parameter_generators(self) -> Dict[str, Callable]:
        """Setup parameter generators for different scenarios"""
        return {
            "valid": self._generate_valid_parameters,
            "invalid": self._generate_invalid_parameters,
            "edge_case": self._generate_edge_case_parameters,
            "boundary": self._generate_boundary_parameters,
            "creative": self._generate_creative_parameters
        }
    
    def generate_tests(self, func: Callable, num_tests: int = 20) -> List[RefactoredTestCase]:
        """Generate refactored test cases with focus on uniqueness, diversity, and intuition"""
        analysis = self._analyze_function(func)
        function_type = self._classify_function(func, analysis)
        
        test_cases = []
        
        # Generate unique tests (40% of total)
        unique_tests = self._generate_unique_tests(func, analysis, function_type, num_tests // 2)
        test_cases.extend(unique_tests)
        
        # Generate diverse tests (40% of total)
        diverse_tests = self._generate_diverse_tests(func, analysis, function_type, num_tests // 2)
        test_cases.extend(diverse_tests)
        
        # Generate intuitive tests (20% of total)
        intuitive_tests = self._generate_intuitive_tests(func, analysis, function_type, num_tests // 5)
        test_cases.extend(intuitive_tests)
        
        # Score and enhance all tests
        for test_case in test_cases:
            self._score_test_case(test_case, analysis)
        
        # Sort by overall quality and return
        test_cases.sort(key=lambda x: x.overall_quality, reverse=True)
        return test_cases[:num_tests]
    
    def _analyze_function(self, func: Callable) -> Dict[str, Any]:
        """Analyze function for test generation"""
        try:
            signature = inspect.signature(func)
            docstring = inspect.getdoc(func) or ""
            source = inspect.getsource(func)
            
            return {
                "name": func.__name__,
                "signature": signature,
                "docstring": docstring,
                "source_code": source,
                "parameters": list(signature.parameters.keys()),
                "return_annotation": str(signature.return_annotation),
                "is_async": inspect.iscoroutinefunction(func),
                "parameter_types": self._get_parameter_types(signature),
                "complexity": self._calculate_complexity(source)
            }
        except Exception as e:
            logger.error(f"Error analyzing function {func.__name__}: {e}")
            return {}
    
    def _classify_function(self, func: Callable, analysis: Dict[str, Any]) -> str:
        """Classify function type"""
        name = func.__name__.lower()
        docstring = analysis.get("docstring", "").lower()
        
        if any(keyword in name or keyword in docstring for keyword in ["validate", "check", "verify", "ensure"]):
            return "validation"
        elif any(keyword in name or keyword in docstring for keyword in ["transform", "convert", "process", "translate"]):
            return "transformation"
        elif any(keyword in name or keyword in docstring for keyword in ["calculate", "compute", "math", "solve"]):
            return "calculation"
        elif any(keyword in name or keyword in docstring for keyword in ["business", "workflow", "rule", "logic"]):
            return "business_logic"
        else:
            return "validation"  # Default to validation
    
    def _generate_unique_tests(self, func: Callable, analysis: Dict[str, Any], 
                             function_type: str, num_tests: int) -> List[RefactoredTestCase]:
        """Generate unique test cases"""
        test_cases = []
        
        if function_type in self.test_patterns:
            scenarios = self.test_patterns[function_type]["scenarios"]
            
            for i, scenario in enumerate(scenarios[:num_tests]):
                test_case = self._create_unique_test(func, analysis, scenario, function_type, i)
                if test_case:
                    test_cases.append(test_case)
        
        return test_cases
    
    def _generate_diverse_tests(self, func: Callable, analysis: Dict[str, Any], 
                              function_type: str, num_tests: int) -> List[RefactoredTestCase]:
        """Generate diverse test cases"""
        test_cases = []
        
        # Generate tests with different parameter types
        param_types = ["valid", "invalid", "edge_case", "boundary", "creative"]
        
        for i, param_type in enumerate(param_types[:num_tests]):
            test_case = self._create_diverse_test(func, analysis, param_type, function_type, i)
            if test_case:
                test_cases.append(test_case)
        
        return test_cases
    
    def _generate_intuitive_tests(self, func: Callable, analysis: Dict[str, Any], 
                                function_type: str, num_tests: int) -> List[RefactoredTestCase]:
        """Generate intuitive test cases"""
        test_cases = []
        
        # Generate tests with different naming strategies
        strategies = list(self.naming_strategies.keys())
        
        for i, strategy in enumerate(strategies[:num_tests]):
            test_case = self._create_intuitive_test(func, analysis, strategy, function_type, i)
            if test_case:
                test_cases.append(test_case)
        
        return test_cases
    
    def _create_unique_test(self, func: Callable, analysis: Dict[str, Any], 
                          scenario: str, function_type: str, index: int) -> Optional[RefactoredTestCase]:
        """Create a unique test case"""
        try:
            name = self._generate_unique_name(func.__name__, scenario, function_type, index)
            description = self._generate_unique_description(func.__name__, scenario, function_type)
            parameters = self._generate_unique_parameters(analysis, scenario)
            assertions = self._generate_assertions(scenario, function_type)
            
            return RefactoredTestCase(
                name=name,
                description=description,
                function_name=func.__name__,
                parameters=parameters,
                assertions=assertions,
                async_test=analysis.get("is_async", False),
                test_type="unique",
                scenario=scenario
            )
        except Exception as e:
            logger.error(f"Error creating unique test: {e}")
            return None
    
    def _create_diverse_test(self, func: Callable, analysis: Dict[str, Any], 
                           param_type: str, function_type: str, index: int) -> Optional[RefactoredTestCase]:
        """Create a diverse test case"""
        try:
            name = self._generate_diverse_name(func.__name__, param_type, function_type, index)
            description = self._generate_diverse_description(func.__name__, param_type, function_type)
            parameters = self._generate_diverse_parameters(analysis, param_type)
            assertions = self._generate_assertions(param_type, function_type)
            
            return RefactoredTestCase(
                name=name,
                description=description,
                function_name=func.__name__,
                parameters=parameters,
                assertions=assertions,
                async_test=analysis.get("is_async", False),
                test_type="diverse",
                scenario=param_type
            )
        except Exception as e:
            logger.error(f"Error creating diverse test: {e}")
            return None
    
    def _create_intuitive_test(self, func: Callable, analysis: Dict[str, Any], 
                             strategy: str, function_type: str, index: int) -> Optional[RefactoredTestCase]:
        """Create an intuitive test case"""
        try:
            name = self._generate_intuitive_name(func.__name__, strategy, function_type, index)
            description = self._generate_intuitive_description(func.__name__, strategy, function_type)
            parameters = self._generate_intuitive_parameters(analysis, strategy)
            assertions = self._generate_assertions(strategy, function_type)
            
            return RefactoredTestCase(
                name=name,
                description=description,
                function_name=func.__name__,
                parameters=parameters,
                assertions=assertions,
                async_test=analysis.get("is_async", False),
                test_type="intuitive",
                scenario=strategy
            )
        except Exception as e:
            logger.error(f"Error creating intuitive test: {e}")
            return None
    
    def _generate_unique_name(self, function_name: str, scenario: str, function_type: str, index: int) -> str:
        """Generate unique test name"""
        if function_type == "validation":
            return f"should_validate_{scenario}_uniquely_{index}"
        elif function_type == "transformation":
            return f"should_transform_{scenario}_creatively_{index}"
        elif function_type == "calculation":
            return f"should_calculate_{scenario}_precisely_{index}"
        elif function_type == "business_logic":
            return f"should_implement_{scenario}_logic_{index}"
        else:
            return f"test_{function_name}_{scenario}_unique_{index}"
    
    def _generate_unique_description(self, function_name: str, scenario: str, function_type: str) -> str:
        """Generate unique test description"""
        return f"Verify {function_name} handles {scenario.replace('_', ' ')} scenario with unique approach"
    
    def _generate_diverse_name(self, function_name: str, param_type: str, function_type: str, index: int) -> str:
        """Generate diverse test name"""
        return f"test_{function_name}_diverse_{param_type}_{index}"
    
    def _generate_diverse_description(self, function_name: str, param_type: str, function_type: str) -> str:
        """Generate diverse test description"""
        return f"Verify {function_name} handles {param_type} parameters with diverse approach"
    
    def _generate_intuitive_name(self, function_name: str, strategy: str, function_type: str, index: int) -> str:
        """Generate intuitive test name"""
        if strategy == "behavior_driven":
            return f"should_{function_name}_behave_correctly_{index}"
        elif strategy == "descriptive":
            return f"test_{function_name}_descriptive_{index}"
        elif strategy == "scenario_based":
            return f"test_{function_name}_scenario_{index}"
        else:
            return f"test_{function_name}_intuitive_{index}"
    
    def _generate_intuitive_description(self, function_name: str, strategy: str, function_type: str) -> str:
        """Generate intuitive test description"""
        return f"Verify {function_name} works intuitively with {strategy.replace('_', ' ')} approach"
    
    def _generate_unique_parameters(self, analysis: Dict[str, Any], scenario: str) -> Dict[str, Any]:
        """Generate unique parameters"""
        parameters = {}
        param_types = analysis.get("parameter_types", {})
        
        for param_name, param_type in param_types.items():
            if "valid" in scenario:
                parameters[param_name] = self._generate_valid_value(param_name, param_type)
            elif "invalid" in scenario:
                parameters[param_name] = self._generate_invalid_value(param_name, param_type)
            elif "edge" in scenario:
                parameters[param_name] = self._generate_edge_case_value(param_name, param_type)
            else:
                parameters[param_name] = self._generate_creative_value(param_name, param_type)
        
        return parameters
    
    def _generate_diverse_parameters(self, analysis: Dict[str, Any], param_type: str) -> Dict[str, Any]:
        """Generate diverse parameters"""
        parameters = {}
        param_types = analysis.get("parameter_types", {})
        
        for param_name, param_type in param_types.items():
            if param_type == "valid":
                parameters[param_name] = self._generate_valid_value(param_name, param_type)
            elif param_type == "invalid":
                parameters[param_name] = self._generate_invalid_value(param_name, param_type)
            elif param_type == "edge_case":
                parameters[param_name] = self._generate_edge_case_value(param_name, param_type)
            elif param_type == "boundary":
                parameters[param_name] = self._generate_boundary_value(param_name, param_type)
            else:
                parameters[param_name] = self._generate_creative_value(param_name, param_type)
        
        return parameters
    
    def _generate_intuitive_parameters(self, analysis: Dict[str, Any], strategy: str) -> Dict[str, Any]:
        """Generate intuitive parameters"""
        parameters = {}
        param_types = analysis.get("parameter_types", {})
        
        for param_name, param_type in param_types.items():
            if strategy == "behavior_driven":
                parameters[param_name] = self._generate_behavior_value(param_name, param_type)
            elif strategy == "descriptive":
                parameters[param_name] = self._generate_descriptive_value(param_name, param_type)
            else:
                parameters[param_name] = self._generate_intuitive_value(param_name, param_type)
        
        return parameters
    
    def _generate_assertions(self, scenario: str, function_type: str) -> List[str]:
        """Generate assertions based on scenario and function type"""
        if function_type in self.test_patterns:
            base_assertions = self.test_patterns[function_type]["assertions"]
        else:
            base_assertions = ["assert result is not None"]
        
        # Add scenario-specific assertions
        if "valid" in scenario:
            base_assertions.append("assert result is valid")
        elif "invalid" in scenario:
            base_assertions.append("assert result indicates error")
        elif "edge" in scenario:
            base_assertions.append("assert result handles edge case correctly")
        
        return base_assertions
    
    def _score_test_case(self, test_case: RefactoredTestCase, analysis: Dict[str, Any]):
        """Score test case for quality metrics"""
        # Uniqueness score
        uniqueness = 0.0
        if test_case.test_type == "unique":
            uniqueness += 0.4
        if "unique" in test_case.name or "creative" in test_case.name:
            uniqueness += 0.3
        if len(test_case.parameters) > 1:
            uniqueness += 0.3
        test_case.uniqueness = min(uniqueness, 1.0)
        
        # Diversity score
        diversity = 0.0
        if test_case.test_type == "diverse":
            diversity += 0.4
        param_types = set(type(v).__name__ for v in test_case.parameters.values())
        diversity += len(param_types) * 0.2
        if test_case.scenario in ["edge_case", "boundary", "creative"]:
            diversity += 0.2
        test_case.diversity = min(diversity, 1.0)
        
        # Intuition score
        intuition = 0.0
        if test_case.test_type == "intuitive":
            intuition += 0.4
        if "should" in test_case.name.lower():
            intuition += 0.3
        if "verify" in test_case.description.lower():
            intuition += 0.3
        test_case.intuition = min(intuition, 1.0)
        
        # Overall quality
        test_case.overall_quality = (
            test_case.uniqueness * 0.4 +
            test_case.diversity * 0.3 +
            test_case.intuition * 0.3
        )
    
    # Parameter generation methods
    def _generate_valid_value(self, param_name: str, param_type: str) -> Any:
        """Generate valid parameter value"""
        if "str" in param_type.lower():
            if "email" in param_name.lower():
                return "user@example.com"
            elif "name" in param_name.lower():
                return "John Doe"
            elif "id" in param_name.lower():
                return "user_123"
            else:
                return "test_value"
        elif "int" in param_type.lower():
            return 42
        elif "float" in param_type.lower():
            return 3.14
        elif "bool" in param_type.lower():
            return True
        elif "list" in param_type.lower():
            return ["item1", "item2"]
        elif "dict" in param_type.lower():
            return {"key": "value"}
        else:
            return "default_value"
    
    def _generate_invalid_value(self, param_name: str, param_type: str) -> Any:
        """Generate invalid parameter value"""
        return None
    
    def _generate_edge_case_value(self, param_name: str, param_type: str) -> Any:
        """Generate edge case parameter value"""
        if "str" in param_type.lower():
            return ""
        elif "int" in param_type.lower():
            return 0
        elif "float" in param_type.lower():
            return 0.0
        else:
            return None
    
    def _generate_boundary_value(self, param_name: str, param_type: str) -> Any:
        """Generate boundary parameter value"""
        if "str" in param_type.lower():
            return "a"  # Single character
        elif "int" in param_type.lower():
            return 1
        elif "float" in param_type.lower():
            return 0.1
        else:
            return self._generate_valid_value(param_name, param_type)
    
    def _generate_creative_value(self, param_name: str, param_type: str) -> Any:
        """Generate creative parameter value"""
        if "str" in param_type.lower():
            return f"creative_{param_name}_{random.randint(1000, 9999)}"
        elif "int" in param_type.lower():
            return random.randint(1, 100)
        elif "float" in param_type.lower():
            return round(random.uniform(0.1, 10.0), 2)
        else:
            return self._generate_valid_value(param_name, param_type)
    
    def _generate_behavior_value(self, param_name: str, param_type: str) -> Any:
        """Generate behavior-driven parameter value"""
        return self._generate_valid_value(param_name, param_type)
    
    def _generate_descriptive_value(self, param_name: str, param_type: str) -> Any:
        """Generate descriptive parameter value"""
        return self._generate_valid_value(param_name, param_type)
    
    def _generate_intuitive_value(self, param_name: str, param_type: str) -> Any:
        """Generate intuitive parameter value"""
        return self._generate_valid_value(param_name, param_type)
    
    def _get_parameter_types(self, signature: inspect.Signature) -> Dict[str, str]:
        """Get parameter types from signature"""
        param_types = {}
        for param_name, param in signature.parameters.items():
            if param.annotation != inspect.Parameter.empty:
                param_types[param_name] = str(param.annotation)
            else:
                param_types[param_name] = "Any"
        return param_types
    
    def _calculate_complexity(self, source: str) -> int:
        """Calculate function complexity"""
        return 1  # Simplified for now


def demonstrate_refactored_generator():
    """Demonstrate the refactored test generator"""
    
    # Example function to test
    def process_user_data(user_data: dict, validation_rules: list, options: dict) -> dict:
        """
        Process user data with validation rules and options.
        
        Args:
            user_data: Dictionary containing user information
            validation_rules: List of validation rules to apply
            options: Dictionary with processing options
            
        Returns:
            Dictionary with processing results and validation status
            
        Raises:
            ValueError: If user_data is invalid or validation_rules is empty
        """
        if not isinstance(user_data, dict):
            raise ValueError("user_data must be a dictionary")
        
        if not validation_rules:
            raise ValueError("validation_rules cannot be empty")
        
        validation_results = []
        warnings = []
        
        # Apply validation rules
        for rule in validation_rules:
            if rule == "email_required" and "email" not in user_data:
                validation_results.append("Email is required")
            elif rule == "age_validation" and user_data.get("age", 0) < 18:
                validation_results.append("Age must be 18 or older")
            elif rule == "username_validation" and len(user_data.get("username", "")) < 3:
                validation_results.append("Username must be at least 3 characters")
        
        # Apply options
        if options.get("normalize_keys", False):
            user_data = {k.lower(): v for k, v in user_data.items()}
        
        if options.get("add_timestamp", False):
            user_data["processed_at"] = datetime.now().isoformat()
        
        return {
            "user_data": user_data,
            "validation_results": validation_results,
            "warnings": warnings,
            "valid": len(validation_results) == 0,
            "processed_at": datetime.now().isoformat()
        }
    
    # Generate refactored tests
    generator = RefactoredTestGenerator()
    test_cases = generator.generate_tests(process_user_data, num_tests=15)
    
    print(f"Generated {len(test_cases)} refactored test cases:")
    print("=" * 60)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"{i}. {test_case.name}")
        print(f"   Description: {test_case.description}")
        print(f"   Type: {test_case.test_type}")
        print(f"   Quality Scores: U={test_case.uniqueness:.2f}, D={test_case.diversity:.2f}, I={test_case.intuition:.2f}")
        print(f"   Overall Quality: {test_case.overall_quality:.2f}")
        print(f"   Parameters: {test_case.parameters}")
        print(f"   Assertions: {test_case.assertions}")
        print()


if __name__ == "__main__":
    demonstrate_refactored_generator()