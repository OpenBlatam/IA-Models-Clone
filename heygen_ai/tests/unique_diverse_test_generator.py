"""
Unique, Diverse, and Intuitive Test Case Generator
================================================

AI-powered test case generation system that creates unique, diverse, and intuitive
unit tests for functions given their signature and docstring.

Key Features:
- Unique test scenarios with varied approaches
- Diverse test cases covering wide range of scenarios  
- Intuitive test naming and structure
- Advanced function analysis and pattern recognition
"""

import ast
import inspect
import re
import random
import string
from typing import Any, Dict, List, Optional, Callable, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class UniqueTestCase:
    """Test case with uniqueness, diversity, and intuition metrics"""
    name: str
    description: str
    test_type: str
    function_name: str
    parameters: Dict[str, Any]
    expected_result: Any
    expected_exception: Optional[type] = None
    setup_code: str = ""
    teardown_code: str = ""
    assertions: List[str] = field(default_factory=list)
    mock_objects: Dict[str, Any] = field(default_factory=dict)
    async_test: bool = False
    parametrize: bool = False
    parametrize_values: List[Dict[str, Any]] = field(default_factory=list)
    # Quality metrics
    uniqueness_score: float = 0.0
    diversity_score: float = 0.0
    intuition_score: float = 0.0
    overall_quality: float = 0.0


class UniqueDiverseTestGenerator:
    """Generator focused on unique, diverse, and intuitive test cases"""
    
    def __init__(self):
        self.test_scenarios = self._load_test_scenarios()
        self.naming_patterns = self._load_naming_patterns()
        self.parameter_generators = self._setup_parameter_generators()
        self.edge_case_detectors = self._setup_edge_case_detectors()
        
    def _load_test_scenarios(self) -> Dict[str, List[str]]:
        """Load diverse test scenarios for different function types"""
        return {
            "validation": [
                "happy_path_validation",
                "boundary_value_testing", 
                "type_coercion_handling",
                "format_compliance_checking",
                "business_rule_validation",
                "cross_field_validation",
                "temporal_validation",
                "geographical_validation",
                "unicode_handling",
                "special_character_processing"
            ],
            "transformation": [
                "identity_transformation",
                "data_aggregation",
                "filtering_operations",
                "sorting_operations", 
                "format_conversion",
                "data_enrichment",
                "normalization",
                "deduplication",
                "data_quality_improvement",
                "schema_validation"
            ],
            "calculation": [
                "mathematical_accuracy",
                "precision_handling",
                "rounding_behavior",
                "overflow_protection",
                "underflow_handling",
                "division_by_zero",
                "floating_point_precision",
                "currency_calculations",
                "percentage_calculations",
                "statistical_operations"
            ],
            "business_logic": [
                "workflow_state_transitions",
                "business_rule_evaluation",
                "decision_tree_logic",
                "approval_workflows",
                "pricing_calculations",
                "user_permission_checks",
                "resource_allocation",
                "quota_management",
                "subscription_logic",
                "billing_operations"
            ],
            "data_processing": [
                "small_dataset_handling",
                "large_dataset_processing",
                "empty_dataset_handling",
                "malformed_data_recovery",
                "data_transformation",
                "batch_processing",
                "stream_processing",
                "real_time_processing",
                "data_validation",
                "error_recovery"
            ]
        }
    
    def _load_naming_patterns(self) -> Dict[str, List[str]]:
        """Load intuitive naming patterns for test cases"""
        return {
            "behavior_driven": [
                "should_{behavior}_when_{condition}",
                "should_{behavior}_given_{context}",
                "should_{behavior}_for_{scenario}",
                "should_{behavior}_with_{input}",
                "should_{behavior}_under_{circumstances}"
            ],
            "scenario_based": [
                "test_{scenario}_scenario",
                "test_{scenario}_case",
                "test_{scenario}_situation",
                "test_{scenario}_context",
                "test_{scenario}_condition"
            ],
            "domain_specific": [
                "test_{domain}_{operation}",
                "test_{domain}_{validation}",
                "test_{domain}_{transformation}",
                "test_{domain}_{calculation}",
                "test_{domain}_{processing}"
            ],
            "user_story": [
                "test_as_{user}_i_can_{action}",
                "test_as_{user}_i_should_{behavior}",
                "test_as_{user}_i_expect_{outcome}",
                "test_as_{user}_i_need_{requirement}",
                "test_as_{user}_i_want_{feature}"
            ]
        }
    
    def _setup_parameter_generators(self) -> Dict[str, Callable]:
        """Setup parameter generators for diverse test data"""
        return {
            "realistic": self._generate_realistic_parameters,
            "edge_case": self._generate_edge_case_parameters,
            "boundary": self._generate_boundary_parameters,
            "stress": self._generate_stress_parameters,
            "error": self._generate_error_parameters,
            "random": self._generate_random_parameters
        }
    
    def _setup_edge_case_detectors(self) -> Dict[str, Callable]:
        """Setup edge case detectors for different parameter types"""
        return {
            "string": self._detect_string_edge_cases,
            "numeric": self._detect_numeric_edge_cases,
            "collection": self._detect_collection_edge_cases,
            "datetime": self._detect_datetime_edge_cases,
            "boolean": self._detect_boolean_edge_cases,
            "custom": self._detect_custom_edge_cases
        }
    
    def generate_unique_diverse_tests(self, func: Callable, num_tests: int = 20) -> List[UniqueTestCase]:
        """Generate unique, diverse, and intuitive test cases"""
        analysis = self._analyze_function(func)
        test_cases = []
        
        # Generate tests based on function characteristics
        function_type = self._classify_function(func, analysis)
        
        if function_type in self.test_scenarios:
            scenarios = self.test_scenarios[function_type]
            
            # Generate tests for each scenario
            for scenario in scenarios[:num_tests//2]:  # Use half for scenarios
                test_case = self._create_scenario_test(func, analysis, scenario, function_type)
                if test_case:
                    test_cases.append(test_case)
        
        # Generate edge case tests
        edge_tests = self._generate_edge_case_tests(func, analysis, num_tests//4)
        test_cases.extend(edge_tests)
        
        # Generate unique combination tests
        unique_tests = self._generate_unique_combination_tests(func, analysis, num_tests//4)
        test_cases.extend(unique_tests)
        
        # Score and rank all tests
        for test_case in test_cases:
            self._score_test_case(test_case, analysis)
        
        # Sort by overall quality and return top tests
        test_cases.sort(key=lambda x: x.overall_quality, reverse=True)
        return test_cases[:num_tests]
    
    def _analyze_function(self, func: Callable) -> Dict[str, Any]:
        """Analyze function for test generation insights"""
        try:
            signature = inspect.signature(func)
            docstring = inspect.getdoc(func) or ""
            source = inspect.getsource(func)
            
            # Parse AST for deeper analysis
            tree = ast.parse(source)
            
            analysis = {
                "name": func.__name__,
                "signature": signature,
                "docstring": docstring,
                "source_code": source,
                "parameters": list(signature.parameters.keys()),
                "return_annotation": str(signature.return_annotation),
                "is_async": inspect.iscoroutinefunction(func),
                "is_generator": inspect.isgeneratorfunction(func),
                "complexity_score": self._calculate_complexity(tree),
                "dependencies": self._find_dependencies(tree),
                "side_effects": self._find_side_effects(tree),
                "business_logic_hints": self._extract_business_logic_hints(docstring, source),
                "domain_context": self._extract_domain_context(func.__name__, docstring),
                "parameter_types": self._analyze_parameter_types(signature),
                "unique_characteristics": self._find_unique_characteristics(func, source, docstring)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing function {func.__name__}: {e}")
            return {}
    
    def _classify_function(self, func: Callable, analysis: Dict[str, Any]) -> str:
        """Classify function type for appropriate test generation"""
        name = func.__name__.lower()
        docstring = analysis.get("docstring", "").lower()
        source = analysis.get("source_code", "").lower()
        
        # Classification keywords
        if any(keyword in name or keyword in docstring for keyword in ["validate", "check", "verify"]):
            return "validation"
        elif any(keyword in name or keyword in docstring for keyword in ["transform", "convert", "process"]):
            return "transformation"
        elif any(keyword in name or keyword in docstring for keyword in ["calculate", "compute", "math"]):
            return "calculation"
        elif any(keyword in name or keyword in docstring for keyword in ["business", "workflow", "rule"]):
            return "business_logic"
        elif any(keyword in name or keyword in docstring for keyword in ["data", "batch", "stream"]):
            return "data_processing"
        else:
            return "general"
    
    def _create_scenario_test(self, func: Callable, analysis: Dict[str, Any], 
                            scenario: str, function_type: str) -> Optional[UniqueTestCase]:
        """Create a test case for a specific scenario"""
        try:
            # Generate intuitive name
            name = self._generate_intuitive_name(func.__name__, scenario, function_type)
            
            # Generate description
            description = self._generate_intuitive_description(func.__name__, scenario, function_type)
            
            # Generate parameters based on scenario
            parameters = self._generate_scenario_parameters(analysis, scenario)
            
            # Generate assertions
            assertions = self._generate_scenario_assertions(scenario, function_type)
            
            # Determine test type
            test_type = self._determine_test_type(scenario)
            
            test_case = UniqueTestCase(
                name=name,
                description=description,
                test_type=test_type,
                function_name=func.__name__,
                parameters=parameters,
                expected_result=None,
                assertions=assertions,
                async_test=analysis.get("is_async", False)
            )
            
            return test_case
            
        except Exception as e:
            logger.error(f"Error creating scenario test for {scenario}: {e}")
            return None
    
    def _generate_intuitive_name(self, function_name: str, scenario: str, function_type: str) -> str:
        """Generate intuitive test name"""
        # Choose naming pattern based on function type
        if function_type == "validation":
            pattern = "should_{behavior}_when_{condition}"
            behavior = "validate_input"
            condition = scenario.replace("_", "_")
        elif function_type == "transformation":
            pattern = "should_{behavior}_for_{scenario}"
            behavior = "transform_data"
        elif function_type == "calculation":
            pattern = "should_{behavior}_with_{input}"
            behavior = "calculate_accurately"
        else:
            pattern = "test_{function}_{scenario}"
        
        # Apply pattern
        if "should" in pattern:
            name = pattern.format(
                behavior=behavior,
                condition=condition if 'condition' in pattern else scenario,
                scenario=scenario,
                input=scenario,
                function=function_name
            )
        else:
            name = pattern.format(
                function=function_name,
                scenario=scenario
            )
        
        return name
    
    def _generate_intuitive_description(self, function_name: str, scenario: str, function_type: str) -> str:
        """Generate intuitive test description"""
        descriptions = {
            "validation": f"Verify {function_name} correctly handles {scenario.replace('_', ' ')} validation",
            "transformation": f"Verify {function_name} performs {scenario.replace('_', ' ')} transformation correctly",
            "calculation": f"Verify {function_name} calculates {scenario.replace('_', ' ')} accurately",
            "business_logic": f"Verify {function_name} implements {scenario.replace('_', ' ')} business logic correctly",
            "data_processing": f"Verify {function_name} processes data for {scenario.replace('_', ' ')} scenario"
        }
        
        return descriptions.get(function_type, f"Test {function_name} for {scenario.replace('_', ' ')} scenario")
    
    def _generate_scenario_parameters(self, analysis: Dict[str, Any], scenario: str) -> Dict[str, Any]:
        """Generate parameters based on scenario"""
        parameters = {}
        param_types = analysis.get("parameter_types", {})
        
        for param_name, param_type in param_types.items():
            if scenario == "happy_path_validation":
                parameters[param_name] = self._generate_happy_path_value(param_name, param_type)
            elif scenario == "boundary_value_testing":
                parameters[param_name] = self._generate_boundary_value(param_name, param_type)
            elif scenario == "edge_case_handling":
                parameters[param_name] = self._generate_edge_case_value(param_name, param_type)
            elif scenario == "stress_testing":
                parameters[param_name] = self._generate_stress_value(param_name, param_type)
            else:
                parameters[param_name] = self._generate_realistic_value(param_name, param_type)
        
        return parameters
    
    def _generate_scenario_assertions(self, scenario: str, function_type: str) -> List[str]:
        """Generate assertions based on scenario and function type"""
        assertions = []
        
        if function_type == "validation":
            if "happy_path" in scenario:
                assertions.extend([
                    "assert result is not None",
                    "assert result is True or result.get('valid', False)"
                ])
            elif "boundary" in scenario:
                assertions.extend([
                    "assert result is not None",
                    "assert isinstance(result, (bool, dict))"
                ])
            elif "error" in scenario:
                assertions.extend([
                    "assert result is not None",
                    "assert 'error' in str(result) or result is False"
                ])
        
        elif function_type == "transformation":
            assertions.extend([
                "assert result is not None",
                "assert result != input_data"
            ])
        
        elif function_type == "calculation":
            assertions.extend([
                "assert result is not None",
                "assert isinstance(result, (int, float, decimal.Decimal))"
            ])
        
        else:
            assertions.extend([
                "assert result is not None",
                "assert result is not False"
            ])
        
        return assertions
    
    def _determine_test_type(self, scenario: str) -> str:
        """Determine test type based on scenario"""
        if "boundary" in scenario or "edge" in scenario:
            return "edge_case"
        elif "error" in scenario or "invalid" in scenario:
            return "error_handling"
        elif "stress" in scenario or "performance" in scenario:
            return "performance"
        elif "happy_path" in scenario:
            return "validation"
        else:
            return "unit"
    
    def _generate_edge_case_tests(self, func: Callable, analysis: Dict[str, Any], num_tests: int) -> List[UniqueTestCase]:
        """Generate edge case tests"""
        test_cases = []
        param_types = analysis.get("parameter_types", {})
        
        for param_name, param_type in list(param_types.items())[:num_tests]:
            edge_cases = self._detect_edge_cases(param_type)
            
            for edge_case in edge_cases[:2]:  # Limit to 2 edge cases per parameter
                test_case = UniqueTestCase(
                    name=f"test_{func.__name__}_handles_{param_name}_edge_case_{edge_case}",
                    description=f"Verify {func.__name__} handles {param_name} edge case: {edge_case}",
                    test_type="edge_case",
                    function_name=func.__name__,
                    parameters={param_name: edge_case},
                    expected_result=None,
                    assertions=["assert result is not None or exception is raised"],
                    async_test=analysis.get("is_async", False)
                )
                test_cases.append(test_case)
        
        return test_cases
    
    def _generate_unique_combination_tests(self, func: Callable, analysis: Dict[str, Any], num_tests: int) -> List[UniqueTestCase]:
        """Generate unique combination tests"""
        test_cases = []
        param_types = analysis.get("parameter_types", {})
        
        # Generate unique combinations of parameter values
        combinations = self._generate_parameter_combinations(param_types, num_tests)
        
        for i, combination in enumerate(combinations):
            test_case = UniqueTestCase(
                name=f"test_{func.__name__}_unique_combination_{i+1}",
                description=f"Verify {func.__name__} handles unique parameter combination {i+1}",
                test_type="combination",
                function_name=func.__name__,
                parameters=combination,
                expected_result=None,
                assertions=["assert result is not None"],
                async_test=analysis.get("is_async", False)
            )
            test_cases.append(test_case)
        
        return test_cases
    
    def _score_test_case(self, test_case: UniqueTestCase, analysis: Dict[str, Any]):
        """Score test case for uniqueness, diversity, and intuition"""
        # Uniqueness score (0-1)
        uniqueness_score = 0.0
        
        # Check for unique parameter combinations
        unique_params = set(str(v) for v in test_case.parameters.values())
        uniqueness_score += len(unique_params) * 0.1
        
        # Check for unique test type
        unique_types = ["edge_case", "combination", "performance", "error_handling"]
        if test_case.test_type in unique_types:
            uniqueness_score += 0.3
        
        # Check for unique naming patterns
        if "should" in test_case.name or "verify" in test_case.description:
            uniqueness_score += 0.2
        
        test_case.uniqueness_score = min(uniqueness_score, 1.0)
        
        # Diversity score (0-1)
        diversity_score = 0.0
        
        # Check parameter type diversity
        param_types = set(type(v).__name__ for v in test_case.parameters.values())
        diversity_score += len(param_types) * 0.2
        
        # Check for diverse scenarios
        diverse_scenarios = ["boundary", "edge", "stress", "error", "combination"]
        if any(scenario in test_case.name for scenario in diverse_scenarios):
            diversity_score += 0.3
        
        test_case.diversity_score = min(diversity_score, 1.0)
        
        # Intuition score (0-1)
        intuition_score = 0.0
        
        # Check for intuitive naming
        if "should" in test_case.name.lower():
            intuition_score += 0.3
        if "verify" in test_case.description.lower():
            intuition_score += 0.2
        if "correctly" in test_case.description.lower():
            intuition_score += 0.2
        
        # Check for descriptive assertions
        if len(test_case.assertions) > 1:
            intuition_score += 0.2
        
        test_case.intuition_score = min(intuition_score, 1.0)
        
        # Overall quality score
        test_case.overall_quality = (
            test_case.uniqueness_score * 0.4 +
            test_case.diversity_score * 0.3 +
            test_case.intuition_score * 0.3
        )
    
    # Helper methods for parameter generation
    def _generate_happy_path_value(self, param_name: str, param_type: str) -> Any:
        """Generate happy path value for parameter"""
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
            return 25
        elif "float" in param_type.lower():
            return 25.5
        elif "bool" in param_type.lower():
            return True
        elif "list" in param_type.lower():
            return ["item1", "item2"]
        elif "dict" in param_type.lower():
            return {"key": "value"}
        else:
            return "default_value"
    
    def _generate_boundary_value(self, param_name: str, param_type: str) -> Any:
        """Generate boundary value for parameter"""
        if "str" in param_type.lower():
            return ""  # Empty string boundary
        elif "int" in param_type.lower():
            return 0  # Zero boundary
        elif "float" in param_type.lower():
            return 0.0
        elif "bool" in param_type.lower():
            return False
        elif "list" in param_type.lower():
            return []  # Empty list boundary
        elif "dict" in param_type.lower():
            return {}  # Empty dict boundary
        else:
            return None
    
    def _generate_edge_case_value(self, param_name: str, param_type: str) -> Any:
        """Generate edge case value for parameter"""
        if "str" in param_type.lower():
            return " "  # Whitespace only
        elif "int" in param_type.lower():
            return -1  # Negative boundary
        elif "float" in param_type.lower():
            return float('inf')  # Infinity
        else:
            return None
    
    def _generate_stress_value(self, param_name: str, param_type: str) -> Any:
        """Generate stress test value for parameter"""
        if "str" in param_type.lower():
            return "x" * 1000  # Very long string
        elif "int" in param_type.lower():
            return 999999999  # Large number
        elif "list" in param_type.lower():
            return list(range(1000))  # Large list
        else:
            return "stress_value"
    
    def _generate_realistic_value(self, param_name: str, param_type: str) -> Any:
        """Generate realistic value for parameter"""
        return self._generate_happy_path_value(param_name, param_type)
    
    def _detect_edge_cases(self, param_type: str) -> List[Any]:
        """Detect edge cases for parameter type"""
        if "str" in param_type.lower():
            return ["", " ", "a" * 1000, "unicode_测试", None]
        elif "int" in param_type.lower():
            return [0, -1, 2**31-1, -2**31]
        elif "float" in param_type.lower():
            return [0.0, -0.0, float('inf'), float('-inf'), float('nan')]
        elif "bool" in param_type.lower():
            return [True, False]
        elif "list" in param_type.lower():
            return [[], [None], [1, 2, 3]]
        elif "dict" in param_type.lower():
            return [{}, {"key": None}]
        else:
            return [None]
    
    def _generate_parameter_combinations(self, param_types: Dict[str, str], num_combinations: int) -> List[Dict[str, Any]]:
        """Generate unique parameter combinations"""
        combinations = []
        
        for i in range(num_combinations):
            combination = {}
            for param_name, param_type in param_types.items():
                # Generate different values for each combination
                if i % 3 == 0:
                    combination[param_name] = self._generate_happy_path_value(param_name, param_type)
                elif i % 3 == 1:
                    combination[param_name] = self._generate_boundary_value(param_name, param_type)
                else:
                    combination[param_name] = self._generate_edge_case_value(param_name, param_type)
            
            combinations.append(combination)
        
        return combinations
    
    def _analyze_parameter_types(self, signature: inspect.Signature) -> Dict[str, str]:
        """Analyze parameter types from signature"""
        param_types = {}
        for param_name, param in signature.parameters.items():
            if param.annotation != inspect.Parameter.empty:
                param_types[param_name] = str(param.annotation)
            else:
                param_types[param_name] = "Any"
        return param_types
    
    def _calculate_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        return complexity
    
    def _find_dependencies(self, tree: ast.AST) -> List[str]:
        """Find external dependencies"""
        dependencies = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    dependencies.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    dependencies.append(node.module)
        return dependencies
    
    def _find_side_effects(self, tree: ast.AST) -> List[str]:
        """Find potential side effects"""
        side_effects = []
        side_effect_patterns = [
            r'\.write\(',
            r'\.append\(',
            r'\.update\(',
            r'\.remove\(',
            r'\.delete\(',
            r'\.save\(',
            r'\.create\(',
            r'print\(',
            r'logging\.',
            r'open\('
        ]
        
        source = ast.unparse(tree)
        for pattern in side_effect_patterns:
            if re.search(pattern, source):
                side_effects.append(pattern)
        
        return side_effects
    
    def _extract_business_logic_hints(self, docstring: str, source: str) -> List[str]:
        """Extract business logic hints"""
        hints = []
        business_keywords = ["business", "rule", "workflow", "process", "logic", "policy"]
        
        for keyword in business_keywords:
            if keyword in docstring.lower() or keyword in source.lower():
                hints.append(keyword)
        
        return hints
    
    def _extract_domain_context(self, function_name: str, docstring: str) -> str:
        """Extract domain context"""
        context = function_name.lower() + " " + docstring.lower()
        
        if any(word in context for word in ["user", "account", "profile"]):
            return "user_management"
        elif any(word in context for word in ["video", "media", "content"]):
            return "media_processing"
        elif any(word in context for word in ["payment", "billing", "subscription"]):
            return "payment_processing"
        elif any(word in context for word in ["notification", "alert", "message"]):
            return "notification_system"
        else:
            return "general"
    
    def _find_unique_characteristics(self, func: Callable, source: str, docstring: str) -> List[str]:
        """Find unique characteristics of the function"""
        characteristics = []
        
        if inspect.iscoroutinefunction(func):
            characteristics.append("async_function")
        if inspect.isgeneratorfunction(func):
            characteristics.append("generator_function")
        if "@" in source:
            characteristics.append("decorated_function")
        if "raise" in source:
            characteristics.append("exception_raising")
        if "logging" in source or "logger" in source:
            characteristics.append("logging_enabled")
        
        return characteristics


def demonstrate_unique_diverse_generation():
    """Demonstrate the unique, diverse, and intuitive test generation"""
    
    # Example function to test
    def process_user_data(user_data: dict, validation_rules: list) -> dict:
        """
        Process and validate user data according to business rules.
        
        Args:
            user_data: Dictionary containing user information
            validation_rules: List of validation rules to apply
            
        Returns:
            Dictionary with processed data and validation results
            
        Raises:
            ValueError: If user_data is invalid
            TypeError: If validation_rules is not a list
        """
        if not isinstance(user_data, dict):
            raise TypeError("user_data must be a dictionary")
        
        if not isinstance(validation_rules, list):
            raise TypeError("validation_rules must be a list")
        
        processed_data = user_data.copy()
        validation_results = []
        
        for rule in validation_rules:
            if rule == "email_required" and "email" not in processed_data:
                validation_results.append("Email is required")
            elif rule == "age_validation" and processed_data.get("age", 0) < 18:
                validation_results.append("Age must be 18 or older")
        
        return {
            "processed_data": processed_data,
            "validation_results": validation_results,
            "is_valid": len(validation_results) == 0,
            "timestamp": datetime.now().isoformat()
        }
    
    # Generate unique, diverse, and intuitive tests
    generator = UniqueDiverseTestGenerator()
    test_cases = generator.generate_unique_diverse_tests(process_user_data, num_tests=15)
    
    print(f"Generated {len(test_cases)} unique, diverse, and intuitive test cases:")
    print("=" * 80)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"{i}. {test_case.name}")
        print(f"   Description: {test_case.description}")
        print(f"   Type: {test_case.test_type}")
        print(f"   Quality Scores: U={test_case.uniqueness_score:.2f}, D={test_case.diversity_score:.2f}, I={test_case.intuition_score:.2f}")
        print(f"   Overall Quality: {test_case.overall_quality:.2f}")
        print(f"   Parameters: {test_case.parameters}")
        print(f"   Assertions: {test_case.assertions}")
        print()


if __name__ == "__main__":
    demonstrate_unique_diverse_generation()
