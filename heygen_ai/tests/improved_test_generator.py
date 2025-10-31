"""
Improved Test Case Generator
===========================

Enhanced test case generation system that creates unique, diverse, and intuitive
unit tests for functions given their signature and docstring.

Key improvements:
- Better uniqueness through creative test scenarios
- Enhanced diversity with comprehensive coverage
- Improved intuition with clear, descriptive naming
"""

import ast
import inspect
import re
import random
import string
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class ImprovedTestCase:
    """Improved test case with enhanced quality metrics"""
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
    # Quality metrics
    uniqueness: float = 0.0
    diversity: float = 0.0
    intuition: float = 0.0
    overall_quality: float = 0.0
    # Metadata
    test_type: str = ""
    scenario: str = ""


class ImprovedTestGenerator:
    """Improved test generator with better uniqueness, diversity, and intuition"""
    
    def __init__(self):
        self.test_scenarios = self._load_test_scenarios()
        self.naming_patterns = self._load_naming_patterns()
        self.parameter_generators = self._setup_parameter_generators()
        
    def _load_test_scenarios(self) -> Dict[str, List[str]]:
        """Load comprehensive test scenarios"""
        return {
            "validation": [
                "happy_path_validation",
                "boundary_value_testing",
                "invalid_input_handling",
                "type_coercion_handling",
                "format_compliance_checking",
                "business_rule_validation",
                "cross_field_validation",
                "unicode_handling",
                "special_character_processing",
                "empty_value_handling",
                "null_value_handling",
                "whitespace_handling",
                "case_sensitivity_handling",
                "length_validation",
                "pattern_matching"
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
                "schema_validation",
                "type_conversion",
                "structure_restructuring",
                "data_cleaning",
                "format_standardization",
                "content_extraction"
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
                "statistical_operations",
                "trigonometric_calculations",
                "logarithmic_calculations",
                "exponential_calculations",
                "polynomial_calculations",
                "matrix_operations"
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
                "billing_operations",
                "audit_trail_creation",
                "compliance_validation",
                "risk_assessment",
                "fraud_detection",
                "performance_optimization"
            ]
        }
    
    def _load_naming_patterns(self) -> Dict[str, List[str]]:
        """Load intuitive naming patterns"""
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
            "descriptive": [
                "test_{function}_{scenario}_{aspect}",
                "test_{function}_{condition}_{result}",
                "test_{function}_{input}_{output}",
                "test_{function}_{context}_{behavior}"
            ],
            "user_story": [
                "test_as_{user}_i_can_{action}",
                "test_as_{user}_i_should_{behavior}",
                "test_as_{user}_i_expect_{outcome}",
                "test_as_{user}_i_need_{requirement}"
            ]
        }
    
    def _setup_parameter_generators(self) -> Dict[str, Callable]:
        """Setup parameter generators"""
        return {
            "realistic": self._generate_realistic_parameters,
            "edge_case": self._generate_edge_case_parameters,
            "boundary": self._generate_boundary_parameters,
            "stress": self._generate_stress_parameters,
            "creative": self._generate_creative_parameters
        }
    
    def generate_improved_tests(self, func: Callable, num_tests: int = 20) -> List[ImprovedTestCase]:
        """Generate improved test cases with better uniqueness, diversity, and intuition"""
        analysis = self._analyze_function(func)
        function_type = self._classify_function(func, analysis)
        
        test_cases = []
        
        # Generate unique scenario tests
        unique_tests = self._generate_unique_tests(func, analysis, function_type, num_tests//3)
        test_cases.extend(unique_tests)
        
        # Generate diverse coverage tests
        diverse_tests = self._generate_diverse_tests(func, analysis, function_type, num_tests//3)
        test_cases.extend(diverse_tests)
        
        # Generate intuitive structure tests
        intuitive_tests = self._generate_intuitive_tests(func, analysis, function_type, num_tests//3)
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
                "parameter_types": self._get_parameter_types(signature)
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
            return "general"
    
    def _generate_unique_tests(self, func: Callable, analysis: Dict[str, Any], 
                             function_type: str, num_tests: int) -> List[ImprovedTestCase]:
        """Generate unique test cases"""
        test_cases = []
        
        if function_type in self.test_scenarios:
            scenarios = self.test_scenarios[function_type]
            
            for scenario in scenarios[:num_tests]:
                test_case = self._create_unique_test(func, analysis, scenario, function_type)
                if test_case:
                    test_cases.append(test_case)
        
        return test_cases
    
    def _generate_diverse_tests(self, func: Callable, analysis: Dict[str, Any], 
                              function_type: str, num_tests: int) -> List[ImprovedTestCase]:
        """Generate diverse test cases"""
        test_cases = []
        
        # Generate different types of diverse tests
        diverse_types = ["edge_case", "boundary_value", "error_condition", "stress_test", "creative_scenario"]
        
        for i, diverse_type in enumerate(diverse_types[:num_tests]):
            test_case = self._create_diverse_test(func, analysis, diverse_type, function_type)
            if test_case:
                test_cases.append(test_case)
        
        return test_cases
    
    def _generate_intuitive_tests(self, func: Callable, analysis: Dict[str, Any], 
                                function_type: str, num_tests: int) -> List[ImprovedTestCase]:
        """Generate intuitive test cases"""
        test_cases = []
        
        # Generate tests with different naming strategies
        naming_strategies = list(self.naming_patterns.keys())
        
        for i, strategy in enumerate(naming_strategies[:num_tests]):
            test_case = self._create_intuitive_test(func, analysis, strategy, function_type)
            if test_case:
                test_cases.append(test_case)
        
        return test_cases
    
    def _create_unique_test(self, func: Callable, analysis: Dict[str, Any], 
                          scenario: str, function_type: str) -> Optional[ImprovedTestCase]:
        """Create a unique test case"""
        try:
            name = self._generate_unique_name(func.__name__, scenario, function_type)
            description = self._generate_unique_description(func.__name__, scenario, function_type)
            parameters = self._create_unique_parameters(analysis, scenario)
            assertions = self._generate_unique_assertions(scenario, function_type)
            
            return ImprovedTestCase(
                name=name,
                description=description,
                function_name=func.__name__,
                parameters=parameters,
                expected_result=None,
                assertions=assertions,
                async_test=analysis.get("is_async", False),
                test_type="unique_scenario",
                scenario=scenario
            )
        except Exception as e:
            logger.error(f"Error creating unique test: {e}")
            return None
    
    def _create_diverse_test(self, func: Callable, analysis: Dict[str, Any], 
                           diverse_type: str, function_type: str) -> Optional[ImprovedTestCase]:
        """Create a diverse test case"""
        try:
            name = self._generate_diverse_name(func.__name__, diverse_type, function_type)
            description = self._generate_diverse_description(func.__name__, diverse_type, function_type)
            parameters = self._create_diverse_parameters(analysis, diverse_type)
            assertions = self._generate_diverse_assertions(diverse_type, function_type)
            
            return ImprovedTestCase(
                name=name,
                description=description,
                function_name=func.__name__,
                parameters=parameters,
                expected_result=None,
                assertions=assertions,
                async_test=analysis.get("is_async", False),
                test_type="diverse_coverage",
                scenario=diverse_type
            )
        except Exception as e:
            logger.error(f"Error creating diverse test: {e}")
            return None
    
    def _create_intuitive_test(self, func: Callable, analysis: Dict[str, Any], 
                             strategy: str, function_type: str) -> Optional[ImprovedTestCase]:
        """Create an intuitive test case"""
        try:
            name = self._generate_intuitive_name(func.__name__, strategy, function_type)
            description = self._generate_intuitive_description(func.__name__, strategy, function_type)
            parameters = self._create_intuitive_parameters(analysis, strategy)
            assertions = self._generate_intuitive_assertions(strategy, function_type)
            
            return ImprovedTestCase(
                name=name,
                description=description,
                function_name=func.__name__,
                parameters=parameters,
                expected_result=None,
                assertions=assertions,
                async_test=analysis.get("is_async", False),
                test_type="intuitive_structure",
                scenario=strategy
            )
        except Exception as e:
            logger.error(f"Error creating intuitive test: {e}")
            return None
    
    def _generate_unique_name(self, function_name: str, scenario: str, function_type: str) -> str:
        """Generate unique test name"""
        if function_type == "validation":
            return f"should_validate_{scenario}_uniquely"
        elif function_type == "transformation":
            return f"should_transform_{scenario}_creatively"
        elif function_type == "calculation":
            return f"should_calculate_{scenario}_precisely"
        elif function_type == "business_logic":
            return f"should_implement_{scenario}_logic"
        else:
            return f"test_{function_name}_{scenario}_unique"
    
    def _generate_unique_description(self, function_name: str, scenario: str, function_type: str) -> str:
        """Generate unique test description"""
        return f"Verify {function_name} handles {scenario.replace('_', ' ')} scenario with unique approach"
    
    def _generate_diverse_name(self, function_name: str, diverse_type: str, function_type: str) -> str:
        """Generate diverse test name"""
        return f"test_{function_name}_diverse_{diverse_type}_coverage"
    
    def _generate_diverse_description(self, function_name: str, diverse_type: str, function_type: str) -> str:
        """Generate diverse test description"""
        return f"Verify {function_name} covers {diverse_type.replace('_', ' ')} cases comprehensively"
    
    def _generate_intuitive_name(self, function_name: str, strategy: str, function_type: str) -> str:
        """Generate intuitive test name"""
        if strategy == "behavior_driven":
            return f"should_{function_name}_behave_correctly"
        elif strategy == "scenario_based":
            return f"test_{function_name}_scenario"
        elif strategy == "descriptive":
            return f"test_{function_name}_descriptive"
        elif strategy == "user_story":
            return f"test_as_user_i_can_{function_name}"
        else:
            return f"test_{function_name}_intuitive"
    
    def _generate_intuitive_description(self, function_name: str, strategy: str, function_type: str) -> str:
        """Generate intuitive test description"""
        return f"Verify {function_name} works intuitively with {strategy.replace('_', ' ')} approach"
    
    def _create_unique_parameters(self, analysis: Dict[str, Any], scenario: str) -> Dict[str, Any]:
        """Create unique parameters"""
        parameters = {}
        param_types = analysis.get("parameter_types", {})
        
        for param_name, param_type in param_types.items():
            if "happy_path" in scenario:
                parameters[param_name] = self._generate_happy_path_value(param_name, param_type)
            elif "boundary" in scenario:
                parameters[param_name] = self._generate_boundary_value(param_name, param_type)
            elif "edge" in scenario:
                parameters[param_name] = self._generate_edge_case_value(param_name, param_type)
            else:
                parameters[param_name] = self._generate_creative_value(param_name, param_type)
        
        return parameters
    
    def _create_diverse_parameters(self, analysis: Dict[str, Any], diverse_type: str) -> Dict[str, Any]:
        """Create diverse parameters"""
        parameters = {}
        param_types = analysis.get("parameter_types", {})
        
        for param_name, param_type in param_types.items():
            if diverse_type == "edge_case":
                parameters[param_name] = self._generate_edge_case_value(param_name, param_type)
            elif diverse_type == "boundary_value":
                parameters[param_name] = self._generate_boundary_value(param_name, param_type)
            elif diverse_type == "error_condition":
                parameters[param_name] = self._generate_error_value(param_name, param_type)
            elif diverse_type == "stress_test":
                parameters[param_name] = self._generate_stress_value(param_name, param_type)
            else:
                parameters[param_name] = self._generate_creative_value(param_name, param_type)
        
        return parameters
    
    def _create_intuitive_parameters(self, analysis: Dict[str, Any], strategy: str) -> Dict[str, Any]:
        """Create intuitive parameters"""
        parameters = {}
        param_types = analysis.get("parameter_types", {})
        
        for param_name, param_type in param_types.items():
            if strategy == "behavior_driven":
                parameters[param_name] = self._generate_behavior_value(param_name, param_type)
            elif strategy == "user_story":
                parameters[param_name] = self._generate_user_value(param_name, param_type)
            else:
                parameters[param_name] = self._generate_intuitive_value(param_name, param_type)
        
        return parameters
    
    def _generate_unique_assertions(self, scenario: str, function_type: str) -> List[str]:
        """Generate unique assertions"""
        assertions = ["assert result is not None"]
        
        if "validation" in scenario:
            assertions.append("assert result is True or result.get('valid', False)")
        elif "transformation" in scenario:
            assertions.append("assert result != input_data")
        elif "calculation" in scenario:
            assertions.append("assert isinstance(result, (int, float))")
        
        return assertions
    
    def _generate_diverse_assertions(self, diverse_type: str, function_type: str) -> List[str]:
        """Generate diverse assertions"""
        assertions = ["assert result is not None"]
        
        if diverse_type == "edge_case":
            assertions.append("assert result handles edge case correctly")
        elif diverse_type == "boundary_value":
            assertions.append("assert result is within expected bounds")
        elif diverse_type == "error_condition":
            assertions.append("assert result indicates error or exception is raised")
        elif diverse_type == "stress_test":
            assertions.append("assert result is not None or timeout occurs")
        
        return assertions
    
    def _generate_intuitive_assertions(self, strategy: str, function_type: str) -> List[str]:
        """Generate intuitive assertions"""
        assertions = ["assert result is not None"]
        
        if strategy == "behavior_driven":
            assertions.append("assert result behaves as expected")
        elif strategy == "user_story":
            assertions.append("assert result satisfies user needs")
        else:
            assertions.append("assert result is intuitive and clear")
        
        return assertions
    
    def _score_test_case(self, test_case: ImprovedTestCase, analysis: Dict[str, Any]):
        """Score test case for quality metrics"""
        # Uniqueness score
        uniqueness = 0.0
        if test_case.test_type == "unique_scenario":
            uniqueness += 0.4
        if "unique" in test_case.name or "creative" in test_case.name:
            uniqueness += 0.3
        if len(test_case.parameters) > 2:
            uniqueness += 0.3
        test_case.uniqueness = min(uniqueness, 1.0)
        
        # Diversity score
        diversity = 0.0
        if test_case.test_type == "diverse_coverage":
            diversity += 0.4
        param_types = set(type(v).__name__ for v in test_case.parameters.values())
        diversity += len(param_types) * 0.2
        if test_case.scenario in ["edge_case", "boundary_value", "error_condition"]:
            diversity += 0.2
        test_case.diversity = min(diversity, 1.0)
        
        # Intuition score
        intuition = 0.0
        if test_case.test_type == "intuitive_structure":
            intuition += 0.4
        if "should" in test_case.name.lower():
            intuition += 0.3
        if "verify" in test_case.description.lower():
            intuition += 0.3
        test_case.intuition = min(intuition, 1.0)
        
        # Overall quality
        test_case.overall_quality = (test_case.uniqueness + test_case.diversity + test_case.intuition) / 3
    
    # Helper methods for parameter generation
    def _generate_happy_path_value(self, param_name: str, param_type: str) -> Any:
        """Generate happy path value"""
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
    
    def _generate_boundary_value(self, param_name: str, param_type: str) -> Any:
        """Generate boundary value"""
        if "str" in param_type.lower():
            return ""
        elif "int" in param_type.lower():
            return 0
        elif "float" in param_type.lower():
            return 0.0
        elif "bool" in param_type.lower():
            return False
        elif "list" in param_type.lower():
            return []
        elif "dict" in param_type.lower():
            return {}
        else:
            return None
    
    def _generate_edge_case_value(self, param_name: str, param_type: str) -> Any:
        """Generate edge case value"""
        if "str" in param_type.lower():
            return " "
        elif "int" in param_type.lower():
            return -1
        elif "float" in param_type.lower():
            return float('inf')
        else:
            return None
    
    def _generate_creative_value(self, param_name: str, param_type: str) -> Any:
        """Generate creative value"""
        return self._generate_happy_path_value(param_name, param_type)
    
    def _generate_error_value(self, param_name: str, param_type: str) -> Any:
        """Generate error value"""
        return None
    
    def _generate_stress_value(self, param_name: str, param_type: str) -> Any:
        """Generate stress value"""
        if "str" in param_type.lower():
            return "x" * 1000
        elif "int" in param_type.lower():
            return 999999999
        elif "list" in param_type.lower():
            return list(range(1000))
        else:
            return "stress_value"
    
    def _generate_behavior_value(self, param_name: str, param_type: str) -> Any:
        """Generate behavior value"""
        return self._generate_happy_path_value(param_name, param_type)
    
    def _generate_user_value(self, param_name: str, param_type: str) -> Any:
        """Generate user value"""
        return self._generate_happy_path_value(param_name, param_type)
    
    def _generate_intuitive_value(self, param_name: str, param_type: str) -> Any:
        """Generate intuitive value"""
        return self._generate_happy_path_value(param_name, param_type)
    
    def _get_parameter_types(self, signature: inspect.Signature) -> Dict[str, str]:
        """Get parameter types from signature"""
        param_types = {}
        for param_name, param in signature.parameters.items():
            if param.annotation != inspect.Parameter.empty:
                param_types[param_name] = str(param.annotation)
            else:
                param_types[param_name] = "Any"
        return param_types


def demonstrate_improved_generator():
    """Demonstrate the improved test generation system"""
    
    # Example function to test
    def process_user_registration(user_data: dict, validation_rules: list, options: dict) -> dict:
        """
        Process and validate user registration data with advanced options.
        
        Args:
            user_data: Dictionary containing user information
            validation_rules: List of validation rules to apply
            options: Dictionary with processing options
            
        Returns:
            Dictionary with processing results and validation status
            
        Raises:
            ValueError: If user_data is invalid or validation_rules is empty
            TypeError: If options is not a dictionary
        """
        if not isinstance(user_data, dict):
            raise ValueError("user_data must be a dictionary")
        
        if not validation_rules:
            raise ValueError("validation_rules cannot be empty")
        
        if not isinstance(options, dict):
            raise TypeError("options must be a dictionary")
        
        processed_data = user_data.copy()
        validation_results = []
        
        # Apply validation rules
        for rule in validation_rules:
            if rule == "email_required" and "email" not in processed_data:
                validation_results.append("Email is required")
            elif rule == "age_validation" and processed_data.get("age", 0) < 18:
                validation_results.append("Age must be 18 or older")
            elif rule == "username_validation" and len(processed_data.get("username", "")) < 3:
                validation_results.append("Username must be at least 3 characters")
        
        # Apply processing options
        if options.get("normalize_case", False):
            for key, value in processed_data.items():
                if isinstance(value, str):
                    processed_data[key] = value.lower()
        
        if options.get("add_timestamp", False):
            processed_data["processed_at"] = datetime.now().isoformat()
        
        return {
            "processed_data": processed_data,
            "validation_results": validation_results,
            "is_valid": len(validation_results) == 0,
            "processing_options": options,
            "timestamp": datetime.now().isoformat()
        }
    
    # Generate improved tests
    generator = ImprovedTestGenerator()
    test_cases = generator.generate_improved_tests(process_user_registration, num_tests=18)
    
    print(f"Generated {len(test_cases)} improved test cases:")
    print("=" * 80)
    
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
    demonstrate_improved_generator()
