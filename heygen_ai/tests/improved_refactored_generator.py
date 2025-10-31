"""
Improved Refactored Test Case Generator
======================================

Enhanced refactored test case generation system that creates unique, diverse, and intuitive
unit tests for functions given their signature and docstring.

This improved version focuses on:
- Enhanced uniqueness with creative patterns
- Better diversity with comprehensive scenarios
- Improved intuition with advanced naming strategies
- Optimized performance and quality
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
    # Enhanced quality metrics
    uniqueness: float = 0.0
    diversity: float = 0.0
    intuition: float = 0.0
    creativity: float = 0.0
    coverage: float = 0.0
    overall_quality: float = 0.0
    # Metadata
    test_type: str = ""
    scenario: str = ""
    complexity: str = ""


class ImprovedRefactoredGenerator:
    """Improved refactored test generator with enhanced capabilities"""
    
    def __init__(self):
        self.test_patterns = self._load_enhanced_patterns()
        self.naming_strategies = self._load_enhanced_naming()
        self.parameter_generators = self._setup_enhanced_generators()
        self.quality_optimizer = self._setup_quality_optimizer()
        
    def _load_enhanced_patterns(self) -> Dict[str, Dict[str, List[str]]]:
        """Load enhanced test patterns"""
        return {
            "validation": {
                "unique_scenarios": [
                    "happy_path_validation", "boundary_value_analysis", "type_coercion_handling",
                    "format_specification_compliance", "business_rule_validation", "cross_field_validation",
                    "temporal_validation", "unicode_handling", "special_character_processing",
                    "empty_value_handling", "null_value_handling", "whitespace_handling",
                    "case_sensitivity_handling", "length_validation", "pattern_matching"
                ],
                "diverse_cases": [
                    "valid_inputs", "invalid_inputs", "edge_cases", "boundary_values",
                    "type_variations", "format_variations", "size_variations", "character_variations",
                    "encoding_variations", "unicode_variations", "special_characters", "control_characters"
                ],
                "intuitive_scenarios": [
                    "user_friendly_validation", "clear_error_messages", "intuitive_workflow",
                    "descriptive_feedback", "helpful_guidance", "contextual_validation"
                ]
            },
            "transformation": {
                "unique_scenarios": [
                    "identity_transformation", "data_aggregation", "filtering_operations",
                    "sorting_operations", "format_conversion", "data_enrichment", "normalization",
                    "deduplication", "data_quality_improvement", "schema_validation", "type_conversion"
                ],
                "diverse_cases": [
                    "small_datasets", "large_datasets", "empty_datasets", "nested_structures",
                    "flat_structures", "mixed_structures", "consistent_data", "inconsistent_data",
                    "malformed_data", "structured_data", "unstructured_data", "semi_structured_data"
                ],
                "intuitive_scenarios": [
                    "clear_transformation", "predictable_output", "intuitive_mapping",
                    "user_understandable", "logical_flow", "transparent_process"
                ]
            },
            "calculation": {
                "unique_scenarios": [
                    "mathematical_accuracy", "precision_handling", "rounding_behavior",
                    "overflow_protection", "underflow_handling", "division_by_zero",
                    "floating_point_precision", "currency_calculations", "percentage_calculations"
                ],
                "diverse_cases": [
                    "positive_numbers", "negative_numbers", "zero_values", "very_small_numbers",
                    "very_large_numbers", "decimal_numbers", "integer_numbers", "fractional_numbers",
                    "irrational_numbers", "complex_numbers", "infinity_values", "nan_values"
                ],
                "intuitive_scenarios": [
                    "clear_calculations", "understandable_formulas", "logical_math",
                    "predictable_results", "transparent_process", "user_friendly_output"
                ]
            },
            "business_logic": {
                "unique_scenarios": [
                    "workflow_state_transitions", "business_rule_evaluation", "decision_tree_logic",
                    "approval_workflows", "pricing_calculations", "user_permission_checks",
                    "resource_allocation", "quota_management", "subscription_logic"
                ],
                "diverse_cases": [
                    "normal_operations", "edge_case_scenarios", "error_conditions",
                    "high_load_scenarios", "concurrent_operations", "data_inconsistencies",
                    "network_failures", "timeout_conditions", "resource_constraints"
                ],
                "intuitive_scenarios": [
                    "user_centric_logic", "business_understandable", "clear_workflow",
                    "intuitive_decisions", "transparent_rules", "logical_process"
                ]
            }
        }
    
    def _load_enhanced_naming(self) -> Dict[str, List[str]]:
        """Load enhanced naming strategies"""
        return {
            "behavior_driven": [
                "should_{behavior}_when_{condition}",
                "should_{behavior}_given_{context}",
                "should_{behavior}_for_{scenario}",
                "should_{behavior}_with_{input}",
                "should_{behavior}_under_{circumstances}",
                "should_{behavior}_in_{environment}"
            ],
            "descriptive": [
                "test_{function}_{scenario}_{aspect}",
                "test_{function}_{condition}_{result}",
                "test_{function}_{input}_{output}",
                "test_{function}_{context}_{behavior}",
                "test_{function}_{environment}_{response}",
                "test_{function}_{circumstances}_{outcome}"
            ],
            "scenario_based": [
                "test_{scenario}_scenario",
                "test_{scenario}_case",
                "test_{scenario}_situation",
                "test_{scenario}_context}",
                "test_{scenario}_condition",
                "test_{scenario}_environment"
            ],
            "user_story": [
                "test_as_{user}_i_can_{action}",
                "test_as_{user}_i_should_{behavior}",
                "test_as_{user}_i_expect_{outcome}",
                "test_as_{user}_i_need_{requirement}",
                "test_as_{user}_i_want_{feature}"
            ],
            "domain_specific": [
                "test_{domain}_{operation}",
                "test_{domain}_{validation}",
                "test_{domain}_{transformation}",
                "test_{domain}_{calculation}",
                "test_{domain}_{processing}"
            ]
        }
    
    def _setup_enhanced_generators(self) -> Dict[str, Callable]:
        """Setup enhanced parameter generators"""
        return {
            "realistic": self._generate_realistic_parameters,
            "edge_case": self._generate_edge_case_parameters,
            "boundary": self._generate_boundary_parameters,
            "creative": self._generate_creative_parameters,
            "diverse": self._generate_diverse_parameters,
            "intuitive": self._generate_intuitive_parameters
        }
    
    def _setup_quality_optimizer(self) -> Dict[str, Callable]:
        """Setup quality optimization functions"""
        return {
            "uniqueness": self._optimize_uniqueness,
            "diversity": self._optimize_diversity,
            "intuition": self._optimize_intuition,
            "creativity": self._optimize_creativity,
            "coverage": self._optimize_coverage
        }
    
    def generate_improved_tests(self, func: Callable, num_tests: int = 25) -> List[ImprovedTestCase]:
        """Generate improved test cases with enhanced uniqueness, diversity, and intuition"""
        analysis = self._analyze_function(func)
        function_type = self._classify_function(func, analysis)
        
        test_cases = []
        
        # Generate unique tests (35% of total)
        unique_tests = self._generate_unique_tests(func, analysis, function_type, int(num_tests * 0.35))
        test_cases.extend(unique_tests)
        
        # Generate diverse tests (35% of total)
        diverse_tests = self._generate_diverse_tests(func, analysis, function_type, int(num_tests * 0.35))
        test_cases.extend(diverse_tests)
        
        # Generate intuitive tests (20% of total)
        intuitive_tests = self._generate_intuitive_tests(func, analysis, function_type, int(num_tests * 0.20))
        test_cases.extend(intuitive_tests)
        
        # Generate creative tests (10% of total)
        creative_tests = self._generate_creative_tests(func, analysis, function_type, int(num_tests * 0.10))
        test_cases.extend(creative_tests)
        
        # Score and optimize all tests
        for test_case in test_cases:
            self._score_improved_test_case(test_case, analysis)
            self._optimize_test_case(test_case, analysis)
        
        # Remove duplicates and sort by quality
        test_cases = self._remove_duplicates(test_cases)
        test_cases.sort(key=lambda x: x.overall_quality, reverse=True)
        
        return test_cases[:num_tests]
    
    def _analyze_function(self, func: Callable) -> Dict[str, Any]:
        """Enhanced function analysis"""
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
                "complexity": self._calculate_complexity(source),
                "dependencies": self._find_dependencies(source),
                "side_effects": self._find_side_effects(source)
            }
        except Exception as e:
            logger.error(f"Error analyzing function {func.__name__}: {e}")
            return {}
    
    def _classify_function(self, func: Callable, analysis: Dict[str, Any]) -> str:
        """Enhanced function classification"""
        name = func.__name__.lower()
        docstring = analysis.get("docstring", "").lower()
        
        if any(keyword in name or keyword in docstring for keyword in ["validate", "check", "verify", "ensure", "confirm"]):
            return "validation"
        elif any(keyword in name or keyword in docstring for keyword in ["transform", "convert", "process", "translate", "migrate"]):
            return "transformation"
        elif any(keyword in name or keyword in docstring for keyword in ["calculate", "compute", "math", "solve", "derive"]):
            return "calculation"
        elif any(keyword in name or keyword in docstring for keyword in ["business", "workflow", "rule", "logic", "policy"]):
            return "business_logic"
        else:
            return "validation"  # Default to validation
    
    def _generate_unique_tests(self, func: Callable, analysis: Dict[str, Any], 
                             function_type: str, num_tests: int) -> List[ImprovedTestCase]:
        """Generate unique test cases with enhanced patterns"""
        test_cases = []
        
        if function_type in self.test_patterns:
            unique_scenarios = self.test_patterns[function_type]["unique_scenarios"]
            
            for i, scenario in enumerate(unique_scenarios[:num_tests]):
                test_case = self._create_unique_test(func, analysis, scenario, function_type, i)
                if test_case:
                    test_cases.append(test_case)
        
        return test_cases
    
    def _generate_diverse_tests(self, func: Callable, analysis: Dict[str, Any], 
                              function_type: str, num_tests: int) -> List[ImprovedTestCase]:
        """Generate diverse test cases with comprehensive coverage"""
        test_cases = []
        
        if function_type in self.test_patterns:
            diverse_cases = self.test_patterns[function_type]["diverse_cases"]
            
            for i, case in enumerate(diverse_cases[:num_tests]):
                test_case = self._create_diverse_test(func, analysis, case, function_type, i)
                if test_case:
                    test_cases.append(test_case)
        
        return test_cases
    
    def _generate_intuitive_tests(self, func: Callable, analysis: Dict[str, Any], 
                                function_type: str, num_tests: int) -> List[ImprovedTestCase]:
        """Generate intuitive test cases with clear naming"""
        test_cases = []
        
        if function_type in self.test_patterns:
            intuitive_scenarios = self.test_patterns[function_type]["intuitive_scenarios"]
            
            for i, scenario in enumerate(intuitive_scenarios[:num_tests]):
                test_case = self._create_intuitive_test(func, analysis, scenario, function_type, i)
                if test_case:
                    test_cases.append(test_case)
        
        return test_cases
    
    def _generate_creative_tests(self, func: Callable, analysis: Dict[str, Any], 
                               function_type: str, num_tests: int) -> List[ImprovedTestCase]:
        """Generate creative test cases with innovative approaches"""
        test_cases = []
        
        creative_approaches = ["innovative", "experimental", "cutting_edge", "revolutionary", "breakthrough"]
        
        for i, approach in enumerate(creative_approaches[:num_tests]):
            test_case = self._create_creative_test(func, analysis, approach, function_type, i)
            if test_case:
                test_cases.append(test_case)
        
        return test_cases
    
    def _create_unique_test(self, func: Callable, analysis: Dict[str, Any], 
                          scenario: str, function_type: str, index: int) -> Optional[ImprovedTestCase]:
        """Create a unique test case with enhanced patterns"""
        try:
            name = self._generate_unique_name(func.__name__, scenario, function_type, index)
            description = self._generate_unique_description(func.__name__, scenario, function_type)
            parameters = self._generate_unique_parameters(analysis, scenario)
            assertions = self._generate_unique_assertions(scenario, function_type)
            
            return ImprovedTestCase(
                name=name,
                description=description,
                function_name=func.__name__,
                parameters=parameters,
                assertions=assertions,
                async_test=analysis.get("is_async", False),
                test_type="unique",
                scenario=scenario,
                complexity="high"
            )
        except Exception as e:
            logger.error(f"Error creating unique test: {e}")
            return None
    
    def _create_diverse_test(self, func: Callable, analysis: Dict[str, Any], 
                           case: str, function_type: str, index: int) -> Optional[ImprovedTestCase]:
        """Create a diverse test case with comprehensive coverage"""
        try:
            name = self._generate_diverse_name(func.__name__, case, function_type, index)
            description = self._generate_diverse_description(func.__name__, case, function_type)
            parameters = self._generate_diverse_parameters(analysis, case)
            assertions = self._generate_diverse_assertions(case, function_type)
            
            return ImprovedTestCase(
                name=name,
                description=description,
                function_name=func.__name__,
                parameters=parameters,
                assertions=assertions,
                async_test=analysis.get("is_async", False),
                test_type="diverse",
                scenario=case,
                complexity="medium"
            )
        except Exception as e:
            logger.error(f"Error creating diverse test: {e}")
            return None
    
    def _create_intuitive_test(self, func: Callable, analysis: Dict[str, Any], 
                             scenario: str, function_type: str, index: int) -> Optional[ImprovedTestCase]:
        """Create an intuitive test case with clear naming"""
        try:
            name = self._generate_intuitive_name(func.__name__, scenario, function_type, index)
            description = self._generate_intuitive_description(func.__name__, scenario, function_type)
            parameters = self._generate_intuitive_parameters(analysis, scenario)
            assertions = self._generate_intuitive_assertions(scenario, function_type)
            
            return ImprovedTestCase(
                name=name,
                description=description,
                function_name=func.__name__,
                parameters=parameters,
                assertions=assertions,
                async_test=analysis.get("is_async", False),
                test_type="intuitive",
                scenario=scenario,
                complexity="low"
            )
        except Exception as e:
            logger.error(f"Error creating intuitive test: {e}")
            return None
    
    def _create_creative_test(self, func: Callable, analysis: Dict[str, Any], 
                            approach: str, function_type: str, index: int) -> Optional[ImprovedTestCase]:
        """Create a creative test case with innovative approach"""
        try:
            name = self._generate_creative_name(func.__name__, approach, function_type, index)
            description = self._generate_creative_description(func.__name__, approach, function_type)
            parameters = self._generate_creative_parameters(analysis, approach)
            assertions = self._generate_creative_assertions(approach, function_type)
            
            return ImprovedTestCase(
                name=name,
                description=description,
                function_name=func.__name__,
                parameters=parameters,
                assertions=assertions,
                async_test=analysis.get("is_async", False),
                test_type="creative",
                scenario=approach,
                complexity="very_high"
            )
        except Exception as e:
            logger.error(f"Error creating creative test: {e}")
            return None
    
    def _score_improved_test_case(self, test_case: ImprovedTestCase, analysis: Dict[str, Any]):
        """Enhanced scoring for improved test cases"""
        # Uniqueness score
        uniqueness = 0.0
        if test_case.test_type == "unique":
            uniqueness += 0.4
        if "unique" in test_case.name or "creative" in test_case.name:
            uniqueness += 0.3
        if len(test_case.parameters) > 2:
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
        
        # Creativity score
        creativity = 0.0
        if test_case.test_type == "creative":
            creativity += 0.4
        if "creative" in test_case.name or "innovative" in test_case.name:
            creativity += 0.3
        if test_case.complexity == "very_high":
            creativity += 0.3
        test_case.creativity = min(creativity, 1.0)
        
        # Coverage score
        coverage = 0.0
        if test_case.test_type == "diverse":
            coverage += 0.4
        if len(test_case.assertions) > 1:
            coverage += 0.3
        if test_case.scenario in ["edge_case", "boundary", "error_condition"]:
            coverage += 0.3
        test_case.coverage = min(coverage, 1.0)
        
        # Overall quality
        test_case.overall_quality = (
            test_case.uniqueness * 0.25 +
            test_case.diversity * 0.25 +
            test_case.intuition * 0.25 +
            test_case.creativity * 0.15 +
            test_case.coverage * 0.10
        )
    
    def _optimize_test_case(self, test_case: ImprovedTestCase, analysis: Dict[str, Any]):
        """Optimize test case for better quality"""
        # Optimize uniqueness
        if test_case.uniqueness < 0.7:
            test_case = self.quality_optimizer["uniqueness"](test_case, analysis)
        
        # Optimize diversity
        if test_case.diversity < 0.7:
            test_case = self.quality_optimizer["diversity"](test_case, analysis)
        
        # Optimize intuition
        if test_case.intuition < 0.7:
            test_case = self.quality_optimizer["intuition"](test_case, analysis)
    
    def _remove_duplicates(self, test_cases: List[ImprovedTestCase]) -> List[ImprovedTestCase]:
        """Remove duplicate test cases"""
        seen = set()
        unique_tests = []
        
        for test_case in test_cases:
            # Create a signature for the test case
            signature = (test_case.name, test_case.test_type, str(test_case.parameters))
            
            if signature not in seen:
                seen.add(signature)
                unique_tests.append(test_case)
        
        return unique_tests
    
    # Enhanced naming methods
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
    
    def _generate_diverse_name(self, function_name: str, case: str, function_type: str, index: int) -> str:
        """Generate diverse test name"""
        return f"test_{function_name}_diverse_{case}_{index}"
    
    def _generate_intuitive_name(self, function_name: str, scenario: str, function_type: str, index: int) -> str:
        """Generate intuitive test name"""
        if "user" in scenario:
            return f"test_as_user_i_can_{function_name}_{index}"
        elif "clear" in scenario:
            return f"test_{function_name}_clearly_{index}"
        else:
            return f"test_{function_name}_intuitively_{index}"
    
    def _generate_creative_name(self, function_name: str, approach: str, function_type: str, index: int) -> str:
        """Generate creative test name"""
        return f"test_{function_name}_creative_{approach}_{index}"
    
    # Enhanced description methods
    def _generate_unique_description(self, function_name: str, scenario: str, function_type: str) -> str:
        """Generate unique test description"""
        return f"Verify {function_name} handles {scenario.replace('_', ' ')} scenario with unique approach"
    
    def _generate_diverse_description(self, function_name: str, case: str, function_type: str) -> str:
        """Generate diverse test description"""
        return f"Verify {function_name} covers {case.replace('_', ' ')} cases comprehensively"
    
    def _generate_intuitive_description(self, function_name: str, scenario: str, function_type: str) -> str:
        """Generate intuitive test description"""
        return f"Verify {function_name} works intuitively with {scenario.replace('_', ' ')} approach"
    
    def _generate_creative_description(self, function_name: str, approach: str, function_type: str) -> str:
        """Generate creative test description"""
        return f"Verify {function_name} uses {approach.replace('_', ' ')} approach for enhanced functionality"
    
    # Enhanced parameter generation methods
    def _generate_unique_parameters(self, analysis: Dict[str, Any], scenario: str) -> Dict[str, Any]:
        """Generate unique parameters"""
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
    
    def _generate_diverse_parameters(self, analysis: Dict[str, Any], case: str) -> Dict[str, Any]:
        """Generate diverse parameters"""
        parameters = {}
        param_types = analysis.get("parameter_types", {})
        
        for param_name, param_type in param_types.items():
            if "valid" in case:
                parameters[param_name] = self._generate_valid_value(param_name, param_type)
            elif "invalid" in case:
                parameters[param_name] = self._generate_invalid_value(param_name, param_type)
            elif "edge" in case:
                parameters[param_name] = self._generate_edge_case_value(param_name, param_type)
            else:
                parameters[param_name] = self._generate_diverse_value(param_name, param_type)
        
        return parameters
    
    def _generate_intuitive_parameters(self, analysis: Dict[str, Any], scenario: str) -> Dict[str, Any]:
        """Generate intuitive parameters"""
        parameters = {}
        param_types = analysis.get("parameter_types", {})
        
        for param_name, param_type in param_types.items():
            if "user" in scenario:
                parameters[param_name] = self._generate_user_value(param_name, param_type)
            elif "clear" in scenario:
                parameters[param_name] = self._generate_clear_value(param_name, param_type)
            else:
                parameters[param_name] = self._generate_intuitive_value(param_name, param_type)
        
        return parameters
    
    def _generate_creative_parameters(self, analysis: Dict[str, Any], approach: str) -> Dict[str, Any]:
        """Generate creative parameters"""
        parameters = {}
        param_types = analysis.get("parameter_types", {})
        
        for param_name, param_type in param_types.items():
            if approach == "innovative":
                parameters[param_name] = self._generate_innovative_value(param_name, param_type)
            elif approach == "experimental":
                parameters[param_name] = self._generate_experimental_value(param_name, param_type)
            else:
                parameters[param_name] = self._generate_creative_value(param_name, param_type)
        
        return parameters
    
    # Enhanced assertion generation methods
    def _generate_unique_assertions(self, scenario: str, function_type: str) -> List[str]:
        """Generate unique assertions"""
        assertions = ["assert result is not None"]
        
        if "validation" in scenario:
            assertions.append("assert result.get('valid', False) is True")
        elif "transformation" in scenario:
            assertions.append("assert result != input_data")
        elif "calculation" in scenario:
            assertions.append("assert isinstance(result, (int, float))")
        
        return assertions
    
    def _generate_diverse_assertions(self, case: str, function_type: str) -> List[str]:
        """Generate diverse assertions"""
        assertions = ["assert result is not None"]
        
        if "valid" in case:
            assertions.append("assert result is valid")
        elif "invalid" in case:
            assertions.append("assert result indicates error")
        elif "edge" in case:
            assertions.append("assert result handles edge case correctly")
        
        return assertions
    
    def _generate_intuitive_assertions(self, scenario: str, function_type: str) -> List[str]:
        """Generate intuitive assertions"""
        assertions = ["assert result is not None"]
        
        if "user" in scenario:
            assertions.append("assert result is user_friendly")
        elif "clear" in scenario:
            assertions.append("assert result is clear and understandable")
        else:
            assertions.append("assert result is intuitive")
        
        return assertions
    
    def _generate_creative_assertions(self, approach: str, function_type: str) -> List[str]:
        """Generate creative assertions"""
        assertions = ["assert result is not None"]
        
        if approach == "innovative":
            assertions.append("assert result demonstrates innovation")
        elif approach == "experimental":
            assertions.append("assert result shows experimental behavior")
        else:
            assertions.append("assert result is creative and unique")
        
        return assertions
    
    # Quality optimization methods
    def _optimize_uniqueness(self, test_case: ImprovedTestCase, analysis: Dict[str, Any]) -> ImprovedTestCase:
        """Optimize test case for uniqueness"""
        # Add more unique characteristics
        if "unique" not in test_case.name:
            test_case.name = f"unique_{test_case.name}"
        
        # Add creative parameters
        for param_name in test_case.parameters:
            test_case.parameters[param_name] = self._generate_creative_value(param_name, "str")
        
        return test_case
    
    def _optimize_diversity(self, test_case: ImprovedTestCase, analysis: Dict[str, Any]) -> ImprovedTestCase:
        """Optimize test case for diversity"""
        # Add more diverse parameter types
        param_types = set(type(v).__name__ for v in test_case.parameters.values())
        if len(param_types) < 3:
            # Add more parameter types
            for i, param_name in enumerate(list(test_case.parameters.keys())[:2]):
                if i == 0:
                    test_case.parameters[f"{param_name}_alt"] = self._generate_edge_case_value(param_name, "str")
                else:
                    test_case.parameters[f"{param_name}_alt"] = self._generate_boundary_value(param_name, "int")
        
        return test_case
    
    def _optimize_intuition(self, test_case: ImprovedTestCase, analysis: Dict[str, Any]) -> ImprovedTestCase:
        """Optimize test case for intuition"""
        # Improve naming
        if "should" not in test_case.name.lower():
            test_case.name = f"should_{test_case.name}"
        
        # Improve description
        if "verify" not in test_case.description.lower():
            test_case.description = f"Verify {test_case.description}"
        
        return test_case
    
    def _optimize_creativity(self, test_case: ImprovedTestCase, analysis: Dict[str, Any]) -> ImprovedTestCase:
        """Optimize test case for creativity"""
        # Add creative elements
        if "creative" not in test_case.name.lower():
            test_case.name = f"creative_{test_case.name}"
        
        return test_case
    
    def _optimize_coverage(self, test_case: ImprovedTestCase, analysis: Dict[str, Any]) -> ImprovedTestCase:
        """Optimize test case for coverage"""
        # Add more assertions
        if len(test_case.assertions) < 3:
            test_case.assertions.extend([
                "assert result is not None",
                "assert isinstance(result, (dict, list, str, int, float, bool))",
                "assert 'error' not in str(result).lower()"
            ])
        
        return test_case
    
    # Parameter value generation methods
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
    
    def _generate_valid_value(self, param_name: str, param_type: str) -> Any:
        """Generate valid value"""
        return self._generate_happy_path_value(param_name, param_type)
    
    def _generate_invalid_value(self, param_name: str, param_type: str) -> Any:
        """Generate invalid value"""
        return None
    
    def _generate_edge_case_value(self, param_name: str, param_type: str) -> Any:
        """Generate edge case value"""
        if "str" in param_type.lower():
            return ""
        elif "int" in param_type.lower():
            return 0
        elif "float" in param_type.lower():
            return 0.0
        else:
            return None
    
    def _generate_boundary_value(self, param_name: str, param_type: str) -> Any:
        """Generate boundary value"""
        if "str" in param_type.lower():
            return "a"  # Single character
        elif "int" in param_type.lower():
            return 1
        elif "float" in param_type.lower():
            return 0.1
        else:
            return self._generate_happy_path_value(param_name, param_type)
    
    def _generate_creative_value(self, param_name: str, param_type: str) -> Any:
        """Generate creative value"""
        if "str" in param_type.lower():
            return f"creative_{param_name}_{random.randint(1000, 9999)}"
        elif "int" in param_type.lower():
            return random.randint(1, 100)
        elif "float" in param_type.lower():
            return round(random.uniform(0.1, 10.0), 2)
        else:
            return self._generate_happy_path_value(param_name, param_type)
    
    def _generate_diverse_value(self, param_name: str, param_type: str) -> Any:
        """Generate diverse value"""
        return self._generate_happy_path_value(param_name, param_type)
    
    def _generate_intuitive_value(self, param_name: str, param_type: str) -> Any:
        """Generate intuitive value"""
        return self._generate_happy_path_value(param_name, param_type)
    
    def _generate_user_value(self, param_name: str, param_type: str) -> Any:
        """Generate user-friendly value"""
        return self._generate_happy_path_value(param_name, param_type)
    
    def _generate_clear_value(self, param_name: str, param_type: str) -> Any:
        """Generate clear value"""
        return self._generate_happy_path_value(param_name, param_type)
    
    def _generate_innovative_value(self, param_name: str, param_type: str) -> Any:
        """Generate innovative value"""
        return self._generate_happy_path_value(param_name, param_type)
    
    def _generate_experimental_value(self, param_name: str, param_type: str) -> Any:
        """Generate experimental value"""
        return self._generate_happy_path_value(param_name, param_type)
    
    def _generate_realistic_parameters(self, analysis: Dict[str, Any], scenario: str) -> Dict[str, Any]:
        """Generate realistic parameters"""
        return self._generate_unique_parameters(analysis, scenario)
    
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
    
    def _find_dependencies(self, source: str) -> List[str]:
        """Find function dependencies"""
        return []  # Simplified for now
    
    def _find_side_effects(self, source: str) -> List[str]:
        """Find function side effects"""
        return []  # Simplified for now


def demonstrate_improved_generator():
    """Demonstrate the improved refactored test generator"""
    
    # Example function to test
    def process_advanced_data(data: dict, processing_rules: list, options: dict) -> dict:
        """
        Process advanced data with complex business rules and options.
        
        Args:
            data: Dictionary containing data to process
            processing_rules: List of processing rules to apply
            options: Dictionary with processing options
            
        Returns:
            Dictionary with processing results and metadata
            
        Raises:
            ValueError: If data is invalid or processing_rules is empty
            TypeError: If options is not a dictionary
        """
        if not isinstance(data, dict):
            raise ValueError("data must be a dictionary")
        
        if not processing_rules:
            raise ValueError("processing_rules cannot be empty")
        
        if not isinstance(options, dict):
            raise TypeError("options must be a dictionary")
        
        processed_data = data.copy()
        processing_results = []
        
        # Apply processing rules
        for rule in processing_rules:
            if rule == "normalize_keys":
                processed_data = {k.lower(): v for k, v in processed_data.items()}
                processing_results.append("Keys normalized to lowercase")
            elif rule == "validate_required_fields":
                required_fields = options.get("required_fields", [])
                missing_fields = [field for field in required_fields if field not in processed_data]
                if missing_fields:
                    processing_results.append(f"Missing required fields: {missing_fields}")
            elif rule == "add_timestamps":
                processed_data["processed_at"] = datetime.now().isoformat()
                processed_data["processing_version"] = "1.0"
                processing_results.append("Timestamps added")
        
        return {
            "processed_data": processed_data,
            "processing_results": processing_results,
            "processing_rules": processing_rules,
            "processing_options": options,
            "metadata": {
                "original_keys": list(data.keys()),
                "processed_keys": list(processed_data.keys()),
                "rules_applied": len(processing_rules),
                "processing_time": datetime.now().isoformat()
            }
        }
    
    # Generate improved tests
    generator = ImprovedRefactoredGenerator()
    test_cases = generator.generate_improved_tests(process_advanced_data, num_tests=20)
    
    print(f"Generated {len(test_cases)} improved test cases:")
    print("=" * 80)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"{i}. {test_case.name}")
        print(f"   Description: {test_case.description}")
        print(f"   Type: {test_case.test_type}")
        print(f"   Quality Scores: U={test_case.uniqueness:.2f}, D={test_case.diversity:.2f}, I={test_case.intuition:.2f}")
        print(f"   Creativity: {test_case.creativity:.2f}, Coverage: {test_case.coverage:.2f}")
        print(f"   Overall Quality: {test_case.overall_quality:.2f}")
        print(f"   Parameters: {test_case.parameters}")
        print(f"   Assertions: {test_case.assertions}")
        print()


if __name__ == "__main__":
    demonstrate_improved_generator()
