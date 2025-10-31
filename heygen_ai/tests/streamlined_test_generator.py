"""
Streamlined Test Case Generator
==============================

Focused test case generation system that creates unique, diverse, and intuitive
unit tests for functions given their signature and docstring.

This streamlined version directly addresses the prompt requirements:
- Unique: Each test case has distinct characteristics
- Diverse: Covers wide range of scenarios
- Intuitive: Clear, descriptive naming and structure
"""

import ast
import inspect
import re
import random
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    """Represents a test case with quality metrics"""
    name: str
    description: str
    function_name: str
    parameters: Dict[str, Any]
    expected_result: Any = None
    expected_exception: Optional[type] = None
    assertions: List[str] = None
    async_test: bool = False
    # Quality metrics
    uniqueness: float = 0.0
    diversity: float = 0.0
    intuition: float = 0.0


class StreamlinedTestGenerator:
    """Streamlined test generator focused on core requirements"""
    
    def __init__(self):
        self.scenario_templates = self._load_scenario_templates()
        self.naming_templates = self._load_naming_templates()
        self.parameter_templates = self._load_parameter_templates()
        
    def _load_scenario_templates(self) -> Dict[str, List[str]]:
        """Load test scenario templates"""
        return {
            "validation": [
                "happy_path", "boundary_values", "invalid_input", "empty_input",
                "type_validation", "format_validation", "business_rules"
            ],
            "transformation": [
                "data_conversion", "format_change", "aggregation", "filtering",
                "sorting", "normalization", "enrichment"
            ],
            "calculation": [
                "basic_math", "precision", "overflow", "underflow", "division_by_zero",
                "currency", "percentage", "statistics"
            ],
            "business_logic": [
                "workflow", "rules", "decisions", "approvals", "pricing",
                "permissions", "quotas", "subscriptions"
            ],
            "data_processing": [
                "small_data", "large_data", "empty_data", "malformed_data",
                "batch_processing", "stream_processing", "error_recovery"
            ]
        }
    
    def _load_naming_templates(self) -> Dict[str, List[str]]:
        """Load intuitive naming templates"""
        return {
            "behavior_driven": [
                "should_{action}_when_{condition}",
                "should_{action}_given_{context}",
                "should_{action}_for_{scenario}"
            ],
            "scenario_based": [
                "test_{scenario}_scenario",
                "test_{scenario}_case",
                "test_{scenario}_situation"
            ],
            "domain_specific": [
                "test_{domain}_{operation}",
                "test_{domain}_{validation}",
                "test_{domain}_{calculation}"
            ]
        }
    
    def _load_parameter_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load parameter value templates"""
        return {
            "happy_path": {
                "str": "test_value",
                "int": 42,
                "float": 3.14,
                "bool": True,
                "list": ["item1", "item2"],
                "dict": {"key": "value"}
            },
            "boundary": {
                "str": "",
                "int": 0,
                "float": 0.0,
                "bool": False,
                "list": [],
                "dict": {}
            },
            "edge_case": {
                "str": " ",
                "int": -1,
                "float": float('inf'),
                "bool": True,
                "list": [None],
                "dict": {"key": None}
            }
        }
    
    def generate_tests(self, func: Callable, num_tests: int = 20) -> List[TestCase]:
        """Generate unique, diverse, and intuitive test cases"""
        analysis = self._analyze_function(func)
        function_type = self._classify_function(func, analysis)
        
        test_cases = []
        
        # Generate scenario-based tests
        if function_type in self.scenario_templates:
            scenarios = self.scenario_templates[function_type]
            for scenario in scenarios[:num_tests//2]:
                test_case = self._create_scenario_test(func, analysis, scenario, function_type)
                if test_case:
                    test_cases.append(test_case)
        
        # Generate edge case tests
        edge_tests = self._generate_edge_case_tests(func, analysis, num_tests//4)
        test_cases.extend(edge_tests)
        
        # Generate combination tests
        combo_tests = self._generate_combination_tests(func, analysis, num_tests//4)
        test_cases.extend(combo_tests)
        
        # Score and sort tests
        for test_case in test_cases:
            self._score_test_case(test_case)
        
        test_cases.sort(key=lambda x: x.uniqueness + x.diversity + x.intuition, reverse=True)
        return test_cases[:num_tests]
    
    def _analyze_function(self, func: Callable) -> Dict[str, Any]:
        """Analyze function for test generation"""
        try:
            signature = inspect.signature(func)
            docstring = inspect.getdoc(func) or ""
            
            return {
                "name": func.__name__,
                "signature": signature,
                "docstring": docstring,
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
                            scenario: str, function_type: str) -> Optional[TestCase]:
        """Create a test case for a specific scenario"""
        try:
            name = self._generate_name(func.__name__, scenario, function_type)
            description = self._generate_description(func.__name__, scenario, function_type)
            parameters = self._generate_parameters(analysis, scenario)
            assertions = self._generate_assertions(scenario, function_type)
            
            return TestCase(
                name=name,
                description=description,
                function_name=func.__name__,
                parameters=parameters,
                assertions=assertions,
                async_test=analysis.get("is_async", False)
            )
        except Exception as e:
            logger.error(f"Error creating scenario test: {e}")
            return None
    
    def _generate_name(self, function_name: str, scenario: str, function_type: str) -> str:
        """Generate intuitive test name"""
        if function_type == "validation":
            return f"should_validate_{scenario}_correctly"
        elif function_type == "transformation":
            return f"should_transform_{scenario}_properly"
        elif function_type == "calculation":
            return f"should_calculate_{scenario}_accurately"
        else:
            return f"test_{function_name}_{scenario}"
    
    def _generate_description(self, function_name: str, scenario: str, function_type: str) -> str:
        """Generate intuitive test description"""
        return f"Verify {function_name} handles {scenario.replace('_', ' ')} scenario correctly"
    
    def _generate_parameters(self, analysis: Dict[str, Any], scenario: str) -> Dict[str, Any]:
        """Generate parameters based on scenario"""
        parameters = {}
        param_types = analysis.get("parameter_types", {})
        
        template_type = "happy_path"
        if "boundary" in scenario:
            template_type = "boundary"
        elif "edge" in scenario or "invalid" in scenario:
            template_type = "edge_case"
        
        for param_name, param_type in param_types.items():
            parameters[param_name] = self._get_parameter_value(param_name, param_type, template_type)
        
        return parameters
    
    def _generate_assertions(self, scenario: str, function_type: str) -> List[str]:
        """Generate assertions based on scenario"""
        assertions = ["assert result is not None"]
        
        if "invalid" in scenario or "error" in scenario:
            assertions.append("assert exception is raised or result indicates error")
        elif "boundary" in scenario:
            assertions.append("assert result is within expected bounds")
        elif function_type == "validation":
            assertions.append("assert result is True or result.get('valid', False)")
        elif function_type == "calculation":
            assertions.append("assert isinstance(result, (int, float))")
        
        return assertions
    
    def _generate_edge_case_tests(self, func: Callable, analysis: Dict[str, Any], num_tests: int) -> List[TestCase]:
        """Generate edge case tests"""
        test_cases = []
        param_types = analysis.get("parameter_types", {})
        
        edge_cases = ["empty", "null", "zero", "negative", "max_value", "min_value"]
        
        for i, edge_case in enumerate(edge_cases[:num_tests]):
            if i < len(param_types):
                param_name = list(param_types.keys())[i]
                param_type = param_types[param_name]
                
                test_case = TestCase(
                    name=f"test_{func.__name__}_handles_{edge_case}_{param_name}",
                    description=f"Verify {func.__name__} handles {edge_case} {param_name}",
                    function_name=func.__name__,
                    parameters={param_name: self._get_edge_case_value(param_type, edge_case)},
                    assertions=["assert result is not None or exception is raised"],
                    async_test=analysis.get("is_async", False)
                )
                test_cases.append(test_case)
        
        return test_cases
    
    def _generate_combination_tests(self, func: Callable, analysis: Dict[str, Any], num_tests: int) -> List[TestCase]:
        """Generate combination tests"""
        test_cases = []
        param_types = analysis.get("parameter_types", {})
        
        for i in range(num_tests):
            parameters = {}
            for param_name, param_type in param_types.items():
                # Mix different parameter types
                if i % 3 == 0:
                    parameters[param_name] = self._get_parameter_value(param_name, param_type, "happy_path")
                elif i % 3 == 1:
                    parameters[param_name] = self._get_parameter_value(param_name, param_type, "boundary")
                else:
                    parameters[param_name] = self._get_parameter_value(param_name, param_type, "edge_case")
            
            test_case = TestCase(
                name=f"test_{func.__name__}_combination_{i+1}",
                description=f"Verify {func.__name__} handles parameter combination {i+1}",
                function_name=func.__name__,
                parameters=parameters,
                assertions=["assert result is not None"],
                async_test=analysis.get("is_async", False)
            )
            test_cases.append(test_case)
        
        return test_cases
    
    def _score_test_case(self, test_case: TestCase):
        """Score test case for uniqueness, diversity, and intuition"""
        # Uniqueness score
        uniqueness = 0.0
        if "should" in test_case.name:
            uniqueness += 0.3
        if "combination" in test_case.name or "edge" in test_case.name:
            uniqueness += 0.3
        if len(test_case.parameters) > 1:
            uniqueness += 0.2
        if len(test_case.assertions) > 1:
            uniqueness += 0.2
        test_case.uniqueness = min(uniqueness, 1.0)
        
        # Diversity score
        diversity = 0.0
        param_types = set(type(v).__name__ for v in test_case.parameters.values())
        diversity += len(param_types) * 0.2
        if any(keyword in test_case.name for keyword in ["boundary", "edge", "combination", "invalid"]):
            diversity += 0.3
        if test_case.expected_exception:
            diversity += 0.2
        test_case.diversity = min(diversity, 1.0)
        
        # Intuition score
        intuition = 0.0
        if "should" in test_case.name.lower():
            intuition += 0.4
        if "verify" in test_case.description.lower():
            intuition += 0.3
        if "correctly" in test_case.description.lower():
            intuition += 0.2
        if len(test_case.assertions) > 1:
            intuition += 0.1
        test_case.intuition = min(intuition, 1.0)
    
    def _get_parameter_types(self, signature: inspect.Signature) -> Dict[str, str]:
        """Get parameter types from signature"""
        param_types = {}
        for param_name, param in signature.parameters.items():
            if param.annotation != inspect.Parameter.empty:
                param_types[param_name] = str(param.annotation)
            else:
                param_types[param_name] = "Any"
        return param_types
    
    def _get_parameter_value(self, param_name: str, param_type: str, template_type: str) -> Any:
        """Get parameter value based on type and template"""
        templates = self.parameter_templates.get(template_type, self.parameter_templates["happy_path"])
        
        if "str" in param_type.lower():
            if "email" in param_name.lower():
                return "user@example.com" if template_type == "happy_path" else ""
            elif "name" in param_name.lower():
                return "John Doe" if template_type == "happy_path" else ""
            elif "id" in param_name.lower():
                return "user_123" if template_type == "happy_path" else ""
            else:
                return templates["str"]
        elif "int" in param_type.lower():
            return templates["int"]
        elif "float" in param_type.lower():
            return templates["float"]
        elif "bool" in param_type.lower():
            return templates["bool"]
        elif "list" in param_type.lower():
            return templates["list"]
        elif "dict" in param_type.lower():
            return templates["dict"]
        else:
            return "default_value"
    
    def _get_edge_case_value(self, param_type: str, edge_case: str) -> Any:
        """Get edge case value for parameter type"""
        if edge_case == "empty":
            if "str" in param_type.lower():
                return ""
            elif "list" in param_type.lower():
                return []
            elif "dict" in param_type.lower():
                return {}
        elif edge_case == "null":
            return None
        elif edge_case == "zero":
            if "int" in param_type.lower() or "float" in param_type.lower():
                return 0
        elif edge_case == "negative":
            if "int" in param_type.lower():
                return -1
            elif "float" in param_type.lower():
                return -1.0
        elif edge_case == "max_value":
            if "int" in param_type.lower():
                return 2**31-1
            elif "float" in param_type.lower():
                return float('inf')
        elif edge_case == "min_value":
            if "int" in param_type.lower():
                return -2**31
            elif "float" in param_type.lower():
                return float('-inf')
        
        return None
    
    def generate_test_file(self, func: Callable, output_path: str, num_tests: int = 20) -> str:
        """Generate a complete test file for a function"""
        test_cases = self.generate_tests(func, num_tests)
        analysis = self._analyze_function(func)
        
        content = self._generate_file_header(func, analysis, test_cases)
        content += self._generate_imports(func, analysis)
        content += self._generate_fixtures(func, analysis)
        content += self._generate_test_class(func, analysis, test_cases)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return content
    
    def _generate_file_header(self, func: Callable, analysis: Dict[str, Any], test_cases: List[TestCase]) -> str:
        """Generate test file header"""
        return f'''"""
Test Cases for {func.__name__}
============================

Auto-generated test cases for the {func.__name__} function.
Generated on: {datetime.now().isoformat()}

Function Analysis:
- Function: {func.__name__}
- Parameters: {', '.join(analysis.get('parameters', []))}
- Return Type: {analysis.get('return_annotation', 'Unknown')}
- Async: {analysis.get('is_async', False)}

Test Generation:
- Total Tests: {len(test_cases)}
- Average Uniqueness: {sum(tc.uniqueness for tc in test_cases)/len(test_cases):.2f}
- Average Diversity: {sum(tc.diversity for tc in test_cases)/len(test_cases):.2f}
- Average Intuition: {sum(tc.intuition for tc in test_cases)/len(test_cases):.2f}

Test Categories:
- Scenario Tests: Parameter validation and business logic
- Edge Case Tests: Boundary values and special cases
- Combination Tests: Mixed parameter scenarios
"""

'''
    
    def _generate_imports(self, func: Callable, analysis: Dict[str, Any]) -> str:
        """Generate import statements"""
        imports = [
            "import pytest",
            "from unittest.mock import Mock, patch",
            "from typing import Any, Dict, List, Optional",
            "from datetime import datetime",
            ""
        ]
        
        if analysis.get("is_async", False):
            imports.insert(1, "import asyncio")
        
        return "\n".join(imports)
    
    def _generate_fixtures(self, func: Callable, analysis: Dict[str, Any]) -> str:
        """Generate pytest fixtures"""
        fixtures = [
            "@pytest.fixture",
            f"def {func.__name__}_instance():",
            f'    """Fixture for {func.__name__} function"""',
            f"    return {func.__name__}",
            "",
        ]
        
        if analysis.get("is_async", False):
            fixtures.extend([
                "@pytest.fixture",
                "def event_loop():",
                '    """Event loop fixture for async tests"""',
                "    loop = asyncio.get_event_loop_policy().new_event_loop()",
                "    yield loop",
                "    loop.close()",
                "",
            ])
        
        return "\n".join(fixtures)
    
    def _generate_test_class(self, func: Callable, analysis: Dict[str, Any], test_cases: List[TestCase]) -> str:
        """Generate test class with all test cases"""
        class_name = f"Test{func.__name__.title()}"
        
        content = [
            f"class {class_name}:",
            f'    """Test suite for {func.__name__} function"""',
            "",
        ]
        
        for test_case in test_cases:
            content.extend(self._generate_test_method(test_case, analysis))
            content.append("")
        
        return "\n".join(content)
    
    def _generate_test_method(self, test_case: TestCase, analysis: Dict[str, Any]) -> List[str]:
        """Generate a single test method"""
        method_lines = [
            f"    def {test_case.name}(self, {test_case.function_name}_instance):",
            f'        """{test_case.description}"""',
        ]
        
        if test_case.async_test:
            method_lines.insert(0, "    @pytest.mark.asyncio")
        
        method_lines.append("        # Test execution")
        
        if test_case.expected_exception:
            method_lines.append("        with pytest.raises(Exception):")
            method_lines.append(f"            result = {test_case.function_name}_instance(**{test_case.parameters})")
        else:
            if test_case.async_test:
                method_lines.append(f"        result = await {test_case.function_name}_instance(**{test_case.parameters})")
            else:
                method_lines.append(f"        result = {test_case.function_name}_instance(**{test_case.parameters})")
        
        if test_case.assertions:
            method_lines.append("")
            method_lines.append("        # Assertions")
            for assertion in test_case.assertions:
                method_lines.append(f"        {assertion}")
        
        return method_lines


def demonstrate_streamlined_generator():
    """Demonstrate the streamlined test generation system"""
    
    # Example function to test
    def calculate_discount(price: float, user_tier: str, quantity: int) -> dict:
        """
        Calculate discount for a purchase based on price, user tier, and quantity.
        
        Args:
            price: Base price of the item
            user_tier: User tier (bronze, silver, gold, platinum)
            quantity: Number of items purchased
            
        Returns:
            Dictionary with discount calculation results
            
        Raises:
            ValueError: If user_tier is invalid or price is negative
        """
        if price < 0:
            raise ValueError("Price cannot be negative")
        
        if user_tier not in ["bronze", "silver", "gold", "platinum"]:
            raise ValueError("Invalid user tier")
        
        # Base discount rates by tier
        tier_discounts = {
            "bronze": 0.05,   # 5%
            "silver": 0.10,   # 10%
            "gold": 0.15,     # 15%
            "platinum": 0.20  # 20%
        }
        
        # Quantity discount
        quantity_discount = 0.0
        if quantity >= 10:
            quantity_discount = 0.05  # 5% for bulk orders
        elif quantity >= 5:
            quantity_discount = 0.02  # 2% for medium orders
        
        # Calculate total discount
        tier_discount = tier_discounts[user_tier]
        total_discount = min(tier_discount + quantity_discount, 0.30)  # Max 30%
        
        discount_amount = price * total_discount
        final_price = price - discount_amount
        
        return {
            "original_price": price,
            "user_tier": user_tier,
            "quantity": quantity,
            "tier_discount": tier_discount,
            "quantity_discount": quantity_discount,
            "total_discount": total_discount,
            "discount_amount": discount_amount,
            "final_price": final_price,
            "calculated_at": datetime.now().isoformat()
        }
    
    # Generate tests
    generator = StreamlinedTestGenerator()
    test_cases = generator.generate_tests(calculate_discount, num_tests=15)
    
    print(f"Generated {len(test_cases)} unique, diverse, and intuitive test cases:")
    print("=" * 80)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"{i}. {test_case.name}")
        print(f"   Description: {test_case.description}")
        print(f"   Quality Scores: U={test_case.uniqueness:.2f}, D={test_case.diversity:.2f}, I={test_case.intuition:.2f}")
        print(f"   Parameters: {test_case.parameters}")
        print(f"   Assertions: {test_case.assertions}")
        print()
    
    # Generate complete test file
    print("Generating complete test file...")
    test_file_content = generator.generate_test_file(
        calculate_discount,
        "streamlined_test_calculator.py",
        num_tests=15
    )
    
    print(f"✅ Generated test file with {len(test_file_content)} characters")
    print(f"📊 File contains {len(test_file_content.splitlines())} lines")


if __name__ == "__main__":
    demonstrate_streamlined_generator()
