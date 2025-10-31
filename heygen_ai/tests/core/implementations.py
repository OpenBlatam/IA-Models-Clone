"""
Concrete Implementations for Test Generation System
==================================================

This module provides concrete implementations of the abstract base classes
defined in the base architecture.
"""

import asyncio
import re
import ast
import inspect
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
import logging

from .base_architecture import (
    BaseTestGenerator, BaseTestPattern, BaseParameterGenerator,
    BaseTestValidator, BaseTestOptimizer, TestCase, TestGenerationConfig,
    TestComplexity, TestCategory, TestType, TestPriority, GenerationMetrics
)

logger = logging.getLogger(__name__)


class EnhancedTestGenerator(BaseTestGenerator):
    """Enhanced test generator with advanced capabilities"""
    
    def __init__(self, config: TestGenerationConfig):
        super().__init__(config)
        self.patterns = self._load_patterns()
        self.parameter_generators = self._load_parameter_generators()
        self.validators = self._load_validators()
        self.optimizers = self._load_optimizers()
    
    def _load_patterns(self) -> Dict[str, BaseTestPattern]:
        """Load test patterns"""
        return {
            "basic": BasicTestPattern(),
            "edge_case": EdgeCaseTestPattern(),
            "performance": PerformanceTestPattern(),
            "security": SecurityTestPattern(),
            "integration": IntegrationTestPattern()
        }
    
    def _load_parameter_generators(self) -> Dict[str, BaseParameterGenerator]:
        """Load parameter generators"""
        return {
            "string": StringParameterGenerator(),
            "integer": IntegerParameterGenerator(),
            "float": FloatParameterGenerator(),
            "boolean": BooleanParameterGenerator(),
            "list": ListParameterGenerator(),
            "dict": DictParameterGenerator()
        }
    
    def _load_validators(self) -> Dict[str, BaseTestValidator]:
        """Load validators"""
        return {
            "syntax": SyntaxValidator(),
            "coverage": CoverageValidator(),
            "quality": QualityValidator()
        }
    
    def _load_optimizers(self) -> Dict[str, BaseTestOptimizer]:
        """Load optimizers"""
        return {
            "deduplication": DeduplicationOptimizer(),
            "performance": PerformanceOptimizer(),
            "coverage": CoverageOptimizer()
        }
    
    async def generate_tests(
        self, 
        function_signature: str, 
        docstring: str, 
        config: TestGenerationConfig
    ) -> List[TestCase]:
        """Generate test cases for a function"""
        
        try:
            # Parse function signature
            func_info = self._parse_function_signature(function_signature)
            
            # Generate test cases based on patterns
            test_cases = []
            
            for pattern_name, pattern in self.patterns.items():
                if self._should_use_pattern(pattern_name, config):
                    pattern_tests = await pattern.generate_tests(func_info, docstring, config)
                    test_cases.extend(pattern_tests)
            
            # Validate and optimize tests
            test_cases = self._validate_tests(test_cases)
            test_cases = self._optimize_tests(test_cases)
            
            # Update metrics
            self.metrics.total_tests_generated += len(test_cases)
            self.metrics.successful_generations += 1
            
            return test_cases
            
        except Exception as e:
            logger.error(f"Test generation failed: {e}")
            self.metrics.failed_generations += 1
            return []
    
    def _parse_function_signature(self, signature: str) -> Dict[str, Any]:
        """Parse function signature to extract information"""
        try:
            # Simple regex-based parsing
            func_match = re.match(r'def\s+(\w+)\s*\((.*?)\)\s*->\s*(.+)?', signature)
            if not func_match:
                raise ValueError("Invalid function signature")
            
            func_name = func_match.group(1)
            params_str = func_match.group(2)
            return_type = func_match.group(3)
            
            # Parse parameters
            parameters = []
            if params_str.strip():
                for param in params_str.split(','):
                    param = param.strip()
                    if ':' in param:
                        name, param_type = param.split(':', 1)
                        parameters.append({
                            'name': name.strip(),
                            'type': param_type.strip(),
                            'default': None
                        })
                    else:
                        parameters.append({
                            'name': param.strip(),
                            'type': 'Any',
                            'default': None
                        })
            
            return {
                'name': func_name,
                'parameters': parameters,
                'return_type': return_type.strip() if return_type else 'Any'
            }
            
        except Exception as e:
            logger.error(f"Failed to parse function signature: {e}")
            return {'name': 'unknown', 'parameters': [], 'return_type': 'Any'}
    
    def _should_use_pattern(self, pattern_name: str, config: TestGenerationConfig) -> bool:
        """Determine if a pattern should be used based on config"""
        if pattern_name == "edge_case" and not config.include_edge_cases:
            return False
        if pattern_name == "performance" and not config.include_performance_tests:
            return False
        if pattern_name == "security" and not config.include_security_tests:
            return False
        return True
    
    def _validate_tests(self, test_cases: List[TestCase]) -> List[TestCase]:
        """Validate generated test cases"""
        valid_tests = []
        
        for test_case in test_cases:
            is_valid = True
            for validator in self.validators.values():
                if not validator.validate_test_case(test_case):
                    is_valid = False
                    break
            
            if is_valid:
                valid_tests.append(test_case)
            else:
                self.metrics.validation_failures += 1
        
        return valid_tests
    
    def _optimize_tests(self, test_cases: List[TestCase]) -> List[TestCase]:
        """Optimize test cases"""
        optimized_tests = test_cases
        
        for optimizer in self.optimizers.values():
            optimized_tests = optimizer.optimize_tests(optimized_tests)
        
        return optimized_tests


class BasicTestPattern(BaseTestPattern):
    """Basic test pattern for standard test cases"""
    
    def __init__(self):
        super().__init__()
        self.pattern_type = "basic"
    
    async def generate_tests(
        self, 
        function_info: Dict[str, Any], 
        docstring: str, 
        config: TestGenerationConfig
    ) -> List[TestCase]:
        """Generate basic test cases"""
        
        test_cases = []
        func_name = function_info['name']
        
        # Generate happy path test
        test_cases.append(TestCase(
            name=f"test_{func_name}_basic",
            description=f"Basic test for {func_name}",
            test_code=f"result = {func_name}()\nassert result is not None",
            setup_code="",
            teardown_code="",
            category=TestCategory.FUNCTIONAL,
            priority=TestPriority.HIGH,
            test_type=TestType.UNIT
        ))
        
        # Generate parameter tests
        for param in function_info['parameters']:
            test_cases.append(TestCase(
                name=f"test_{func_name}_with_{param['name']}",
                description=f"Test {func_name} with {param['name']} parameter",
                test_code=f"result = {func_name}({param['name']}=test_value)\nassert result is not None",
                setup_code=f"test_value = self._generate_{param['type']}_value()",
                teardown_code="",
                category=TestCategory.FUNCTIONAL,
                priority=TestPriority.MEDIUM,
                test_type=TestType.UNIT
            ))
        
        return test_cases


class EdgeCaseTestPattern(BaseTestPattern):
    """Edge case test pattern"""
    
    def __init__(self):
        super().__init__()
        self.pattern_type = "edge_case"
    
    async def generate_tests(
        self, 
        function_info: Dict[str, Any], 
        docstring: str, 
        config: TestGenerationConfig
    ) -> List[TestCase]:
        """Generate edge case test cases"""
        
        test_cases = []
        func_name = function_info['name']
        
        # Generate edge case tests
        edge_cases = [
            ("empty", "Empty input"),
            ("none", "None input"),
            ("zero", "Zero value"),
            ("negative", "Negative value"),
            ("max_value", "Maximum value"),
            ("min_value", "Minimum value")
        ]
        
        for case_name, description in edge_cases:
            test_cases.append(TestCase(
                name=f"test_{func_name}_{case_name}",
                description=f"Test {func_name} with {description}",
                test_code=f"result = {func_name}({case_name}_input)\n# Add appropriate assertions",
                setup_code=f"{case_name}_input = self._get_{case_name}_input()",
                teardown_code="",
                category=TestCategory.EDGE_CASE,
                priority=TestPriority.MEDIUM,
                test_type=TestType.UNIT
            ))
        
        return test_cases


class PerformanceTestPattern(BaseTestPattern):
    """Performance test pattern"""
    
    def __init__(self):
        super().__init__()
        self.pattern_type = "performance"
    
    async def generate_tests(
        self, 
        function_info: Dict[str, Any], 
        docstring: str, 
        config: TestGenerationConfig
    ) -> List[TestCase]:
        """Generate performance test cases"""
        
        test_cases = []
        func_name = function_info['name']
        
        # Generate performance tests
        test_cases.append(TestCase(
            name=f"test_{func_name}_performance",
            description=f"Performance test for {func_name}",
            test_code="""import time
start_time = time.time()
result = {func_name}()
end_time = time.time()
execution_time = end_time - start_time
assert execution_time < 1.0  # 1 second threshold""".format(func_name=func_name),
            setup_code="",
            teardown_code="",
            category=TestCategory.PERFORMANCE,
            priority=TestPriority.LOW,
            test_type=TestType.PERFORMANCE
        ))
        
        return test_cases


class SecurityTestPattern(BaseTestPattern):
    """Security test pattern"""
    
    def __init__(self):
        super().__init__()
        self.pattern_type = "security"
    
    async def generate_tests(
        self, 
        function_info: Dict[str, Any], 
        docstring: str, 
        config: TestGenerationConfig
    ) -> List[TestCase]:
        """Generate security test cases"""
        
        test_cases = []
        func_name = function_info['name']
        
        # Generate security tests
        security_tests = [
            ("sql_injection", "SQL injection attempt"),
            ("xss", "XSS attack attempt"),
            ("path_traversal", "Path traversal attempt"),
            ("buffer_overflow", "Buffer overflow attempt")
        ]
        
        for test_name, description in security_tests:
            test_cases.append(TestCase(
                name=f"test_{func_name}_security_{test_name}",
                description=f"Security test: {description}",
                test_code=f"# Test {description}\nmalicious_input = self._get_{test_name}_input()\n# Verify function handles malicious input safely",
                setup_code=f"malicious_input = self._get_{test_name}_input()",
                teardown_code="",
                category=TestCategory.SECURITY,
                priority=TestPriority.HIGH,
                test_type=TestType.SECURITY
            ))
        
        return test_cases


class IntegrationTestPattern(BaseTestPattern):
    """Integration test pattern"""
    
    def __init__(self):
        super().__init__()
        self.pattern_type = "integration"
    
    async def generate_tests(
        self, 
        function_info: Dict[str, Any], 
        docstring: str, 
        config: TestGenerationConfig
    ) -> List[TestCase]:
        """Generate integration test cases"""
        
        test_cases = []
        func_name = function_info['name']
        
        # Generate integration tests
        test_cases.append(TestCase(
            name=f"test_{func_name}_integration",
            description=f"Integration test for {func_name}",
            test_code=f"# Test integration with other components\nresult = {func_name}()\n# Verify integration behavior",
            setup_code="self._setup_integration_environment()",
            teardown_code="self._cleanup_integration_environment()",
            category=TestCategory.INTEGRATION,
            priority=TestPriority.MEDIUM,
            test_type=TestType.INTEGRATION
        ))
        
        return test_cases


class StringParameterGenerator(BaseParameterGenerator):
    """String parameter generator"""
    
    def __init__(self):
        super().__init__()
        self.param_type = "string"
    
    def generate_parameters(self, constraints: Dict[str, Any]) -> List[str]:
        """Generate string parameters"""
        params = []
        
        # Basic strings
        params.extend([
            "hello",
            "world",
            "test",
            "example",
            ""
        ])
        
        # Edge cases
        if constraints.get("include_edge_cases", True):
            params.extend([
                "a" * 1000,  # Long string
                "🚀🌟💫",  # Unicode
                "  spaced  ",  # Whitespace
                "newline\nstring"  # Newlines
            ])
        
        return params


class IntegerParameterGenerator(BaseParameterGenerator):
    """Integer parameter generator"""
    
    def __init__(self):
        super().__init__()
        self.param_type = "integer"
    
    def generate_parameters(self, constraints: Dict[str, Any]) -> List[int]:
        """Generate integer parameters"""
        params = []
        
        # Basic integers
        params.extend([0, 1, -1, 10, -10, 100, -100])
        
        # Edge cases
        if constraints.get("include_edge_cases", True):
            params.extend([
                2**31 - 1,  # Max int32
                -2**31,     # Min int32
                2**63 - 1,  # Max int64
                -2**63      # Min int64
            ])
        
        return params


class FloatParameterGenerator(BaseParameterGenerator):
    """Float parameter generator"""
    
    def __init__(self):
        super().__init__()
        self.param_type = "float"
    
    def generate_parameters(self, constraints: Dict[str, Any]) -> List[float]:
        """Generate float parameters"""
        params = []
        
        # Basic floats
        params.extend([0.0, 1.0, -1.0, 3.14, -3.14, 0.001, -0.001])
        
        # Edge cases
        if constraints.get("include_edge_cases", True):
            params.extend([
                float('inf'),
                float('-inf'),
                float('nan')
            ])
        
        return params


class BooleanParameterGenerator(BaseParameterGenerator):
    """Boolean parameter generator"""
    
    def __init__(self):
        super().__init__()
        self.param_type = "boolean"
    
    def generate_parameters(self, constraints: Dict[str, Any]) -> List[bool]:
        """Generate boolean parameters"""
        return [True, False]


class ListParameterGenerator(BaseParameterGenerator):
    """List parameter generator"""
    
    def __init__(self):
        super().__init__()
        self.param_type = "list"
    
    def generate_parameters(self, constraints: Dict[str, Any]) -> List[List[Any]]:
        """Generate list parameters"""
        params = []
        
        # Basic lists
        params.extend([
            [],
            [1, 2, 3],
            ["a", "b", "c"],
            [1, "mixed", True]
        ])
        
        # Edge cases
        if constraints.get("include_edge_cases", True):
            params.extend([
                [None],
                [[]],  # Nested empty list
                [[1, 2], [3, 4]]  # Nested lists
            ])
        
        return params


class DictParameterGenerator(BaseParameterGenerator):
    """Dictionary parameter generator"""
    
    def __init__(self):
        super().__init__()
        self.param_type = "dict"
    
    def generate_parameters(self, constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate dictionary parameters"""
        params = []
        
        # Basic dictionaries
        params.extend([
            {},
            {"key": "value"},
            {"a": 1, "b": 2, "c": 3}
        ])
        
        # Edge cases
        if constraints.get("include_edge_cases", True):
            params.extend([
                {"": "empty_key"},
                {"key": None},
                {"nested": {"inner": "value"}}
            ])
        
        return params


class SyntaxValidator(BaseTestValidator):
    """Syntax validator for test cases"""
    
    def __init__(self):
        super().__init__()
        self.validator_type = "syntax"
    
    def validate_test_case(self, test_case: TestCase) -> bool:
        """Validate test case syntax"""
        try:
            # Check if test code is valid Python
            ast.parse(test_case.test_code)
            return True
        except SyntaxError:
            return False
    
    def get_validation_errors(self, test_case: TestCase) -> List[str]:
        """Get validation errors for a test case"""
        errors = []
        try:
            ast.parse(test_case.test_code)
        except SyntaxError as e:
            errors.append(f"Syntax error: {e}")
        return errors


class CoverageValidator(BaseTestValidator):
    """Coverage validator for test cases"""
    
    def __init__(self):
        super().__init__()
        self.validator_type = "coverage"
    
    def validate_test_case(self, test_case: TestCase) -> bool:
        """Validate test case coverage"""
        # Simple coverage check - ensure test has assertions
        return "assert" in test_case.test_code.lower()
    
    def get_validation_errors(self, test_case: TestCase) -> List[str]:
        """Get validation errors for a test case"""
        errors = []
        if "assert" not in test_case.test_code.lower():
            errors.append("Test case lacks assertions")
        return errors


class QualityValidator(BaseTestValidator):
    """Quality validator for test cases"""
    
    def __init__(self):
        super().__init__()
        self.validator_type = "quality"
    
    def validate_test_case(self, test_case: TestCase) -> bool:
        """Validate test case quality"""
        # Check for basic quality indicators
        has_description = bool(test_case.description.strip())
        has_meaningful_name = len(test_case.name) > 10
        has_test_code = bool(test_case.test_code.strip())
        
        return has_description and has_meaningful_name and has_test_code
    
    def get_validation_errors(self, test_case: TestCase) -> List[str]:
        """Get validation errors for a test case"""
        errors = []
        
        if not test_case.description.strip():
            errors.append("Missing description")
        
        if len(test_case.name) <= 10:
            errors.append("Test name too short")
        
        if not test_case.test_code.strip():
            errors.append("Missing test code")
        
        return errors


class DeduplicationOptimizer(BaseTestOptimizer):
    """Deduplication optimizer for test cases"""
    
    def __init__(self):
        super().__init__()
        self.optimizer_type = "deduplication"
    
    def optimize_tests(self, test_cases: List[TestCase]) -> List[TestCase]:
        """Remove duplicate test cases"""
        seen = set()
        unique_tests = []
        
        for test_case in test_cases:
            # Create a hash of the test case content
            test_hash = hash((
                test_case.name,
                test_case.test_code,
                test_case.category,
                test_case.test_type
            ))
            
            if test_hash not in seen:
                seen.add(test_hash)
                unique_tests.append(test_case)
        
        return unique_tests


class PerformanceOptimizer(BaseTestOptimizer):
    """Performance optimizer for test cases"""
    
    def __init__(self):
        super().__init__()
        self.optimizer_type = "performance"
    
    def optimize_tests(self, test_cases: List[TestCase]) -> List[TestCase]:
        """Optimize test cases for performance"""
        # Sort by priority (HIGH first)
        priority_order = {TestPriority.HIGH: 0, TestPriority.MEDIUM: 1, TestPriority.LOW: 2}
        
        return sorted(
            test_cases,
            key=lambda tc: priority_order.get(tc.priority, 3)
        )


class CoverageOptimizer(BaseTestOptimizer):
    """Coverage optimizer for test cases"""
    
    def __init__(self):
        super().__init__()
        self.optimizer_type = "coverage"
    
    def optimize_tests(self, test_cases: List[TestCase]) -> List[TestCase]:
        """Optimize test cases for coverage"""
        # Group by category and ensure we have tests for each category
        category_tests = {}
        
        for test_case in test_cases:
            category = test_case.category
            if category not in category_tests:
                category_tests[category] = []
            category_tests[category].append(test_case)
        
        # Ensure we have at least one test per category
        optimized_tests = []
        for category, tests in category_tests.items():
            # Take the highest priority test from each category
            best_test = max(tests, key=lambda tc: tc.priority.value)
            optimized_tests.append(best_test)
        
        return optimized_tests









