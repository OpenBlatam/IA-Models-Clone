"""
Test Case Generator for HeyGen AI
=================================

An AI-powered test case generation system that creates unique, diverse,
and intuitive unit tests for functions given their signature and docstring.

This module provides:
- Automated test case generation
- Test pattern recognition
- Edge case identification
- Parameter validation testing
- Mock data generation
- Test coverage analysis
"""

import ast
import inspect
import re
import types
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import random
import string
from datetime import datetime, timedelta
import pytest
from unittest.mock import Mock, patch, MagicMock
import asyncio
import logging

logger = logging.getLogger(__name__)


class TypeEnum(Enum):
    """Types of tests that can be generated"""
    UNIT = "unit"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"
    EDGE_CASE = "edge_case"
    ERROR_HANDLING = "error_handling"
    VALIDATION = "validation"
    MOCK = "mock"
    ASYNC = "async"


class ComplexityEnum(Enum):
    """Complexity levels for test generation"""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    ENTERPRISE = "enterprise"


@dataclass
class CaseData:
    """Represents a generated test case"""
    name: str
    description: str
    test_type: TypeEnum
    complexity: ComplexityEnum
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


@dataclass
class FunctionAnalysis:
    """Analysis of a function for test generation"""
    name: str
    signature: inspect.Signature
    docstring: str
    source_code: str
    parameters: List[str]
    return_annotation: Any
    is_async: bool
    is_generator: bool
    is_class_method: bool
    is_static_method: bool
    decorators: List[str]
    complexity_score: int
    dependencies: List[str]
    side_effects: List[str]


class CaseGeneratorClass:
    """Main test case generator class"""
    
    def __init__(self):
        self.test_patterns = self._load_test_patterns()
        self.mock_generators = self._setup_mock_generators()
        self.edge_case_generators = self._setup_edge_case_generators()
        self.validation_patterns = self._setup_validation_patterns()
        
    def _load_test_patterns(self) -> Dict[str, List[str]]:
        """Load common test patterns for different function types"""
        return {
            "validation": [
                "test_{function}_valid_input",
                "test_{function}_invalid_input",
                "test_{function}_empty_input",
                "test_{function}_null_input",
                "test_{function}_boundary_values"
            ],
            "async": [
                "test_{function}_async_execution",
                "test_{function}_async_timeout",
                "test_{function}_async_cancellation",
                "test_{function}_async_error_handling"
            ],
            "error_handling": [
                "test_{function}_raises_exception",
                "test_{function}_handles_error_gracefully",
                "test_{function}_error_message_format",
                "test_{function}_error_recovery"
            ],
            "edge_cases": [
                "test_{function}_maximum_values",
                "test_{function}_minimum_values",
                "test_{function}_zero_values",
                "test_{function}_negative_values",
                "test_{function}_special_characters"
            ],
            "integration": [
                "test_{function}_with_real_dependencies",
                "test_{function}_end_to_end_workflow",
                "test_{function}_cross_component_interaction"
            ],
            "performance": [
                "test_{function}_performance_benchmark",
                "test_{function}_memory_usage",
                "test_{function}_concurrent_execution"
            ]
        }
    
    def _setup_mock_generators(self) -> Dict[str, Callable]:
        """Setup mock data generators for different types"""
        return {
            "string": lambda: self._generate_random_string(),
            "int": lambda: random.randint(1, 1000),
            "float": lambda: round(random.uniform(0.1, 100.0), 2),
            "bool": lambda: random.choice([True, False]),
            "list": lambda: [self._generate_random_string() for _ in range(random.randint(1, 5))],
            "dict": lambda: {self._generate_random_string(): self._generate_random_string() for _ in range(3)},
            "datetime": lambda: datetime.now() + timedelta(days=random.randint(-30, 30)),
            "uuid": lambda: str(uuid4()),
            "email": lambda: f"{self._generate_random_string()}@example.com",
            "url": lambda: f"https://example.com/{self._generate_random_string()}",
            "json": lambda: {"key": "value", "number": random.randint(1, 100)},
            "file_path": lambda: f"/tmp/{self._generate_random_string()}.txt",
            "user_id": lambda: f"user_{random.randint(1000, 9999)}",
            "video_id": lambda: f"video_{random.randint(1000, 9999)}",
            "avatar_id": lambda: f"avatar_{random.randint(1000, 9999)}",
            "voice_id": lambda: f"voice_{random.randint(1000, 9999)}"
        }
    
    def _setup_edge_case_generators(self) -> Dict[str, List[Any]]:
        """Setup edge case values for different types"""
        return {
            "string": ["", " ", "a" * 1000, "special!@#$%^&*()", "unicode_测试", "null", "undefined"],
            "int": [0, -1, 1, 2**31-1, -2**31, 999999999],
            "float": [0.0, -0.0, float('inf'), float('-inf'), float('nan'), 0.000001, 999999.999999],
            "bool": [True, False],
            "list": [[], [None], [1, 2, 3], ["a", "b", "c"], [1, "mixed", True]],
            "dict": [{}, {"key": None}, {"nested": {"deep": "value"}}],
            "none": [None],
            "empty": ["", [], {}, None]
        }
    
    def _setup_validation_patterns(self) -> Dict[str, List[str]]:
        """Setup validation patterns for different parameter types"""
        return {
            "email": ["valid@example.com", "invalid-email", "@example.com", "test@"],
            "url": ["https://example.com", "invalid-url", "ftp://test.com", "not-a-url"],
            "uuid": ["550e8400-e29b-41d4-a716-446655440000", "invalid-uuid", "not-a-uuid"],
            "positive_int": [1, 100, 0, -1, "not-a-number"],
            "non_empty_string": ["valid", "", "   ", None],
            "valid_json": ['{"key": "value"}', "invalid-json", "{incomplete", "not-json"]
        }
    
    def _generate_random_string(self, length: int = 10) -> str:
        """Generate a random string for testing"""
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    
    def analyze_function(self, func: Callable) -> FunctionAnalysis:
        """Analyze a function to understand its structure and requirements"""
        try:
            signature = inspect.signature(func)
            source = inspect.getsource(func)
            docstring = inspect.getdoc(func) or ""
            
            # Parse AST to get more details
            tree = ast.parse(source)
            func_node = tree.body[0] if tree.body else None
            
            # Analyze parameters
            parameters = list(signature.parameters.keys())
            
            # Check if async
            is_async = inspect.iscoroutinefunction(func)
            
            # Check if generator
            is_generator = inspect.isgeneratorfunction(func)
            
            # Check if class method
            is_class_method = inspect.ismethod(func)
            is_static_method = inspect.isfunction(func) and not is_class_method
            
            # Get decorators
            decorators = []
            if func_node and hasattr(func_node, 'decorator_list'):
                for decorator in func_node.decorator_list:
                    if isinstance(decorator, ast.Name):
                        decorators.append(decorator.id)
                    elif isinstance(decorator, ast.Attribute):
                        decorators.append(f"{decorator.attr}")
            
            # Calculate complexity score
            complexity_score = self._calculate_complexity(func, source)
            
            # Find dependencies
            dependencies = self._find_dependencies(func)
            
            # Find side effects
            side_effects = self._find_side_effects(func)
            
            return FunctionAnalysis(
                name=func.__name__,
                signature=signature,
                docstring=docstring,
                source_code=source,
                parameters=parameters,
                return_annotation=signature.return_annotation,
                is_async=is_async,
                is_generator=is_generator,
                is_class_method=is_class_method,
                is_static_method=is_static_method,
                decorators=decorators,
                complexity_score=complexity_score,
                dependencies=dependencies,
                side_effects=side_effects
            )
            
        except Exception as e:
            logger.error(f"Error analyzing function {func.__name__}: {e}")
            raise
    
    def _calculate_complexity(self, func: Callable, source: str) -> int:
        """Calculate cyclomatic complexity of a function"""
        try:
            tree = ast.parse(source)
            complexity = 1  # Base complexity
            
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
    
    def _find_dependencies(self, func: Callable) -> List[str]:
        """Find external dependencies of a function"""
        dependencies = []
        try:
            source = inspect.getsource(func)
            tree = ast.parse(source)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        dependencies.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        dependencies.append(node.module)
        except:
            pass
        
        return dependencies
    
    def _find_side_effects(self, func: Callable) -> List[str]:
        """Find potential side effects of a function"""
        side_effects = []
        try:
            source = inspect.getsource(func)
            
            # Look for common side effect patterns
            side_effect_patterns = [
                r'\.write\(',
                r'\.append\(',
                r'\.update\(',
                r'\.remove\(',
                r'\.delete\(',
                r'\.save\(',
                r'\.create\(',
                r'\.insert\(',
                r'print\(',
                r'logging\.',
                r'logger\.',
                r'open\(',
                r'requests\.',
                r'http\.',
                r'database\.',
                r'db\.'
            ]
            
            for pattern in side_effect_patterns:
                if re.search(pattern, source):
                    side_effects.append(pattern)
        except:
            pass
        
        return side_effects
    
    def generate_test_cases(self, func: Callable, num_cases: int = 10) -> List[CaseData]:
        """Generate comprehensive test cases for a function"""
        analysis = self.analyze_function(func)
        test_cases = []
        
        # Generate different types of test cases
        test_cases.extend(self._generate_validation_tests(analysis))
        test_cases.extend(self._generate_edge_case_tests(analysis))
        test_cases.extend(self._generate_error_handling_tests(analysis))
        test_cases.extend(self._generate_integration_tests(analysis))
        test_cases.extend(self._generate_performance_tests(analysis))
        
        # Limit to requested number
        return test_cases[:num_cases]
    
    def _generate_validation_tests(self, analysis: FunctionAnalysis) -> List[CaseData]:
        """Generate validation test cases"""
        test_cases = []
        
        for param_name, param in analysis.signature.parameters.items():
            if param.annotation != inspect.Parameter.empty:
                param_type = self._get_type_name(param.annotation)
                
                # Valid input test
                test_cases.append(CaseData(
                    name=f"test_{analysis.name}_valid_{param_name}",
                    description=f"Test {analysis.name} with valid {param_name}",
                    test_type=TestType.VALIDATION,
                    complexity=TestComplexity.SIMPLE,
                    function_name=analysis.name,
                    parameters={param_name: self._generate_valid_value(param_type)},
                    expected_result=None,
                    assertions=[f"assert result is not None", f"assert isinstance(result, {param_type})"]
                ))
                
                # Invalid input test
                test_cases.append(CaseData(
                    name=f"test_{analysis.name}_invalid_{param_name}",
                    description=f"Test {analysis.name} with invalid {param_name}",
                    test_type=TestType.ERROR_HANDLING,
                    complexity=TestComplexity.SIMPLE,
                    function_name=analysis.name,
                    parameters={param_name: self._generate_invalid_value(param_type)},
                    expected_exception=ValueError,
                    assertions=["assert exception is raised"]
                ))
        
        return test_cases
    
    def _generate_edge_case_tests(self, analysis: FunctionAnalysis) -> List[CaseData]:
        """Generate edge case test cases"""
        test_cases = []
        
        for param_name, param in analysis.signature.parameters.items():
            if param.annotation != inspect.Parameter.empty:
                param_type = self._get_type_name(param.annotation)
                edge_cases = self.edge_case_generators.get(param_type, [])
                
                for i, edge_case in enumerate(edge_cases[:3]):  # Limit to 3 edge cases per parameter
                    test_cases.append(CaseData(
                        name=f"test_{analysis.name}_edge_case_{param_name}_{i}",
                        description=f"Test {analysis.name} with edge case {param_name}: {edge_case}",
                        test_type=TestType.EDGE_CASE,
                        complexity=TestComplexity.MEDIUM,
                        function_name=analysis.name,
                        parameters={param_name: edge_case},
                        expected_result=None,
                        assertions=[f"assert result is not None or exception is raised"]
                    ))
        
        return test_cases
    
    def _generate_error_handling_tests(self, analysis: FunctionAnalysis) -> List[CaseData]:
        """Generate error handling test cases"""
        test_cases = []
        
        # Test with None values
        test_cases.append(CaseData(
            name=f"test_{analysis.name}_none_parameters",
            description=f"Test {analysis.name} with None parameters",
            test_type=TestType.ERROR_HANDLING,
            complexity=TestComplexity.SIMPLE,
            function_name=analysis.name,
            parameters={param: None for param in analysis.parameters},
            expected_exception=TypeError,
            assertions=["assert TypeError is raised"]
        ))
        
        # Test with wrong types
        for param_name in analysis.parameters:
            test_cases.append(CaseData(
                name=f"test_{analysis.name}_wrong_type_{param_name}",
                description=f"Test {analysis.name} with wrong type for {param_name}",
                test_type=TestType.ERROR_HANDLING,
                complexity=TestComplexity.SIMPLE,
                function_name=analysis.name,
                parameters={param_name: "wrong_type"},
                expected_exception=TypeError,
                assertions=["assert TypeError is raised"]
            ))
        
        return test_cases
    
    def _generate_integration_tests(self, analysis: FunctionAnalysis) -> List[CaseData]:
        """Generate integration test cases"""
        test_cases = []
        
        if analysis.dependencies:
            test_cases.append(CaseData(
                name=f"test_{analysis.name}_integration",
                description=f"Test {analysis.name} with real dependencies",
                test_type=TestType.INTEGRATION,
                complexity=TestComplexity.COMPLEX,
                function_name=analysis.name,
                parameters=self._generate_integration_parameters(analysis),
                expected_result=None,
                assertions=["assert result is not None", "assert no exceptions raised"],
                setup_code=self._generate_integration_setup(analysis)
            ))
        
        return test_cases
    
    def _generate_performance_tests(self, analysis: FunctionAnalysis) -> List[CaseData]:
        """Generate performance test cases"""
        test_cases = []
        
        if analysis.complexity_score > 3:  # Only for complex functions
            test_cases.append(CaseData(
                name=f"test_{analysis.name}_performance",
                description=f"Test {analysis.name} performance",
                test_type=TestType.PERFORMANCE,
                complexity=TestComplexity.COMPLEX,
                function_name=analysis.name,
                parameters=self._generate_performance_parameters(analysis),
                expected_result=None,
                assertions=["assert execution_time < threshold"],
                setup_code="import time; start_time = time.time()"
            ))
        
        return test_cases
    
    def _get_type_name(self, annotation: Any) -> str:
        """Get string representation of type annotation"""
        if hasattr(annotation, '__name__'):
            return annotation.__name__
        elif hasattr(annotation, '__origin__'):
            return str(annotation.__origin__)
        else:
            return str(annotation)
    
    def _generate_valid_value(self, type_name: str) -> Any:
        """Generate a valid value for a given type"""
        generator = self.mock_generators.get(type_name.lower())
        if generator:
            return generator()
        
        # Fallback generators
        if 'str' in type_name.lower():
            return self.mock_generators['string']()
        elif 'int' in type_name.lower():
            return self.mock_generators['int']()
        elif 'float' in type_name.lower():
            return self.mock_generators['float']()
        elif 'bool' in type_name.lower():
            return self.mock_generators['bool']()
        elif 'list' in type_name.lower():
            return self.mock_generators['list']()
        elif 'dict' in type_name.lower():
            return self.mock_generators['dict']()
        else:
            return None
    
    def _generate_invalid_value(self, type_name: str) -> Any:
        """Generate an invalid value for a given type"""
        if 'str' in type_name.lower():
            return 123  # Wrong type
        elif 'int' in type_name.lower():
            return "not_a_number"
        elif 'float' in type_name.lower():
            return "not_a_float"
        elif 'bool' in type_name.lower():
            return "not_a_bool"
        else:
            return "invalid_value"
    
    def _generate_integration_parameters(self, analysis: FunctionAnalysis) -> Dict[str, Any]:
        """Generate parameters for integration tests"""
        params = {}
        for param_name in analysis.parameters:
            params[param_name] = self._generate_valid_value("string")  # Default to string
        return params
    
    def _generate_integration_setup(self, analysis: FunctionAnalysis) -> str:
        """Generate setup code for integration tests"""
        setup_lines = []
        for dep in analysis.dependencies:
            setup_lines.append(f"# Setup {dep}")
            setup_lines.append(f"# Mock or configure {dep} as needed")
        return "\n".join(setup_lines)
    
    def _generate_performance_parameters(self, analysis: FunctionAnalysis) -> Dict[str, Any]:
        """Generate parameters for performance tests"""
        params = {}
        for param_name in analysis.parameters:
            # Use larger datasets for performance testing
            if 'list' in str(analysis.signature.parameters[param_name].annotation).lower():
                params[param_name] = list(range(1000))
            else:
                params[param_name] = self._generate_valid_value("string")
        return params
    
    def generate_test_file(self, func: Callable, output_file: str, num_cases: int = 10) -> str:
        """Generate a complete test file for a function"""
        test_cases = self.generate_test_cases(func, num_cases)
        analysis = self.analyze_function(func)
        
        # Generate test file content
        content = self._generate_test_file_header(analysis)
        content += self._generate_imports(analysis)
        content += self._generate_fixtures(analysis)
        content += self._generate_test_class(analysis, test_cases)
        
        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return content
    
    def _generate_test_file_header(self, analysis: FunctionAnalysis) -> str:
        """Generate test file header"""
        return f'''"""
Generated Test Cases for {analysis.name}
=====================================

Auto-generated test cases for the {analysis.name} function.
Generated on: {datetime.now().isoformat()}

Function Analysis:
- Complexity Score: {analysis.complexity_score}
- Parameters: {', '.join(analysis.parameters)}
- Return Type: {analysis.return_annotation}
- Async: {analysis.is_async}
- Dependencies: {', '.join(analysis.dependencies)}
- Side Effects: {', '.join(analysis.side_effects)}

Test Coverage:
- Validation Tests: Parameter validation and type checking
- Edge Case Tests: Boundary values and special cases
- Error Handling Tests: Exception scenarios
- Integration Tests: Real dependency interaction
- Performance Tests: Execution time and resource usage
"""

'''
    
    def _generate_imports(self, analysis: FunctionAnalysis) -> str:
        """Generate import statements"""
        imports = [
            "import pytest",
            "import asyncio",
            "from unittest.mock import Mock, patch, MagicMock",
            "from typing import Any, Dict, List, Optional",
            "import time",
            "import logging",
            ""
        ]
        
        # Add specific imports based on dependencies
        for dep in analysis.dependencies:
            if dep not in ['pytest', 'asyncio', 'unittest', 'typing', 'time', 'logging']:
                imports.append(f"from {dep} import *")
        
        imports.append("")
        return "\n".join(imports)
    
    def _generate_fixtures(self, analysis: FunctionAnalysis) -> str:
        """Generate pytest fixtures"""
        fixtures = [
            "@pytest.fixture",
            f"def {analysis.name}_instance():",
            f'    """Fixture for {analysis.name} function"""',
            f"    # Return the function or create instance as needed",
            f"    return {analysis.name}",
            "",
            "@pytest.fixture",
            "def sample_data():",
            '    """Sample data for testing"""',
            "    return {",
        ]
        
        # Add sample data based on parameters
        for param_name in analysis.parameters:
            fixtures.append(f'        "{param_name}": "sample_value",')
        
        fixtures.extend([
            "    }",
            "",
        ])
        
        return "\n".join(fixtures)
    
    def _generate_test_class(self, analysis: FunctionAnalysis, test_cases: List[CaseData]) -> str:
        """Generate test class with all test cases"""
        content = [
            f"class Test{analysis.name.title()}:",
            f'    """Test suite for {analysis.name} function"""',
            "",
        ]
        
        # Group test cases by type
        test_groups = {}
        for test_case in test_cases:
            test_type = test_case.test_type.value
            if test_type not in test_groups:
                test_groups[test_type] = []
            test_groups[test_type].append(test_case)
        
        # Generate test methods for each group
        for test_type, cases in test_groups.items():
            content.append(f"    # {test_type.title()} Tests")
            content.append("")
            
            for test_case in cases:
                content.extend(self._generate_test_method(test_case, analysis))
                content.append("")
        
        return "\n".join(content)
    
    def _generate_test_method(self, test_case: CaseData, analysis: FunctionAnalysis) -> List[str]:
        """Generate a single test method"""
        method_lines = [
            f"    @pytest.mark.{test_case.test_type.value}",
        ]
        
        if test_case.async_test or analysis.is_async:
            method_lines.append("    @pytest.mark.asyncio")
        
        if test_case.parametrize:
            method_lines.append("    @pytest.mark.parametrize")
        
        method_lines.extend([
            f"    async def {test_case.name}(self, {analysis.name}_instance, sample_data):",
            f'        """{test_case.description}"""',
        ])
        
        # Add setup code
        if test_case.setup_code:
            method_lines.append("        # Setup")
            for line in test_case.setup_code.split('\n'):
                if line.strip():
                    method_lines.append(f"        {line}")
            method_lines.append("")
        
        # Add mock setup
        if test_case.mock_objects:
            method_lines.append("        # Mock setup")
            for mock_name, mock_value in test_case.mock_objects.items():
                method_lines.append(f"        {mock_name} = Mock(return_value={mock_value})")
            method_lines.append("")
        
        # Add test execution
        method_lines.append("        # Test execution")
        if test_case.expected_exception:
            method_lines.append("        with pytest.raises(Exception):")
            method_lines.append(f"            result = await {analysis.name}_instance(**{test_case.parameters})")
        else:
            if analysis.is_async:
                method_lines.append(f"        result = await {analysis.name}_instance(**{test_case.parameters})")
            else:
                method_lines.append(f"        result = {analysis.name}_instance(**{test_case.parameters})")
        
        # Add assertions
        if test_case.assertions:
            method_lines.append("")
            method_lines.append("        # Assertions")
            for assertion in test_case.assertions:
                method_lines.append(f"        {assertion}")
        
        # Add teardown code
        if test_case.teardown_code:
            method_lines.append("")
            method_lines.append("        # Teardown")
            for line in test_case.teardown_code.split('\n'):
                if line.strip():
                    method_lines.append(f"        {line}")
        
        return method_lines


class CoverageAnalyzerClass:
    """Analyze test coverage and suggest improvements"""
    
    def __init__(self):
        self.coverage_patterns = {
            "line_coverage": r"TOTAL\s+(\d+)\s+(\d+)\s+(\d+)%",
            "branch_coverage": r"TOTAL\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)%",
            "function_coverage": r"TOTAL\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)%"
        }
    
    def analyze_coverage(self, coverage_report: str) -> Dict[str, Any]:
        """Analyze coverage report and return metrics"""
        coverage_data = {
            "line_coverage": 0,
            "branch_coverage": 0,
            "function_coverage": 0,
            "missing_lines": [],
            "suggestions": []
        }
        
        # Parse coverage report (simplified)
        lines = coverage_report.split('\n')
        for line in lines:
            if "TOTAL" in line and "%" in line:
                # Extract percentage
                import re
                match = re.search(r'(\d+)%', line)
                if match:
                    coverage_data["line_coverage"] = int(match.group(1))
                    break
        
        # Generate suggestions based on coverage
        if coverage_data["line_coverage"] < 80:
            coverage_data["suggestions"].append("Add more test cases to improve line coverage")
        
        if coverage_data["line_coverage"] < 60:
            coverage_data["suggestions"].append("Critical: Line coverage is below 60%")
        
        return coverage_data


class QualityGateClass:
    """Quality gate for test validation"""
    
    def __init__(self):
        self.quality_thresholds = {
            "min_coverage": 80.0,
            "max_complexity": 10,
            "min_test_cases": 5,
            "max_execution_time": 300.0
        }
    
    def validate_test_quality(self, test_cases: List[CaseData], coverage_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate test quality against thresholds"""
        quality_report = {
            "passed": True,
            "score": 0.0,
            "issues": [],
            "recommendations": []
        }
        
        # Check coverage
        if coverage_data["line_coverage"] < self.quality_thresholds["min_coverage"]:
            quality_report["passed"] = False
            quality_report["issues"].append(f"Coverage {coverage_data['line_coverage']}% below threshold {self.quality_thresholds['min_coverage']}%")
        
        # Check test case count
        if len(test_cases) < self.quality_thresholds["min_test_cases"]:
            quality_report["issues"].append(f"Only {len(test_cases)} test cases, minimum is {self.quality_thresholds['min_test_cases']}")
        
        # Calculate quality score
        coverage_score = min(coverage_data["line_coverage"] / 100.0, 1.0)
        test_count_score = min(len(test_cases) / self.quality_thresholds["min_test_cases"], 1.0)
        quality_report["score"] = (coverage_score + test_count_score) / 2.0
        
        # Generate recommendations
        if quality_report["score"] < 0.8:
            quality_report["recommendations"].append("Add more comprehensive test cases")
            quality_report["recommendations"].append("Improve test coverage")
        
        return quality_report


# Example usage and demonstration
def demonstrate_test_generation():
    """Demonstrate the test case generation system"""
    
    # Example function to test
    def example_function(name: str, age: int, email: str = None) -> Dict[str, Any]:
        """
        Example function for demonstration.
        
        Args:
            name: User's name
            age: User's age
            email: User's email (optional)
            
        Returns:
            Dictionary with user information
            
        Raises:
            ValueError: If name is empty or age is negative
        """
        if not name:
            raise ValueError("Name cannot be empty")
        if age < 0:
            raise ValueError("Age cannot be negative")
        
        return {
            "name": name,
            "age": age,
            "email": email,
            "created_at": datetime.now().isoformat()
        }
    
    # Generate test cases
    generator = CaseDataGenerator()
    test_cases = generator.generate_test_cases(example_function, num_cases=15)
    
    print(f"Generated {len(test_cases)} test cases:")
    for i, test_case in enumerate(test_cases, 1):
        print(f"{i}. {test_case.name} - {test_case.description}")
        print(f"   Type: {test_case.test_type.value}, Complexity: {test_case.complexity.value}")
        print(f"   Parameters: {test_case.parameters}")
        if test_case.expected_exception:
            print(f"   Expected Exception: {test_case.expected_exception.__name__}")
        print()
    
    # Generate complete test file
    test_file_content = generator.generate_test_file(
        example_function, 
        "generated_test_example.py", 
        num_cases=10
    )
    
    print("Generated test file content:")
    print(test_file_content[:500] + "..." if len(test_file_content) > 500 else test_file_content)


if __name__ == "__main__":
    demonstrate_test_generation()
