"""
Comprehensive Test Case Generator for HeyGen AI
==============================================

Advanced test case generation system that creates unique, diverse, and intuitive
unit tests for functions given their signature and docstring.

This is the main entry point that integrates all test generation capabilities:
- Unique test scenarios with varied approaches
- Diverse test cases covering wide range of scenarios  
- Intuitive test naming and structure
- Advanced function analysis and pattern recognition
- Comprehensive coverage of edge cases and error conditions
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
from pathlib import Path

# Import our enhanced generators
from intelligent_test_generator import IntelligentTestGenerator, TestDiversity, TestIntuition
from unique_diverse_test_generator import UniqueDiverseTestGenerator

logger = logging.getLogger(__name__)


@dataclass
class ComprehensiveTestCase:
    """Comprehensive test case with all quality metrics"""
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
    coverage_score: float = 0.0
    complexity_score: float = 0.0
    overall_quality: float = 0.0
    # Metadata
    generation_method: str = ""
    test_category: str = ""
    priority: str = "medium"


class ComprehensiveTestGenerator:
    """Main comprehensive test generator that combines all capabilities"""
    
    def __init__(self, 
                 diversity_level: TestDiversity = TestDiversity.COMPREHENSIVE,
                 intuition_level: TestIntuition = TestIntuition.DESCRIPTIVE):
        self.diversity_level = diversity_level
        self.intuition_level = intuition_level
        
        # Initialize sub-generators
        self.intelligent_generator = IntelligentTestGenerator(diversity_level, intuition_level)
        self.unique_diverse_generator = UniqueDiverseTestGenerator()
        
        # Test generation strategies
        self.generation_strategies = self._setup_generation_strategies()
        self.test_categories = self._setup_test_categories()
        self.quality_weights = self._setup_quality_weights()
        
    def _setup_generation_strategies(self) -> Dict[str, Callable]:
        """Setup different test generation strategies"""
        return {
            "intelligent": self._generate_intelligent_tests,
            "unique_diverse": self._generate_unique_diverse_tests,
            "comprehensive": self._generate_comprehensive_tests,
            "focused": self._generate_focused_tests,
            "exploratory": self._generate_exploratory_tests
        }
    
    def _setup_test_categories(self) -> Dict[str, Dict[str, Any]]:
        """Setup test categories with specific requirements"""
        return {
            "validation": {
                "required_scenarios": ["happy_path", "boundary_values", "invalid_input", "edge_cases"],
                "priority": "high",
                "coverage_target": 0.95
            },
            "transformation": {
                "required_scenarios": ["identity", "data_processing", "format_conversion", "error_handling"],
                "priority": "high", 
                "coverage_target": 0.90
            },
            "calculation": {
                "required_scenarios": ["accuracy", "precision", "overflow", "underflow", "edge_cases"],
                "priority": "critical",
                "coverage_target": 0.98
            },
            "business_logic": {
                "required_scenarios": ["workflow", "rules", "decisions", "approvals", "edge_cases"],
                "priority": "critical",
                "coverage_target": 0.95
            },
            "data_processing": {
                "required_scenarios": ["small_data", "large_data", "empty_data", "malformed_data"],
                "priority": "high",
                "coverage_target": 0.90
            },
            "api": {
                "required_scenarios": ["success", "failure", "timeout", "authentication", "authorization"],
                "priority": "high",
                "coverage_target": 0.85
            }
        }
    
    def _setup_quality_weights(self) -> Dict[str, float]:
        """Setup weights for quality scoring"""
        return {
            "uniqueness": 0.25,
            "diversity": 0.25,
            "intuition": 0.20,
            "coverage": 0.15,
            "complexity": 0.15
        }
    
    def generate_comprehensive_tests(self, func: Callable, 
                                   strategy: str = "comprehensive",
                                   num_tests: int = 25) -> List[ComprehensiveTestCase]:
        """Generate comprehensive test cases using specified strategy"""
        
        if strategy not in self.generation_strategies:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Generate tests using the specified strategy
        generator_func = self.generation_strategies[strategy]
        test_cases = generator_func(func, num_tests)
        
        # Enhance and score all test cases
        enhanced_cases = []
        for test_case in test_cases:
            enhanced_case = self._enhance_test_case(test_case, func)
            enhanced_cases.append(enhanced_case)
        
        # Sort by overall quality and return
        enhanced_cases.sort(key=lambda x: x.overall_quality, reverse=True)
        return enhanced_cases[:num_tests]
    
    def _generate_intelligent_tests(self, func: Callable, num_tests: int) -> List[ComprehensiveTestCase]:
        """Generate tests using intelligent generator"""
        intelligent_tests = self.intelligent_generator.generate_intelligent_tests(func, num_tests)
        
        # Convert to comprehensive format
        comprehensive_tests = []
        for test in intelligent_tests:
            comp_test = ComprehensiveTestCase(
                name=test.name,
                description=test.description,
                test_type=test.test_type,
                function_name=test.function_name,
                parameters=test.parameters,
                expected_result=test.expected_result,
                expected_exception=test.expected_exception,
                setup_code=test.setup_code,
                teardown_code=test.teardown_code,
                assertions=test.assertions,
                mock_objects=test.mock_objects,
                async_test=test.async_test,
                parametrize=test.parametrize,
                parametrize_values=test.parametrize_values,
                uniqueness_score=test.uniqueness_score,
                diversity_score=test.diversity_score,
                intuition_score=test.intuition_score,
                generation_method="intelligent"
            )
            comprehensive_tests.append(comp_test)
        
        return comprehensive_tests
    
    def _generate_unique_diverse_tests(self, func: Callable, num_tests: int) -> List[ComprehensiveTestCase]:
        """Generate tests using unique diverse generator"""
        unique_tests = self.unique_diverse_generator.generate_unique_diverse_tests(func, num_tests)
        
        # Convert to comprehensive format
        comprehensive_tests = []
        for test in unique_tests:
            comp_test = ComprehensiveTestCase(
                name=test.name,
                description=test.description,
                test_type=test.test_type,
                function_name=test.function_name,
                parameters=test.parameters,
                expected_result=test.expected_result,
                expected_exception=test.expected_exception,
                setup_code=test.setup_code,
                teardown_code=test.teardown_code,
                assertions=test.assertions,
                mock_objects=test.mock_objects,
                async_test=test.async_test,
                parametrize=test.parametrize,
                parametrize_values=test.parametrize_values,
                uniqueness_score=test.uniqueness_score,
                diversity_score=test.diversity_score,
                intuition_score=test.intuition_score,
                generation_method="unique_diverse"
            )
            comprehensive_tests.append(comp_test)
        
        return comprehensive_tests
    
    def _generate_comprehensive_tests(self, func: Callable, num_tests: int) -> List[ComprehensiveTestCase]:
        """Generate comprehensive tests combining multiple strategies"""
        # Use both generators and combine results
        intelligent_tests = self._generate_intelligent_tests(func, num_tests // 2)
        unique_tests = self._generate_unique_diverse_tests(func, num_tests // 2)
        
        # Combine and deduplicate
        all_tests = intelligent_tests + unique_tests
        unique_tests_dict = {}
        
        for test in all_tests:
            # Use name as key for deduplication
            if test.name not in unique_tests_dict:
                unique_tests_dict[test.name] = test
            else:
                # Keep the one with higher quality
                existing = unique_tests_dict[test.name]
                if test.overall_quality > existing.overall_quality:
                    unique_tests_dict[test.name] = test
        
        return list(unique_tests_dict.values())
    
    def _generate_focused_tests(self, func: Callable, num_tests: int) -> List[ComprehensiveTestCase]:
        """Generate focused tests for specific scenarios"""
        analysis = self._analyze_function(func)
        function_type = self._classify_function(func, analysis)
        
        if function_type not in self.test_categories:
            return self._generate_comprehensive_tests(func, num_tests)
        
        category_config = self.test_categories[function_type]
        required_scenarios = category_config["required_scenarios"]
        
        test_cases = []
        tests_per_scenario = num_tests // len(required_scenarios)
        
        for scenario in required_scenarios:
            scenario_tests = self._generate_scenario_focused_tests(func, analysis, scenario, tests_per_scenario)
            test_cases.extend(scenario_tests)
        
        return test_cases
    
    def _generate_exploratory_tests(self, func: Callable, num_tests: int) -> List[ComprehensiveTestCase]:
        """Generate exploratory tests for edge cases and unusual scenarios"""
        analysis = self._analyze_function(func)
        test_cases = []
        
        # Generate unusual parameter combinations
        unusual_combinations = self._generate_unusual_combinations(analysis, num_tests // 3)
        for combination in unusual_combinations:
            test_case = ComprehensiveTestCase(
                name=f"test_{func.__name__}_unusual_combination_{len(test_cases)+1}",
                description=f"Exploratory test for {func.__name__} with unusual parameter combination",
                test_type="exploratory",
                function_name=func.__name__,
                parameters=combination,
                expected_result=None,
                assertions=["assert result is not None or exception is raised"],
                async_test=analysis.get("is_async", False),
                generation_method="exploratory",
                test_category="edge_case",
                priority="low"
            )
            test_cases.append(test_case)
        
        # Generate stress tests
        stress_tests = self._generate_stress_tests(func, analysis, num_tests // 3)
        test_cases.extend(stress_tests)
        
        # Generate boundary exploration tests
        boundary_tests = self._generate_boundary_exploration_tests(func, analysis, num_tests // 3)
        test_cases.extend(boundary_tests)
        
        return test_cases
    
    def _enhance_test_case(self, test_case: ComprehensiveTestCase, func: Callable) -> ComprehensiveTestCase:
        """Enhance test case with additional metrics and improvements"""
        # Calculate additional scores
        test_case.coverage_score = self._calculate_coverage_score(test_case, func)
        test_case.complexity_score = self._calculate_complexity_score(test_case, func)
        
        # Determine test category and priority
        test_case.test_category = self._determine_test_category(test_case)
        test_case.priority = self._determine_priority(test_case)
        
        # Enhance assertions if needed
        if not test_case.assertions:
            test_case.assertions = self._generate_enhanced_assertions(test_case, func)
        
        # Calculate overall quality
        test_case.overall_quality = self._calculate_overall_quality(test_case)
        
        return test_case
    
    def _calculate_coverage_score(self, test_case: ComprehensiveTestCase, func: Callable) -> float:
        """Calculate coverage score for the test case"""
        score = 0.0
        
        # Base score for test type
        type_scores = {
            "validation": 0.8,
            "edge_case": 0.9,
            "transformation": 0.7,
            "calculation": 0.8,
            "business_logic": 0.8,
            "exploratory": 0.6,
            "stress": 0.7
        }
        score += type_scores.get(test_case.test_type, 0.5)
        
        # Bonus for comprehensive parameter coverage
        if len(test_case.parameters) > 1:
            score += 0.1
        
        # Bonus for exception testing
        if test_case.expected_exception:
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_complexity_score(self, test_case: ComprehensiveTestCase, func: Callable) -> float:
        """Calculate complexity score for the test case"""
        score = 0.0
        
        # Base score for test complexity
        if test_case.async_test:
            score += 0.2
        if test_case.parametrize:
            score += 0.2
        if len(test_case.parameters) > 3:
            score += 0.2
        if len(test_case.assertions) > 2:
            score += 0.2
        if test_case.setup_code or test_case.teardown_code:
            score += 0.2
        
        return min(score, 1.0)
    
    def _determine_test_category(self, test_case: ComprehensiveTestCase) -> str:
        """Determine test category based on test case characteristics"""
        if "validation" in test_case.test_type or "happy_path" in test_case.name:
            return "validation"
        elif "edge_case" in test_case.test_type or "boundary" in test_case.name:
            return "edge_case"
        elif "exploratory" in test_case.test_type or "unusual" in test_case.name:
            return "exploratory"
        elif "stress" in test_case.test_type or "performance" in test_case.name:
            return "performance"
        else:
            return "unit"
    
    def _determine_priority(self, test_case: ComprehensiveTestCase) -> str:
        """Determine test priority based on test case characteristics"""
        if test_case.test_category in ["validation", "edge_case"]:
            return "high"
        elif test_case.test_category in ["exploratory", "performance"]:
            return "medium"
        else:
            return "low"
    
    def _generate_enhanced_assertions(self, test_case: ComprehensiveTestCase, func: Callable) -> List[str]:
        """Generate enhanced assertions for the test case"""
        assertions = []
        
        # Basic assertions
        assertions.append("assert result is not None")
        
        # Type-specific assertions
        if test_case.expected_exception:
            assertions.append("assert exception is raised")
        elif "validation" in test_case.test_type:
            assertions.append("assert result is True or result.get('valid', False)")
        elif "calculation" in test_case.test_type:
            assertions.append("assert isinstance(result, (int, float, decimal.Decimal))")
        elif "transformation" in test_case.test_type:
            assertions.append("assert result != input_data")
        
        return assertions
    
    def _calculate_overall_quality(self, test_case: ComprehensiveTestCase) -> float:
        """Calculate overall quality score"""
        weights = self.quality_weights
        
        overall_quality = (
            test_case.uniqueness_score * weights["uniqueness"] +
            test_case.diversity_score * weights["diversity"] +
            test_case.intuition_score * weights["intuition"] +
            test_case.coverage_score * weights["coverage"] +
            test_case.complexity_score * weights["complexity"]
        )
        
        return overall_quality
    
    def generate_test_file(self, func: Callable, output_path: str, 
                          strategy: str = "comprehensive", num_tests: int = 25) -> str:
        """Generate a complete test file for a function"""
        test_cases = self.generate_comprehensive_tests(func, strategy, num_tests)
        analysis = self._analyze_function(func)
        
        # Generate file content
        content = self._generate_file_header(func, analysis, test_cases)
        content += self._generate_imports(func, analysis)
        content += self._generate_fixtures(func, analysis)
        content += self._generate_test_class(func, analysis, test_cases)
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return content
    
    def _generate_file_header(self, func: Callable, analysis: Dict[str, Any], 
                            test_cases: List[ComprehensiveTestCase]) -> str:
        """Generate comprehensive test file header"""
        return f'''"""
Comprehensive Test Cases for {func.__name__}
==========================================

Auto-generated comprehensive test cases for the {func.__name__} function.
Generated on: {datetime.now().isoformat()}

Function Analysis:
- Function: {func.__name__}
- Parameters: {', '.join(analysis.get('parameters', []))}
- Return Type: {analysis.get('return_annotation', 'Unknown')}
- Async: {analysis.get('is_async', False)}
- Complexity: {analysis.get('complexity_score', 1)}
- Domain: {analysis.get('domain_context', 'general')}

Test Generation:
- Strategy: Comprehensive
- Total Tests: {len(test_cases)}
- Test Categories: {', '.join(set(tc.test_category for tc in test_cases))}
- Quality Scores: U={sum(tc.uniqueness_score for tc in test_cases)/len(test_cases):.2f}, D={sum(tc.diversity_score for tc in test_cases)/len(test_cases):.2f}, I={sum(tc.intuition_score for tc in test_cases)/len(test_cases):.2f}

Test Coverage:
- Validation Tests: Parameter validation and type checking
- Edge Case Tests: Boundary values and special cases
- Error Handling Tests: Exception scenarios and error recovery
- Integration Tests: Real dependency interaction
- Performance Tests: Execution time and resource usage
- Exploratory Tests: Unusual scenarios and stress testing
"""

'''
    
    def _generate_imports(self, func: Callable, analysis: Dict[str, Any]) -> str:
        """Generate import statements"""
        imports = [
            "import pytest",
            "import asyncio",
            "from unittest.mock import Mock, patch, MagicMock, AsyncMock",
            "from typing import Any, Dict, List, Optional, Union",
            "from datetime import datetime, timedelta",
            "import json",
            "import uuid",
            "import decimal",
            ""
        ]
        
        # Add async imports if needed
        if analysis.get("is_async", False):
            imports.insert(2, "import asyncio")
        
        # Add domain-specific imports
        domain = analysis.get("domain_context", "general")
        if domain == "user_management":
            imports.append("from core.enterprise_features import EnterpriseFeatures")
        elif domain == "media_processing":
            imports.append("from core.media_processor import MediaProcessor")
        
        imports.append("")
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
        
        # Add async fixture if needed
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
    
    def _generate_test_class(self, func: Callable, analysis: Dict[str, Any], 
                           test_cases: List[ComprehensiveTestCase]) -> str:
        """Generate test class with all test cases"""
        class_name = f"Test{func.__name__.title()}"
        
        content = [
            f"class {class_name}:",
            f'    """Comprehensive test suite for {func.__name__} function"""',
            "",
        ]
        
        # Group test cases by category
        categories = {}
        for test_case in test_cases:
            category = test_case.test_category
            if category not in categories:
                categories[category] = []
            categories[category].append(test_case)
        
        # Generate test methods for each category
        for category, cases in categories.items():
            content.append(f"    # {category.title()} Tests")
            content.append("")
            
            for test_case in cases:
                content.extend(self._generate_test_method(test_case, analysis))
                content.append("")
        
        return "\n".join(content)
    
    def _generate_test_method(self, test_case: ComprehensiveTestCase, analysis: Dict[str, Any]) -> List[str]:
        """Generate a single test method"""
        method_lines = [
            f"    @pytest.mark.{test_case.test_type}",
            f"    @pytest.mark.{test_case.test_category}",
            f"    @pytest.mark.{test_case.priority}",
        ]
        
        if test_case.async_test:
            method_lines.append("    @pytest.mark.asyncio")
        
        if test_case.parametrize:
            method_lines.append("    @pytest.mark.parametrize")
        
        method_lines.extend([
            f"    async def {test_case.name}(self, {test_case.function_name}_instance):",
            f'        """{test_case.description}"""',
        ])
        
        # Add setup code
        if test_case.setup_code:
            method_lines.append("        # Setup")
            for line in test_case.setup_code.split('\n'):
                if line.strip():
                    method_lines.append(f"        {line}")
            method_lines.append("")
        
        # Add test execution
        method_lines.append("        # Test execution")
        if test_case.expected_exception:
            method_lines.append("        with pytest.raises(Exception):")
            method_lines.append(f"            result = await {test_case.function_name}_instance(**{test_case.parameters})")
        else:
            if test_case.async_test:
                method_lines.append(f"        result = await {test_case.function_name}_instance(**{test_case.parameters})")
            else:
                method_lines.append(f"        result = {test_case.function_name}_instance(**{test_case.parameters})")
        
        # Add assertions
        if test_case.assertions:
            method_lines.append("")
            method_lines.append("        # Assertions")
            for assertion in test_case.assertions:
                method_lines.append(f"        {assertion}")
        
        return method_lines
    
    # Helper methods
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
                "complexity_score": 1,  # Simplified
                "domain_context": "general"  # Simplified
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
        else:
            return "general"
    
    def _generate_scenario_focused_tests(self, func: Callable, analysis: Dict[str, Any], 
                                       scenario: str, num_tests: int) -> List[ComprehensiveTestCase]:
        """Generate focused tests for a specific scenario"""
        # Simplified implementation
        test_cases = []
        for i in range(num_tests):
            test_case = ComprehensiveTestCase(
                name=f"test_{func.__name__}_{scenario}_{i+1}",
                description=f"Test {func.__name__} for {scenario} scenario",
                test_type="focused",
                function_name=func.__name__,
                parameters={},
                expected_result=None,
                assertions=["assert result is not None"],
                generation_method="focused"
            )
            test_cases.append(test_case)
        return test_cases
    
    def _generate_unusual_combinations(self, analysis: Dict[str, Any], num_combinations: int) -> List[Dict[str, Any]]:
        """Generate unusual parameter combinations"""
        combinations = []
        for i in range(num_combinations):
            combination = {"param": f"unusual_value_{i}"}
            combinations.append(combination)
        return combinations
    
    def _generate_stress_tests(self, func: Callable, analysis: Dict[str, Any], num_tests: int) -> List[ComprehensiveTestCase]:
        """Generate stress tests"""
        test_cases = []
        for i in range(num_tests):
            test_case = ComprehensiveTestCase(
                name=f"test_{func.__name__}_stress_{i+1}",
                description=f"Stress test for {func.__name__}",
                test_type="stress",
                function_name=func.__name__,
                parameters={"stress_param": "large_value"},
                expected_result=None,
                assertions=["assert result is not None"],
                generation_method="exploratory"
            )
            test_cases.append(test_case)
        return test_cases
    
    def _generate_boundary_exploration_tests(self, func: Callable, analysis: Dict[str, Any], num_tests: int) -> List[ComprehensiveTestCase]:
        """Generate boundary exploration tests"""
        test_cases = []
        for i in range(num_tests):
            test_case = ComprehensiveTestCase(
                name=f"test_{func.__name__}_boundary_{i+1}",
                description=f"Boundary exploration test for {func.__name__}",
                test_type="boundary",
                function_name=func.__name__,
                parameters={"boundary_param": "edge_value"},
                expected_result=None,
                assertions=["assert result is not None"],
                generation_method="exploratory"
            )
            test_cases.append(test_case)
        return test_cases


def demonstrate_comprehensive_generation():
    """Demonstrate the comprehensive test generation system"""
    
    # Example function to test
    def calculate_user_credits(user_tier: str, video_duration: int, quality: str) -> dict:
        """
        Calculate video generation credits for a user based on their tier and video specifications.
        
        Args:
            user_tier: User subscription tier (free, premium, enterprise)
            video_duration: Video duration in seconds
            quality: Video quality (standard, hd, 4k)
            
        Returns:
            Dictionary with credit calculation results
            
        Raises:
            ValueError: If user_tier is invalid
            TypeError: If video_duration is not an integer
        """
        if user_tier not in ["free", "premium", "enterprise"]:
            raise ValueError("Invalid user tier")
        
        if not isinstance(video_duration, int):
            raise TypeError("Video duration must be an integer")
        
        # Base credit calculation
        base_credits = 1
        
        # Tier multipliers
        tier_multipliers = {
            "free": 1.0,
            "premium": 0.7,
            "enterprise": 0.5
        }
        
        # Quality multipliers
        quality_multipliers = {
            "standard": 1.0,
            "hd": 1.5,
            "4k": 2.0
        }
        
        # Duration factor (credits increase with duration)
        duration_factor = max(1, video_duration // 30)  # 1 credit per 30 seconds
        
        # Calculate total credits
        total_credits = base_credits * tier_multipliers[user_tier] * quality_multipliers[quality] * duration_factor
        
        return {
            "user_tier": user_tier,
            "video_duration": video_duration,
            "quality": quality,
            "base_credits": base_credits,
            "tier_multiplier": tier_multipliers[user_tier],
            "quality_multiplier": quality_multipliers[quality],
            "duration_factor": duration_factor,
            "total_credits": total_credits,
            "calculated_at": datetime.now().isoformat()
        }
    
    # Generate comprehensive tests
    generator = ComprehensiveTestGenerator()
    
    print("Generating comprehensive test cases...")
    print("=" * 60)
    
    # Test different strategies
    strategies = ["intelligent", "unique_diverse", "comprehensive", "focused", "exploratory"]
    
    for strategy in strategies:
        print(f"\n{strategy.upper()} STRATEGY:")
        print("-" * 30)
        
        test_cases = generator.generate_comprehensive_tests(
            calculate_user_credits, 
            strategy=strategy, 
            num_tests=10
        )
        
        print(f"Generated {len(test_cases)} test cases:")
        
        for i, test_case in enumerate(test_cases[:5], 1):  # Show first 5
            print(f"{i}. {test_case.name}")
            print(f"   Quality: {test_case.overall_quality:.2f} (U:{test_case.uniqueness_score:.2f}, D:{test_case.diversity_score:.2f}, I:{test_case.intuition_score:.2f})")
            print(f"   Category: {test_case.test_category}, Priority: {test_case.priority}")
            print(f"   Method: {test_case.generation_method}")
        
        if len(test_cases) > 5:
            print(f"   ... and {len(test_cases) - 5} more tests")
    
    # Generate complete test file
    print(f"\n{'='*60}")
    print("GENERATING COMPLETE TEST FILE")
    print("=" * 60)
    
    test_file_content = generator.generate_test_file(
        calculate_user_credits,
        "comprehensive_test_calculator.py",
        strategy="comprehensive",
        num_tests=20
    )
    
    print("Generated comprehensive test file:")
    print(f"File size: {len(test_file_content)} characters")
    print(f"Lines: {len(test_file_content.splitlines())}")
    print("\nFirst 500 characters:")
    print(test_file_content[:500] + "..." if len(test_file_content) > 500 else test_file_content)


if __name__ == "__main__":
    demonstrate_comprehensive_generation()
