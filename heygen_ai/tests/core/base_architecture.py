"""
Base Architecture for Refactored Test Generation System
======================================================

This module provides the foundational architecture for the refactored test generation system,
including common interfaces, abstract classes, and base implementations.
"""

import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Type, Union, Protocol
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class TestComplexity(Enum):
    """Test complexity levels"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    ADVANCED = "advanced"
    EXPERT = "expert"


class TestCategory(Enum):
    """Test categories"""
    UNIT = "unit"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"
    SECURITY = "security"
    API = "api"
    DATABASE = "database"
    NETWORK = "network"
    UI = "ui"
    ENTERPRISE = "enterprise"
    CORE = "core"


class TestPriority(Enum):
    """Test priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TestType(Enum):
    """Test types"""
    VALIDATION = "validation"
    TRANSFORMATION = "transformation"
    ASYNC = "async"
    ERROR_HANDLING = "error_handling"
    EDGE_CASE = "edge_case"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"
    SECURITY = "security"


@dataclass
class TestCase:
    """Base test case structure"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    test_type: TestType = TestType.VALIDATION
    category: TestCategory = TestCategory.UNIT
    priority: TestPriority = TestPriority.MEDIUM
    complexity: TestComplexity = TestComplexity.SIMPLE
    function_signature: str = ""
    docstring: str = ""
    test_code: str = ""
    expected_result: Any = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    assertions: List[str] = field(default_factory=list)
    setup_code: str = ""
    teardown_code: str = ""
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestGenerationConfig:
    """Configuration for test generation"""
    target_coverage: float = 0.8
    max_test_cases: int = 100
    include_edge_cases: bool = True
    include_performance_tests: bool = False
    include_security_tests: bool = False
    complexity_level: TestComplexity = TestComplexity.MODERATE
    naming_convention: str = "descriptive"
    code_style: str = "pytest"
    mock_strategy: str = "comprehensive"
    documentation_level: str = "detailed"
    parallel_generation: bool = True
    custom_patterns: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationMetrics:
    """Metrics for test generation"""
    total_tests_generated: int = 0
    generation_time: float = 0.0
    coverage_achieved: float = 0.0
    uniqueness_score: float = 0.0
    diversity_index: float = 0.0
    intuition_rating: float = 0.0
    quality_score: float = 0.0
    errors_encountered: int = 0
    warnings_generated: int = 0


class TestGeneratorInterface(Protocol):
    """Interface for test generators"""
    
    async def generate_tests(
        self, 
        function_signature: str, 
        docstring: str, 
        config: TestGenerationConfig
    ) -> List[TestCase]:
        """Generate test cases for a function"""
        ...


class TestPatternInterface(Protocol):
    """Interface for test patterns"""
    
    def get_patterns(self, test_type: TestType) -> List[str]:
        """Get test patterns for a specific type"""
        ...


class ParameterGeneratorInterface(Protocol):
    """Interface for parameter generators"""
    
    def generate_parameters(
        self, 
        param_type: str, 
        constraints: Dict[str, Any]
    ) -> List[Any]:
        """Generate test parameters for a given type"""
        ...


class TestValidatorInterface(Protocol):
    """Interface for test validators"""
    
    def validate_test_case(self, test_case: TestCase) -> bool:
        """Validate a test case"""
        ...


class TestOptimizerInterface(Protocol):
    """Interface for test optimizers"""
    
    def optimize_tests(self, test_cases: List[TestCase]) -> List[TestCase]:
        """Optimize a list of test cases"""
        ...


class BaseTestGenerator(ABC):
    """Abstract base class for test generators"""
    
    def __init__(self, config: TestGenerationConfig):
        self.config = config
        self.metrics = GenerationMetrics()
        self.logger = logging.getLogger(self.__class__.__name__)
        
    @abstractmethod
    async def generate_tests(
        self, 
        function_signature: str, 
        docstring: str, 
        config: Optional[TestGenerationConfig] = None
    ) -> List[TestCase]:
        """Generate test cases for a function"""
        pass
    
    def _create_base_test_case(
        self, 
        name: str, 
        description: str, 
        test_type: TestType,
        function_signature: str,
        docstring: str
    ) -> TestCase:
        """Create a base test case with common properties"""
        return TestCase(
            name=name,
            description=description,
            test_type=test_type,
            function_signature=function_signature,
            docstring=docstring,
            complexity=self.config.complexity_level,
            tags=self._generate_tags(test_type)
        )
    
    def _generate_tags(self, test_type: TestType) -> List[str]:
        """Generate tags for a test type"""
        tags = [test_type.value]
        if self.config.include_edge_cases:
            tags.append("edge_cases")
        if self.config.include_performance_tests:
            tags.append("performance")
        if self.config.include_security_tests:
            tags.append("security")
        return tags
    
    def _update_metrics(self, test_cases: List[TestCase], generation_time: float):
        """Update generation metrics"""
        self.metrics.total_tests_generated = len(test_cases)
        self.metrics.generation_time = generation_time
        self.metrics.coverage_achieved = self._calculate_coverage(test_cases)
        self.metrics.uniqueness_score = self._calculate_uniqueness(test_cases)
        self.metrics.diversity_index = self._calculate_diversity(test_cases)
        self.metrics.intuition_rating = self._calculate_intuition(test_cases)
        self.metrics.quality_score = self._calculate_quality(test_cases)
    
    def _calculate_coverage(self, test_cases: List[TestCase]) -> float:
        """Calculate test coverage score"""
        if not test_cases:
            return 0.0
        
        # Simple coverage calculation based on test types and categories
        unique_types = len(set(tc.test_type for tc in test_cases))
        unique_categories = len(set(tc.category for tc in test_cases))
        
        return min((unique_types + unique_categories) / 10.0, 1.0)
    
    def _calculate_uniqueness(self, test_cases: List[TestCase]) -> float:
        """Calculate uniqueness score"""
        if not test_cases:
            return 0.0
        
        unique_names = len(set(tc.name for tc in test_cases))
        total_tests = len(test_cases)
        
        return unique_names / total_tests if total_tests > 0 else 0.0
    
    def _calculate_diversity(self, test_cases: List[TestCase]) -> float:
        """Calculate diversity index"""
        if not test_cases:
            return 0.0
        
        # Calculate diversity based on different test characteristics
        type_diversity = len(set(tc.test_type for tc in test_cases)) / len(TestType)
        category_diversity = len(set(tc.category for tc in test_cases)) / len(TestCategory)
        complexity_diversity = len(set(tc.complexity for tc in test_cases)) / len(TestComplexity)
        
        return (type_diversity + category_diversity + complexity_diversity) / 3.0
    
    def _calculate_intuition(self, test_cases: List[TestCase]) -> float:
        """Calculate intuition rating"""
        if not test_cases:
            return 0.0
        
        # Calculate intuition based on descriptive names and clear descriptions
        descriptive_names = sum(1 for tc in test_cases if len(tc.name.split('_')) >= 3)
        clear_descriptions = sum(1 for tc in test_cases if len(tc.description) > 20)
        
        total_tests = len(test_cases)
        return (descriptive_names + clear_descriptions) / (2 * total_tests) if total_tests > 0 else 0.0
    
    def _calculate_quality(self, test_cases: List[TestCase]) -> float:
        """Calculate overall quality score"""
        if not test_cases:
            return 0.0
        
        # Calculate quality based on multiple factors
        has_assertions = sum(1 for tc in test_cases if tc.assertions)
        has_setup = sum(1 for tc in test_cases if tc.setup_code)
        has_teardown = sum(1 for tc in test_cases if tc.teardown_code)
        
        total_tests = len(test_cases)
        quality_factors = [
            has_assertions / total_tests,
            has_setup / total_tests,
            has_teardown / total_tests,
            self._calculate_coverage(test_cases),
            self._calculate_uniqueness(test_cases),
            self._calculate_diversity(test_cases),
            self._calculate_intuition(test_cases)
        ]
        
        return sum(quality_factors) / len(quality_factors)
    
    def get_metrics(self) -> GenerationMetrics:
        """Get current generation metrics"""
        return self.metrics
    
    def export_metrics(self, file_path: str):
        """Export metrics to a file"""
        metrics_dict = {
            "total_tests_generated": self.metrics.total_tests_generated,
            "generation_time": self.metrics.generation_time,
            "coverage_achieved": self.metrics.coverage_achieved,
            "uniqueness_score": self.metrics.uniqueness_score,
            "diversity_index": self.metrics.diversity_index,
            "intuition_rating": self.metrics.intuition_rating,
            "quality_score": self.metrics.quality_score,
            "errors_encountered": self.metrics.errors_encountered,
            "warnings_generated": self.metrics.warnings_generated
        }
        
        with open(file_path, 'w') as f:
            json.dump(metrics_dict, f, indent=2)


class BaseTestPattern(ABC):
    """Abstract base class for test patterns"""
    
    @abstractmethod
    def get_patterns(self, test_type: TestType) -> List[str]:
        """Get test patterns for a specific type"""
        pass
    
    @abstractmethod
    def get_pattern_description(self, pattern: str) -> str:
        """Get description for a specific pattern"""
        pass


class BaseParameterGenerator(ABC):
    """Abstract base class for parameter generators"""
    
    @abstractmethod
    def generate_parameters(
        self, 
        param_type: str, 
        constraints: Dict[str, Any]
    ) -> List[Any]:
        """Generate test parameters for a given type"""
        pass
    
    @abstractmethod
    def get_supported_types(self) -> List[str]:
        """Get list of supported parameter types"""
        pass


class BaseTestValidator(ABC):
    """Abstract base class for test validators"""
    
    @abstractmethod
    def validate_test_case(self, test_case: TestCase) -> bool:
        """Validate a test case"""
        pass
    
    @abstractmethod
    def get_validation_errors(self, test_case: TestCase) -> List[str]:
        """Get validation errors for a test case"""
        pass


class BaseTestOptimizer(ABC):
    """Abstract base class for test optimizers"""
    
    @abstractmethod
    def optimize_tests(self, test_cases: List[TestCase]) -> List[TestCase]:
        """Optimize a list of test cases"""
        pass
    
    @abstractmethod
    def get_optimization_suggestions(self, test_cases: List[TestCase]) -> List[str]:
        """Get optimization suggestions for test cases"""
        pass


class TestGenerationError(Exception):
    """Base exception for test generation errors"""
    pass


class ConfigurationError(TestGenerationError):
    """Exception for configuration-related errors"""
    pass


class PatternError(TestGenerationError):
    """Exception for pattern-related errors"""
    pass


class ParameterGenerationError(TestGenerationError):
    """Exception for parameter generation errors"""
    pass


class ValidationError(TestGenerationError):
    """Exception for validation errors"""
    pass


class OptimizationError(TestGenerationError):
    """Exception for optimization errors"""
    pass
