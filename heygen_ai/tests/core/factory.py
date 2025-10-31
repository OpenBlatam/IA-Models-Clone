"""
Factory Pattern Implementation for Test Generation System
========================================================

This module provides factory classes for creating test generators, patterns, 
parameter generators, validators, and optimizers.
"""

import importlib
import inspect
from abc import ABC, abstractmethod
from typing import Dict, List, Type, Any, Optional, Callable
import logging

from .base_architecture import (
    BaseTestGenerator, BaseTestPattern, BaseParameterGenerator,
    BaseTestValidator, BaseTestOptimizer, TestGenerationConfig,
    TestGeneratorInterface, TestPatternInterface, ParameterGeneratorInterface,
    TestValidatorInterface, TestOptimizerInterface
)

logger = logging.getLogger(__name__)


class GeneratorType:
    """Generator type constants"""
    BASIC = "basic"
    ENHANCED = "enhanced"
    AI_POWERED = "ai_powered"
    QUANTUM = "quantum"
    CONSCIOUSNESS = "consciousness"
    TRANSCENDENT = "transcendent"
    DIVINE = "divine"
    ULTIMATE = "ultimate"
    CONTINUATION = "continuation"
    IMPROVEMENT = "improvement"
    REALITY = "reality"
    OMNIPOTENT = "omnipotent"
    INFINITE = "infinite"


class PatternType:
    """Pattern type constants"""
    VALIDATION = "validation"
    TRANSFORMATION = "transformation"
    ASYNC = "async"
    ERROR_HANDLING = "error_handling"
    EDGE_CASE = "edge_case"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"
    SECURITY = "security"
    CUSTOM = "custom"


class ParameterType:
    """Parameter type constants"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    LIST = "list"
    DICT = "dict"
    DATETIME = "datetime"
    UUID = "uuid"
    CUSTOM = "custom"


class ValidatorType:
    """Validator type constants"""
    SYNTAX = "syntax"
    SEMANTIC = "semantic"
    COVERAGE = "coverage"
    QUALITY = "quality"
    PERFORMANCE = "performance"
    SECURITY = "security"
    CUSTOM = "custom"


class OptimizerType:
    """Optimizer type constants"""
    DEDUPLICATION = "deduplication"
    PERFORMANCE = "performance"
    COVERAGE = "coverage"
    QUALITY = "quality"
    PARALLELIZATION = "parallelization"
    CUSTOM = "custom"


class TestGeneratorFactory:
    """Factory for creating test generators"""
    
    _generators: Dict[str, Type[BaseTestGenerator]] = {}
    _registered_modules: Dict[str, str] = {}
    
    @classmethod
    def register_generator(
        cls, 
        generator_type: str, 
        generator_class: Type[BaseTestGenerator],
        module_path: Optional[str] = None
    ):
        """Register a test generator class"""
        cls._generators[generator_type] = generator_class
        if module_path:
            cls._registered_modules[generator_type] = module_path
        logger.info(f"Registered generator: {generator_type} -> {generator_class.__name__}")
    
    @classmethod
    def create_generator(
        cls, 
        generator_type: str, 
        config: TestGenerationConfig
    ) -> BaseTestGenerator:
        """Create a test generator instance"""
        if generator_type not in cls._generators:
            # Try to load from registered module
            if generator_type in cls._registered_modules:
                cls._load_generator_from_module(generator_type)
            else:
                raise ValueError(f"Unknown generator type: {generator_type}")
        
        generator_class = cls._generators[generator_type]
        return generator_class(config)
    
    @classmethod
    def _load_generator_from_module(cls, generator_type: str):
        """Load generator class from module"""
        module_path = cls._registered_modules[generator_type]
        try:
            module = importlib.import_module(module_path)
            # Find the generator class in the module
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (issubclass(obj, BaseTestGenerator) and 
                    obj != BaseTestGenerator and
                    not inspect.isabstract(obj)):
                    cls._generators[generator_type] = obj
                    logger.info(f"Loaded generator {name} from {module_path}")
                    return
            raise ImportError(f"No generator class found in {module_path}")
        except ImportError as e:
            logger.error(f"Failed to load generator from {module_path}: {e}")
            raise
    
    @classmethod
    def get_available_generators(cls) -> List[str]:
        """Get list of available generator types"""
        return list(cls._generators.keys())
    
    @classmethod
    def get_generator_info(cls, generator_type: str) -> Dict[str, Any]:
        """Get information about a generator type"""
        if generator_type not in cls._generators:
            return {}
        
        generator_class = cls._generators[generator_type]
        return {
            "name": generator_class.__name__,
            "module": generator_class.__module__,
            "docstring": generator_class.__doc__,
            "methods": [method for method in dir(generator_class) 
                       if not method.startswith('_')],
            "is_abstract": inspect.isabstract(generator_class)
        }


class TestPatternFactory:
    """Factory for creating test patterns"""
    
    _patterns: Dict[str, Type[BaseTestPattern]] = {}
    
    @classmethod
    def register_pattern(
        cls, 
        pattern_type: str, 
        pattern_class: Type[BaseTestPattern]
    ):
        """Register a test pattern class"""
        cls._patterns[pattern_type] = pattern_class
        logger.info(f"Registered pattern: {pattern_type} -> {pattern_class.__name__}")
    
    @classmethod
    def create_pattern(cls, pattern_type: str) -> BaseTestPattern:
        """Create a test pattern instance"""
        if pattern_type not in cls._patterns:
            raise ValueError(f"Unknown pattern type: {pattern_type}")
        
        pattern_class = cls._patterns[pattern_type]
        return pattern_class()
    
    @classmethod
    def get_available_patterns(cls) -> List[str]:
        """Get list of available pattern types"""
        return list(cls._patterns.keys())


class ParameterGeneratorFactory:
    """Factory for creating parameter generators"""
    
    _generators: Dict[str, Type[BaseParameterGenerator]] = {}
    
    @classmethod
    def register_generator(
        cls, 
        param_type: str, 
        generator_class: Type[BaseParameterGenerator]
    ):
        """Register a parameter generator class"""
        cls._generators[param_type] = generator_class
        logger.info(f"Registered parameter generator: {param_type} -> {generator_class.__name__}")
    
    @classmethod
    def create_generator(cls, param_type: str) -> BaseParameterGenerator:
        """Create a parameter generator instance"""
        if param_type not in cls._generators:
            raise ValueError(f"Unknown parameter type: {param_type}")
        
        generator_class = cls._generators[param_type]
        return generator_class()
    
    @classmethod
    def get_available_types(cls) -> List[str]:
        """Get list of available parameter types"""
        return list(cls._generators.keys())


class TestValidatorFactory:
    """Factory for creating test validators"""
    
    _validators: Dict[str, Type[BaseTestValidator]] = {}
    
    @classmethod
    def register_validator(
        cls, 
        validator_type: str, 
        validator_class: Type[BaseTestValidator]
    ):
        """Register a test validator class"""
        cls._validators[validator_type] = validator_class
        logger.info(f"Registered validator: {validator_type} -> {validator_class.__name__}")
    
    @classmethod
    def create_validator(cls, validator_type: str) -> BaseTestValidator:
        """Create a test validator instance"""
        if validator_type not in cls._validators:
            raise ValueError(f"Unknown validator type: {validator_type}")
        
        validator_class = cls._validators[validator_type]
        return validator_class()
    
    @classmethod
    def get_available_validators(cls) -> List[str]:
        """Get list of available validator types"""
        return list(cls._validators.keys())


class TestOptimizerFactory:
    """Factory for creating test optimizers"""
    
    _optimizers: Dict[str, Type[BaseTestOptimizer]] = {}
    
    @classmethod
    def register_optimizer(
        cls, 
        optimizer_type: str, 
        optimizer_class: Type[BaseTestOptimizer]
    ):
        """Register a test optimizer class"""
        cls._optimizers[optimizer_type] = optimizer_class
        logger.info(f"Registered optimizer: {optimizer_type} -> {optimizer_class.__name__}")
    
    @classmethod
    def create_optimizer(cls, optimizer_type: str) -> BaseTestOptimizer:
        """Create a test optimizer instance"""
        if optimizer_type not in cls._optimizers:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        
        optimizer_class = cls._optimizers[optimizer_type]
        return optimizer_class()
    
    @classmethod
    def get_available_optimizers(cls) -> List[str]:
        """Get list of available optimizer types"""
        return list(cls._optimizers.keys())


class ComponentRegistry:
    """Registry for all test generation components"""
    
    def __init__(self):
        self.generator_factory = TestGeneratorFactory()
        self.pattern_factory = TestPatternFactory()
        self.parameter_factory = ParameterGeneratorFactory()
        self.validator_factory = TestValidatorFactory()
        self.optimizer_factory = TestOptimizerFactory()
    
    def register_all_components(self):
        """Register all available components"""
        self._register_default_generators()
        self._register_default_patterns()
        self._register_default_parameter_generators()
        self._register_default_validators()
        self._register_default_optimizers()
    
    def _register_default_generators(self):
        """Register default test generators"""
        # This will be populated with actual generator classes
        pass
    
    def _register_default_patterns(self):
        """Register default test patterns"""
        # This will be populated with actual pattern classes
        pass
    
    def _register_default_parameter_generators(self):
        """Register default parameter generators"""
        # This will be populated with actual parameter generator classes
        pass
    
    def _register_default_validators(self):
        """Register default validators"""
        # This will be populated with actual validator classes
        pass
    
    def _register_default_optimizers(self):
        """Register default optimizers"""
        # This will be populated with actual optimizer classes
        pass
    
    def get_component_info(self) -> Dict[str, Any]:
        """Get information about all registered components"""
        return {
            "generators": {
                "available": self.generator_factory.get_available_generators(),
                "count": len(self.generator_factory.get_available_generators())
            },
            "patterns": {
                "available": self.pattern_factory.get_available_patterns(),
                "count": len(self.pattern_factory.get_available_patterns())
            },
            "parameter_generators": {
                "available": self.parameter_factory.get_available_types(),
                "count": len(self.parameter_factory.get_available_types())
            },
            "validators": {
                "available": self.validator_factory.get_available_validators(),
                "count": len(self.validator_factory.get_available_validators())
            },
            "optimizers": {
                "available": self.optimizer_factory.get_available_optimizers(),
                "count": len(self.optimizer_factory.get_available_optimizers())
            }
        }


# Global registry instance
component_registry = ComponentRegistry()
