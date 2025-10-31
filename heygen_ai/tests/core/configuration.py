"""
Configuration System for Test Generation
========================================

This module provides a comprehensive configuration system for managing
test generation parameters, settings, and preferences.
"""

import json
import yaml
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging

from .base_architecture import (
    TestGenerationConfig, TestComplexity, TestCategory, TestPriority, TestType
)

logger = logging.getLogger(__name__)


class ConfigFormat(Enum):
    """Configuration file formats"""
    JSON = "json"
    YAML = "yaml"
    TOML = "toml"
    INI = "ini"


class ConfigSource(Enum):
    """Configuration sources"""
    FILE = "file"
    ENVIRONMENT = "environment"
    DEFAULTS = "defaults"
    CLI = "cli"
    API = "api"


@dataclass
class GeneratorSettings:
    """Settings for specific generators"""
    generator_type: str
    enabled: bool = True
    priority: int = 0
    custom_config: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PatternSettings:
    """Settings for test patterns"""
    pattern_type: str
    enabled: bool = True
    weight: float = 1.0
    custom_patterns: List[str] = field(default_factory=list)
    exclusions: List[str] = field(default_factory=list)
    modifications: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParameterSettings:
    """Settings for parameter generation"""
    param_type: str
    enabled: bool = True
    generation_strategy: str = "comprehensive"
    constraints: Dict[str, Any] = field(default_factory=dict)
    custom_generators: List[str] = field(default_factory=list)
    validation_rules: List[str] = field(default_factory=list)


@dataclass
class ValidationSettings:
    """Settings for test validation"""
    validator_type: str
    enabled: bool = True
    strict_mode: bool = False
    custom_rules: List[str] = field(default_factory=list)
    error_threshold: float = 0.1
    warning_threshold: float = 0.2


@dataclass
class OptimizationSettings:
    """Settings for test optimization"""
    optimizer_type: str
    enabled: bool = True
    optimization_level: str = "balanced"
    custom_optimizations: List[str] = field(default_factory=list)
    performance_targets: Dict[str, float] = field(default_factory=dict)


@dataclass
class AdvancedConfig:
    """Advanced configuration options"""
    # Performance settings
    max_parallel_workers: int = 4
    memory_limit_mb: int = 1024
    timeout_seconds: int = 300
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600
    
    # Quality settings
    min_coverage_threshold: float = 0.8
    max_complexity_score: float = 10.0
    quality_gates_enabled: bool = True
    auto_optimization: bool = True
    
    # Output settings
    output_format: str = "pytest"
    include_metadata: bool = True
    generate_documentation: bool = True
    export_metrics: bool = True
    
    # Debugging settings
    debug_mode: bool = False
    verbose_logging: bool = False
    profile_performance: bool = False
    save_intermediate_results: bool = False


class ConfigurationManager:
    """Manages configuration for test generation system"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config = self._load_default_config()
        self.generator_settings: Dict[str, GeneratorSettings] = {}
        self.pattern_settings: Dict[str, PatternSettings] = {}
        self.parameter_settings: Dict[str, ParameterSettings] = {}
        self.validation_settings: Dict[str, ValidationSettings] = {}
        self.optimization_settings: Dict[str, OptimizationSettings] = {}
        self.advanced_config = AdvancedConfig()
        
        if config_path:
            self.load_from_file(config_path)
    
    def _load_default_config(self) -> TestGenerationConfig:
        """Load default configuration"""
        return TestGenerationConfig(
            target_coverage=0.8,
            max_test_cases=100,
            include_edge_cases=True,
            include_performance_tests=False,
            include_security_tests=False,
            complexity_level=TestComplexity.MODERATE,
            naming_convention="descriptive",
            code_style="pytest",
            mock_strategy="comprehensive",
            documentation_level="detailed",
            parallel_generation=True,
            custom_patterns={}
        )
    
    def load_from_file(self, file_path: str):
        """Load configuration from file"""
        try:
            path = Path(file_path)
            if not path.exists():
                logger.warning(f"Configuration file not found: {file_path}")
                return
            
            with open(path, 'r') as f:
                if path.suffix.lower() == '.json':
                    config_data = json.load(f)
                elif path.suffix.lower() in ['.yml', '.yaml']:
                    config_data = yaml.safe_load(f)
                else:
                    logger.error(f"Unsupported configuration format: {path.suffix}")
                    return
            
            self._apply_config_data(config_data)
            logger.info(f"Configuration loaded from: {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration from {file_path}: {e}")
    
    def save_to_file(self, file_path: str, format: ConfigFormat = ConfigFormat.JSON):
        """Save configuration to file"""
        try:
            config_data = self._export_config_data()
            
            with open(file_path, 'w') as f:
                if format == ConfigFormat.JSON:
                    json.dump(config_data, f, indent=2)
                elif format == ConfigFormat.YAML:
                    yaml.dump(config_data, f, default_flow_style=False)
                else:
                    logger.error(f"Unsupported save format: {format}")
                    return
            
            logger.info(f"Configuration saved to: {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration to {file_path}: {e}")
    
    def _apply_config_data(self, config_data: Dict[str, Any]):
        """Apply configuration data to internal structures"""
        # Apply main config
        if 'main_config' in config_data:
            main_config = config_data['main_config']
            for key, value in main_config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
        
        # Apply generator settings
        if 'generators' in config_data:
            for gen_type, settings in config_data['generators'].items():
                self.generator_settings[gen_type] = GeneratorSettings(
                    generator_type=gen_type,
                    **settings
                )
        
        # Apply pattern settings
        if 'patterns' in config_data:
            for pattern_type, settings in config_data['patterns'].items():
                self.pattern_settings[pattern_type] = PatternSettings(
                    pattern_type=pattern_type,
                    **settings
                )
        
        # Apply parameter settings
        if 'parameters' in config_data:
            for param_type, settings in config_data['parameters'].items():
                self.parameter_settings[param_type] = ParameterSettings(
                    param_type=param_type,
                    **settings
                )
        
        # Apply validation settings
        if 'validation' in config_data:
            for validator_type, settings in config_data['validation'].items():
                self.validation_settings[validator_type] = ValidationSettings(
                    validator_type=validator_type,
                    **settings
                )
        
        # Apply optimization settings
        if 'optimization' in config_data:
            for optimizer_type, settings in config_data['optimization'].items():
                self.optimization_settings[optimizer_type] = OptimizationSettings(
                    optimizer_type=optimizer_type,
                    **settings
                )
        
        # Apply advanced config
        if 'advanced' in config_data:
            for key, value in config_data['advanced'].items():
                if hasattr(self.advanced_config, key):
                    setattr(self.advanced_config, key, value)
    
    def _export_config_data(self) -> Dict[str, Any]:
        """Export configuration data to dictionary"""
        return {
            'main_config': asdict(self.config),
            'generators': {k: asdict(v) for k, v in self.generator_settings.items()},
            'patterns': {k: asdict(v) for k, v in self.pattern_settings.items()},
            'parameters': {k: asdict(v) for k, v in self.parameter_settings.items()},
            'validation': {k: asdict(v) for k, v in self.validation_settings.items()},
            'optimization': {k: asdict(v) for k, v in self.optimization_settings.items()},
            'advanced': asdict(self.advanced_config)
        }
    
    def load_from_environment(self):
        """Load configuration from environment variables"""
        env_mappings = {
            'TEST_GEN_TARGET_COVERAGE': ('config', 'target_coverage', float),
            'TEST_GEN_MAX_TEST_CASES': ('config', 'max_test_cases', int),
            'TEST_GEN_INCLUDE_EDGE_CASES': ('config', 'include_edge_cases', bool),
            'TEST_GEN_INCLUDE_PERFORMANCE': ('config', 'include_performance_tests', bool),
            'TEST_GEN_INCLUDE_SECURITY': ('config', 'include_security_tests', bool),
            'TEST_GEN_COMPLEXITY_LEVEL': ('config', 'complexity_level', str),
            'TEST_GEN_NAMING_CONVENTION': ('config', 'naming_convention', str),
            'TEST_GEN_CODE_STYLE': ('config', 'code_style', str),
            'TEST_GEN_MOCK_STRATEGY': ('config', 'mock_strategy', str),
            'TEST_GEN_DOCUMENTATION_LEVEL': ('config', 'documentation_level', str),
            'TEST_GEN_PARALLEL_GENERATION': ('config', 'parallel_generation', bool),
            'TEST_GEN_MAX_WORKERS': ('advanced_config', 'max_parallel_workers', int),
            'TEST_GEN_MEMORY_LIMIT': ('advanced_config', 'memory_limit_mb', int),
            'TEST_GEN_TIMEOUT': ('advanced_config', 'timeout_seconds', int),
            'TEST_GEN_CACHE_ENABLED': ('advanced_config', 'cache_enabled', bool),
            'TEST_GEN_DEBUG_MODE': ('advanced_config', 'debug_mode', bool),
            'TEST_GEN_VERBOSE_LOGGING': ('advanced_config', 'verbose_logging', bool)
        }
        
        for env_var, (target, attr, type_func) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    if type_func == bool:
                        typed_value = value.lower() in ('true', '1', 'yes', 'on')
                    else:
                        typed_value = type_func(value)
                    
                    if target == 'config':
                        setattr(self.config, attr, typed_value)
                    elif target == 'advanced_config':
                        setattr(self.advanced_config, attr, typed_value)
                    
                    logger.info(f"Loaded {env_var} = {typed_value}")
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid value for {env_var}: {value} ({e})")
    
    def get_generator_config(self, generator_type: str) -> Optional[GeneratorSettings]:
        """Get configuration for a specific generator"""
        return self.generator_settings.get(generator_type)
    
    def set_generator_config(self, generator_type: str, settings: GeneratorSettings):
        """Set configuration for a specific generator"""
        self.generator_settings[generator_type] = settings
    
    def get_pattern_config(self, pattern_type: str) -> Optional[PatternSettings]:
        """Get configuration for a specific pattern"""
        return self.pattern_settings.get(pattern_type)
    
    def set_pattern_config(self, pattern_type: str, settings: PatternSettings):
        """Set configuration for a specific pattern"""
        self.pattern_settings[pattern_type] = settings
    
    def get_parameter_config(self, param_type: str) -> Optional[ParameterSettings]:
        """Get configuration for a specific parameter type"""
        return self.parameter_settings.get(param_type)
    
    def set_parameter_config(self, param_type: str, settings: ParameterSettings):
        """Set configuration for a specific parameter type"""
        self.parameter_settings[param_type] = settings
    
    def get_validation_config(self, validator_type: str) -> Optional[ValidationSettings]:
        """Get configuration for a specific validator"""
        return self.validation_settings.get(validator_type)
    
    def set_validation_config(self, validator_type: str, settings: ValidationSettings):
        """Set configuration for a specific validator"""
        self.validation_settings[validator_type] = settings
    
    def get_optimization_config(self, optimizer_type: str) -> Optional[OptimizationSettings]:
        """Get configuration for a specific optimizer"""
        return self.optimization_settings.get(optimizer_type)
    
    def set_optimization_config(self, optimizer_type: str, settings: OptimizationSettings):
        """Set configuration for a specific optimizer"""
        self.optimization_settings[optimizer_type] = settings
    
    def validate_config(self) -> List[str]:
        """Validate current configuration and return any issues"""
        issues = []
        
        # Validate main config
        if self.config.target_coverage < 0 or self.config.target_coverage > 1:
            issues.append("target_coverage must be between 0 and 1")
        
        if self.config.max_test_cases < 1:
            issues.append("max_test_cases must be at least 1")
        
        if self.config.complexity_level not in [level.value for level in TestComplexity]:
            issues.append(f"invalid complexity_level: {self.config.complexity_level}")
        
        # Validate advanced config
        if self.advanced_config.max_parallel_workers < 1:
            issues.append("max_parallel_workers must be at least 1")
        
        if self.advanced_config.memory_limit_mb < 1:
            issues.append("memory_limit_mb must be at least 1")
        
        if self.advanced_config.timeout_seconds < 1:
            issues.append("timeout_seconds must be at least 1")
        
        return issues
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of current configuration"""
        return {
            "main_config": {
                "target_coverage": self.config.target_coverage,
                "max_test_cases": self.config.max_test_cases,
                "complexity_level": self.config.complexity_level,
                "naming_convention": self.config.naming_convention,
                "code_style": self.config.code_style,
                "parallel_generation": self.config.parallel_generation
            },
            "generators": {
                "count": len(self.generator_settings),
                "enabled": sum(1 for s in self.generator_settings.values() if s.enabled)
            },
            "patterns": {
                "count": len(self.pattern_settings),
                "enabled": sum(1 for s in self.pattern_settings.values() if s.enabled)
            },
            "parameters": {
                "count": len(self.parameter_settings),
                "enabled": sum(1 for s in self.parameter_settings.values() if s.enabled)
            },
            "validation": {
                "count": len(self.validation_settings),
                "enabled": sum(1 for s in self.validation_settings.values() if s.enabled)
            },
            "optimization": {
                "count": len(self.optimization_settings),
                "enabled": sum(1 for s in self.optimization_settings.values() if s.enabled)
            },
            "advanced": {
                "max_parallel_workers": self.advanced_config.max_parallel_workers,
                "memory_limit_mb": self.advanced_config.memory_limit_mb,
                "timeout_seconds": self.advanced_config.timeout_seconds,
                "cache_enabled": self.advanced_config.cache_enabled,
                "debug_mode": self.advanced_config.debug_mode
            }
        }


# Global configuration manager instance
config_manager = ConfigurationManager()
