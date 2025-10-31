"""
Test Configuration for Brand Voice AI System
============================================

This module provides configuration settings and utilities for running
comprehensive tests on the Brand Voice AI system.
"""

import os
import tempfile
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import json
import yaml

@dataclass
class TestConfig:
    """Test configuration class"""
    
    # Test execution settings
    test_mode: bool = True
    parallel_execution: bool = False
    max_workers: int = 4
    timeout: int = 300  # 5 minutes
    
    # Performance test settings
    performance_tests: bool = True
    load_tests: bool = True
    stress_tests: bool = False
    memory_profiling: bool = True
    benchmark_tests: bool = True
    
    # Test data settings
    generate_test_data: bool = True
    test_data_size: str = "small"  # small, medium, large
    cleanup_after_tests: bool = True
    
    # Mock settings
    mock_external_apis: bool = True
    mock_database: bool = True
    mock_redis: bool = True
    mock_web3: bool = True
    
    # Output settings
    generate_reports: bool = True
    report_format: str = "json"  # json, html, pdf
    save_logs: bool = True
    log_level: str = "INFO"
    
    # System settings
    use_gpu: bool = False
    max_memory_usage: float = 2048.0  # MB
    max_cpu_usage: float = 80.0  # Percentage
    
    # Test thresholds
    max_execution_time: float = 10.0  # seconds
    max_memory_increase: float = 500.0  # MB
    min_success_rate: float = 0.95  # 95%
    max_error_rate: float = 0.05  # 5%
    
    # Specific test settings
    transformer_tests: bool = True
    computer_vision_tests: bool = True
    sentiment_analysis_tests: bool = True
    voice_cloning_tests: bool = True
    collaboration_tests: bool = True
    automation_tests: bool = True
    blockchain_tests: bool = True
    crisis_management_tests: bool = True
    
    # Integration test settings
    integration_tests: bool = True
    end_to_end_tests: bool = True
    cross_module_tests: bool = True
    
    # Security test settings
    security_tests: bool = True
    authentication_tests: bool = True
    authorization_tests: bool = True
    input_validation_tests: bool = True
    
    # Deployment test settings
    deployment_tests: bool = True
    container_tests: bool = True
    kubernetes_tests: bool = False
    cloud_tests: bool = False
    
    # Monitoring test settings
    monitoring_tests: bool = True
    alerting_tests: bool = True
    metrics_tests: bool = True
    
    # Database settings
    test_database_url: str = "sqlite:///:memory:"
    test_redis_url: str = "redis://localhost:6379/15"
    
    # API settings
    test_api_host: str = "localhost"
    test_api_port: int = 8000
    test_api_timeout: int = 30
    
    # File paths
    test_data_dir: Optional[str] = None
    test_output_dir: Optional[str] = None
    test_reports_dir: Optional[str] = None
    test_logs_dir: Optional[str] = None
    
    def __post_init__(self):
        """Initialize paths after object creation"""
        if self.test_data_dir is None:
            self.test_data_dir = tempfile.mkdtemp(prefix="brand_ai_test_data_")
        
        if self.test_output_dir is None:
            self.test_output_dir = tempfile.mkdtemp(prefix="brand_ai_test_output_")
        
        if self.test_reports_dir is None:
            self.test_reports_dir = tempfile.mkdtemp(prefix="brand_ai_test_reports_")
        
        if self.test_logs_dir is None:
            self.test_logs_dir = tempfile.mkdtemp(prefix="brand_ai_test_logs_")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'test_mode': self.test_mode,
            'parallel_execution': self.parallel_execution,
            'max_workers': self.max_workers,
            'timeout': self.timeout,
            'performance_tests': self.performance_tests,
            'load_tests': self.load_tests,
            'stress_tests': self.stress_tests,
            'memory_profiling': self.memory_profiling,
            'benchmark_tests': self.benchmark_tests,
            'generate_test_data': self.generate_test_data,
            'test_data_size': self.test_data_size,
            'cleanup_after_tests': self.cleanup_after_tests,
            'mock_external_apis': self.mock_external_apis,
            'mock_database': self.mock_database,
            'mock_redis': self.mock_redis,
            'mock_web3': self.mock_web3,
            'generate_reports': self.generate_reports,
            'report_format': self.report_format,
            'save_logs': self.save_logs,
            'log_level': self.log_level,
            'use_gpu': self.use_gpu,
            'max_memory_usage': self.max_memory_usage,
            'max_cpu_usage': self.max_cpu_usage,
            'max_execution_time': self.max_execution_time,
            'max_memory_increase': self.max_memory_increase,
            'min_success_rate': self.min_success_rate,
            'max_error_rate': self.max_error_rate,
            'transformer_tests': self.transformer_tests,
            'computer_vision_tests': self.computer_vision_tests,
            'sentiment_analysis_tests': self.sentiment_analysis_tests,
            'voice_cloning_tests': self.voice_cloning_tests,
            'collaboration_tests': self.collaboration_tests,
            'automation_tests': self.automation_tests,
            'blockchain_tests': self.blockchain_tests,
            'crisis_management_tests': self.crisis_management_tests,
            'integration_tests': self.integration_tests,
            'end_to_end_tests': self.end_to_end_tests,
            'cross_module_tests': self.cross_module_tests,
            'security_tests': self.security_tests,
            'authentication_tests': self.authentication_tests,
            'authorization_tests': self.authorization_tests,
            'input_validation_tests': self.input_validation_tests,
            'deployment_tests': self.deployment_tests,
            'container_tests': self.container_tests,
            'kubernetes_tests': self.kubernetes_tests,
            'cloud_tests': self.cloud_tests,
            'monitoring_tests': self.monitoring_tests,
            'alerting_tests': self.alerting_tests,
            'metrics_tests': self.metrics_tests,
            'test_database_url': self.test_database_url,
            'test_redis_url': self.test_redis_url,
            'test_api_host': self.test_api_host,
            'test_api_port': self.test_api_port,
            'test_api_timeout': self.test_api_timeout,
            'test_data_dir': self.test_data_dir,
            'test_output_dir': self.test_output_dir,
            'test_reports_dir': self.test_reports_dir,
            'test_logs_dir': self.test_logs_dir
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TestConfig':
        """Create config from dictionary"""
        return cls(**config_dict)
    
    @classmethod
    def from_file(cls, config_path: str) -> 'TestConfig':
        """Load config from file"""
        config_path = Path(config_path)
        
        if config_path.suffix == '.json':
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
        elif config_path.suffix in ['.yaml', '.yml']:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        return cls.from_dict(config_dict)
    
    def save_to_file(self, config_path: str):
        """Save config to file"""
        config_path = Path(config_path)
        config_dict = self.to_dict()
        
        if config_path.suffix == '.json':
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        elif config_path.suffix in ['.yaml', '.yml']:
            with open(config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")

# Predefined test configurations
def get_quick_test_config() -> TestConfig:
    """Get configuration for quick tests"""
    config = TestConfig()
    config.test_data_size = "small"
    config.performance_tests = False
    config.load_tests = False
    config.stress_tests = False
    config.memory_profiling = False
    config.benchmark_tests = False
    config.timeout = 60
    return config

def get_comprehensive_test_config() -> TestConfig:
    """Get configuration for comprehensive tests"""
    config = TestConfig()
    config.test_data_size = "large"
    config.performance_tests = True
    config.load_tests = True
    config.stress_tests = True
    config.memory_profiling = True
    config.benchmark_tests = True
    config.timeout = 600  # 10 minutes
    return config

def get_performance_test_config() -> TestConfig:
    """Get configuration for performance tests only"""
    config = TestConfig()
    config.performance_tests = True
    config.load_tests = True
    config.stress_tests = True
    config.memory_profiling = True
    config.benchmark_tests = True
    config.transformer_tests = False
    config.computer_vision_tests = False
    config.sentiment_analysis_tests = False
    config.voice_cloning_tests = False
    config.collaboration_tests = False
    config.automation_tests = False
    config.blockchain_tests = False
    config.crisis_management_tests = False
    config.integration_tests = False
    config.end_to_end_tests = False
    config.cross_module_tests = False
    config.security_tests = False
    config.deployment_tests = False
    config.monitoring_tests = False
    return config

def get_integration_test_config() -> TestConfig:
    """Get configuration for integration tests only"""
    config = TestConfig()
    config.performance_tests = False
    config.load_tests = False
    config.stress_tests = False
    config.memory_profiling = False
    config.benchmark_tests = False
    config.transformer_tests = False
    config.computer_vision_tests = False
    config.sentiment_analysis_tests = False
    config.voice_cloning_tests = False
    config.collaboration_tests = False
    config.automation_tests = False
    config.blockchain_tests = False
    config.crisis_management_tests = False
    config.integration_tests = True
    config.end_to_end_tests = True
    config.cross_module_tests = True
    config.security_tests = False
    config.deployment_tests = False
    config.monitoring_tests = False
    return config

def get_security_test_config() -> TestConfig:
    """Get configuration for security tests only"""
    config = TestConfig()
    config.performance_tests = False
    config.load_tests = False
    config.stress_tests = False
    config.memory_profiling = False
    config.benchmark_tests = False
    config.transformer_tests = False
    config.computer_vision_tests = False
    config.sentiment_analysis_tests = False
    config.voice_cloning_tests = False
    config.collaboration_tests = False
    config.automation_tests = False
    config.blockchain_tests = False
    config.crisis_management_tests = False
    config.integration_tests = False
    config.end_to_end_tests = False
    config.cross_module_tests = False
    config.security_tests = True
    config.authentication_tests = True
    config.authorization_tests = True
    config.input_validation_tests = True
    config.deployment_tests = False
    config.monitoring_tests = False
    return config

def get_deployment_test_config() -> TestConfig:
    """Get configuration for deployment tests only"""
    config = TestConfig()
    config.performance_tests = False
    config.load_tests = False
    config.stress_tests = False
    config.memory_profiling = False
    config.benchmark_tests = False
    config.transformer_tests = False
    config.computer_vision_tests = False
    config.sentiment_analysis_tests = False
    config.voice_cloning_tests = False
    config.collaboration_tests = False
    config.automation_tests = False
    config.blockchain_tests = False
    config.crisis_management_tests = False
    config.integration_tests = False
    config.end_to_end_tests = False
    config.cross_module_tests = False
    config.security_tests = False
    config.deployment_tests = True
    config.container_tests = True
    config.kubernetes_tests = True
    config.cloud_tests = True
    config.monitoring_tests = False
    return config

def get_monitoring_test_config() -> TestConfig:
    """Get configuration for monitoring tests only"""
    config = TestConfig()
    config.performance_tests = False
    config.load_tests = False
    config.stress_tests = False
    config.memory_profiling = False
    config.benchmark_tests = False
    config.transformer_tests = False
    config.computer_vision_tests = False
    config.sentiment_analysis_tests = False
    config.voice_cloning_tests = False
    config.collaboration_tests = False
    config.automation_tests = False
    config.blockchain_tests = False
    config.crisis_management_tests = False
    config.integration_tests = False
    config.end_to_end_tests = False
    config.cross_module_tests = False
    config.security_tests = False
    config.deployment_tests = False
    config.monitoring_tests = True
    config.alerting_tests = True
    config.metrics_tests = True
    return config

# Test environment setup
def setup_test_environment(config: TestConfig) -> Dict[str, str]:
    """Set up test environment based on configuration"""
    # Create directories
    os.makedirs(config.test_data_dir, exist_ok=True)
    os.makedirs(config.test_output_dir, exist_ok=True)
    os.makedirs(config.test_reports_dir, exist_ok=True)
    os.makedirs(config.test_logs_dir, exist_ok=True)
    
    # Set environment variables
    env_vars = {
        'BRAND_AI_TEST_MODE': 'true',
        'BRAND_AI_TEST_DATA_DIR': config.test_data_dir,
        'BRAND_AI_TEST_OUTPUT_DIR': config.test_output_dir,
        'BRAND_AI_TEST_REPORTS_DIR': config.test_reports_dir,
        'BRAND_AI_TEST_LOGS_DIR': config.test_logs_dir,
        'BRAND_AI_MOCK_APIS': str(config.mock_external_apis).lower(),
        'BRAND_AI_MOCK_DATABASE': str(config.mock_database).lower(),
        'BRAND_AI_MOCK_REDIS': str(config.mock_redis).lower(),
        'BRAND_AI_MOCK_WEB3': str(config.mock_web3).lower(),
        'BRAND_AI_USE_GPU': str(config.use_gpu).lower(),
        'BRAND_AI_LOG_LEVEL': config.log_level
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
    
    return env_vars

def teardown_test_environment(config: TestConfig):
    """Tear down test environment"""
    # Remove environment variables
    env_vars = [
        'BRAND_AI_TEST_MODE',
        'BRAND_AI_TEST_DATA_DIR',
        'BRAND_AI_TEST_OUTPUT_DIR',
        'BRAND_AI_TEST_REPORTS_DIR',
        'BRAND_AI_TEST_LOGS_DIR',
        'BRAND_AI_MOCK_APIS',
        'BRAND_AI_MOCK_DATABASE',
        'BRAND_AI_MOCK_REDIS',
        'BRAND_AI_MOCK_WEB3',
        'BRAND_AI_USE_GPU',
        'BRAND_AI_LOG_LEVEL'
    ]
    
    for var in env_vars:
        if var in os.environ:
            del os.environ[var]
    
    # Clean up directories if requested
    if config.cleanup_after_tests:
        import shutil
        
        for dir_path in [
            config.test_data_dir,
            config.test_output_dir,
            config.test_reports_dir,
            config.test_logs_dir
        ]:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path, ignore_errors=True)

# Test data generation configuration
def get_test_data_config(config: TestConfig) -> Dict[str, Any]:
    """Get test data generation configuration"""
    size_multipliers = {
        'small': 1,
        'medium': 5,
        'large': 10
    }
    
    multiplier = size_multipliers.get(config.test_data_size, 1)
    
    return {
        'text_data_size': 10 * multiplier,
        'image_count': 3 * multiplier,
        'audio_duration': 2.0 * multiplier,
        'brand_data_count': 5 * multiplier,
        'workflow_count': 3 * multiplier,
        'crisis_data_count': 2 * multiplier,
        'verification_data_count': 2 * multiplier,
        'performance_data_count': 5 * multiplier,
        'collaboration_data_count': 2 * multiplier
    }

# Test execution configuration
def get_test_execution_config(config: TestConfig) -> Dict[str, Any]:
    """Get test execution configuration"""
    return {
        'parallel_execution': config.parallel_execution,
        'max_workers': config.max_workers,
        'timeout': config.timeout,
        'test_mode': config.test_mode,
        'generate_reports': config.generate_reports,
        'report_format': config.report_format,
        'save_logs': config.save_logs,
        'log_level': config.log_level
    }

# Performance thresholds
def get_performance_thresholds(config: TestConfig) -> Dict[str, float]:
    """Get performance thresholds"""
    return {
        'max_execution_time': config.max_execution_time,
        'max_memory_increase': config.max_memory_increase,
        'min_success_rate': config.min_success_rate,
        'max_error_rate': config.max_error_rate,
        'max_memory_usage': config.max_memory_usage,
        'max_cpu_usage': config.max_cpu_usage
    }

# Test module configuration
def get_test_modules_config(config: TestConfig) -> Dict[str, bool]:
    """Get test modules configuration"""
    return {
        'transformer_tests': config.transformer_tests,
        'computer_vision_tests': config.computer_vision_tests,
        'sentiment_analysis_tests': config.sentiment_analysis_tests,
        'voice_cloning_tests': config.voice_cloning_tests,
        'collaboration_tests': config.collaboration_tests,
        'automation_tests': config.automation_tests,
        'blockchain_tests': config.blockchain_tests,
        'crisis_management_tests': config.crisis_management_tests,
        'integration_tests': config.integration_tests,
        'end_to_end_tests': config.end_to_end_tests,
        'cross_module_tests': config.cross_module_tests,
        'security_tests': config.security_tests,
        'authentication_tests': config.authentication_tests,
        'authorization_tests': config.authorization_tests,
        'input_validation_tests': config.input_validation_tests,
        'deployment_tests': config.deployment_tests,
        'container_tests': config.container_tests,
        'kubernetes_tests': config.kubernetes_tests,
        'cloud_tests': config.cloud_tests,
        'monitoring_tests': config.monitoring_tests,
        'alerting_tests': config.alerting_tests,
        'metrics_tests': config.metrics_tests,
        'performance_tests': config.performance_tests,
        'load_tests': config.load_tests,
        'stress_tests': config.stress_tests,
        'memory_profiling': config.memory_profiling,
        'benchmark_tests': config.benchmark_tests
    }

# Export all functions and classes
__all__ = [
    'TestConfig',
    'get_quick_test_config',
    'get_comprehensive_test_config',
    'get_performance_test_config',
    'get_integration_test_config',
    'get_security_test_config',
    'get_deployment_test_config',
    'get_monitoring_test_config',
    'setup_test_environment',
    'teardown_test_environment',
    'get_test_data_config',
    'get_test_execution_config',
    'get_performance_thresholds',
    'get_test_modules_config'
]
























