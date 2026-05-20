"""
Enhanced Library Configuration - Intelligent Library Management
============================================================

Intelligent configuration and management for enhanced libraries.
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
import yaml

logger = logging.getLogger(__name__)


@dataclass
class LibraryConfig:
    """Library configuration settings."""
    name: str
    version: str
    enabled: bool = True
    priority: int = 0
    dependencies: List[str] = field(default_factory=list)
    environment_variables: Dict[str, str] = field(default_factory=dict)
    configuration: Dict[str, Any] = field(default_factory=dict)
    performance_settings: Dict[str, Any] = field(default_factory=dict)
    health_check_url: Optional[str] = None
    documentation_url: Optional[str] = None


@dataclass
class SystemOptimization:
    """System optimization settings."""
    enable_gpu_acceleration: bool = True
    enable_mkl_optimization: bool = True
    enable_avx_optimization: bool = True
    enable_parallel_processing: bool = True
    max_workers: int = 8
    memory_limit_gb: int = 8
    cache_size_mb: int = 1024
    compression_level: int = 6


class EnhancedLibraryManager:
    """Enhanced library manager with intelligent configuration."""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "library_config.yaml"
        self.libraries: Dict[str, LibraryConfig] = {}
        self.system_optimization = SystemOptimization()
        self._load_configuration()
    
    def _load_configuration(self):
        """Load library configuration."""
        config_path = Path(self.config_file)
        
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                # Load system optimization
                if 'system_optimization' in config_data:
                    opt_data = config_data['system_optimization']
                    self.system_optimization = SystemOptimization(**opt_data)
                
                # Load library configurations
                if 'libraries' in config_data:
                    for lib_name, lib_data in config_data['libraries'].items():
                        self.libraries[lib_name] = LibraryConfig(
                            name=lib_name,
                            **lib_data
                        )
                
                logger.info(f"Loaded library configuration from {config_path}")
                
            except Exception as e:
                logger.error(f"Failed to load configuration: {e}")
                self._create_default_configuration()
        else:
            self._create_default_configuration()
    
    def _create_default_configuration(self):
        """Create default library configuration."""
        logger.info("Creating default library configuration")
        
        # Core libraries
        self.libraries['fastapi'] = LibraryConfig(
            name='fastapi',
            version='0.104.1',
            priority=1,
            configuration={
                'docs_url': '/docs',
                'redoc_url': '/redoc',
                'openapi_url': '/openapi.json'
            }
        )
        
        self.libraries['uvicorn'] = LibraryConfig(
            name='uvicorn',
            version='0.24.0',
            priority=1,
            configuration={
                'host': '0.0.0.0',
                'port': 8001,
                'workers': 1,
                'loop': 'uvloop'
            }
        )
        
        self.libraries['pydantic'] = LibraryConfig(
            name='pydantic',
            version='2.5.0',
            priority=1,
            configuration={
                'validate_assignment': True,
                'use_enum_values': True
            }
        )
        
        # Performance libraries
        self.libraries['redis'] = LibraryConfig(
            name='redis',
            version='5.0.1',
            priority=2,
            configuration={
                'host': 'localhost',
                'port': 6379,
                'db': 0,
                'decode_responses': True
            }
        )
        
        self.libraries['orjson'] = LibraryConfig(
            name='orjson',
            version='3.9.10',
            priority=2,
            configuration={
                'option': 'OPT_SERIALIZE_NUMPY'
            }
        )
        
        # AI libraries
        self.libraries['openai'] = LibraryConfig(
            name='openai',
            version='1.3.0',
            priority=3,
            environment_variables={
                'OPENAI_API_KEY': 'your-api-key-here'
            },
            configuration={
                'model': 'gpt-3.5-turbo',
                'max_tokens': 2000,
                'temperature': 0.7
            }
        )
        
        self.libraries['torch'] = LibraryConfig(
            name='torch',
            version='2.1.1',
            priority=3,
            performance_settings={
                'enable_gpu': True,
                'enable_mkl': True,
                'num_threads': 8
            }
        )
        
        self.libraries['transformers'] = LibraryConfig(
            name='transformers',
            version='4.36.0',
            priority=3,
            configuration={
                'cache_dir': './cache',
                'use_auth_token': False
            }
        )
        
        # Document processing libraries
        self.libraries['pymupdf'] = LibraryConfig(
            name='pymupdf',
            version='1.23.0',
            priority=4,
            configuration={
                'enable_gpu': True
            }
        )
        
        self.libraries['opencv'] = LibraryConfig(
            name='opencv-python',
            version='4.8.1.78',
            priority=4,
            configuration={
                'enable_gpu': True,
                'threads': 8
            }
        )
        
        # NLP libraries
        self.libraries['spacy'] = LibraryConfig(
            name='spacy',
            version='3.7.0',
            priority=4,
            configuration={
                'model': 'en_core_web_sm',
                'disable': ['ner']
            }
        )
        
        self.libraries['nltk'] = LibraryConfig(
            name='nltk',
            version='3.8.0',
            priority=4,
            configuration={
                'data_path': './nltk_data'
            }
        )
        
        # Monitoring libraries
        self.libraries['prometheus'] = LibraryConfig(
            name='prometheus-client',
            version='0.19.0',
            priority=5,
            configuration={
                'port': 9090,
                'path': '/metrics'
            }
        )
        
        self.libraries['sentry'] = LibraryConfig(
            name='sentry-sdk',
            version='1.38.0',
            priority=5,
            environment_variables={
                'SENTRY_DSN': 'your-sentry-dsn-here'
            }
        )
        
        # Save default configuration
        self.save_configuration()
    
    def get_library_config(self, library_name: str) -> Optional[LibraryConfig]:
        """Get library configuration."""
        return self.libraries.get(library_name)
    
    def update_library_config(self, library_name: str, config: LibraryConfig):
        """Update library configuration."""
        self.libraries[library_name] = config
        self.save_configuration()
    
    def enable_library(self, library_name: str):
        """Enable a library."""
        if library_name in self.libraries:
            self.libraries[library_name].enabled = True
            self.save_configuration()
    
    def disable_library(self, library_name: str):
        """Disable a library."""
        if library_name in self.libraries:
            self.libraries[library_name].enabled = False
            self.save_configuration()
    
    def get_enabled_libraries(self) -> List[LibraryConfig]:
        """Get list of enabled libraries."""
        return [lib for lib in self.libraries.values() if lib.enabled]
    
    def get_libraries_by_priority(self) -> List[LibraryConfig]:
        """Get libraries sorted by priority."""
        return sorted(self.libraries.values(), key=lambda x: x.priority)
    
    def apply_system_optimizations(self):
        """Apply system optimizations."""
        logger.info("Applying system optimizations...")
        
        # Set environment variables for optimizations
        if self.system_optimization.enable_gpu_acceleration:
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            os.environ['TORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
        
        if self.system_optimization.enable_mkl_optimization:
            os.environ['MKL_NUM_THREADS'] = str(self.system_optimization.max_workers)
            os.environ['OMP_NUM_THREADS'] = str(self.system_optimization.max_workers)
        
        if self.system_optimization.enable_avx_optimization:
            os.environ['NUMPY_DISABLE_CPU_FEATURES'] = '0'
        
        if self.system_optimization.enable_parallel_processing:
            os.environ['NUMEXPR_NUM_THREADS'] = str(self.system_optimization.max_workers)
            os.environ['OPENBLAS_NUM_THREADS'] = str(self.system_optimization.max_workers)
        
        # Set memory limits
        os.environ['PYTHONHASHSEED'] = '0'
        os.environ['PYTHONUNBUFFERED'] = '1'
        
        logger.info("System optimizations applied")
    
    def configure_library(self, library_name: str, **kwargs):
        """Configure a specific library."""
        if library_name in self.libraries:
            lib_config = self.libraries[library_name]
            
            # Update configuration
            for key, value in kwargs.items():
                if hasattr(lib_config, key):
                    setattr(lib_config, key, value)
                else:
                    lib_config.configuration[key] = value
            
            self.save_configuration()
            logger.info(f"Configured library: {library_name}")
        else:
            logger.warning(f"Library not found: {library_name}")
    
    def get_library_environment_variables(self) -> Dict[str, str]:
        """Get all library environment variables."""
        env_vars = {}
        
        for lib_config in self.libraries.values():
            if lib_config.enabled:
                env_vars.update(lib_config.environment_variables)
        
        return env_vars
    
    def validate_configuration(self) -> List[str]:
        """Validate library configuration."""
        errors = []
        
        for lib_name, lib_config in self.libraries.items():
            if not lib_config.name:
                errors.append(f"Library {lib_name} has no name")
            
            if not lib_config.version:
                errors.append(f"Library {lib_name} has no version")
            
            # Check for required environment variables
            for env_var in lib_config.environment_variables:
                if not os.getenv(env_var) and lib_config.environment_variables[env_var] == 'your-api-key-here':
                    errors.append(f"Library {lib_name} requires environment variable: {env_var}")
        
        return errors
    
    def save_configuration(self):
        """Save library configuration to file."""
        config_data = {
            'system_optimization': {
                'enable_gpu_acceleration': self.system_optimization.enable_gpu_acceleration,
                'enable_mkl_optimization': self.system_optimization.enable_mkl_optimization,
                'enable_avx_optimization': self.system_optimization.enable_avx_optimization,
                'enable_parallel_processing': self.system_optimization.enable_parallel_processing,
                'max_workers': self.system_optimization.max_workers,
                'memory_limit_gb': self.system_optimization.memory_limit_gb,
                'cache_size_mb': self.system_optimization.cache_size_mb,
                'compression_level': self.system_optimization.compression_level
            },
            'libraries': {}
        }
        
        for lib_name, lib_config in self.libraries.items():
            config_data['libraries'][lib_name] = {
                'version': lib_config.version,
                'enabled': lib_config.enabled,
                'priority': lib_config.priority,
                'dependencies': lib_config.dependencies,
                'environment_variables': lib_config.environment_variables,
                'configuration': lib_config.configuration,
                'performance_settings': lib_config.performance_settings,
                'health_check_url': lib_config.health_check_url,
                'documentation_url': lib_config.documentation_url
            }
        
        try:
            with open(self.config_file, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
            
            logger.info(f"Saved library configuration to {self.config_file}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
    
    def get_performance_recommendations(self) -> List[str]:
        """Get performance recommendations based on system."""
        recommendations = []
        
        # Check system capabilities
        try:
            import psutil
            cpu_count = psutil.cpu_count()
            memory_gb = psutil.virtual_memory().total / (1024**3)
            
            if cpu_count >= 8:
                recommendations.append("Consider increasing max_workers for better parallel processing")
            
            if memory_gb >= 16:
                recommendations.append("Enable large memory optimizations for data processing")
            
            if memory_gb >= 32:
                recommendations.append("Consider enabling advanced caching and vector databases")
            
        except ImportError:
            recommendations.append("Install psutil for system monitoring and optimization")
        
        # Check GPU availability
        try:
            import torch
            if torch.cuda.is_available():
                recommendations.append("GPU detected - enable GPU acceleration for AI libraries")
            else:
                recommendations.append("No GPU detected - using CPU-optimized versions")
        except ImportError:
            recommendations.append("Install PyTorch for GPU detection and optimization")
        
        return recommendations
    
    def get_library_stats(self) -> Dict[str, Any]:
        """Get library statistics."""
        total_libraries = len(self.libraries)
        enabled_libraries = len([lib for lib in self.libraries.values() if lib.enabled])
        
        priority_counts = {}
        for lib in self.libraries.values():
            priority_counts[lib.priority] = priority_counts.get(lib.priority, 0) + 1
        
        return {
            'total_libraries': total_libraries,
            'enabled_libraries': enabled_libraries,
            'disabled_libraries': total_libraries - enabled_libraries,
            'priority_distribution': priority_counts,
            'system_optimization': {
                'gpu_acceleration': self.system_optimization.enable_gpu_acceleration,
                'mkl_optimization': self.system_optimization.enable_mkl_optimization,
                'avx_optimization': self.system_optimization.enable_avx_optimization,
                'parallel_processing': self.system_optimization.enable_parallel_processing,
                'max_workers': self.system_optimization.max_workers
            }
        }


# Global library manager instance
_library_manager: Optional[EnhancedLibraryManager] = None


def get_library_manager() -> EnhancedLibraryManager:
    """Get global library manager instance."""
    global _library_manager
    if _library_manager is None:
        _library_manager = EnhancedLibraryManager()
    return _library_manager


def configure_library(library_name: str, **kwargs):
    """Configure a library."""
    manager = get_library_manager()
    manager.configure_library(library_name, **kwargs)


def apply_optimizations():
    """Apply system optimizations."""
    manager = get_library_manager()
    manager.apply_system_optimizations()


def get_library_config(library_name: str) -> Optional[LibraryConfig]:
    """Get library configuration."""
    manager = get_library_manager()
    return manager.get_library_config(library_name)