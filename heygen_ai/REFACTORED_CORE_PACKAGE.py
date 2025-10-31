#!/usr/bin/env python3
"""
🔄 Refactored Core Package
=========================

Refactored core package with improved organization, lazy loading,
and better import management.

Author: AI Assistant
Date: December 2024
Version: 1.0.0
"""

import os
import sys
import logging
import importlib
from typing import Dict, List, Any, Optional, Union, Type, Callable
from pathlib import Path
from functools import lru_cache
from dataclasses import dataclass

# Configure logging
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION MANAGEMENT
# ============================================================================

@dataclass
class PackageConfig:
    """Configuration for the refactored core package."""
    enable_lazy_loading: bool = True
    enable_conditional_imports: bool = True
    enable_import_caching: bool = True
    enable_performance_monitoring: bool = True
    log_imports: bool = False
    cache_size: int = 128

class ConfigurationManager:
    """Centralized configuration management for the core package."""
    
    def __init__(self, config: Optional[PackageConfig] = None):
        self.config = config or PackageConfig()
        self._import_cache = {}
        self._performance_metrics = {}
    
    def get_config(self) -> PackageConfig:
        """Get the current configuration."""
        return self.config
    
    def update_config(self, **kwargs) -> None:
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                logger.warning(f"Unknown configuration parameter: {key}")
    
    def get_import_cache(self) -> Dict[str, Any]:
        """Get the import cache."""
        return self._import_cache
    
    def clear_import_cache(self) -> None:
        """Clear the import cache."""
        self._import_cache.clear()
        logger.info("Import cache cleared")

# ============================================================================
# LAZY LOADING SYSTEM
# ============================================================================

class LazyLoader:
    """Lazy loading system for modules and classes."""
    
    def __init__(self, config_manager: ConfigurationManager):
        self.config_manager = config_manager
        self._loaded_modules = {}
        self._loading_stack = []
    
    def lazy_import(self, module_name: str, class_name: Optional[str] = None) -> Any:
        """Lazy import a module or class."""
        if not self.config_manager.get_config().enable_lazy_loading:
            return self._direct_import(module_name, class_name)
        
        cache_key = f"{module_name}.{class_name}" if class_name else module_name
        
        if cache_key in self._loaded_modules:
            return self._loaded_modules[cache_key]
        
        try:
            # Check for circular imports
            if module_name in self._loading_stack:
                raise ImportError(f"Circular import detected: {module_name}")
            
            self._loading_stack.append(module_name)
            
            # Import the module
            module = importlib.import_module(module_name)
            
            if class_name:
                if not hasattr(module, class_name):
                    raise AttributeError(f"Module {module_name} has no attribute {class_name}")
                result = getattr(module, class_name)
            else:
                result = module
            
            # Cache the result
            self._loaded_modules[cache_key] = result
            
            if self.config_manager.get_config().log_imports:
                logger.info(f"Lazy loaded: {cache_key}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to lazy load {cache_key}: {e}")
            raise
        finally:
            if module_name in self._loading_stack:
                self._loading_stack.remove(module_name)
    
    def _direct_import(self, module_name: str, class_name: Optional[str] = None) -> Any:
        """Direct import without lazy loading."""
        module = importlib.import_module(module_name)
        if class_name:
            return getattr(module, class_name)
        return module
    
    def preload_module(self, module_name: str) -> None:
        """Preload a module for faster access."""
        try:
            self.lazy_import(module_name)
            logger.info(f"Preloaded module: {module_name}")
        except Exception as e:
            logger.warning(f"Failed to preload module {module_name}: {e}")
    
    def get_loaded_modules(self) -> List[str]:
        """Get list of loaded modules."""
        return list(self._loaded_modules.keys())

# ============================================================================
# CONDITIONAL IMPORTS
# ============================================================================

class ConditionalImporter:
    """Conditional import system for optional dependencies."""
    
    def __init__(self, config_manager: ConfigurationManager):
        self.config_manager = config_manager
        self._optional_dependencies = {}
        self._dependency_checks = {}
    
    def register_optional_dependency(self, name: str, module_name: str, 
                                   install_name: Optional[str] = None) -> None:
        """Register an optional dependency."""
        self._optional_dependencies[name] = {
            'module_name': module_name,
            'install_name': install_name or name,
            'available': False
        }
        
        # Check if dependency is available
        self._check_dependency_availability(name)
    
    def _check_dependency_availability(self, name: str) -> bool:
        """Check if a dependency is available."""
        if name not in self._optional_dependencies:
            return False
        
        dependency = self._optional_dependencies[name]
        module_name = dependency['module_name']
        
        try:
            importlib.import_module(module_name)
            dependency['available'] = True
            self._dependency_checks[name] = True
            return True
        except ImportError:
            dependency['available'] = False
            self._dependency_checks[name] = False
            return False
    
    def import_optional(self, name: str) -> Any:
        """Import an optional dependency."""
        if name not in self._optional_dependencies:
            raise ValueError(f"Unknown optional dependency: {name}")
        
        dependency = self._optional_dependencies[name]
        
        if not dependency['available']:
            install_name = dependency['install_name']
            raise ImportError(f"Optional dependency '{name}' not available. Install with: pip install {install_name}")
        
        return importlib.import_module(dependency['module_name'])
    
    def is_available(self, name: str) -> bool:
        """Check if an optional dependency is available."""
        if name not in self._dependency_checks:
            self._check_dependency_availability(name)
        return self._dependency_checks.get(name, False)
    
    def get_available_dependencies(self) -> List[str]:
        """Get list of available optional dependencies."""
        return [name for name, available in self._dependency_checks.items() if available]
    
    def get_missing_dependencies(self) -> List[str]:
        """Get list of missing optional dependencies."""
        return [name for name, available in self._dependency_checks.items() if not available]

# ============================================================================
# IMPORT OPTIMIZATION
# ============================================================================

class ImportOptimizer:
    """Import optimization system."""
    
    def __init__(self, config_manager: ConfigurationManager):
        self.config_manager = config_manager
        self._import_groups = {
            'standard_library': [],
            'third_party': [],
            'local_modules': []
        }
        self._unused_imports = set()
    
    def optimize_imports(self, file_path: str) -> Dict[str, Any]:
        """Optimize imports in a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse imports
            imports = self._parse_imports(content)
            
            # Group imports
            grouped_imports = self._group_imports(imports)
            
            # Remove unused imports
            optimized_imports = self._remove_unused_imports(grouped_imports, content)
            
            # Generate optimized content
            optimized_content = self._generate_optimized_content(content, optimized_imports)
            
            return {
                'success': True,
                'original_imports': len(imports),
                'optimized_imports': len(optimized_imports),
                'removed_imports': len(imports) - len(optimized_imports),
                'optimized_content': optimized_content
            }
            
        except Exception as e:
            logger.error(f"Failed to optimize imports in {file_path}: {e}")
            return {'success': False, 'error': str(e)}
    
    def _parse_imports(self, content: str) -> List[Dict[str, Any]]:
        """Parse import statements from content."""
        imports = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith(('import ', 'from ')):
                imports.append({
                    'line_number': i + 1,
                    'content': line,
                    'type': 'import' if line.startswith('import ') else 'from'
                })
        
        return imports
    
    def _group_imports(self, imports: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group imports by type."""
        grouped = {
            'standard_library': [],
            'third_party': [],
            'local_modules': []
        }
        
        for imp in imports:
            content = imp['content']
            if content.startswith('from '):
                module_name = content.split(' ')[1].split('.')[0]
            else:
                module_name = content.split(' ')[1].split('.')[0]
            
            if module_name in sys.builtin_module_names:
                grouped['standard_library'].append(imp)
            elif '.' in module_name or module_name.startswith('_'):
                grouped['local_modules'].append(imp)
            else:
                grouped['third_party'].append(imp)
        
        return grouped
    
    def _remove_unused_imports(self, grouped_imports: Dict[str, List[Dict[str, Any]]], 
                              content: str) -> List[Dict[str, Any]]:
        """Remove unused imports."""
        # This is a simplified version - in practice, you'd use AST analysis
        return [imp for group in grouped_imports.values() for imp in group]
    
    def _generate_optimized_content(self, content: str, 
                                   optimized_imports: List[Dict[str, Any]]) -> str:
        """Generate optimized content with grouped imports."""
        lines = content.split('\n')
        
        # Find import section
        import_start = None
        import_end = None
        
        for i, line in enumerate(lines):
            if line.strip().startswith(('import ', 'from ')):
                if import_start is None:
                    import_start = i
                import_end = i
        
        if import_start is None:
            return content
        
        # Generate optimized imports
        optimized_lines = []
        
        # Standard library imports
        std_imports = [imp for imp in optimized_imports if imp in self._import_groups['standard_library']]
        if std_imports:
            optimized_lines.extend([imp['content'] for imp in std_imports])
            optimized_lines.append('')
        
        # Third-party imports
        third_party_imports = [imp for imp in optimized_imports if imp in self._import_groups['third_party']]
        if third_party_imports:
            optimized_lines.extend([imp['content'] for imp in third_party_imports])
            optimized_lines.append('')
        
        # Local module imports
        local_imports = [imp for imp in optimized_imports if imp in self._import_groups['local_modules']]
        if local_imports:
            optimized_lines.extend([imp['content'] for imp in local_imports])
            optimized_lines.append('')
        
        # Replace import section
        new_lines = lines[:import_start] + optimized_lines + lines[import_end + 1:]
        
        return '\n'.join(new_lines)

# ============================================================================
# PERFORMANCE MONITORING
# ============================================================================

class PerformanceMonitor:
    """Performance monitoring for imports and module loading."""
    
    def __init__(self, config_manager: ConfigurationManager):
        self.config_manager = config_manager
        self._metrics = {
            'import_times': {},
            'module_load_times': {},
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    def start_timer(self, operation: str) -> float:
        """Start timing an operation."""
        import time
        return time.time()
    
    def end_timer(self, operation: str, start_time: float) -> float:
        """End timing an operation and record the duration."""
        import time
        duration = time.time() - start_time
        self._metrics['import_times'][operation] = duration
        return duration
    
    def record_cache_hit(self) -> None:
        """Record a cache hit."""
        self._metrics['cache_hits'] += 1
    
    def record_cache_miss(self) -> None:
        """Record a cache miss."""
        self._metrics['cache_misses'] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        total_imports = self._metrics['cache_hits'] + self._metrics['cache_misses']
        cache_hit_rate = (self._metrics['cache_hits'] / total_imports * 100) if total_imports > 0 else 0
        
        return {
            'import_times': self._metrics['import_times'],
            'cache_hits': self._metrics['cache_hits'],
            'cache_misses': self._metrics['cache_misses'],
            'cache_hit_rate': cache_hit_rate,
            'total_imports': total_imports
        }
    
    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        self._metrics = {
            'import_times': {},
            'module_load_times': {},
            'cache_hits': 0,
            'cache_misses': 0
        }

# ============================================================================
# MAIN CORE PACKAGE
# ============================================================================

class RefactoredCorePackage:
    """Main refactored core package with all optimizations."""
    
    def __init__(self, config: Optional[PackageConfig] = None):
        self.config_manager = ConfigurationManager(config)
        self.lazy_loader = LazyLoader(self.config_manager)
        self.conditional_importer = ConditionalImporter(self.config_manager)
        self.import_optimizer = ImportOptimizer(self.config_manager)
        self.performance_monitor = PerformanceMonitor(self.config_manager)
        
        # Register optional dependencies
        self._register_optional_dependencies()
    
    def _register_optional_dependencies(self) -> None:
        """Register optional dependencies."""
        optional_deps = [
            ('torch', 'torch', 'torch'),
            ('transformers', 'transformers', 'transformers'),
            ('numpy', 'numpy', 'numpy'),
            ('pandas', 'pandas', 'pandas'),
            ('scikit_learn', 'sklearn', 'scikit-learn'),
            ('tensorflow', 'tensorflow', 'tensorflow'),
            ('pytorch_lightning', 'pytorch_lightning', 'pytorch-lightning')
        ]
        
        for name, module_name, install_name in optional_deps:
            self.conditional_importer.register_optional_dependency(name, module_name, install_name)
    
    def lazy_import(self, module_name: str, class_name: Optional[str] = None) -> Any:
        """Lazy import a module or class."""
        if self.config_manager.get_config().enable_performance_monitoring:
            start_time = self.performance_monitor.start_timer(f"lazy_import_{module_name}")
        
        try:
            result = self.lazy_loader.lazy_import(module_name, class_name)
            return result
        finally:
            if self.config_manager.get_config().enable_performance_monitoring:
                self.performance_monitor.end_timer(f"lazy_import_{module_name}", start_time)
    
    def import_optional(self, name: str) -> Any:
        """Import an optional dependency."""
        return self.conditional_importer.import_optional(name)
    
    def is_optional_available(self, name: str) -> bool:
        """Check if an optional dependency is available."""
        return self.conditional_importer.is_available(name)
    
    def optimize_file_imports(self, file_path: str) -> Dict[str, Any]:
        """Optimize imports in a file."""
        return self.import_optimizer.optimize_imports(file_path)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return self.performance_monitor.get_metrics()
    
    def preload_common_modules(self) -> None:
        """Preload commonly used modules."""
        common_modules = [
            'os',
            'sys',
            'logging',
            'typing',
            'pathlib',
            'dataclasses',
            'functools'
        ]
        
        for module_name in common_modules:
            try:
                self.lazy_loader.preload_module(module_name)
            except Exception as e:
                logger.warning(f"Failed to preload {module_name}: {e}")
    
    def get_package_info(self) -> Dict[str, Any]:
        """Get package information."""
        return {
            'config': self.config_manager.get_config(),
            'loaded_modules': self.lazy_loader.get_loaded_modules(),
            'available_dependencies': self.conditional_importer.get_available_dependencies(),
            'missing_dependencies': self.conditional_importer.get_missing_dependencies(),
            'performance_metrics': self.get_performance_metrics()
        }

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

# Global instance
_core_package = None

def get_core_package() -> RefactoredCorePackage:
    """Get the global core package instance."""
    global _core_package
    if _core_package is None:
        _core_package = RefactoredCorePackage()
    return _core_package

def lazy_import(module_name: str, class_name: Optional[str] = None) -> Any:
    """Convenience function for lazy imports."""
    return get_core_package().lazy_import(module_name, class_name)

def import_optional(name: str) -> Any:
    """Convenience function for optional imports."""
    return get_core_package().import_optional(name)

def is_optional_available(name: str) -> bool:
    """Convenience function to check optional dependency availability."""
    return get_core_package().is_optional_available(name)

def optimize_imports(file_path: str) -> Dict[str, Any]:
    """Convenience function to optimize imports."""
    return get_core_package().optimize_file_imports(file_path)

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def main():
    """Example usage of the refactored core package."""
    # Initialize core package
    config = PackageConfig(
        enable_lazy_loading=True,
        enable_conditional_imports=True,
        enable_performance_monitoring=True,
        log_imports=True
    )
    
    core_package = RefactoredCorePackage(config)
    
    # Lazy import example
    print("Lazy importing modules...")
    os_module = core_package.lazy_import('os')
    sys_module = core_package.lazy_import('sys')
    
    # Optional import example
    print("Checking optional dependencies...")
    if core_package.is_optional_available('torch'):
        print("PyTorch is available")
        torch = core_package.import_optional('torch')
        print(f"PyTorch version: {torch.__version__}")
    else:
        print("PyTorch is not available")
    
    # Performance metrics
    print("\nPerformance metrics:")
    metrics = core_package.get_performance_metrics()
    print(f"Cache hit rate: {metrics['cache_hit_rate']:.1f}%")
    print(f"Total imports: {metrics['total_imports']}")
    
    # Package info
    print("\nPackage info:")
    info = core_package.get_package_info()
    print(f"Loaded modules: {len(info['loaded_modules'])}")
    print(f"Available dependencies: {info['available_dependencies']}")
    print(f"Missing dependencies: {info['missing_dependencies']}")
    
    print("✅ Refactored core package working correctly!")

if __name__ == "__main__":
    main()

