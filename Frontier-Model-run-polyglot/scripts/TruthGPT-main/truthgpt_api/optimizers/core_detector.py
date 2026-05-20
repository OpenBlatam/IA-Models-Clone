"""
Core Detection Module - Detects and configures optimization_core availability.

Single Responsibility: Detect optimization_core and configure imports.
Separated from adapters.py to improve testability and maintainability.

Refactored to:
- Separate path detection from import checking
- Centralize all detection logic
- Support factories and registries
- Improve logging and error handling
"""

import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# Configure logging
logger = logging.getLogger(__name__)

# Module-level state (initialized once)
_OPTIMIZATION_CORE_PATH: Optional[str] = None
_OPTIMIZATION_CORE_AVAILABLE: bool = False
_OPTIMIZATION_CORE_IMPORTS: Dict[str, bool] = {}
_OPTIMIZATION_CORE_FACTORIES: Dict[str, Any] = {}
_OPTIMIZATION_CORE_REGISTRIES: Dict[str, Dict[str, Any]] = {}


def _find_optimization_core_path() -> Optional[str]:
    """
    Find the path to optimization_core module.
    
    Returns:
        Path to optimization_core parent directory, or None if not found
    """
    current_dir = Path(__file__).parent
    possible_paths = [
        current_dir.parent.parent / "optimization_core",
        current_dir.parent.parent.parent / "optimization_core",
        Path(__file__).parent.parent.parent.parent / "optimization_core",
        Path.cwd() / "optimization_core",  # Also try absolute paths
    ]
    
    for path in possible_paths:
        abs_path = path.resolve()
        if abs_path.exists() and (abs_path / "__init__.py").exists():
            logger.info(f"✅ Found optimization_core at: {abs_path}")
            return str(abs_path.parent)
    
    return None


def _setup_optimization_core_path(core_path: str) -> None:
    """
    Add optimization_core path to sys.path if not already present.
    
    Args:
        core_path: Path to optimization_core parent directory
    """
    if core_path not in sys.path:
        sys.path.insert(0, core_path)


def _check_imports() -> Dict[str, bool]:
    """
    Check which optimization_core modules are available.
    
    Returns:
        Dictionary mapping module names to availability status
    """
    imports_status: Dict[str, bool] = {}
    
    # Check tensorflow optimizers
    try:
        from optimization_core.optimizers.tensorflow import (
            tensorflow_inspired_optimizer,
            advanced_tensorflow_optimizer
        )
        imports_status['tensorflow'] = True
        logger.debug("✅ TensorFlow optimizers available")
    except (ImportError, ModuleNotFoundError) as e:
        imports_status['tensorflow'] = False
        logger.debug(f"⚠️ TensorFlow optimizers not available: {e}")
    
    # Check core optimizers
    try:
        from optimization_core.optimizers.core import (
            unified_optimizer,
            generic_optimizer
        )
        imports_status['core'] = True
        logger.debug("✅ Core optimizers available")
    except (ImportError, ModuleNotFoundError) as e:
        imports_status['core'] = False
        logger.debug(f"⚠️ Core optimizers not available: {e}")
    
    # Check truthgpt optimizers
    try:
        from optimization_core.optimizers.truthgpt import (
            truthgpt_dynamo_optimizer,
            truthgpt_inductor_optimizer
        )
        imports_status['truthgpt'] = True
        logger.debug("✅ TruthGPT optimizers available")
    except (ImportError, ModuleNotFoundError) as e:
        imports_status['truthgpt'] = False
        logger.debug(f"⚠️ TruthGPT optimizers not available: {e}")
    
    # Check specialized optimizers
    try:
        from optimization_core.optimizers.quantum import quantum_truthgpt_optimizer
        imports_status['quantum'] = True
        logger.debug("✅ Quantum optimizers available")
    except (ImportError, ModuleNotFoundError):
        imports_status['quantum'] = False
    
    try:
        from optimization_core.optimizers.kv_cache import kv_cache_optimizer
        imports_status['kv_cache'] = True
        logger.debug("✅ KV cache optimizers available")
    except (ImportError, ModuleNotFoundError):
        imports_status['kv_cache'] = False
    
    try:
        from optimization_core.optimizers.production import production_optimizer
        imports_status['production'] = True
        logger.debug("✅ Production optimizers available")
    except (ImportError, ModuleNotFoundError):
        imports_status['production'] = False
    
    return imports_status


def _check_factories() -> Dict[str, Any]:
    """
    Check which optimization_core factory functions are available.
    
    Returns:
        Dictionary mapping factory names to factory objects
    """
    factories: Dict[str, Any] = {}
    
    try:
        from optimization_core.factories import optimizer_factory
        factories['main'] = optimizer_factory
        logger.debug("✅ Optimization factory available")
    except (ImportError, ModuleNotFoundError):
        try:
            from optimization_core.optimizers import create_optimizer
            factories['create_optimizer'] = create_optimizer
            logger.debug("✅ Optimizer creation function available")
        except (ImportError, ModuleNotFoundError):
            logger.debug("⚠️ No factory functions found")
    
    return factories


def _check_registries() -> Dict[str, Dict[str, Any]]:
    """
    Check which optimization_core registries are available.
    
    Returns:
        Dictionary mapping registry names to registry functions
    """
    registries: Dict[str, Dict[str, Any]] = {}
    
    try:
        from optimization_core.optimizers.optimization_cores import (
            list_available_cores,
            get_core_info
        )
        registries['optimization_cores'] = {
            'list': list_available_cores,
            'info': get_core_info
        }
        logger.debug("✅ Optimization cores registry available")
    except (ImportError, ModuleNotFoundError):
        pass
    
    try:
        from optimization_core.optimizers.truthgpt import (
            list_available_truthgpt_optimizers,
            get_truthgpt_optimizer_info
        )
        registries['truthgpt'] = {
            'list': list_available_truthgpt_optimizers,
            'info': get_truthgpt_optimizer_info
        }
        logger.debug("✅ TruthGPT optimizers registry available")
    except (ImportError, ModuleNotFoundError):
        pass
    
    try:
        from optimization_core.optimizers.specialized import (
            list_available_specialized_optimizers,
            get_specialized_optimizer_info
        )
        registries['specialized'] = {
            'list': list_available_specialized_optimizers,
            'info': get_specialized_optimizer_info
        }
        logger.debug("✅ Specialized optimizers registry available")
    except (ImportError, ModuleNotFoundError):
        pass
    
    return registries


def initialize_optimization_core() -> None:
    """
    Initialize optimization_core detection and configuration.
    
    This function should be called once at module import time.
    It detects optimization_core, sets up paths, and checks available imports.
    """
    global _OPTIMIZATION_CORE_PATH, _OPTIMIZATION_CORE_AVAILABLE
    global _OPTIMIZATION_CORE_IMPORTS, _OPTIMIZATION_CORE_FACTORIES, _OPTIMIZATION_CORE_REGISTRIES
    
    core_path = _find_optimization_core_path()
    
    if core_path:
        _OPTIMIZATION_CORE_PATH = core_path
        _setup_optimization_core_path(core_path)
        _OPTIMIZATION_CORE_AVAILABLE = True
        _OPTIMIZATION_CORE_IMPORTS = _check_imports()
        _OPTIMIZATION_CORE_FACTORIES = _check_factories()
        _OPTIMIZATION_CORE_REGISTRIES = _check_registries()
    else:
        _OPTIMIZATION_CORE_AVAILABLE = False
        _OPTIMIZATION_CORE_IMPORTS = {}
        _OPTIMIZATION_CORE_FACTORIES = {}
        _OPTIMIZATION_CORE_REGISTRIES = {}
        logger.info("ℹ️ optimization_core not found, using PyTorch fallback")


def is_optimization_core_available() -> bool:
    """
    Check if optimization_core is available.
    
    Returns:
        True if optimization_core is available, False otherwise
    """
    return _OPTIMIZATION_CORE_AVAILABLE


def get_optimization_core_path() -> Optional[str]:
    """
    Get the path to optimization_core if available.
    
    Returns:
        Path to optimization_core parent directory, or None if not available
    """
    return _OPTIMIZATION_CORE_PATH


def get_available_imports() -> Dict[str, bool]:
    """
    Get status of available optimization_core imports.
    
    Returns:
        Dictionary mapping module names to availability status
    """
    return _OPTIMIZATION_CORE_IMPORTS.copy()


def is_import_available(module_name: str) -> bool:
    """
    Check if a specific optimization_core module is available.
    
    Args:
        module_name: Name of the module ('tensorflow', 'core', 'truthgpt', etc.)
    
    Returns:
        True if the module is available, False otherwise
    """
    return _OPTIMIZATION_CORE_IMPORTS.get(module_name, False)


def get_available_factories() -> Dict[str, Any]:
    """
    Get available factory functions.
    
    Returns:
        Dictionary mapping factory names to factory objects
    """
    return _OPTIMIZATION_CORE_FACTORIES.copy()


def get_available_registries() -> Dict[str, Dict[str, Any]]:
    """
    Get available registry functions.
    
    Returns:
        Dictionary mapping registry names to registry functions
    """
    return _OPTIMIZATION_CORE_REGISTRIES.copy()


# Initialize on module import
initialize_optimization_core()
