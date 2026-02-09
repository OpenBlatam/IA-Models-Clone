"""
Module Loader
Dynamic module loading and initialization
"""

import logging
import os
from typing import List, Optional
from core.module_registry import ModuleRegistry, get_registry

logger = logging.getLogger(__name__)


class ModuleLoader:
    """
    Module loader for dynamic module discovery and loading
    
    Features:
    - Auto-discovery of modules
    - Configuration-based loading
    - Environment-based module selection
    """
    
    def __init__(self, registry: Optional[ModuleRegistry] = None):
        self.registry = registry or get_registry()
        self._loaded_modules: List[str] = []
    
    def load_from_config(self, config: dict) -> None:
        """Load modules from configuration"""
        enabled_modules = config.get("enabled_modules", [])
        module_paths = config.get("module_paths", {})
        
        for module_name in enabled_modules:
            module_path = module_paths.get(module_name, f"modules.{module_name}_module")
            self.load_module(module_path, module_name)
    
    def load_module(self, module_path: str, module_name: Optional[str] = None) -> bool:
        """Load a single module"""
        try:
            module = self.registry.load_module(module_path)
            if module:
                if module_name:
                    logger.info(f"Loaded module {module_name} from {module_path}")
                self._loaded_modules.append(module.name)
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to load module {module_path}: {str(e)}")
            return False
    
    def load_all_from_package(self, package: str = "modules") -> List[str]:
        """Load all modules from a package"""
        discovered = self.registry.discover_modules(package)
        return [module.name for module in discovered]
    
    def load_default_modules(self) -> None:
        """Load default modules"""
        default_modules = [
            "modules.storage_module.StorageModule",
            "modules.cache_module.CacheModule",
            "modules.observability_module.ObservabilityModule",
            "modules.security_module.SecurityModule",
            "modules.messaging_module.MessagingModule",
            "modules.api_module.APIModule",
            "modules.recovery_api_module.RecoveryAPIModule",
        ]
        
        for module_path in default_modules:
            self.load_module(module_path)
    
    def load_from_environment(self) -> None:
        """Load modules based on environment variables"""
        enabled_modules = os.getenv("ENABLED_MODULES", "").split(",")
        enabled_modules = [m.strip() for m in enabled_modules if m.strip()]
        
        if not enabled_modules:
            # Load all default modules
            self.load_default_modules()
        else:
            # Load only specified modules
            for module_name in enabled_modules:
                module_path = f"modules.{module_name}_module"
                self.load_module(module_path, module_name)
    
    def initialize_all(self) -> None:
        """Initialize all loaded modules"""
        self.registry.initialize_all()
    
    def shutdown_all(self) -> None:
        """Shutdown all loaded modules"""
        self.registry.shutdown_all()
    
    def get_loaded_modules(self) -> List[str]:
        """Get list of loaded modules"""
        return self._loaded_modules.copy()


# Global loader instance
_loader: Optional[ModuleLoader] = None


def get_loader() -> ModuleLoader:
    """Get global module loader"""
    global _loader
    if _loader is None:
        _loader = ModuleLoader()
    return _loader















