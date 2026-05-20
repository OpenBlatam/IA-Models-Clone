"""
Module Loader for Dynamic Module Loading
Supports lazy loading and module discovery
"""

import importlib
import importlib.util
from typing import Optional, Dict, Any, List, Type
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ModuleLoader:
    """
    Dynamic module loader with lazy loading support.
    Optimizes cold start by loading modules only when needed.
    """
    
    def __init__(self, base_path: Optional[str] = None):
        self.base_path = base_path
        self.loaded_modules: Dict[str, Any] = {}
        self.module_cache: Dict[str, Any] = {}
    
    def load_module(self, module_path: str, lazy: bool = True) -> Optional[Any]:
        """
        Load a module
        
        Args:
            module_path: Full module path (e.g., 'api.routers.analysis_router')
            lazy: If True, return a lazy loader proxy
            
        Returns:
            Loaded module or lazy proxy
        """
        if module_path in self.loaded_modules:
            return self.loaded_modules[module_path]
        
        if lazy:
            return self._create_lazy_proxy(module_path)
        
        try:
            module = importlib.import_module(module_path)
            self.loaded_modules[module_path] = module
            logger.debug(f"Loaded module: {module_path}")
            return module
        except Exception as e:
            logger.error(f"Failed to load module {module_path}: {e}", exc_info=True)
            return None
    
    def _create_lazy_proxy(self, module_path: str):
        """Create a lazy loading proxy for a module"""
        class LazyModule:
            def __init__(self, loader, path):
                self._loader = loader
                self._path = path
                self._module = None
            
            def _ensure_loaded(self):
                if self._module is None:
                    self._module = self._loader._load_module_direct(self._path)
                return self._module
            
            def __getattr__(self, name):
                module = self._ensure_loaded()
                return getattr(module, name)
            
            def __call__(self, *args, **kwargs):
                module = self._ensure_loaded()
                return module(*args, **kwargs)
        
        return LazyModule(self, module_path)
    
    def _load_module_direct(self, module_path: str) -> Optional[Any]:
        """Directly load a module (used by lazy proxy)"""
        if module_path in self.loaded_modules:
            return self.loaded_modules[module_path]
        
        try:
            module = importlib.import_module(module_path)
            self.loaded_modules[module_path] = module
            return module
        except Exception as e:
            logger.error(f"Failed to load module {module_path}: {e}", exc_info=True)
            return None
    
    def load_from_file(self, file_path: str, module_name: Optional[str] = None) -> Optional[Any]:
        """Load module from a file path"""
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return None
        
        module_name = module_name or file_path.stem
        
        try:
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None or spec.loader is None:
                logger.error(f"Failed to create spec for {file_path}")
                return None
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            self.loaded_modules[module_name] = module
            logger.debug(f"Loaded module from file: {file_path}")
            return module
            
        except Exception as e:
            logger.error(f"Failed to load module from file {file_path}: {e}", exc_info=True)
            return None
    
    def discover_modules(
        self,
        package_path: str,
        pattern: str = "*.py",
        recursive: bool = True
    ) -> List[str]:
        """
        Discover modules in a package
        
        Args:
            package_path: Path to package
            pattern: File pattern to match
            recursive: Search recursively
            
        Returns:
            List of module paths
        """
        package = Path(package_path)
        if not package.exists():
            logger.warning(f"Package path {package_path} does not exist")
            return []
        
        modules = []
        
        if recursive:
            for py_file in package.rglob(pattern):
                if py_file.name.startswith("_"):
                    continue
                
                # Convert file path to module path
                relative_path = py_file.relative_to(package.parent)
                module_path = str(relative_path.with_suffix("")).replace("/", ".").replace("\\", ".")
                modules.append(module_path)
        else:
            for py_file in package.glob(pattern):
                if py_file.name.startswith("_"):
                    continue
                
                module_path = f"{package_path}.{py_file.stem}"
                modules.append(module_path)
        
        return modules
    
    def preload_modules(self, module_paths: List[str]):
        """Preload a list of modules"""
        for module_path in module_paths:
            self.load_module(module_path, lazy=False)
    
    def clear_cache(self):
        """Clear module cache"""
        self.loaded_modules.clear()
        self.module_cache.clear()
        logger.info("Module cache cleared")


# Global module loader
_module_loader: Optional[ModuleLoader] = None


def get_module_loader(base_path: Optional[str] = None) -> ModuleLoader:
    """Get or create global module loader"""
    global _module_loader
    if _module_loader is None:
        _module_loader = ModuleLoader(base_path)
    return _module_loader















