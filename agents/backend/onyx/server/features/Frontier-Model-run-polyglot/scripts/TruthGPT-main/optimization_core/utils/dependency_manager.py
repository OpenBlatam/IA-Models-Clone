"""
Dependency management utilities for optimization_core.

Provides utilities for managing dependencies and versions.
"""
import logging
import importlib
import pkg_resources
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Dependency:
    """Dependency information."""
    name: str
    version: str
    required: bool = True
    installed: bool = False
    installed_version: Optional[str] = None
    
    def __str__(self) -> str:
        """String representation."""
        status = "✓" if self.installed else "✗"
        if self.installed and self.installed_version:
            return f"{status} {self.name}=={self.version} (installed: {self.installed_version})"
        return f"{status} {self.name}=={self.version}"


class DependencyManager:
    """Manager for dependencies."""
    
    def __init__(self):
        """Initialize dependency manager."""
        self.dependencies: Dict[str, Dependency] = {}
        self._check_installed()
    
    def register(
        self,
        name: str,
        version: str,
        required: bool = True
    ):
        """
        Register a dependency.
        
        Args:
            name: Package name
            version: Required version
            required: Whether dependency is required
        """
        self.dependencies[name] = Dependency(
            name=name,
            version=version,
            required=required
        )
        self._check_installed()
    
    def _check_installed(self):
        """Check which dependencies are installed."""
        for dep in self.dependencies.values():
            try:
                dist = pkg_resources.get_distribution(dep.name)
                dep.installed = True
                dep.installed_version = dist.version
            except pkg_resources.DistributionNotFound:
                dep.installed = False
                dep.installed_version = None
    
    def check_all(self) -> Tuple[bool, List[str]]:
        """
        Check all dependencies.
        
        Returns:
            Tuple of (all_ok, missing_deps)
        """
        missing = []
        
        for dep in self.dependencies.values():
            if dep.required and not dep.installed:
                missing.append(f"{dep.name}=={dep.version}")
        
        return len(missing) == 0, missing
    
    def get_missing(self) -> List[Dependency]:
        """
        Get missing required dependencies.
        
        Returns:
            List of missing dependencies
        """
        return [dep for dep in self.dependencies.values() if dep.required and not dep.installed]
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get dependency status.
        
        Returns:
            Status dictionary
        """
        all_deps = list(self.dependencies.values())
        installed = [dep for dep in all_deps if dep.installed]
        missing = [dep for dep in all_deps if not dep.installed and dep.required]
        
        return {
            "total": len(all_deps),
            "installed": len(installed),
            "missing": len(missing),
            "dependencies": {dep.name: {
                "version": dep.version,
                "installed": dep.installed,
                "installed_version": dep.installed_version,
                "required": dep.required,
            } for dep in all_deps}
        }
    
    def import_safe(
        self,
        module_name: str,
        package_name: Optional[str] = None
    ) -> Optional[Any]:
        """
        Safely import a module.
        
        Args:
            module_name: Module name
            package_name: Package name (for checking dependency)
        
        Returns:
            Imported module or None
        """
        if package_name:
            dep = self.dependencies.get(package_name)
            if dep and dep.required and not dep.installed:
                logger.warning(f"Required dependency '{package_name}' not installed")
                return None
        
        try:
            return importlib.import_module(module_name)
        except ImportError as e:
            logger.warning(f"Failed to import '{module_name}': {e}")
            return None


# Global dependency manager
_global_dependency_manager = DependencyManager()


def get_dependency_manager() -> DependencyManager:
    """Get global dependency manager."""
    return _global_dependency_manager


def register_dependency(name: str, version: str, required: bool = True):
    """
    Register a dependency.
    
    Args:
        name: Package name
        version: Required version
        required: Whether dependency is required
    """
    _global_dependency_manager.register(name, version, required)












