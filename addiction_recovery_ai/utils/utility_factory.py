"""
Utility Factory - Centralized utility creation and management
"""

from typing import Dict, Type, Any, Optional
from utils.categories import get_utility, get_all_utilities, auto_discover_utilities

_utility_instances: Dict[str, Any] = {}


class UtilityFactory:
    """Factory for creating and managing utility instances"""
    
    def __init__(self):
        """Initialize the utility factory"""
        auto_discover_utilities()
        self._instances: Dict[str, Any] = {}
    
    def get_utility(self, category: str, utility_name: str, singleton: bool = True) -> Any:
        """
        Get a utility instance
        
        Args:
            category: Utility category (e.g., 'data', 'async', 'validation')
            utility_name: Name of the utility
            singleton: Whether to return a singleton instance
        
        Returns:
            Utility instance
        """
        key = f"{category}.{utility_name}"
        
        if singleton and key in self._instances:
            return self._instances[key]
        
        try:
            utility = get_utility(category, utility_name)
            if singleton:
                self._instances[key] = utility
            return utility
        except ValueError as e:
            raise ValueError(f"Utility {key} not found: {e}")
    
    def register_utility_instance(self, category: str, utility_name: str, instance: Any) -> None:
        """Register a utility instance"""
        key = f"{category}.{utility_name}"
        self._instances[key] = instance
    
    def list_available_utilities(self) -> Dict[str, Type[Any]]:
        """List all available utilities"""
        return get_all_utilities()
    
    def clear_cache(self) -> None:
        """Clear all cached utility instances"""


_global_factory: Optional[UtilityFactory] = None


def get_utility_factory() -> UtilityFactory:
    """Get the global utility factory instance"""
    global _global_factory
    if _global_factory is None:
        _global_factory = UtilityFactory()
    return _global_factory


def get_utility_instance(category: str, utility_name: str, singleton: bool = True) -> Any:
    """Convenience function to get a utility instance"""
    return get_utility_factory().get_utility(category, utility_name, singleton)



