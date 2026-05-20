"""
Lens utilities
Functional lenses for immutable updates
"""

from typing import TypeVar, Callable, Any
from copy import deepcopy

T = TypeVar('T')
U = TypeVar('U')


class Lens:
    """
    Lens for getting and setting nested values immutably
    """
    
    def __init__(
        self,
        getter: Callable[[T], U],
        setter: Callable[[T, U], T]
    ):
        self.getter = getter
        self.setter = setter
    
    def get(self, obj: T) -> U:
        """Get value using lens"""
        return self.getter(obj)
    
    def set(self, obj: T, value: U) -> T:
        """Set value using lens (immutable)"""
        return self.setter(obj, value)
    
    def modify(self, obj: T, func: Callable[[U], U]) -> T:
        """Modify value using lens"""
        current = self.get(obj)
        new_value = func(current)
        return self.set(obj, new_value)
    
    def compose(self, other: 'Lens') -> 'Lens':
        """Compose two lenses"""
        def getter(obj: T) -> Any:
            return other.get(self.get(obj))
        
        def setter(obj: T, value: Any) -> T:
            inner = self.get(obj)
            updated_inner = other.set(inner, value)
            return self.set(obj, updated_inner)
        
        return Lens(getter, setter)


def lens(getter: Callable[[T], U], setter: Callable[[T, U], T]) -> Lens:
    """
    Create a lens
    
    Args:
        getter: Function to get value
        setter: Function to set value
    
    Returns:
        Lens instance
    """
    return Lens(getter, setter)


def prop_lens(prop_name: str) -> Lens:
    """
    Create lens for object property
    
    Args:
        prop_name: Property name
    
    Returns:
        Lens for property
    """
    def getter(obj: dict) -> Any:
        return obj.get(prop_name)
    
    def setter(obj: dict, value: Any) -> dict:
        new_obj = deepcopy(obj)
        new_obj[prop_name] = value
        return new_obj
    
    return Lens(getter, setter)


def path_lens(path: str, separator: str = ".") -> Lens:
    """
    Create lens for nested path
    
    Args:
        path: Dot-separated path
        separator: Path separator
    
    Returns:
        Lens for path
    """
    keys = path.split(separator)
    
    def getter(obj: dict) -> Any:
        current = obj
        for key in keys:
            if isinstance(current, dict):
                current = current.get(key)
            else:
                return None
        return current
    
    def setter(obj: dict, value: Any) -> dict:
        new_obj = deepcopy(obj)
        current = new_obj
        
        for i, key in enumerate(keys[:-1]):
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
        return new_obj
    
    return Lens(getter, setter)

