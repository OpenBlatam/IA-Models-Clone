"""Builder pattern utilities."""

from typing import Any, Dict, Optional, Callable
from copy import deepcopy


class Builder:
    """Generic builder."""
    
    def __init__(self, target_class: type):
        self._target_class = target_class
        self._data: Dict[str, Any] = {}
    
    def set(self, key: str, value: Any) -> 'Builder':
        """
        Set builder value.
        
        Args:
            key: Key
            value: Value
            
        Returns:
            Self for chaining
        """
        self._data[key] = value
        return self
    
    def update(self, data: Dict[str, Any]) -> 'Builder':
        """
        Update builder with dictionary.
        
        Args:
            data: Dictionary of values
            
        Returns:
            Self for chaining
        """
        self._data.update(data)
        return self
    
    def build(self) -> Any:
        """
        Build instance.
        
        Returns:
            Built instance
        """
        return self._target_class(**self._data)
    
    def reset(self) -> 'Builder':
        """
        Reset builder.
        
        Returns:
            Self for chaining
        """
        self._data.clear()
        return self


class FluentBuilder:
    """Fluent builder with method chaining."""
    
    def __init__(self, target_class: type):
        self._target_class = target_class
        self._data: Dict[str, Any] = {}
    
    def __getattr__(self, name: str) -> Callable:
        """Create fluent setter method."""
        def setter(value: Any) -> 'FluentBuilder':
            self._data[name] = value
            return self
        return setter
    
    def build(self) -> Any:
        """
        Build instance.
        
        Returns:
            Built instance
        """
        return self._target_class(**self._data)


class StepBuilder:
    """Step-by-step builder."""
    
    def __init__(self):
        self._steps: list = []
        self._current_step: Optional[Dict] = None
    
    def step(self, name: str) -> 'StepBuilder':
        """
        Start new step.
        
        Args:
            name: Step name
            
        Returns:
            Self for chaining
        """
        if self._current_step:
            self._steps.append(self._current_step)
        self._current_step = {'name': name, 'data': {}}
        return self
    
    def add(self, key: str, value: Any) -> 'StepBuilder':
        """
        Add data to current step.
        
        Args:
            key: Key
            value: Value
            
        Returns:
            Self for chaining
        """
        if self._current_step:
            self._current_step['data'][key] = value
        return self
    
    def build(self, builder_func: Callable) -> Any:
        """
        Build using builder function.
        
        Args:
            builder_func: Function(steps) -> result
            
        Returns:
            Built result
        """
        if self._current_step:
            self._steps.append(self._current_step)
        return builder_func(self._steps)
    
    def get_steps(self) -> list:
        """Get all steps."""
        return deepcopy(self._steps)



