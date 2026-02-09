"""
Builder Pattern Utilities for Piel Mejorador AI SAM3
===================================================

Unified builder pattern implementation utilities.
"""

import logging
from typing import TypeVar, Generic, Callable, Optional, Dict, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class BuilderStep:
    """Builder step definition."""
    name: str
    setter: Callable[[Any], None]
    validator: Optional[Callable[[Any], bool]] = None
    default: Any = None
    required: bool = False


class GenericBuilder(Generic[T]):
    """Generic builder with fluent interface."""
    
    def __init__(self, build_func: Callable[[], T]):
        """
        Initialize builder.
        
        Args:
            build_func: Function to build final object
        """
        self._build_func = build_func
        self._values: Dict[str, Any] = {}
        self._steps: Dict[str, BuilderStep] = {}
        self._validated: bool = False
    
    def register_step(
        self,
        name: str,
        setter: Callable[[Any], None],
        validator: Optional[Callable[[Any], bool]] = None,
        default: Any = None,
        required: bool = False
    ) -> "GenericBuilder[T]":
        """
        Register builder step.
        
        Args:
            name: Step name
            setter: Setter function
            validator: Optional validator function
            default: Optional default value
            required: Whether step is required
            
        Returns:
            Self for chaining
        """
        self._steps[name] = BuilderStep(
            name=name,
            setter=setter,
            validator=validator,
            default=default,
            required=required
        )
        return self
    
    def set(self, name: str, value: Any) -> "GenericBuilder[T]":
        """
        Set builder value.
        
        Args:
            name: Step name
            value: Value to set
            
        Returns:
            Self for chaining
        """
        if name not in self._steps:
            logger.warning(f"Unknown builder step: {name}")
            self._values[name] = value
            return self
        
        step = self._steps[name]
        
        # Validate if validator provided
        if step.validator and not step.validator(value):
            raise ValueError(f"Invalid value for {name}: {value}")
        
        # Set value
        step.setter(value)
        self._values[name] = value
        self._validated = False
        
        return self
    
    def validate(self) -> bool:
        """
        Validate builder state.
        
        Returns:
            True if valid
        """
        for name, step in self._steps.items():
            if step.required and name not in self._values:
                if step.default is None:
                    raise ValueError(f"Required step not set: {name}")
                # Use default
                step.setter(step.default)
                self._values[name] = step.default
        
        self._validated = True
        return True
    
    def build(self) -> T:
        """
        Build final object.
        
        Returns:
            Built object
            
        Raises:
            ValueError: If validation fails
        """
        if not self._validated:
            self.validate()
        
        return self._build_func()
    
    def reset(self) -> "GenericBuilder[T]":
        """
        Reset builder state.
        
        Returns:
            Self for chaining
        """
        self._values.clear()
        self._validated = False
        return self
    
    def get_value(self, name: str) -> Any:
        """
        Get builder value.
        
        Args:
            name: Step name
            
        Returns:
            Value or None
        """
        return self._values.get(name)
    
    def has_value(self, name: str) -> bool:
        """
        Check if value is set.
        
        Args:
            name: Step name
            
        Returns:
            True if set
        """
        return name in self._values


class BuilderUtils:
    """Unified builder pattern utilities."""
    
    @staticmethod
    def create_builder(build_func: Callable[[], T]) -> GenericBuilder[T]:
        """
        Create generic builder.
        
        Args:
            build_func: Function to build final object
            
        Returns:
            GenericBuilder instance
        """
        return GenericBuilder(build_func)
    
    @staticmethod
    def create_step(
        name: str,
        setter: Callable[[Any], None],
        validator: Optional[Callable[[Any], bool]] = None,
        default: Any = None,
        required: bool = False
    ) -> BuilderStep:
        """
        Create builder step.
        
        Args:
            name: Step name
            setter: Setter function
            validator: Optional validator
            default: Optional default
            required: Whether required
            
        Returns:
            BuilderStep object
        """
        return BuilderStep(
            name=name,
            setter=setter,
            validator=validator,
            default=default,
            required=required
        )


# Convenience functions
def create_builder(build_func: Callable[[], T]) -> GenericBuilder[T]:
    """Create builder."""
    return BuilderUtils.create_builder(build_func)


def create_step(name: str, setter: Callable[[Any], None], **kwargs) -> BuilderStep:
    """Create builder step."""
    return BuilderUtils.create_step(name, setter, **kwargs)




