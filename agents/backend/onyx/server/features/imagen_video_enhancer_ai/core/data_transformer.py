"""
Data Transformer
================

Advanced data transformation system.
"""

import logging
from typing import Dict, Any, Optional, List, Callable, Type
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class TransformDirection(Enum):
    """Transformation direction."""
    IN = "in"  # Transform input data
    OUT = "out"  # Transform output data
    BOTH = "both"  # Transform both directions


@dataclass
class TransformRule:
    """Transformation rule."""
    field_name: str
    transformer: Callable[[Any], Any]
    direction: TransformDirection = TransformDirection.BOTH
    condition: Optional[Callable[[Any], bool]] = None


class DataTransformer:
    """Data transformer with multiple rules."""
    
    def __init__(self, name: str = "DataTransformer"):
        """
        Initialize data transformer.
        
        Args:
            name: Transformer name
        """
        self.name = name
        self.rules: List[TransformRule] = []
    
    def add_rule(
        self,
        field_name: str,
        transformer: Callable[[Any], Any],
        direction: TransformDirection = TransformDirection.BOTH,
        condition: Optional[Callable[[Any], bool]] = None
    ) -> "DataTransformer":
        """
        Add transformation rule.
        
        Args:
            field_name: Field name to transform
            transformer: Transformation function
            direction: Transformation direction
            condition: Optional condition for transformation
            
        Returns:
            Self for chaining
        """
        rule = TransformRule(field_name, transformer, direction, condition)
        self.rules.append(rule)
        return self
    
    def transform(
        self,
        data: Dict[str, Any],
        direction: TransformDirection = TransformDirection.BOTH
    ) -> Dict[str, Any]:
        """
        Transform data.
        
        Args:
            data: Data to transform
            direction: Transformation direction
            
        Returns:
            Transformed data
        """
        result = data.copy()
        
        for rule in self.rules:
            if rule.field_name not in result:
                continue
            
            # Check direction
            if direction != TransformDirection.BOTH and rule.direction != direction:
                continue
            
            # Check condition
            if rule.condition:
                if not rule.condition(result[rule.field_name]):
                    continue
            
            # Apply transformation
            try:
                result[rule.field_name] = rule.transformer(result[rule.field_name])
            except Exception as e:
                logger.warning(f"Error transforming field '{rule.field_name}': {e}")
        
        return result


# Common transformers
def to_lowercase(value: Any) -> Any:
    """Transform to lowercase."""
    if isinstance(value, str):
        return value.lower()
    return value


def to_uppercase(value: Any) -> Any:
    """Transform to uppercase."""
    if isinstance(value, str):
        return value.upper()
    return value


def to_datetime(value: Any) -> datetime:
    """Transform to datetime."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    raise ValueError(f"Cannot convert {type(value)} to datetime")


def to_string(value: Any) -> str:
    """Transform to string."""
    return str(value)


def to_int(value: Any) -> int:
    """Transform to integer."""
    return int(value)


def to_float(value: Any) -> float:
    """Transform to float."""
    return float(value)


def strip_whitespace(value: Any) -> Any:
    """Strip whitespace from string."""
    if isinstance(value, str):
        return value.strip()
    return value




