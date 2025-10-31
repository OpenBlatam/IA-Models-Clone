from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from typing import Union, List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, List, Dict, Optional
import logging
import asyncio
#!/usr/bin/env python3
"""
Math Operation Definitions
Core data structures and enums for mathematical operations.
"""



class OperationType(Enum):
    """Supported mathematical operations."""
    ADD = "add"
    MULTIPLY = "multiply"
    DIVIDE = "divide"
    POWER = "power"
    SQRT = "sqrt"
    LOG = "log"


class CalculationMethod(Enum):
    """Available calculation methods."""
    BASIC = "basic"
    NUMPY = "numpy"
    MATH = "math"
    DECIMAL = "decimal"


@dataclass
class MathOperation:
    """Mathematical operation definition."""
    operation_type: OperationType
    operands: List[Union[int, float]]
    method: CalculationMethod = CalculationMethod.BASIC
    precision: int = 10
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> Any:
        if self.metadata is None:
            self.metadata = {}


@dataclass
class MathResult:
    """Result of mathematical operation."""
    value: Union[int, float, List[Union[int, float]]]
    operation: MathOperation
    execution_time: float
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    cache_hit: Optional[bool] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self) -> Any:
        if self.metadata is None:
            self.metadata = {} 