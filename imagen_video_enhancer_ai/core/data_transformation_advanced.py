"""
Advanced Data Transformation System
=====================================

Advanced system for data transformation with pipelines and rules.
"""

import logging
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class TransformDirection(Enum):
    """Transform direction."""
    IN = "in"  # Input transformation
    OUT = "out"  # Output transformation
    BOTH = "both"  # Both directions


@dataclass
class TransformRule:
    """Transform rule definition."""
    field: str
    transformer: Callable
    direction: TransformDirection = TransformDirection.BOTH
    condition: Optional[Callable] = None


@dataclass
class TransformResult:
    """Transform result."""
    data: Dict[str, Any]
    applied_transforms: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdvancedDataTransformer:
    """Advanced data transformer with pipeline support."""
    
    def __init__(self):
        """Initialize advanced data transformer."""
        self.rules: Dict[str, List[TransformRule]] = {}
        self.pipelines: Dict[str, List[Callable]] = {}
        self.transformers: Dict[str, Callable] = {}
        self._register_default_transformers()
    
    def _register_default_transformers(self):
        """Register default transformers."""
        self.transformers['to_lower'] = lambda x: x.lower() if isinstance(x, str) else x
        self.transformers['to_upper'] = lambda x: x.upper() if isinstance(x, str) else x
        self.transformers['to_int'] = lambda x: int(x) if x is not None else None
        self.transformers['to_float'] = lambda x: float(x) if x is not None else None
        self.transformers['to_bool'] = lambda x: bool(x) if x is not None else None
        self.transformers['to_string'] = lambda x: str(x) if x is not None else None
        self.transformers['to_datetime'] = lambda x: datetime.fromisoformat(x) if isinstance(x, str) else x
        self.transformers['strip'] = lambda x: x.strip() if isinstance(x, str) else x
    
    def register_transformer(self, name: str, transformer: Callable):
        """
        Register a transformer function.
        
        Args:
            name: Transformer name
            transformer: Transformer function
        """
        self.transformers[name] = transformer
        logger.info(f"Registered transformer: {name}")
    
    def add_rule(self, endpoint: str, rule: TransformRule):
        """
        Add transform rule for endpoint.
        
        Args:
            endpoint: Endpoint path
            rule: Transform rule
        """
        if endpoint not in self.rules:
            self.rules[endpoint] = []
        self.rules[endpoint].append(rule)
    
    def register_pipeline(self, name: str, transformers: List[Callable]):
        """
        Register a transformation pipeline.
        
        Args:
            name: Pipeline name
            transformers: List of transformer functions
        """
        self.pipelines[name] = transformers
        logger.info(f"Registered pipeline: {name}")
    
    def transform(
        self,
        data: Dict[str, Any],
        direction: TransformDirection = TransformDirection.IN,
        endpoint: Optional[str] = None,
        pipeline: Optional[str] = None
    ) -> TransformResult:
        """
        Transform data.
        
        Args:
            data: Data to transform
            direction: Transform direction
            endpoint: Optional endpoint path
            pipeline: Optional pipeline name
            
        Returns:
            Transform result
        """
        transformed_data = data.copy()
        applied_transforms = []
        
        # Apply pipeline
        if pipeline and pipeline in self.pipelines:
            for transformer in self.pipelines[pipeline]:
                try:
                    transformed_data = transformer(transformed_data)
                    applied_transforms.append(f"pipeline:{pipeline}")
                except Exception as e:
                    logger.warning(f"Pipeline transformer failed: {e}")
        
        # Apply rules
        if endpoint and endpoint in self.rules:
            for rule in self.rules[endpoint]:
                if rule.direction in (direction, TransformDirection.BOTH):
                    if rule.condition is None or rule.condition(transformed_data):
                        field_value = transformed_data.get(rule.field)
                        if field_value is not None:
                            try:
                                transformed_data[rule.field] = rule.transformer(field_value)
                                applied_transforms.append(f"rule:{rule.field}")
                            except Exception as e:
                                logger.warning(f"Transform rule failed for '{rule.field}': {e}")
        
        return TransformResult(
            data=transformed_data,
            applied_transforms=applied_transforms
        )
    
    def transform_field(
        self,
        field_name: str,
        value: Any,
        transformer_name: Optional[str] = None,
        transformer: Optional[Callable] = None
    ) -> Any:
        """
        Transform a single field.
        
        Args:
            field_name: Field name
            value: Field value
            transformer_name: Optional transformer name
            transformer: Optional transformer function
            
        Returns:
            Transformed value
        """
        if transformer:
            return transformer(value)
        elif transformer_name and transformer_name in self.transformers:
            return self.transformers[transformer_name](value)
        return value
    
    def chain_transformers(self, *transformer_names: str) -> Callable:
        """
        Chain multiple transformers.
        
        Args:
            *transformer_names: Transformer names
            
        Returns:
            Chained transformer function
        """
        def chained(value: Any) -> Any:
            result = value
            for name in transformer_names:
                if name in self.transformers:
                    result = self.transformers[name](result)
            return result
        return chained



