"""
Transformation Engine for Color Grading AI
===========================================

Advanced data transformation and conversion engine.
"""

import logging
import json
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class TransformationType(Enum):
    """Transformation types."""
    MAP = "map"  # Map values
    FILTER = "filter"  # Filter values
    REDUCE = "reduce"  # Reduce values
    TRANSFORM = "transform"  # Transform structure
    VALIDATE = "validate"  # Validate data
    ENRICH = "enrich"  # Enrich data


@dataclass
class TransformationRule:
    """Transformation rule."""
    name: str
    transformation_type: TransformationType
    function: Callable
    condition: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TransformationResult:
    """Transformation result."""
    success: bool
    input_data: Any
    output_data: Any
    transformations_applied: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class TransformationEngine:
    """
    Transformation engine.
    
    Features:
    - Multiple transformation types
    - Rule-based transformations
    - Pipeline transformations
    - Conditional transformations
    - Data validation
    - Data enrichment
    """
    
    def __init__(self):
        """Initialize transformation engine."""
        self._rules: Dict[str, TransformationRule] = {}
        self._pipelines: Dict[str, List[str]] = {}
    
    def register_rule(
        self,
        name: str,
        transformation_type: TransformationType,
        function: Callable,
        condition: Optional[Callable] = None
    ):
        """
        Register transformation rule.
        
        Args:
            name: Rule name
            transformation_type: Transformation type
            function: Transformation function
            condition: Optional condition function
        """
        rule = TransformationRule(
            name=name,
            transformation_type=transformation_type,
            function=function,
            condition=condition
        )
        self._rules[name] = rule
        logger.info(f"Registered transformation rule: {name}")
    
    def create_pipeline(
        self,
        name: str,
        rule_names: List[str]
    ):
        """
        Create transformation pipeline.
        
        Args:
            name: Pipeline name
            rule_names: List of rule names in order
        """
        self._pipelines[name] = rule_names
        logger.info(f"Created transformation pipeline: {name}")
    
    def transform(
        self,
        data: Any,
        rules: Optional[List[str]] = None,
        pipeline: Optional[str] = None
    ) -> TransformationResult:
        """
        Transform data using rules or pipeline.
        
        Args:
            data: Input data
            rules: Optional list of rule names
            pipeline: Optional pipeline name
            
        Returns:
            Transformation result
        """
        start_time = datetime.now()
        current_data = data
        applied_transformations = []
        errors = []
        
        try:
            # Get rules to apply
            if pipeline:
                rule_names = self._pipelines.get(pipeline, [])
            elif rules:
                rule_names = rules
            else:
                return TransformationResult(
                    success=False,
                    input_data=data,
                    output_data=data,
                    errors=["No rules or pipeline specified"]
                )
            
            # Apply transformations
            for rule_name in rule_names:
                rule = self._rules.get(rule_name)
                if not rule:
                    errors.append(f"Rule not found: {rule_name}")
                    continue
                
                # Check condition
                if rule.condition and not rule.condition(current_data):
                    continue
                
                # Apply transformation
                try:
                    current_data = rule.function(current_data)
                    applied_transformations.append(rule_name)
                except Exception as e:
                    errors.append(f"Error applying rule {rule_name}: {e}")
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return TransformationResult(
                success=len(errors) == 0,
                input_data=data,
                output_data=current_data,
                transformations_applied=applied_transformations,
                errors=errors,
                execution_time=execution_time
            )
        
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return TransformationResult(
                success=False,
                input_data=data,
                output_data=data,
                errors=[str(e)],
                execution_time=execution_time
            )
    
    def map_values(
        self,
        data: List[Any],
        map_function: Callable
    ) -> List[Any]:
        """
        Map values in a list.
        
        Args:
            data: Input data list
            map_function: Mapping function
            
        Returns:
            Mapped data
        """
        return [map_function(item) for item in data]
    
    def filter_values(
        self,
        data: List[Any],
        filter_function: Callable
    ) -> List[Any]:
        """
        Filter values in a list.
        
        Args:
            data: Input data list
            filter_function: Filter function
            
        Returns:
            Filtered data
        """
        return [item for item in data if filter_function(item)]
    
    def reduce_values(
        self,
        data: List[Any],
        reduce_function: Callable,
        initial: Any = None
    ) -> Any:
        """
        Reduce values in a list.
        
        Args:
            data: Input data list
            reduce_function: Reduce function
            initial: Initial value
            
        Returns:
            Reduced value
        """
        result = initial
        for item in data:
            result = reduce_function(result, item)
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get transformation statistics."""
        return {
            "total_rules": len(self._rules),
            "total_pipelines": len(self._pipelines),
            "rule_types": {
                rule_type.value: sum(1 for r in self._rules.values() if r.transformation_type == rule_type)
                for rule_type in TransformationType
            }
        }


