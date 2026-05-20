"""
Statistics helper utilities for calculating aggregations and metrics.
"""

from typing import Dict, Any, List, Optional, Callable
from sqlalchemy.orm import Query
from sqlalchemy import func
import logging

logger = logging.getLogger(__name__)


def calculate_basic_stats(
    items: List[Any],
    value_field: str,
    include_count: bool = True
) -> Dict[str, Any]:
    """
    Calculate basic statistics from a list of items.
    
    Args:
        items: List of items to calculate stats from
        value_field: Name of the field to calculate stats on
        include_count: Whether to include count in results
        
    Returns:
        Dictionary with statistics (count, sum, avg, min, max)
    """
    if not items:
        stats = {}
        if include_count:
            stats['count'] = 0
        return stats
    
    values = []
    for item in items:
        if hasattr(item, value_field):
            value = getattr(item, value_field)
            if value is not None:
                values.append(value)
    
    if not values:
        stats = {}
        if include_count:
            stats['count'] = len(items)
        return stats
    
    stats = {
        'sum': sum(values),
        'average': sum(values) / len(values),
        'min': min(values),
        'max': max(values)
    }
    
    if include_count:
        stats['count'] = len(items)
    
    return stats


def calculate_field_stats(
    items: List[Any],
    field_configs: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Calculate statistics for multiple fields.
    
    Args:
        items: List of items to calculate stats from
        field_configs: Dictionary mapping field names to config
            Example: {
                'vote_count': {'type': 'sum'},
                'score': {'type': 'avg', 'round': 2}
            }
        
    Returns:
        Dictionary with statistics for each field
    """
    stats = {}
    
    for field_name, config in field_configs.items():
        stat_type = config.get('type', 'sum')
        round_digits = config.get('round')
        
        values = []
        for item in items:
            if hasattr(item, field_name):
                value = getattr(item, field_name)
                if value is not None:
                    values.append(value)
        
        if not values:
            stats[field_name] = 0
            continue
        
        if stat_type == 'sum':
            result = sum(values)
        elif stat_type == 'avg':
            result = sum(values) / len(values)
        elif stat_type == 'min':
            result = min(values)
        elif stat_type == 'max':
            result = max(values)
        elif stat_type == 'count':
            result = len(values)
        else:
            result = sum(values)
        
        if round_digits is not None:
            result = round(result, round_digits)
        
        stats[field_name] = result
    
    return stats


def count_by_condition(
    items: List[Any],
    condition_func: Callable[[Any], bool]
) -> int:
    """
    Count items that match a condition.
    
    Args:
        items: List of items to count
        condition_func: Function that returns True for items to count
        
    Returns:
        Count of matching items
    """
    return len([item for item in items if condition_func(item)])


def calculate_percentage(
    part: float,
    total: float,
    round_digits: int = 2
) -> float:
    """
    Calculate percentage.
    
    Args:
        part: Part value
        total: Total value
        round_digits: Number of decimal places
        
    Returns:
        Percentage value
    """
    if total == 0:
        return 0.0
    
    percentage = (part / total) * 100
    return round(percentage, round_digits)


def calculate_query_stats(
    query: Query,
    model_class: Any,
    stat_fields: List[str]
) -> Dict[str, Any]:
    """
    Calculate statistics directly from a SQL query using aggregation.
    
    Args:
        query: SQLAlchemy query
        model_class: Model class
        stat_fields: List of field names to calculate stats for
        
    Returns:
        Dictionary with statistics
    """
    stats = {}
    
    for field_name in stat_fields:
        if not hasattr(model_class, field_name):
            continue
        
        attr = getattr(model_class, field_name)
        
        # Calculate multiple aggregations
        result = query.with_entities(
            func.count().label('count'),
            func.sum(attr).label('sum'),
            func.avg(attr).label('avg'),
            func.min(attr).label('min'),
            func.max(attr).label('max')
        ).first()
        
        if result:
            stats[field_name] = {
                'count': result.count or 0,
                'sum': float(result.sum) if result.sum is not None else 0,
                'average': float(result.avg) if result.avg is not None else 0,
                'min': float(result.min) if result.min is not None else 0,
                'max': float(result.max) if result.max is not None else 0
            }
    
    return stats


def group_and_count(
    items: List[Any],
    group_by_field: str
) -> Dict[str, int]:
    """
    Group items by a field and count occurrences.
    
    Args:
        items: List of items to group
        group_by_field: Field name to group by
        
    Returns:
        Dictionary mapping field values to counts
    """
    groups = {}
    
    for item in items:
        if hasattr(item, group_by_field):
            value = getattr(item, group_by_field)
            if value is not None:
                key = str(value)
                groups[key] = groups.get(key, 0) + 1
    
    return groups


def calculate_trend(
    current_value: float,
    previous_value: float,
    round_digits: int = 2
) -> Dict[str, Any]:
    """
    Calculate trend between two values.
    
    Args:
        current_value: Current value
        previous_value: Previous value
        round_digits: Number of decimal places
        
    Returns:
        Dictionary with trend information
    """
    if previous_value == 0:
        change_percent = 100.0 if current_value > 0 else 0.0
    else:
        change_percent = ((current_value - previous_value) / previous_value) * 100
    
    return {
        'current': round(current_value, round_digits),
        'previous': round(previous_value, round_digits),
        'change': round(current_value - previous_value, round_digits),
        'change_percent': round(change_percent, round_digits),
        'direction': 'up' if change_percent > 0 else 'down' if change_percent < 0 else 'stable'
    }






