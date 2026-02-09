"""Aggregation utilities."""

from typing import List, Dict, Callable, Any, Optional
from collections import defaultdict


def group_by(items: List[Dict], key: str) -> Dict[Any, List[Dict]]:
    """
    Group items by key.
    
    Args:
        items: List of dictionaries
        key: Key to group by
        
    Returns:
        Dictionary grouped by key
    """
    grouped = defaultdict(list)
    for item in items:
        grouped[item.get(key)].append(item)
    return dict(grouped)


def aggregate_sum(items: List[Dict], field: str) -> float:
    """
    Sum values of field.
    
    Args:
        items: List of dictionaries
        field: Field to sum
        
    Returns:
        Sum of values
    """
    return sum(item.get(field, 0) for item in items)


def aggregate_avg(items: List[Dict], field: str) -> float:
    """
    Average values of field.
    
    Args:
        items: List of dictionaries
        field: Field to average
        
    Returns:
        Average value
    """
    if not items:
        return 0.0
    return aggregate_sum(items, field) / len(items)


def aggregate_min(items: List[Dict], field: str) -> Optional[Any]:
    """
    Minimum value of field.
    
    Args:
        items: List of dictionaries
        field: Field to find minimum
        
    Returns:
        Minimum value or None
    """
    if not items:
        return None
    values = [item.get(field) for item in items if field in item]
    return min(values) if values else None


def aggregate_max(items: List[Dict], field: str) -> Optional[Any]:
    """
    Maximum value of field.
    
    Args:
        items: List of dictionaries
        field: Field to find maximum
        
    Returns:
        Maximum value or None
    """
    if not items:
        return None
    values = [item.get(field) for item in items if field in item]
    return max(values) if values else None


def aggregate_count(items: List[Dict], field: Optional[str] = None) -> int:
    """
    Count items (optionally by field).
    
    Args:
        items: List of dictionaries
        field: Optional field to count (counts non-None values)
        
    Returns:
        Count
    """
    if field:
        return sum(1 for item in items if item.get(field) is not None)
    return len(items)


def aggregate_by_group(
    items: List[Dict],
    group_key: str,
    aggregations: Dict[str, Callable]
) -> List[Dict]:
    """
    Aggregate items by group with multiple aggregations.
    
    Args:
        items: List of dictionaries
        group_key: Key to group by
        aggregations: Dictionary of field:aggregation_function
        
    Returns:
        List of aggregated dictionaries
    """
    grouped = group_by(items, group_key)
    result = []
    
    for key, group_items in grouped.items():
        agg_dict = {group_key: key}
        for field, agg_func in aggregations.items():
            agg_dict[field] = agg_func(group_items, field)
        result.append(agg_dict)
    
    return result

