"""
Data aggregation and counting helper functions.
Consolidates repetitive patterns of dictionary counting, grouping, and aggregation.
"""

from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from collections import defaultdict
from datetime import datetime, timedelta


def count_by_key(
    items: List[Dict[str, Any]],
    key: str,
    default_value: str = "unknown"
) -> Dict[str, int]:
    """
    Count items by a dictionary key.
    
    Args:
        items: List of dictionaries to count
        key: Key to count by
        default_value: Default value if key is missing
    
    Returns:
        Dictionary with counts by key value
    """
    counts = defaultdict(int)
    for item in items:
        value = item.get(key, default_value)
        counts[value] += 1
    return dict(counts)


def count_by_function(
    items: List[Any],
    key_func: Callable[[Any], str],
    default_value: str = "unknown"
) -> Dict[str, int]:
    """
    Count items by a function that extracts a key.
    
    Args:
        items: List of items to count
        key_func: Function that extracts key from item
        default_value: Default value if key_func returns None/empty
    
    Returns:
        Dictionary with counts by key value
    """
    counts = defaultdict(int)
    for item in items:
        value = key_func(item) or default_value
        counts[value] += 1
    return dict(counts)


def count_by_field(
    items: List[Dict[str, Any]],
    field: str,
    default: str = "unknown"
) -> Dict[str, int]:
    """
    Count items by a field value.
    
    Args:
        items: List of dictionaries to count
        field: Field name to count by
        default: Default value if field is missing
    
    Returns:
        Dictionary with counts by field value
    """
    return count_by_key(items, field, default)


def get_most_common(counts: Dict[str, int]) -> Optional[Tuple[str, int]]:
    """
    Get the most common item from a count dictionary.
    
    Args:
        counts: Dictionary with counts
    
    Returns:
        Tuple of (key, count) for most common item, or None if empty
    """
    if not counts:
        return None
    max_key = max(counts.items(), key=lambda x: x[1])
    return max_key


def get_most_common_key(counts: Dict[str, int]) -> Optional[str]:
    """
    Get the key of the most common item from a count dictionary.
    
    Args:
        counts: Dictionary with counts
    
    Returns:
        Key of most common item, or None if empty
    """
    result = get_most_common(counts)
    return result[0] if result else None


def find_most_common(counts: Dict[str, int]) -> Optional[Tuple[str, int]]:
    """
    Find the most common item from a counts dictionary.
    Alias for get_most_common for backward compatibility.
    
    Args:
        counts: Dictionary with counts
    
    Returns:
        Tuple of (key, count) for most common item, or None if empty
    """
    return get_most_common(counts)


def count_matching(
    items: List[Any],
    condition: Callable[[Any], bool]
) -> int:
    """
    Count items that match a condition.
    More Pythonic and readable than sum(1 for item in items if condition(item)).
    
    Args:
        items: List of items to check
        condition: Function that returns True if item matches condition
    
    Returns:
        Number of items that match the condition
    
    Example:
        >>> items = [{"status": "active"}, {"status": "inactive"}, {"status": "active"}]
        >>> count_matching(items, lambda x: x.get("status") == "active")
        2
    """
    return sum(1 for item in items if condition(item))


def parse_time_range(time_range: str) -> Tuple[datetime, datetime]:
    """
    Parse a time range string into start and end dates.
    
    Args:
        time_range: Time range string (e.g., "1d", "7d", "30d", "90d", "1y", "all")
    
    Returns:
        Tuple of (start_date, end_date)
    """
    end_date = datetime.now()
    
    if time_range == "1d":
        start_date = end_date - timedelta(days=1)
    elif time_range == "7d":
        start_date = end_date - timedelta(days=7)
    elif time_range == "30d":
        start_date = end_date - timedelta(days=30)
    elif time_range == "90d":
        start_date = end_date - timedelta(days=90)
    elif time_range == "1y":
        start_date = end_date - timedelta(days=365)
    elif time_range == "all":
        start_date = datetime(2000, 1, 1)  # All time
    else:
        # Default to 30 days
        start_date = end_date - timedelta(days=30)
    
    return start_date, end_date


def calculate_average_interval(dates: List[datetime]) -> float:
    """
    Calculate average interval between sorted dates.
    
    Args:
        dates: List of sorted datetime objects
    
    Returns:
        Average interval in days, or 0 if insufficient data
    """
    if len(dates) < 2:
        return 0.0
    
    intervals = [(dates[i+1] - dates[i]).days for i in range(len(dates) - 1)]
    return sum(intervals) / len(intervals) if intervals else 0.0


def calculate_intervals(dates: List[Any]) -> List[float]:
    """
    Calculate intervals between consecutive dates.
    
    Args:
        dates: List of date objects (datetime, etc.)
    
    Returns:
        List of interval values in days
    """
    if len(dates) < 2:
        return []
    
    intervals = []
    for i in range(len(dates) - 1):
        interval = (dates[i + 1] - dates[i]).days
        intervals.append(interval)
    
    return intervals


def calculate_frequency_per_month(total: int, days_span: int) -> float:
    """
    Calculate frequency per month from total count and days span.
    
    Args:
        total: Total count
        days_span: Number of days in the period
    
    Returns:
        Frequency per month
    """
    if days_span <= 0:
        return 0.0
    return (total / days_span) * 30


def safe_average(values: List[float], default: float = 0.0) -> float:
    """
    Calculate average of values safely, handling empty lists.
    
    Args:
        values: List of numeric values
        default: Default value if list is empty
    
    Returns:
        Average value or default
    """
    if not values:
        return default
    return sum(values) / len(values)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Divide two numbers safely, handling division by zero.
    
    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value if denominator is zero
    
    Returns:
        Division result or default
    """
    if denominator == 0:
        return default
    return numerator / denominator


def group_by(
    items: List[Dict[str, Any]],
    key_func: Callable[[Dict[str, Any]], str],
    default: str = "unknown"
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group items by a key extracted from each item.
    
    Args:
        items: List of dictionaries to group
        key_func: Function to extract key from each item
        default: Default value if key is missing
    
    Returns:
        Dictionary with lists of items grouped by key
    """
    groups = defaultdict(list)
    for item in items:
        key = key_func(item) if key_func(item) else default
        groups[key].append(item)
    return dict(groups)


def filter_by_date_range(
    items: List[Dict[str, Any]],
    start_date: datetime,
    end_date: datetime,
    date_field: str = "created_at",
    parse_date_func: Optional[Callable[[str], Optional[datetime]]] = None
) -> List[Dict[str, Any]]:
    """
    Filter items by date range.
    
    Args:
        items: List of dictionaries to filter
        start_date: Start date (inclusive)
        end_date: End date (inclusive)
        date_field: Field name containing the date string
        parse_date_func: Optional function to parse date string (defaults to parse_iso_date)
    
    Returns:
        Filtered list of items within date range
    """
    from ..utils.file_helpers import parse_iso_date
    
    if parse_date_func is None:
        parse_date_func = parse_iso_date
    
    filtered = []
    for item in items:
        date_str = item.get(date_field)
        if date_str:
            parsed_date = parse_date_func(date_str)
            if parsed_date and start_date <= parsed_date <= end_date:
                filtered.append(item)
    
    return filtered


def extract_sorted_dates(
    items: List[Dict[str, Any]],
    date_field: str = "created_at",
    parse_date_func: Optional[Callable[[str], Optional[datetime]]] = None
) -> List[datetime]:
    """
    Extract and sort dates from items.
    
    Args:
        items: List of dictionaries with date fields
        date_field: Field name containing the date string
        parse_date_func: Optional function to parse date string (defaults to parse_iso_date)
    
    Returns:
        Sorted list of datetime objects
    """
    from ..utils.file_helpers import parse_iso_date
    
    if parse_date_func is None:
        parse_date_func = parse_iso_date
    
    dates = []
    for item in items:
        date_str = item.get(date_field)
        if date_str:
            parsed_date = parse_date_func(date_str)
            if parsed_date:
                dates.append(parsed_date)
    
    return sorted(dates)


def filter_by_fields(
    items: List[Dict[str, Any]],
    filters: Dict[str, Any],
    default_values: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Filter items by multiple field values.
    
    Args:
        items: List of dictionaries to filter
        filters: Dictionary of field_name -> value to filter by (None values are ignored)
        default_values: Optional dictionary of default values for missing fields
    
    Returns:
        Filtered list of items matching all specified filters
    """
    if not filters:
        return items
    
    default_values = default_values or {}
    filtered = items
    
    for field, value in filters.items():
        if value is None:
            continue
        
        # Handle special case: filtering by value in a list (e.g., tags)
        if field.endswith("_in") and isinstance(value, (list, tuple)):
            # Special handling for "field_in" pattern
            base_field = field[:-3]  # Remove "_in" suffix
            filtered = [
                item for item in filtered
                if item.get(base_field) in value
            ]
        elif isinstance(value, (list, tuple)):
            # Filter where field value is in the provided list
            filtered = [
                item for item in filtered
                if item.get(field, default_values.get(field)) in value
            ]
        else:
            # Standard equality filter
            filtered = [
                item for item in filtered
                if item.get(field, default_values.get(field)) == value
            ]
    
    return filtered


def filter_by_field_contains(
    items: List[Dict[str, Any]],
    field: str,
    value: Any
) -> List[Dict[str, Any]]:
    """
    Filter items where a field contains a value (for list fields).
    
    Args:
        items: List of dictionaries to filter
        field: Field name to check
        value: Value to search for in the field (if field is a list)
    
    Returns:
        Filtered list of items where field contains value
    """
    filtered = []
    for item in items:
        field_value = item.get(field, [])
        if isinstance(field_value, (list, tuple)) and value in field_value:
            filtered.append(item)
    return filtered


def ensure_resource_exists(
    resource_id: str,
    store: Dict[str, Any],
    resource_name: str = "Resource"
) -> None:
    """
    Ensure a resource exists in a store, raising NotFoundError if not.
    
    Args:
        resource_id: ID of the resource to check
        store: Dictionary store to check in
        resource_name: Name of the resource type for error message
    
    Raises:
        NotFoundError: If resource is not found
    """
    if resource_id not in store:
        from ..api.exceptions import NotFoundError
        raise NotFoundError(resource_name, resource_id)


def sort_by_field(
    items: List[Dict[str, Any]],
    field: str,
    reverse: bool = False,
    default_value: Any = None
) -> List[Dict[str, Any]]:
    """
    Sort items by a field value.
    
    Args:
        items: List of dictionaries to sort
        field: Field name to sort by
        reverse: Whether to sort in reverse order
        default_value: Default value if field is missing
    
    Returns:
        Sorted list of items
    """
    return sorted(
        items,
        key=lambda x: x.get(field, default_value),
        reverse=reverse
    )


def sort_by_function(
    items: List[Any],
    key_func: Callable[[Any], Any],
    reverse: bool = False
) -> List[Any]:
    """
    Sort items by a function that extracts a key.
    
    Args:
        items: List of items to sort
        key_func: Function that extracts key from item
        reverse: Whether to sort in reverse order
    
    Returns:
        Sorted list of items
    """
    return sorted(items, key=key_func, reverse=reverse)


def paginate_items(
    items: List[Any],
    offset: int = 0,
    limit: int = 10
) -> Tuple[List[Any], int, int]:
    """
    Paginate a list of items.
    
    Args:
        items: List of items to paginate
        offset: Starting index (0-based)
        limit: Number of items per page
    
    Returns:
        Tuple of (paginated_items, total, page_number)
    """
    total = len(items)
    paginated = items[offset:offset + limit]
    page = (offset // limit) + 1 if limit > 0 else 1
    return paginated, total, page


def find_max_by_key(
    items: Dict[str, Any],
    key_func: Callable[[Any], Any],
    default: Optional[Any] = None
) -> Optional[Tuple[str, Any]]:
    """
    Find the item with maximum value based on a key function.
    
    Args:
        items: Dictionary of items to search
        key_func: Function that extracts the value to compare from each item
        default: Default value if items is empty
    
    Returns:
        Tuple of (key, item) with maximum value, or None if empty
    """
    if not items:
        return default
    
    max_key = max(items.items(), key=lambda x: key_func(x[1]))
    return max_key


def find_min_by_key(
    items: Dict[str, Any],
    key_func: Callable[[Any], Any],
    default: Optional[Any] = None
) -> Optional[Tuple[str, Any]]:
    """
    Find the item with minimum value based on a key function.
    
    Args:
        items: Dictionary of items to search
        key_func: Function that extracts the value to compare from each item
        default: Default value if items is empty
    
    Returns:
        Tuple of (key, item) with minimum value, or None if empty
    """
    if not items:
        return default
    
    min_key = min(items.items(), key=lambda x: key_func(x[1]))
    return min_key


def remove_sensitive_fields(
    data: Dict[str, Any],
    fields_to_remove: List[str]
) -> Dict[str, Any]:
    """
    Remove sensitive fields from a dictionary (creates a copy).
    
    Args:
        data: Dictionary to clean
        fields_to_remove: List of field names to remove
    
    Returns:
        Copy of dictionary with sensitive fields removed
    """
    cleaned = data.copy()
    for field in fields_to_remove:
        cleaned.pop(field, None)
    return cleaned


def round_decimal(value: float, decimals: int = 2) -> float:
    """
    Round a number to specified decimal places.
    
    Args:
        value: Number to round
        decimals: Number of decimal places (default: 2)
    
    Returns:
        Rounded number
    """
    return round(value, decimals)


def get_nested_value(data: Dict[str, Any], *keys, default: Any = None) -> Any:
    """
    Safely get a nested value from a dictionary using multiple keys.
    
    Args:
        data: Dictionary to search
        *keys: Variable number of keys to traverse
        default: Default value if key path doesn't exist
    
    Returns:
        Value at the nested key path or default
    
    Example:
        >>> data = {"cache_stats": {"hit_rate": "85%"}}
        >>> get_nested_value(data, "cache_stats", "hit_rate", default="0%")
        "85%"
        >>> get_nested_value(data, "cache_stats", "miss_rate", default="0%")
        "0%"
    """
    current = data
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current if current is not None else default


def ensure_minimum(value: float, minimum: float = 1.0) -> float:
    """
    Ensure a value is at least a minimum value.
    Commonly used to prevent division by zero.
    
    Args:
        value: Value to check
        minimum: Minimum value (default: 1.0)
    
    Returns:
        Value if >= minimum, otherwise minimum
    
    Example:
        >>> ensure_minimum(0, 1)
        1.0
        >>> ensure_minimum(5, 1)
        5.0
        >>> ensure_minimum(0.5, 1)
        1.0
    """
    return max(value, minimum)
