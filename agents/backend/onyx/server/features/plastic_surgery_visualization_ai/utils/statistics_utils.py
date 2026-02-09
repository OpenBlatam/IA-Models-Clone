"""Statistics utilities."""

from typing import List, Optional
import math
from collections import Counter


def mean(values: List[float]) -> float:
    """
    Calculate mean (average).
    
    Args:
        values: List of values
        
    Returns:
        Mean value
    """
    if not values:
        return 0.0
    return sum(values) / len(values)


def median(values: List[float]) -> float:
    """
    Calculate median.
    
    Args:
        values: List of values
        
    Returns:
        Median value
    """
    if not values:
        return 0.0
    
    sorted_values = sorted(values)
    n = len(sorted_values)
    
    if n % 2 == 0:
        return (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2
    else:
        return sorted_values[n // 2]


def mode(values: List[float]) -> Optional[float]:
    """
    Calculate mode (most frequent value).
    
    Args:
        values: List of values
        
    Returns:
        Mode value or None
    """
    if not values:
        return None
    
    counter = Counter(values)
    most_common = counter.most_common(1)
    
    if most_common:
        return most_common[0][0]
    return None


def variance(values: List[float], sample: bool = True) -> float:
    """
    Calculate variance.
    
    Args:
        values: List of values
        sample: Use sample variance (n-1) or population (n)
        
    Returns:
        Variance
    """
    if not values:
        return 0.0
    
    avg = mean(values)
    n = len(values)
    divisor = n - 1 if sample and n > 1 else n
    
    return sum((x - avg) ** 2 for x in values) / divisor


def standard_deviation(values: List[float], sample: bool = True) -> float:
    """
    Calculate standard deviation.
    
    Args:
        values: List of values
        sample: Use sample std dev or population
        
    Returns:
        Standard deviation
    """
    return math.sqrt(variance(values, sample))


def percentile(values: List[float], p: float) -> float:
    """
    Calculate percentile.
    
    Args:
        values: List of values
        p: Percentile (0.0 to 1.0)
        
    Returns:
        Percentile value
    """
    if not values:
        return 0.0
    
    sorted_values = sorted(values)
    index = p * (len(sorted_values) - 1)
    
    if index.is_integer():
        return sorted_values[int(index)]
    else:
        lower = sorted_values[int(index)]
        upper = sorted_values[int(index) + 1]
        return lower + (upper - lower) * (index - int(index))


def quartiles(values: List[float]) -> dict:
    """
    Calculate quartiles.
    
    Args:
        values: List of values
        
    Returns:
        Dictionary with Q1, Q2 (median), Q3
    """
    return {
        "q1": percentile(values, 0.25),
        "q2": median(values),
        "q3": percentile(values, 0.75),
    }


def range_values(values: List[float]) -> float:
    """
    Calculate range (max - min).
    
    Args:
        values: List of values
        
    Returns:
        Range
    """
    if not values:
        return 0.0
    return max(values) - min(values)


def correlation(x: List[float], y: List[float]) -> float:
    """
    Calculate correlation coefficient.
    
    Args:
        x: First variable
        y: Second variable
        
    Returns:
        Correlation coefficient (-1.0 to 1.0)
    """
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    
    mean_x = mean(x)
    mean_y = mean(y)
    
    numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x)))
    denominator_x = sum((x[i] - mean_x) ** 2 for i in range(len(x)))
    denominator_y = sum((y[i] - mean_y) ** 2 for i in range(len(y)))
    
    if denominator_x == 0 or denominator_y == 0:
        return 0.0
    
    return numerator / math.sqrt(denominator_x * denominator_y)

