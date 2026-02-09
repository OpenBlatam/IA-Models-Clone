"""
Algorithm Utilities for Piel Mejorador AI SAM3
==============================================

Unified algorithm and mathematical utilities.
"""

import logging
from typing import List, Callable, Optional, TypeVar, Any
from functools import reduce
import math

logger = logging.getLogger(__name__)

T = TypeVar('T')


class AlgorithmUtils:
    """Unified algorithm utilities."""
    
    @staticmethod
    def binary_search(
        items: List[T],
        target: T,
        key: Optional[Callable[[T], Any]] = None
    ) -> Optional[int]:
        """
        Binary search in sorted list.
        
        Args:
            items: Sorted list
            target: Target value
            key: Optional key function
            
        Returns:
            Index or None
        """
        left, right = 0, len(items) - 1
        
        while left <= right:
            mid = (left + right) // 2
            mid_value = items[mid]
            
            if key:
                mid_key = key(mid_value)
                target_key = key(target) if callable(key) else target
            else:
                mid_key = mid_value
                target_key = target
            
            if mid_key == target_key:
                return mid
            elif mid_key < target_key:
                left = mid + 1
            else:
                right = mid - 1
        
        return None
    
    @staticmethod
    def find_peak(
        values: List[float],
        threshold: Optional[float] = None
    ) -> Optional[int]:
        """
        Find peak value in list.
        
        Args:
            values: List of values
            threshold: Optional threshold (only peaks above threshold)
            
        Returns:
            Index of peak or None
        """
        if not values:
            return None
        
        max_value = max(values)
        if threshold and max_value < threshold:
            return None
        
        return values.index(max_value)
    
    @staticmethod
    def find_valley(
        values: List[float],
        threshold: Optional[float] = None
    ) -> Optional[int]:
        """
        Find valley (minimum) value in list.
        
        Args:
            values: List of values
            threshold: Optional threshold (only valleys below threshold)
            
        Returns:
            Index of valley or None
        """
        if not values:
            return None
        
        min_value = min(values)
        if threshold and min_value > threshold:
            return None
        
        return values.index(min_value)
    
    @staticmethod
    def moving_average(
        values: List[float],
        window_size: int
    ) -> List[float]:
        """
        Calculate moving average.
        
        Args:
            values: List of values
            window_size: Window size
            
        Returns:
            List of moving averages
        """
        if not values or window_size <= 0:
            return []
        
        if window_size >= len(values):
            avg = sum(values) / len(values)
            return [avg] * len(values)
        
        result = []
        for i in range(len(values)):
            start = max(0, i - window_size + 1)
            window = values[start:i + 1]
            avg = sum(window) / len(window)
            result.append(avg)
        
        return result
    
    @staticmethod
    def exponential_moving_average(
        values: List[float],
        alpha: float = 0.3
    ) -> List[float]:
        """
        Calculate exponential moving average.
        
        Args:
            values: List of values
            alpha: Smoothing factor (0-1)
            
        Returns:
            List of EMA values
        """
        if not values:
            return []
        
        result = [values[0]]
        for i in range(1, len(values)):
            ema = alpha * values[i] + (1 - alpha) * result[-1]
            result.append(ema)
        
        return result
    
    @staticmethod
    def calculate_gradient(
        values: List[float]
    ) -> List[float]:
        """
        Calculate gradient (rate of change).
        
        Args:
            values: List of values
            
        Returns:
            List of gradients
        """
        if len(values) < 2:
            return [0.0] * len(values)
        
        gradients = [0.0]
        for i in range(1, len(values)):
            gradient = values[i] - values[i - 1]
            gradients.append(gradient)
        
        return gradients
    
    @staticmethod
    def detect_trend(
        values: List[float],
        threshold: float = 0.1
    ) -> str:
        """
        Detect trend (increasing, decreasing, stable).
        
        Args:
            values: List of values
            threshold: Threshold for trend detection
            
        Returns:
            "increasing", "decreasing", or "stable"
        """
        if len(values) < 2:
            return "stable"
        
        gradients = AlgorithmUtils.calculate_gradient(values)
        avg_gradient = sum(gradients) / len(gradients)
        
        if avg_gradient > threshold:
            return "increasing"
        elif avg_gradient < -threshold:
            return "decreasing"
        else:
            return "stable"
    
    @staticmethod
    def smooth(
        values: List[float],
        method: str = "moving_average",
        **kwargs
    ) -> List[float]:
        """
        Smooth values using various methods.
        
        Args:
            values: List of values
            method: "moving_average" or "exponential"
            **kwargs: Method-specific arguments
            
        Returns:
            Smoothed values
        """
        if method == "moving_average":
            window_size = kwargs.get("window_size", 5)
            return AlgorithmUtils.moving_average(values, window_size)
        elif method == "exponential":
            alpha = kwargs.get("alpha", 0.3)
            return AlgorithmUtils.exponential_moving_average(values, alpha)
        else:
            return values
    
    @staticmethod
    def normalize_min_max(
        values: List[float],
        min_val: Optional[float] = None,
        max_val: Optional[float] = None
    ) -> List[float]:
        """
        Normalize values to 0-1 range using min-max.
        
        Args:
            values: List of values
            min_val: Optional minimum (uses actual min if None)
            max_val: Optional maximum (uses actual max if None)
            
        Returns:
            Normalized values
        """
        if not values:
            return []
        
        if min_val is None:
            min_val = min(values)
        if max_val is None:
            max_val = max(values)
        
        if max_val == min_val:
            return [0.5] * len(values)
        
        return [
            (v - min_val) / (max_val - min_val)
            for v in values
        ]
    
    @staticmethod
    def calculate_correlation(
        x: List[float],
        y: List[float]
    ) -> float:
        """
        Calculate Pearson correlation coefficient.
        
        Args:
            x: First variable
            y: Second variable
            
        Returns:
            Correlation coefficient (-1 to 1)
        """
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        sum_y2 = sum(y[i] ** 2 for i in range(n))
        
        numerator = n * sum_xy - sum_x * sum_y
        denominator = math.sqrt(
            (n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2)
        )
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator


# Convenience functions
def binary_search(items: List[T], target: T, **kwargs) -> Optional[int]:
    """Binary search."""
    return AlgorithmUtils.binary_search(items, target, **kwargs)


def moving_average(values: List[float], window_size: int) -> List[float]:
    """Calculate moving average."""
    return AlgorithmUtils.moving_average(values, window_size)


def detect_trend(values: List[float], **kwargs) -> str:
    """Detect trend."""
    return AlgorithmUtils.detect_trend(values, **kwargs)




