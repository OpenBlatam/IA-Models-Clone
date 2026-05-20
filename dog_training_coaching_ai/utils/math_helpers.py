"""
Math Helpers
============
Utilidades matemáticas.
"""

from typing import List, Optional


def clamp(value: float, min_value: float, max_value: float) -> float:
    """
    Limitar valor entre mínimo y máximo.
    
    Args:
        value: Valor a limitar
        min_value: Valor mínimo
        max_value: Valor máximo
        
    Returns:
        Valor limitado
    """
    return max(min_value, min(value, max_value))


def percentage(value: float, total: float) -> float:
    """
    Calcular porcentaje.
    
    Args:
        value: Valor
        total: Total
        
    Returns:
        Porcentaje (0-100)
    """
    if total == 0:
        return 0.0
    return (value / total) * 100


def round_to(value: float, decimals: int = 2) -> float:
    """
    Redondear a número de decimales.
    
    Args:
        value: Valor a redondear
        decimals: Número de decimales
        
    Returns:
        Valor redondeado
    """
    return round(value, decimals)


def calculate_percentile(values: List[float], percentile: float) -> Optional[float]:
    """
    Calcular percentil de una lista de valores.
    
    Args:
        values: Lista de valores
        percentile: Percentil (0-100)
        
    Returns:
        Valor del percentil o None si la lista está vacía
    """
    if not values:
        return None
    
    sorted_values = sorted(values)
    index = (percentile / 100) * (len(sorted_values) - 1)
    
    if index.is_integer():
        return sorted_values[int(index)]
    
    lower = sorted_values[int(index)]
    upper = sorted_values[int(index) + 1]
    return lower + (upper - lower) * (index - int(index))


def calculate_median(values: List[float]) -> Optional[float]:
    """
    Calcular mediana de una lista de valores.
    
    Args:
        values: Lista de valores
        
    Returns:
        Mediana o None si la lista está vacía
    """
    if not values:
        return None
    
    sorted_values = sorted(values)
    n = len(sorted_values)
    
    if n % 2 == 0:
        return (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2
    else:
        return sorted_values[n // 2]


def calculate_standard_deviation(values: List[float]) -> Optional[float]:
    """
    Calcular desviación estándar.
    
    Args:
        values: Lista de valores
        
    Returns:
        Desviación estándar o None si la lista está vacía
    """
    if not values or len(values) < 2:
        return None
    
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return variance ** 0.5

