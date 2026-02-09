"""
Metrics - Métricas y telemetría
"""

from typing import Dict, Optional, List
from collections import defaultdict
import time


class Metrics:
    """Sistema de métricas y telemetría"""

    def __init__(self):
        """Inicializa el sistema de métricas"""
        self._metrics: Dict[str, List[float]] = defaultdict(list)
        self._counters: Dict[str, int] = defaultdict(int)

    def record(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Registra una métrica"""
        self._metrics[name].append(value)

    def increment(self, name: str, value: int = 1) -> None:
        """Incrementa un contador"""
        self._counters[name] += value

    def get_metric(self, name: str) -> List[float]:
        """Obtiene valores de una métrica"""
        return self._metrics.get(name, [])

    def get_counter(self, name: str) -> int:
        """Obtiene valor de un contador"""
        return self._counters.get(name, 0)

