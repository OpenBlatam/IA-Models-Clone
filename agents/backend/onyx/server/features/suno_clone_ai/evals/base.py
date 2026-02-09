"""Base Evaluator - Clase base para evaluadores"""
from abc import ABC, abstractmethod
class BaseEvaluator(ABC):
    @abstractmethod
    async def evaluate(self, *args, **kwargs): pass

