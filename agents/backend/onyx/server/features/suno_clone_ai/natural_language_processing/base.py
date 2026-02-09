"""Base NLP - Clase base para NLP"""
from abc import ABC, abstractmethod
class BaseNLP(ABC):
    @abstractmethod
    async def analyze(self, text: str): pass

