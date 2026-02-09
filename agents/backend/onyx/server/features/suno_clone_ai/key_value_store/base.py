"""Base Key Value Store - Clase base para KV store"""
from abc import ABC, abstractmethod
class BaseKeyValueStore(ABC):
    @abstractmethod
    def get(self, key: str): pass
    @abstractmethod
    def set(self, key: str, value: any): pass

