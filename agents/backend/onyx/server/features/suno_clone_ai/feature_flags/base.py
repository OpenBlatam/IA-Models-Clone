"""Base Feature Flag - Clase base para feature flags"""
from abc import ABC, abstractmethod
class BaseFeatureFlag(ABC):
    @abstractmethod
    def is_enabled(self, flag_name: str, user_id: str = None) -> bool: pass

