"""Base Bot - Clase base para bot"""
from abc import ABC, abstractmethod
class BaseBot(ABC):
    @abstractmethod
    async def handle_message(self, message: str): pass

