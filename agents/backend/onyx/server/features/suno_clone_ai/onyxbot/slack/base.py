"""Base Slack - Clase base para Slack"""
from abc import ABC, abstractmethod
class BaseSlack(ABC):
    @abstractmethod
    async def send_message(self, channel: str, message: str): pass

