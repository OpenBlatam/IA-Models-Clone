"""Base Seeder - Clase base para seeders"""
from abc import ABC, abstractmethod
class BaseSeeder(ABC):
    @abstractmethod
    async def seed(self): pass

