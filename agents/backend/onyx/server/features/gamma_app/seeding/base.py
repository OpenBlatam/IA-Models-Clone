"""
Seeding Base Classes and Interfaces
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime
from uuid import uuid4


class SeedData:
    """Seed data definition"""
    
    def __init__(
        self,
        table_name: str,
        data: List[Dict[str, Any]],
        clear_existing: bool = False
    ):
        self.table_name = table_name
        self.data = data
        self.clear_existing = clear_existing


class Seeder:
    """Seeder definition"""
    
    def __init__(
        self,
        name: str,
        seed_data: List[SeedData],
        dependencies: Optional[List[str]] = None
    ):
        self.id = str(uuid4())
        self.name = name
        self.seed_data = seed_data
        self.dependencies = dependencies or []
        self.created_at = datetime.utcnow()


class SeederBase(ABC):
    """Base interface for seeding"""
    
    @abstractmethod
    async def seed(self, seeder: Seeder) -> bool:
        """Run seeder"""
        pass
    
    @abstractmethod
    async def clear_table(self, table_name: str) -> bool:
        """Clear table data"""
        pass
    
    @abstractmethod
    async def load_fixtures(self, fixture_path: str) -> bool:
        """Load fixtures from file"""
        pass

