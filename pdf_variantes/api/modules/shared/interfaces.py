"""
Shared Interfaces
Common interfaces used across all modules
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, List
from datetime import datetime


class IEntity(ABC):
    """Base entity interface"""
    
    @property
    @abstractmethod
    def id(self) -> str:
        """Entity ID"""
        pass
    
    @property
    @abstractmethod
    def created_at(self) -> datetime:
        """Creation timestamp"""
        pass


class IRepository(ABC):
    """Base repository interface"""
    
    @abstractmethod
    async def get_by_id(self, id: str) -> Optional[IEntity]:
        """Get entity by ID"""
        pass
    
    @abstractmethod
    async def save(self, entity: IEntity) -> IEntity:
        """Save entity"""
        pass
    
    @abstractmethod
    async def delete(self, id: str) -> bool:
        """Delete entity"""
        pass
    
    @abstractmethod
    async def exists(self, id: str) -> bool:
        """Check if entity exists"""
        pass


class IUseCase(ABC):
    """Base use case interface"""
    
    @abstractmethod
    async def execute(self, *args, **kwargs) -> Any:
        """Execute use case"""
        pass


class IController(ABC):
    """Base controller interface"""
    
    @abstractmethod
    async def handle(self, request: Any, *args, **kwargs) -> Any:
        """Handle request"""
        pass


class IPresenter(ABC):
    """Base presenter interface"""
    
    @abstractmethod
    def present(self, entity: IEntity) -> dict:
        """Format entity for response"""
        pass


class IValidator(ABC):
    """Base validator interface"""
    
    @abstractmethod
    def validate(self, value: Any) -> tuple[bool, Optional[str]]:
        """Validate value, return (is_valid, error_message)"""
        pass


class IModule(ABC):
    """Base module interface"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Module name"""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """Module version"""
        pass
    
    @abstractmethod
    def initialize(self, config: dict) -> None:
        """Initialize module"""
        pass
    
    @abstractmethod
    def get_controller(self) -> IController:
        """Get module controller"""
        pass
    
    @abstractmethod
    def get_repository(self) -> IRepository:
        """Get module repository"""
        pass






