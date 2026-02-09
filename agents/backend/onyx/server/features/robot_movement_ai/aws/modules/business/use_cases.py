"""
Use Cases
=========

Business use cases following Clean Architecture.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class UseCaseRequest(BaseModel):
    """Base class for use case requests."""
    pass


class UseCaseResponse(BaseModel):
    """Base class for use case responses."""
    success: bool = True
    message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None


class UseCase(ABC):
    """Base class for use cases."""
    
    @abstractmethod
    async def execute(self, request: UseCaseRequest) -> UseCaseResponse:
        """Execute use case."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get use case name."""
        pass


class UseCaseExecutor:
    """Executor for use cases with dependency injection."""
    
    def __init__(self, dependencies: Optional[Dict[str, Any]] = None):
        self.dependencies = dependencies or {}
        self._use_cases: Dict[str, UseCase] = {}
    
    def register(self, use_case: UseCase):
        """Register a use case."""
        self._use_cases[use_case.get_name()] = use_case
        logger.info(f"Registered use case: {use_case.get_name()}")
    
    async def execute(self, use_case_name: str, request: UseCaseRequest) -> UseCaseResponse:
        """Execute a use case."""
        if use_case_name not in self._use_cases:
            raise ValueError(f"Use case not found: {use_case_name}")
        
        use_case = self._use_cases[use_case_name]
        
        try:
            # Inject dependencies
            if hasattr(use_case, "set_dependencies"):
                use_case.set_dependencies(self.dependencies)
            
            return await use_case.execute(request)
        except Exception as e:
            logger.error(f"Use case execution failed: {use_case_name} - {e}", exc_info=True)
            return UseCaseResponse(
                success=False,
                message=str(e),
                data=None
            )
    
    def get_use_case(self, name: str) -> Optional[UseCase]:
        """Get use case by name."""
        return self._use_cases.get(name)















