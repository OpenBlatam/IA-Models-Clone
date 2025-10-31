"""
Workflow Repository Interface
============================

Repository interface for workflow persistence operations.
Follows the Repository pattern for clean separation of concerns.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime

from ..entities.workflow_chain import WorkflowChain
from ..value_objects.workflow_id import WorkflowId
from ..value_objects.workflow_status import WorkflowStatus


class WorkflowRepository(ABC):
    """
    Abstract repository for workflow persistence operations
    
    This interface defines the contract for workflow data access,
    allowing for different implementations (SQL, NoSQL, in-memory, etc.)
    """
    
    @abstractmethod
    async def save(self, workflow: WorkflowChain) -> None:
        """
        Save a workflow
        
        Args:
            workflow: The workflow to save
        """
        pass
    
    @abstractmethod
    async def get_by_id(self, workflow_id: WorkflowId) -> Optional[WorkflowChain]:
        """
        Get a workflow by ID
        
        Args:
            workflow_id: The workflow ID
            
        Returns:
            Optional[WorkflowChain]: The workflow if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def get_by_name(self, name: str) -> Optional[WorkflowChain]:
        """
        Get a workflow by name
        
        Args:
            name: The workflow name
            
        Returns:
            Optional[WorkflowChain]: The workflow if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def get_all(self, limit: int = 100, offset: int = 0) -> List[WorkflowChain]:
        """
        Get all workflows with pagination
        
        Args:
            limit: Maximum number of workflows to return
            offset: Number of workflows to skip
            
        Returns:
            List[WorkflowChain]: List of workflows
        """
        pass
    
    @abstractmethod
    async def get_by_status(self, status: WorkflowStatus, limit: int = 100, offset: int = 0) -> List[WorkflowChain]:
        """
        Get workflows by status
        
        Args:
            status: The workflow status
            limit: Maximum number of workflows to return
            offset: Number of workflows to skip
            
        Returns:
            List[WorkflowChain]: List of workflows with the specified status
        """
        pass
    
    @abstractmethod
    async def get_by_created_date_range(
        self, 
        start_date: datetime, 
        end_date: datetime, 
        limit: int = 100, 
        offset: int = 0
    ) -> List[WorkflowChain]:
        """
        Get workflows created within a date range
        
        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            limit: Maximum number of workflows to return
            offset: Number of workflows to skip
            
        Returns:
            List[WorkflowChain]: List of workflows created within the date range
        """
        pass
    
    @abstractmethod
    async def get_by_updated_date_range(
        self, 
        start_date: datetime, 
        end_date: datetime, 
        limit: int = 100, 
        offset: int = 0
    ) -> List[WorkflowChain]:
        """
        Get workflows updated within a date range
        
        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            limit: Maximum number of workflows to return
            offset: Number of workflows to skip
            
        Returns:
            List[WorkflowChain]: List of workflows updated within the date range
        """
        pass
    
    @abstractmethod
    async def search_by_name(self, name_pattern: str, limit: int = 100, offset: int = 0) -> List[WorkflowChain]:
        """
        Search workflows by name pattern
        
        Args:
            name_pattern: The name pattern to search for
            limit: Maximum number of workflows to return
            offset: Number of workflows to skip
            
        Returns:
            List[WorkflowChain]: List of workflows matching the name pattern
        """
        pass
    
    @abstractmethod
    async def count(self) -> int:
        """
        Get total count of workflows
        
        Returns:
            int: Total number of workflows
        """
        pass
    
    @abstractmethod
    async def count_by_status(self, status: WorkflowStatus) -> int:
        """
        Get count of workflows by status
        
        Args:
            status: The workflow status
            
        Returns:
            int: Number of workflows with the specified status
        """
        pass
    
    @abstractmethod
    async def delete(self, workflow_id: WorkflowId) -> bool:
        """
        Delete a workflow
        
        Args:
            workflow_id: The workflow ID to delete
            
        Returns:
            bool: True if deleted, False if not found
        """
        pass
    
    @abstractmethod
    async def exists(self, workflow_id: WorkflowId) -> bool:
        """
        Check if a workflow exists
        
        Args:
            workflow_id: The workflow ID to check
            
        Returns:
            bool: True if exists, False otherwise
        """
        pass
    
    @abstractmethod
    async def exists_by_name(self, name: str) -> bool:
        """
        Check if a workflow with the given name exists
        
        Args:
            name: The workflow name to check
            
        Returns:
            bool: True if exists, False otherwise
        """
        pass
    
    @abstractmethod
    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get repository statistics
        
        Returns:
            Dict[str, Any]: Repository statistics
        """
        pass




