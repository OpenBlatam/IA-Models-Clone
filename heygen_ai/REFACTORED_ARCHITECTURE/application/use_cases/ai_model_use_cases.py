"""
AI Model Use Cases

This module defines the use cases for AI model operations in the HeyGen AI system.
Use cases represent the application's business logic and orchestrate domain services.
"""

from typing import Any, Dict, List, Optional
from uuid import UUID

from ...domain.entities.ai_model import AIModel, ModelStatus, ModelType
from ...domain.services.ai_model_service import AIModelService


class CreateModelUseCase:
    """Use case for creating a new AI model."""
    
    def __init__(self, model_service: AIModelService):
        self._model_service = model_service
    
    async def execute(
        self,
        name: str,
        model_type: ModelType,
        version: str = "1.0.0",
        description: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        created_by: Optional[str] = None
    ) -> AIModel:
        """
        Execute the create model use case.
        
        Args:
            name: Name of the model
            model_type: Type of the model
            version: Version of the model
            description: Optional description
            config: Optional model configuration
            created_by: Creator of the model
            
        Returns:
            The created AI model
            
        Raises:
            ValueError: If model name and version already exist
        """
        return await self._model_service.create_model(
            name=name,
            model_type=model_type,
            version=version,
            description=description,
            config=config,
            created_by=created_by
        )


class GetModelUseCase:
    """Use case for getting an AI model by ID."""
    
    def __init__(self, model_service: AIModelService):
        self._model_service = model_service
    
    async def execute(self, model_id: UUID) -> Optional[AIModel]:
        """
        Execute the get model use case.
        
        Args:
            model_id: ID of the model to retrieve
            
        Returns:
            The model if found, None otherwise
        """
        return await self._model_service.get_model(model_id)


class ListModelsUseCase:
    """Use case for listing AI models with optional filtering."""
    
    def __init__(self, model_service: AIModelService):
        self._model_service = model_service
    
    async def execute(
        self,
        model_type: Optional[ModelType] = None,
        status: Optional[ModelStatus] = None,
        created_by: Optional[str] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[AIModel]:
        """
        Execute the list models use case.
        
        Args:
            model_type: Optional model type filter
            status: Optional status filter
            created_by: Optional creator filter
            skip: Number of models to skip
            limit: Maximum number of models to return
            
        Returns:
            List of models matching the filters
        """
        return await self._model_service.list_models(
            model_type=model_type,
            status=status,
            created_by=created_by,
            skip=skip,
            limit=limit
        )


class UpdateModelUseCase:
    """Use case for updating an AI model."""
    
    def __init__(self, model_service: AIModelService):
        self._model_service = model_service
    
    async def execute(
        self,
        model_id: UUID,
        updates: Dict[str, Any]
    ) -> Optional[AIModel]:
        """
        Execute the update model use case.
        
        Args:
            model_id: ID of the model to update
            updates: Dictionary of updates to apply
            
        Returns:
            The updated model if found, None otherwise
        """
        return await self._model_service.update_model(model_id, updates)


class DeleteModelUseCase:
    """Use case for deleting an AI model."""
    
    def __init__(self, model_service: AIModelService):
        self._model_service = model_service
    
    async def execute(self, model_id: UUID) -> bool:
        """
        Execute the delete model use case.
        
        Args:
            model_id: ID of the model to delete
            
        Returns:
            True if the model was deleted, False otherwise
        """
        return await self._model_service.delete_model(model_id)


class TrainModelUseCase:
    """Use case for training an AI model."""
    
    def __init__(self, model_service: AIModelService):
        self._model_service = model_service
    
    async def execute(
        self,
        model_id: UUID,
        training_config: Dict[str, Any],
        metrics: Optional[Dict[str, float]] = None
    ) -> Optional[AIModel]:
        """
        Execute the train model use case.
        
        Args:
            model_id: ID of the model to train
            training_config: Training configuration
            metrics: Optional training metrics
            
        Returns:
            The updated model if found, None otherwise
        """
        return await self._model_service.train_model(
            model_id=model_id,
            training_config=training_config,
            metrics=metrics
        )


class CompleteTrainingUseCase:
    """Use case for completing model training."""
    
    def __init__(self, model_service: AIModelService):
        self._model_service = model_service
    
    async def execute(
        self,
        model_id: UUID,
        metrics: Dict[str, float],
        model_path: Optional[str] = None,
        model_size: Optional[int] = None
    ) -> Optional[AIModel]:
        """
        Execute the complete training use case.
        
        Args:
            model_id: ID of the model
            metrics: Training metrics
            model_path: Optional path to the trained model file
            model_size: Optional size of the model file in bytes
            
        Returns:
            The updated model if found, None otherwise
        """
        return await self._model_service.complete_training(
            model_id=model_id,
            metrics=metrics,
            model_path=model_path,
            model_size=model_size
        )


class DeployModelUseCase:
    """Use case for deploying an AI model."""
    
    def __init__(self, model_service: AIModelService):
        self._model_service = model_service
    
    async def execute(self, model_id: UUID) -> Optional[AIModel]:
        """
        Execute the deploy model use case.
        
        Args:
            model_id: ID of the model to deploy
            
        Returns:
            The updated model if found, None otherwise
            
        Raises:
            ValueError: If model is not trained
        """
        return await self._model_service.deploy_model(model_id)


class ArchiveModelUseCase:
    """Use case for archiving an AI model."""
    
    def __init__(self, model_service: AIModelService):
        self._model_service = model_service
    
    async def execute(self, model_id: UUID) -> Optional[AIModel]:
        """
        Execute the archive model use case.
        
        Args:
            model_id: ID of the model to archive
            
        Returns:
            The updated model if found, None otherwise
        """
        return await self._model_service.archive_model(model_id)


class SearchModelsUseCase:
    """Use case for searching AI models."""
    
    def __init__(self, model_service: AIModelService):
        self._model_service = model_service
    
    async def execute(
        self,
        query: str,
        model_type: Optional[ModelType] = None,
        status: Optional[ModelStatus] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[AIModel]:
        """
        Execute the search models use case.
        
        Args:
            query: Search query
            model_type: Optional model type filter
            status: Optional status filter
            skip: Number of models to skip
            limit: Maximum number of models to return
            
        Returns:
            List of models matching the search query
        """
        return await self._model_service.search_models(
            query=query,
            model_type=model_type,
            status=status,
            skip=skip,
            limit=limit
        )


class GetModelVersionsUseCase:
    """Use case for getting all versions of a model."""
    
    def __init__(self, model_service: AIModelService):
        self._model_service = model_service
    
    async def execute(self, name: str) -> List[AIModel]:
        """
        Execute the get model versions use case.
        
        Args:
            name: Name of the model
            
        Returns:
            List of all versions of the model
        """
        return await self._model_service.get_model_versions(name)


class GetLatestModelVersionUseCase:
    """Use case for getting the latest version of a model."""
    
    def __init__(self, model_service: AIModelService):
        self._model_service = model_service
    
    async def execute(self, name: str) -> Optional[AIModel]:
        """
        Execute the get latest model version use case.
        
        Args:
            name: Name of the model
            
        Returns:
            The latest version of the model if found, None otherwise
        """
        return await self._model_service.get_latest_model_version(name)