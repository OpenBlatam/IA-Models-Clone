"""
AI Model Service

This module provides the AI model service that handles business logic
for AI model operations in the HeyGen AI system.
"""

from typing import Any, Dict, List, Optional
from uuid import UUID

from ..entities.ai_model import AIModel, ModelStatus, ModelType
from ..repositories.base_repository import BaseRepository


class AIModelService:
    """
    Service class for AI model operations.
    
    This service provides business logic for:
    - Model creation and management
    - Model training and deployment
    - Model versioning and lifecycle management
    - Model search and filtering
    """
    
    def __init__(self, model_repository: BaseRepository[AIModel]):
        """
        Initialize the AI model service.
        
        Args:
            model_repository: Repository for AI model persistence
        """
        self._model_repository = model_repository
    
    async def create_model(
        self,
        name: str,
        model_type: ModelType,
        version: str = "1.0.0",
        description: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        created_by: Optional[str] = None
    ) -> AIModel:
        """
        Create a new AI model.
        
        Args:
            name: Name of the model
            model_type: Type of the model
            version: Version of the model
            description: Optional description
            config: Optional model configuration
            created_by: Creator of the model
            
        Returns:
            The created AI model
        """
        # Validate model name uniqueness
        existing_models = await self._model_repository.search(
            filters={"name": name, "version": version}
        )
        if existing_models:
            raise ValueError(f"Model with name '{name}' and version '{version}' already exists")
        
        # Create the model
        model = AIModel(
            name=name,
            model_type=model_type,
            version=version,
            description=description
        )
        
        # Set optional fields
        if config:
            model.update_config(config)
        if created_by:
            model.set_created_by(created_by)
        
        # Save the model
        return await self._model_repository.create(model)
    
    async def get_model(self, model_id: UUID) -> Optional[AIModel]:
        """
        Get a model by its ID.
        
        Args:
            model_id: ID of the model
            
        Returns:
            The model if found, None otherwise
        """
        return await self._model_repository.get_by_id(model_id)
    
    async def get_model_by_name_and_version(
        self,
        name: str,
        version: str
    ) -> Optional[AIModel]:
        """
        Get a model by name and version.
        
        Args:
            name: Name of the model
            version: Version of the model
            
        Returns:
            The model if found, None otherwise
        """
        models = await self._model_repository.search(
            filters={"name": name, "version": version}
        )
        return models[0] if models else None
    
    async def list_models(
        self,
        model_type: Optional[ModelType] = None,
        status: Optional[ModelStatus] = None,
        created_by: Optional[str] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[AIModel]:
        """
        List models with optional filtering.
        
        Args:
            model_type: Optional model type filter
            status: Optional status filter
            created_by: Optional creator filter
            skip: Number of models to skip
            limit: Maximum number of models to return
            
        Returns:
            List of models matching the filters
        """
        filters = {}
        if model_type:
            filters["model_type"] = model_type.value
        if status:
            filters["status"] = status.value
        if created_by:
            filters["created_by"] = created_by
        
        return await self._model_repository.search(filters, skip, limit)
    
    async def update_model(
        self,
        model_id: UUID,
        updates: Dict[str, Any]
    ) -> Optional[AIModel]:
        """
        Update a model with the given updates.
        
        Args:
            model_id: ID of the model to update
            updates: Dictionary of updates to apply
            
        Returns:
            The updated model if found, None otherwise
        """
        model = await self._model_repository.get_by_id(model_id)
        if not model:
            return None
        
        # Apply updates
        for key, value in updates.items():
            if key == "name":
                model.set_name(value)
            elif key == "description":
                model.set_description(value)
            elif key == "status":
                model.set_status(ModelStatus(value))
            elif key == "config":
                model.update_config(value)
            elif key == "metrics":
                model.update_metrics(value)
            elif key == "tags":
                if isinstance(value, list):
                    for tag in value:
                        model.add_tag(tag)
            elif key == "path":
                model.set_path(value)
            elif key == "size_bytes":
                model.set_size_bytes(value)
        
        return await self._model_repository.update(model)
    
    async def delete_model(self, model_id: UUID) -> bool:
        """
        Delete a model by its ID.
        
        Args:
            model_id: ID of the model to delete
            
        Returns:
            True if the model was deleted, False otherwise
        """
        return await self._model_repository.delete(model_id)
    
    async def train_model(
        self,
        model_id: UUID,
        training_config: Dict[str, Any],
        metrics: Optional[Dict[str, float]] = None
    ) -> Optional[AIModel]:
        """
        Mark a model as being trained and update its configuration.
        
        Args:
            model_id: ID of the model to train
            training_config: Training configuration
            metrics: Optional training metrics
            
        Returns:
            The updated model if found, None otherwise
        """
        model = await self._model_repository.get_by_id(model_id)
        if not model:
            return None
        
        # Update status to training
        model.set_status(ModelStatus.TRAINING)
        
        # Update configuration
        model.update_config(training_config)
        
        # Update metrics if provided
        if metrics:
            model.update_metrics(metrics)
        
        return await self._model_repository.update(model)
    
    async def complete_training(
        self,
        model_id: UUID,
        metrics: Dict[str, float],
        model_path: Optional[str] = None,
        model_size: Optional[int] = None
    ) -> Optional[AIModel]:
        """
        Mark a model as having completed training.
        
        Args:
            model_id: ID of the model
            metrics: Training metrics
            model_path: Optional path to the trained model file
            model_size: Optional size of the model file in bytes
            
        Returns:
            The updated model if found, None otherwise
        """
        model = await self._model_repository.get_by_id(model_id)
        if not model:
            return None
        
        # Update status to trained
        model.set_status(ModelStatus.TRAINED)
        
        # Update metrics
        model.update_metrics(metrics)
        
        # Set model path and size if provided
        if model_path:
            model.set_path(model_path)
        if model_size:
            model.set_size_bytes(model_size)
        
        return await self._model_repository.update(model)
    
    async def deploy_model(self, model_id: UUID) -> Optional[AIModel]:
        """
        Deploy a model.
        
        Args:
            model_id: ID of the model to deploy
            
        Returns:
            The updated model if found, None otherwise
        """
        model = await self._model_repository.get_by_id(model_id)
        if not model:
            return None
        
        # Check if model is trained
        if model.status != ModelStatus.TRAINED:
            raise ValueError("Model must be trained before deployment")
        
        # Update status to deployed
        model.set_status(ModelStatus.DEPLOYED)
        
        return await self._model_repository.update(model)
    
    async def archive_model(self, model_id: UUID) -> Optional[AIModel]:
        """
        Archive a model.
        
        Args:
            model_id: ID of the model to archive
            
        Returns:
            The updated model if found, None otherwise
        """
        model = await self._model_repository.get_by_id(model_id)
        if not model:
            return None
        
        # Update status to archived
        model.set_status(ModelStatus.ARCHIVED)
        
        return await self._model_repository.update(model)
    
    async def get_model_versions(self, name: str) -> List[AIModel]:
        """
        Get all versions of a model by name.
        
        Args:
            name: Name of the model
            
        Returns:
            List of all versions of the model
        """
        return await self._model_repository.search(filters={"name": name})
    
    async def get_latest_model_version(self, name: str) -> Optional[AIModel]:
        """
        Get the latest version of a model by name.
        
        Args:
            name: Name of the model
            
        Returns:
            The latest version of the model if found, None otherwise
        """
        models = await self.get_model_versions(name)
        if not models:
            return None
        
        # Sort by version and return the latest
        models.sort(key=lambda m: m.version, reverse=True)
        return models[0]
    
    async def search_models(
        self,
        query: str,
        model_type: Optional[ModelType] = None,
        status: Optional[ModelStatus] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[AIModel]:
        """
        Search models by query string.
        
        Args:
            query: Search query
            model_type: Optional model type filter
            status: Optional status filter
            skip: Number of models to skip
            limit: Maximum number of models to return
            
        Returns:
            List of models matching the search query
        """
        filters = {"search": query}
        if model_type:
            filters["model_type"] = model_type.value
        if status:
            filters["status"] = status.value
        
        return await self._model_repository.search(filters, skip, limit)
    
    async def get_model_count(
        self,
        model_type: Optional[ModelType] = None,
        status: Optional[ModelStatus] = None
    ) -> int:
        """
        Get the count of models with optional filtering.
        
        Args:
            model_type: Optional model type filter
            status: Optional status filter
            
        Returns:
            Number of models matching the filters
        """
        filters = {}
        if model_type:
            filters["model_type"] = model_type.value
        if status:
            filters["status"] = status.value
        
        return await self._model_repository.count(filters)