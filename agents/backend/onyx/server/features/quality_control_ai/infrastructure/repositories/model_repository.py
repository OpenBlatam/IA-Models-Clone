"""
Model Repository

Repository for storing and retrieving ML models.
"""

import logging
from typing import List, Optional, Dict, Any
from pathlib import Path
import json

from ...domain.exceptions import ModelException, ModelLoadException

logger = logging.getLogger(__name__)


class ModelRepository:
    """
    Repository for ML model storage and retrieval.
    
    Handles model persistence, versioning, and metadata management.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize repository.
        
        Args:
            storage_path: Base path for model storage
        """
        self.storage_path = Path(storage_path) if storage_path else Path("./models")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._model_registry = {}  # In-memory registry
    
    def save_model(
        self,
        model_type: str,
        model_path: str,
        metadata: Dict[str, Any],
        version: Optional[str] = None,
    ) -> str:
        """
        Save a model with metadata.
        
        Args:
            model_type: Type of model (e.g., 'autoencoder', 'classifier')
            model_path: Path to the model file
            metadata: Model metadata
            version: Optional version string
        
        Returns:
            Model ID
        
        Raises:
            ModelException: If save fails
        """
        try:
            import uuid
            model_id = str(uuid.uuid4())
            
            # Create model directory
            model_dir = self.storage_path / model_type / model_id
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy model file
            import shutil
            source_path = Path(model_path)
            dest_path = model_dir / source_path.name
            shutil.copy2(source_path, dest_path)
            
            # Save metadata
            metadata_file = model_dir / "metadata.json"
            metadata_data = {
                "model_id": model_id,
                "model_type": model_type,
                "model_path": str(dest_path),
                "version": version or "1.0.0",
                **metadata,
            }
            with open(metadata_file, 'w') as f:
                json.dump(metadata_data, f, indent=2)
            
            # Register in memory
            self._model_registry[model_id] = metadata_data
            
            logger.info(f"Model {model_id} saved: type={model_type}, version={version}")
            return model_id
        
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}", exc_info=True)
            raise ModelException(f"Failed to save model: {str(e)}")
    
    def load_model(self, model_id: str) -> Dict[str, Any]:
        """
        Load model metadata and path.
        
        Args:
            model_id: Model ID
        
        Returns:
            Dictionary with model information
        
        Raises:
            ModelLoadException: If model not found
        """
        try:
            # Check in-memory registry first
            if model_id in self._model_registry:
                return self._model_registry[model_id]
            
            # Search in storage
            for model_dir in self.storage_path.rglob("metadata.json"):
                with open(model_dir, 'r') as f:
                    metadata = json.load(f)
                    if metadata.get('model_id') == model_id:
                        self._model_registry[model_id] = metadata
                        return metadata
            
            raise ModelLoadException(model_id, "Model not found")
        
        except ModelLoadException:
            raise
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}", exc_info=True)
            raise ModelLoadException(model_id, str(e))
    
    def list_models(
        self,
        model_type: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        List all models with optional filtering.
        
        Args:
            model_type: Filter by model type
            limit: Maximum number of results
        
        Returns:
            List of model metadata dictionaries
        """
        try:
            models = []
            
            # Search in storage
            for metadata_file in self.storage_path.rglob("metadata.json"):
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    if model_type is None or metadata.get('model_type') == model_type:
                        models.append(metadata)
            
            # Sort by version or date
            models.sort(key=lambda x: x.get('version', '0.0.0'), reverse=True)
            
            if limit:
                models = models[:limit]
            
            return models
        
        except Exception as e:
            logger.error(f"Failed to list models: {str(e)}", exc_info=True)
            return []
    
    def get_latest_model(self, model_type: str) -> Optional[Dict[str, Any]]:
        """
        Get the latest version of a model type.
        
        Args:
            model_type: Type of model
        
        Returns:
            Model metadata or None if not found
        """
        models = self.list_models(model_type=model_type, limit=1)
        return models[0] if models else None
    
    def delete_model(self, model_id: str) -> bool:
        """
        Delete a model.
        
        Args:
            model_id: Model ID
        
        Returns:
            True if deleted, False if not found
        """
        try:
            # Find model directory
            for model_dir in self.storage_path.rglob("metadata.json"):
                with open(model_dir, 'r') as f:
                    metadata = json.load(f)
                    if metadata.get('model_id') == model_id:
                        # Delete entire model directory
                        import shutil
                        shutil.rmtree(model_dir.parent)
                        
                        # Remove from registry
                        if model_id in self._model_registry:
                            del self._model_registry[model_id]
                        
                        logger.info(f"Model {model_id} deleted")
                        return True
            
            return False
        
        except Exception as e:
            logger.error(f"Failed to delete model: {str(e)}", exc_info=True)
            return False



