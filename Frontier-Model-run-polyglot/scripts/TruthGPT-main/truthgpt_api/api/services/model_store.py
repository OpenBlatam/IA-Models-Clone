"""
Model Store Service
==================

Service for managing model storage and retrieval.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid
import logging
import truthgpt as tg
from ..exceptions import ModelNotFoundError, ModelNotCompiledError
from ..constants import MAX_OPERATION_HISTORY, DEFAULT_OPERATION_HISTORY_LIMIT

logger = logging.getLogger(__name__)


class ModelStore:
    """In-memory model storage service."""
    
    def __init__(self):
        """Initialize the model store."""
        self._models: Dict[str, Dict[str, Any]] = {}
        self._operation_history: Dict[str, List[Dict[str, Any]]] = {}
        logger.info("ModelStore initialized")
    
    def create(self, model: Any, name: Optional[str] = None) -> str:
        """
        Create a new model entry.
        
        Args:
            model: Model instance
            name: Optional model name
            
        Returns:
            Model ID
        """
        model_id = str(uuid.uuid4())
        self._models[model_id] = {
            "model": model,
            "compiled": False,
            "name": name or f"Model_{model_id[:8]}",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        self._record_operation(model_id, "create", {"name": self._models[model_id]['name']})
        logger.info(f"Model created: {model_id} - {self._models[model_id]['name']}")
        return model_id
    
    def get(self, model_id: str) -> Dict[str, Any]:
        """
        Get model data by ID.
        
        Args:
            model_id: Model ID
            
        Returns:
            Model data dictionary
            
        Raises:
            ModelNotFoundError: If model not found
        """
        if model_id not in self._models:
            raise ModelNotFoundError(model_id)
        return self._models[model_id]
    
    def get_model(self, model_id: str) -> Any:
        """
        Get model instance by ID.
        
        Args:
            model_id: Model ID
            
        Returns:
            Model instance
            
        Raises:
            ModelNotFoundError: If model not found
        """
        return self.get(model_id)["model"]
    
    def update(self, model_id: str, **kwargs) -> None:
        """
        Update model data.
        
        Args:
            model_id: Model ID
            **kwargs: Fields to update
        """
        if model_id not in self._models:
            raise ModelNotFoundError(model_id)
        self._models[model_id].update(kwargs)
        self._models[model_id]["updated_at"] = datetime.now().isoformat()
        self._record_operation(model_id, "update", kwargs)
        logger.debug(f"Model updated: {model_id}")
    
    def mark_compiled(self, model_id: str) -> None:
        """
        Mark a model as compiled.
        
        Args:
            model_id: Model ID
        """
        self.update(model_id, compiled=True)
    
    def delete(self, model_id: str) -> None:
        """
        Delete a model.
        
        Args:
            model_id: Model ID
            
        Raises:
            ModelNotFoundError: If model not found
        """
        if model_id not in self._models:
            raise ModelNotFoundError(model_id)
        model_name = self._models[model_id]["name"]
        del self._models[model_id]
        logger.info(f"Model deleted: {model_id} - {model_name}")
    
    def list_all(self) -> Dict[str, Dict[str, Any]]:
        """
        List all models.
        
        Returns:
            Dictionary of all models
        """
        return self._models.copy()
    
    def exists(self, model_id: str) -> bool:
        """
        Check if a model exists.
        
        Args:
            model_id: Model ID
            
        Returns:
            True if model exists, False otherwise
        """
        return model_id in self._models
    
    def is_compiled(self, model_id: str) -> bool:
        """
        Check if a model is compiled.
        
        Args:
            model_id: Model ID
            
        Returns:
            True if compiled, False otherwise
            
        Raises:
            ModelNotFoundError: If model not found
        """
        return self.get(model_id)["compiled"]
    
    def require_compiled(self, model_id: str) -> None:
        """
        Require that a model is compiled, raise error if not.
        
        Args:
            model_id: Model ID
            
        Raises:
            ModelNotCompiledError: If model is not compiled
            ModelNotFoundError: If model not found
        """
        if not self.is_compiled(model_id):
            raise ModelNotCompiledError(model_id)
    
    def load_model(self, filepath: str, model_id: Optional[str] = None) -> str:
        """
        Load a model from disk and add to store.
        
        Args:
            filepath: Path to model file
            model_id: Optional model ID (generated if not provided)
            
        Returns:
            Model ID
        """
        model = tg.load_model(filepath, model_class=tg.Sequential)
        
        if not model_id:
            model_id = str(uuid.uuid4())
        
        self._models[model_id] = {
            "model": model,
            "compiled": hasattr(model, '_compiled') and model._compiled,
            "name": f"Loaded_{model_id[:8]}",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "filepath": filepath
        }
        
        self._record_operation(model_id, "load", {"filepath": filepath})
        logger.info(f"Model loaded from disk: {model_id} - {filepath}")
        return model_id
    
    def _record_operation(self, model_id: str, operation: str, details: Optional[Dict[str, Any]] = None) -> None:
        """
        Record an operation in the model's history.
        
        Args:
            model_id: Model ID
            operation: Operation type
            details: Optional operation details
        """
        if model_id not in self._operation_history:
            self._operation_history[model_id] = []
        
        self._operation_history[model_id].append({
            "operation": operation,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        })
        
        if len(self._operation_history[model_id]) > MAX_OPERATION_HISTORY:
            self._operation_history[model_id] = self._operation_history[model_id][-MAX_OPERATION_HISTORY:]
    
    def get_operation_history(self, model_id: str, limit: int = DEFAULT_OPERATION_HISTORY_LIMIT) -> List[Dict[str, Any]]:
        """
        Get operation history for a model.
        
        Args:
            model_id: Model ID
            limit: Maximum number of operations to return
            
        Returns:
            List of operation records
        """
        if model_id not in self._operation_history:
            return []
        return self._operation_history[model_id][-limit:]


model_store = ModelStore()

