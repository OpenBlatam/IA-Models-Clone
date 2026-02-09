"""
Model Repository - Store and retrieve models
"""

from typing import Optional, Dict, Any, List
import torch
import logging
from pathlib import Path

from .repository import BaseRepository

logger = logging.getLogger(__name__)


class ModelRepository(BaseRepository):
    """
    Repository for model storage and retrieval
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        super().__init__("ModelRepository")
        self.storage_path = Path(storage_path) if storage_path else Path("./models")
        self.storage_path.mkdir(parents=True, exist_ok=True)
    
    def save_model(self, model_id: str, model: torch.nn.Module, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Save model to disk"""
        try:
            model_path = self.storage_path / f"{model_id}.pth"
            torch.save({
                "model_state_dict": model.state_dict(),
                "metadata": metadata or {}
            }, model_path)
            
            logger.info(f"Saved model {model_id} to {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model {model_id}: {str(e)}")
            return False
    
    def load_model(self, model_id: str, model_class: Optional[type] = None) -> Optional[torch.nn.Module]:
        """Load model from disk"""
        try:
            model_path = self.storage_path / f"{model_id}.pth"
            
            if not model_path.exists():
                logger.warning(f"Model {model_id} not found at {model_path}")
                return None
            
            checkpoint = torch.load(model_path, map_location="cpu")
            
            if model_class:
                model = model_class()
                model.load_state_dict(checkpoint["model_state_dict"])
                return model
            else:
                # Return state dict if no model class provided
                return checkpoint["model_state_dict"]
        
        except Exception as e:
            logger.error(f"Error loading model {model_id}: {str(e)}")
            return None
    
    def list_models(self) -> List[str]:
        """List all saved models"""
        return [f.stem for f in self.storage_path.glob("*.pth")]








