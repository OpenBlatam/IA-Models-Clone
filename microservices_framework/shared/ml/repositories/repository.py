"""
Repository Pattern
Data access abstraction layer.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, List, Dict
import logging

logger = logging.getLogger(__name__)


class Repository(ABC):
    """Base repository interface."""
    
    @abstractmethod
    def save(self, entity: Any, **kwargs) -> bool:
        """Save an entity."""
        pass
    
    @abstractmethod
    def get(self, identifier: str, **kwargs) -> Optional[Any]:
        """Get an entity by identifier."""
        pass
    
    @abstractmethod
    def delete(self, identifier: str, **kwargs) -> bool:
        """Delete an entity."""
        pass
    
    @abstractmethod
    def list(self, **kwargs) -> List[Any]:
        """List all entities."""
        pass


class ModelRepository(Repository):
    """Repository for model storage."""
    
    def __init__(self, storage_path: str = "./models"):
        self.storage_path = storage_path
        from pathlib import Path
        Path(storage_path).mkdir(parents=True, exist_ok=True)
    
    def save(self, entity: Any, identifier: str, **kwargs) -> bool:
        """Save a model."""
        import torch
        from pathlib import Path
        
        model_path = Path(self.storage_path) / f"{identifier}.pt"
        
        if isinstance(entity, dict):
            torch.save(entity, model_path)
        else:
            # Assume it's a model
            torch.save(entity.state_dict(), model_path)
        
        logger.info(f"Model saved: {model_path}")
        return True
    
    def get(self, identifier: str, **kwargs) -> Optional[Any]:
        """Get a model by identifier."""
        import torch
        from pathlib import Path
        
        model_path = Path(self.storage_path) / f"{identifier}.pt"
        
        if not model_path.exists():
            return None
        
        return torch.load(model_path, map_location=kwargs.get("map_location", "cpu"))
    
    def delete(self, identifier: str, **kwargs) -> bool:
        """Delete a model."""
        from pathlib import Path
        
        model_path = Path(self.storage_path) / f"{identifier}.pt"
        
        if model_path.exists():
            model_path.unlink()
            logger.info(f"Model deleted: {model_path}")
            return True
        
        return False
    
    def list(self, **kwargs) -> List[str]:
        """List all models."""
        from pathlib import Path
        
        model_dir = Path(self.storage_path)
        return [f.stem for f in model_dir.glob("*.pt")]


class CheckpointRepository(Repository):
    """Repository for checkpoint storage."""
    
    def __init__(self, storage_path: str = "./checkpoints"):
        self.storage_path = storage_path
        from pathlib import Path
        Path(storage_path).mkdir(parents=True, exist_ok=True)
    
    def save(self, entity: Any, identifier: str, **kwargs) -> bool:
        """Save a checkpoint."""
        import torch
        from pathlib import Path
        
        checkpoint_path = Path(self.storage_path) / f"{identifier}.pt"
        torch.save(entity, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        return True
    
    def get(self, identifier: str, **kwargs) -> Optional[Any]:
        """Get a checkpoint by identifier."""
        import torch
        from pathlib import Path
        
        checkpoint_path = Path(self.storage_path) / f"{identifier}.pt"
        
        if not checkpoint_path.exists():
            return None
        
        return torch.load(checkpoint_path, map_location=kwargs.get("map_location", "cpu"))
    
    def delete(self, identifier: str, **kwargs) -> bool:
        """Delete a checkpoint."""
        from pathlib import Path
        
        checkpoint_path = Path(self.storage_path) / f"{identifier}.pt"
        
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            logger.info(f"Checkpoint deleted: {checkpoint_path}")
            return True
        
        return False
    
    def list(self, **kwargs) -> List[str]:
        """List all checkpoints."""
        from pathlib import Path
        
        checkpoint_dir = Path(self.storage_path)
        return [f.stem for f in checkpoint_dir.glob("*.pt")]


class ConfigRepository(Repository):
    """Repository for configuration storage."""
    
    def __init__(self, storage_path: str = "./configs"):
        self.storage_path = storage_path
        from pathlib import Path
        Path(storage_path).mkdir(parents=True, exist_ok=True)
    
    def save(self, entity: Any, identifier: str, **kwargs) -> bool:
        """Save a configuration."""
        import yaml
        from pathlib import Path
        
        config_path = Path(self.storage_path) / f"{identifier}.yaml"
        
        with open(config_path, "w") as f:
            yaml.dump(entity, f)
        
        logger.info(f"Config saved: {config_path}")
        return True
    
    def get(self, identifier: str, **kwargs) -> Optional[Any]:
        """Get a configuration by identifier."""
        import yaml
        from pathlib import Path
        
        config_path = Path(self.storage_path) / f"{identifier}.yaml"
        
        if not config_path.exists():
            return None
        
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    
    def delete(self, identifier: str, **kwargs) -> bool:
        """Delete a configuration."""
        from pathlib import Path
        
        config_path = Path(self.storage_path) / f"{identifier}.yaml"
        
        if config_path.exists():
            config_path.unlink()
            logger.info(f"Config deleted: {config_path}")
            return True
        
        return False
    
    def list(self, **kwargs) -> List[str]:
        """List all configurations."""
        from pathlib import Path
        
        config_dir = Path(self.storage_path)
        return [f.stem for f in config_dir.glob("*.yaml")]


class RepositoryManager:
    """Manager for repositories."""
    
    def __init__(self):
        self._repositories: Dict[str, Repository] = {}
    
    def register(self, name: str, repository: Repository):
        """Register a repository."""
        self._repositories[name] = repository
        logger.info(f"Repository '{name}' registered")
    
    def get(self, name: str) -> Optional[Repository]:
        """Get a repository by name."""
        return self._repositories.get(name)
    
    def list_repositories(self) -> List[str]:
        """List all registered repositories."""
        return list(self._repositories.keys())



