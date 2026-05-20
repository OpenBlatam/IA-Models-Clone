"""
Model Commands - Commands for model operations
"""

from typing import Any, Optional, Dict
import logging
import torch
from pathlib import Path

from .command import ICommand, CommandResult

logger = logging.getLogger(__name__)


class TrainModelCommand(ICommand):
    """
    Command to train a model
    """
    
    def __init__(
        self,
        trainer,
        dataloader,
        num_epochs: int = 1
    ):
        self.trainer = trainer
        self.dataloader = dataloader
        self.num_epochs = num_epochs
        self._original_state = None
        self._training_results = []
    
    @property
    def name(self) -> str:
        return "TrainModel"
    
    def execute(self) -> Dict[str, Any]:
        """Execute training"""
        # Save original state for undo
        if hasattr(self.trainer, 'model'):
            self._original_state = self.trainer.model.state_dict().copy()
        
        results = []
        for epoch in range(self.num_epochs):
            metrics = self.trainer.train_epoch(self.dataloader, epoch)
            results.append(metrics)
        
        self._training_results = results
        return {"results": results, "epochs": self.num_epochs}
    
    def undo(self) -> Dict[str, Any]:
        """Undo training (restore original state)"""
        if self._original_state and hasattr(self.trainer, 'model'):
            self.trainer.model.load_state_dict(self._original_state)
            logger.info("Training undone - model state restored")
        return {"status": "undone"}


class EvaluateModelCommand(ICommand):
    """
    Command to evaluate a model
    """
    
    def __init__(self, trainer, dataloader):
        self.trainer = trainer
        self.dataloader = dataloader
    
    @property
    def name(self) -> str:
        return "EvaluateModel"
    
    def execute(self) -> Dict[str, Any]:
        """Execute evaluation"""
        metrics = self.trainer.evaluate(self.dataloader)
        return metrics
    
    def undo(self) -> Any:
        """Evaluation cannot be undone"""
        logger.warning("Evaluation cannot be undone")
        return None


class SaveModelCommand(ICommand):
    """
    Command to save a model
    """
    
    def __init__(self, model, path: str, metadata: Optional[Dict] = None):
        self.model = model
        self.path = Path(path)
        self.metadata = metadata or {}
        self._saved = False
    
    @property
    def name(self) -> str:
        return "SaveModel"
    
    def execute(self) -> Dict[str, Any]:
        """Execute save"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "metadata": self.metadata
        }, self.path)
        self._saved = True
        logger.info(f"Model saved to {self.path}")
        return {"path": str(self.path), "saved": True}
    
    def undo(self) -> Dict[str, Any]:
        """Undo save (delete file)"""
        if self._saved and self.path.exists():
            self.path.unlink()
            logger.info(f"Model file deleted: {self.path}")
            return {"deleted": True}
        return {"deleted": False}








