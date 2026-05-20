"""
Model Ensemble Service - Ensembles de modelos
==============================================

Sistema para crear y gestionar ensembles de modelos.
"""

import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")


class EnsembleMethod(str):
    """Métodos de ensemble"""
    VOTING = "voting"
    AVERAGING = "averaging"
    WEIGHTED_AVERAGING = "weighted_averaging"
    STACKING = "stacking"


@dataclass
class EnsembleConfig:
    """Configuración de ensemble"""
    method: str = "averaging"
    weights: Optional[List[float]] = None
    voting_strategy: str = "hard"  # hard, soft


class ModelEnsembleService:
    """Servicio de ensembles de modelos"""
    
    def __init__(self):
        """Inicializar servicio"""
        self.ensembles: Dict[str, List[Any]] = {}
        self.configs: Dict[str, EnsembleConfig] = {}
        logger.info("ModelEnsembleService initialized")
    
    def create_ensemble(
        self,
        ensemble_id: str,
        models: List[Any],
        config: EnsembleConfig
    ) -> bool:
        """Crear ensemble de modelos"""
        if len(models) == 0:
            return False
        
        # Normalize weights if provided
        if config.weights and len(config.weights) != len(models):
            logger.warning("Weights length doesn't match models, using equal weights")
            config.weights = None
        
        if config.weights:
            # Normalize weights
            total = sum(config.weights)
            config.weights = [w / total for w in config.weights]
        
        self.ensembles[ensemble_id] = models
        self.configs[ensemble_id] = config
        
        logger.info(f"Ensemble {ensemble_id} created with {len(models)} models")
        return True
    
    async def predict_ensemble(
        self,
        ensemble_id: str,
        inputs: Any,
        device: Optional[torch.device] = None
    ) -> np.ndarray:
        """Predecir con ensemble"""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        
        models = self.ensembles.get(ensemble_id)
        config = self.configs.get(ensemble_id)
        
        if not models or not config:
            raise ValueError(f"Ensemble {ensemble_id} not found")
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        predictions = []
        
        for model in models:
            model.eval()
            with torch.no_grad():
                if isinstance(inputs, torch.Tensor):
                    inputs_device = inputs.to(device)
                else:
                    inputs_device = torch.tensor(inputs).to(device)
                
                output = model(inputs_device)
                
                if isinstance(output, torch.Tensor):
                    predictions.append(output.cpu().numpy())
                else:
                    predictions.append(output)
        
        # Combine predictions
        if config.method == "averaging":
            combined = np.mean(predictions, axis=0)
        elif config.method == "weighted_averaging":
            if config.weights:
                combined = np.average(predictions, axis=0, weights=config.weights)
            else:
                combined = np.mean(predictions, axis=0)
        elif config.method == "voting":
            if config.voting_strategy == "hard":
                # Hard voting: majority class
                pred_classes = [np.argmax(p, axis=1) if p.ndim > 1 else p for p in predictions]
                combined = np.array([np.bincount(preds).argmax() for preds in zip(*pred_classes)])
            else:
                # Soft voting: average probabilities
                combined = np.mean(predictions, axis=0)
        else:
            combined = np.mean(predictions, axis=0)
        
        return combined
    
    def get_ensemble_info(self, ensemble_id: str) -> Dict[str, Any]:
        """Obtener información del ensemble"""
        models = self.ensembles.get(ensemble_id)
        config = self.configs.get(ensemble_id)
        
        if not models or not config:
            return {"error": "Ensemble not found"}
        
        return {
            "ensemble_id": ensemble_id,
            "num_models": len(models),
            "method": config.method,
            "weights": config.weights,
        }




