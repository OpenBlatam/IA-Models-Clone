"""
Model Reproducibility Manager - Gestor de reproducibilidad de modelos
======================================================================
"""

import torch
import random
import numpy as np
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json

from .core_utils import get_logger

logger = get_logger(__name__)


@dataclass
class ReproducibilityConfig:
    """Configuración de reproducibilidad"""
    seed: int = 42
    deterministic: bool = True
    cudnn_deterministic: bool = True
    cudnn_benchmark: bool = False


class ModelReproducibilityManager:
    """Gestor de reproducibilidad"""
    
    def __init__(self, config: ReproducibilityConfig):
        self.config = config
        self.reproducibility_info: Dict[str, Any] = {}
    
    def set_seed(self, seed: Optional[int] = None):
        """Establece semilla para reproducibilidad"""
        seed = seed or self.config.seed
        
        # Python random
        random.seed(seed)
        
        # NumPy
        np.random.seed(seed)
        
        # PyTorch
        torch.manual_seed(seed)
        
        # CUDA
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            
            if self.config.cudnn_deterministic:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            else:
                torch.backends.cudnn.benchmark = self.config.cudnn_benchmark
        
        # Deterministic operations
        if self.config.deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
        
        logger.info(f"Semilla establecida: {seed}")
    
    def save_reproducibility_info(
        self,
        model_info: Dict[str, Any],
        training_config: Dict[str, Any],
        filepath: str
    ):
        """Guarda información de reproducibilidad"""
        info = {
            "seed": self.config.seed,
            "python_version": self._get_python_version(),
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "model_info": model_info,
            "training_config": training_config,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(info, f, indent=2)
        
        self.reproducibility_info = info
        logger.info(f"Información de reproducibilidad guardada: {filepath}")
    
    def _get_python_version(self) -> str:
        """Obtiene versión de Python"""
        import sys
        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    
    def load_reproducibility_info(self, filepath: str) -> Dict[str, Any]:
        """Carga información de reproducibilidad"""
        with open(filepath, 'r') as f:
            info = json.load(f)
        
        self.reproducibility_info = info
        self.config.seed = info.get("seed", 42)
        
        logger.info(f"Información de reproducibilidad cargada: {filepath}")
        return info
    
    def verify_reproducibility(
        self,
        model: torch.nn.Module,
        example_input: torch.Tensor,
        expected_output: Optional[torch.Tensor] = None
    ) -> bool:
        """Verifica reproducibilidad"""
        # Primera ejecución
        self.set_seed()
        model.eval()
        with torch.no_grad():
            output1 = model(example_input)
        
        # Segunda ejecución con misma semilla
        self.set_seed()
        model.eval()
        with torch.no_grad():
            output2 = model(example_input)
        
        # Comparar
        if hasattr(output1, 'logits'):
            output1 = output1.logits
        if hasattr(output2, 'logits'):
            output2 = output2.logits
        
        are_equal = torch.allclose(output1, output2, atol=1e-6)
        
        if are_equal:
            logger.info("Reproducibilidad verificada: outputs idénticos")
        else:
            logger.warning("Reproducibilidad no verificada: outputs diferentes")
        
        return are_equal




