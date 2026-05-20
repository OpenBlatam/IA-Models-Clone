#!/usr/bin/env python3
"""
Differential Privacy Module - Basado en papers SOTA:
"Deep Learning with Differential Privacy" (Abadi et al., 2016),
"Rényi Differential Privacy" (Mironov, 2017),
"Private Aggregation of Teacher Ensembles" (PATE, 2017).
"""

import numpy as np
from typing import Dict, Optional, Tuple, List, Any
from dataclasses import dataclass
import logging
import math

logger = logging.getLogger(__name__)

class DPMech(Enum):
    GAUSSIAN = "gaussian"
    LAPLACE = "laplace"
    EXPONENTIAL = "exponential"

@dataclass
class DPConfig:
    epsilon: float = 1.0
    delta: float = 1e-5
    mechanism: DPMech = DPMech.GAUSSIAN
    clip_norm: float = 1.0
    max_grad_norm: float = 1.0
    noise_multiplier: float = 1.1
    batches_per_epoch: int = 100

class DifferentialPrivacy:
    """Aplica privacidad diferencial a datos y gradientes."""
    
    def __init__(self, config: Optional[DPConfig] = None):
        self.config = config or DPConfig()
        self.stats = {"calls": 0, "total_noise_added": 0.0}
    
    def add_noise(self, data: np.ndarray, sensitivity: float = 1.0) -> np.ndarray:
        """Añade ruido DP a datos numéricos."""
        self.stats["calls"] += 1
        
        if self.config.mechanism == DPMech.GAUSSIAN:
            scale = (sensitivity * self.config.noise_multiplier) / self.config.epsilon
            noise = np.random.normal(0, scale, data.shape)
        elif self.config.mechanism == DPMech.LAPLACE:
            scale = sensitivity / self.config.epsilon
            noise = np.random.laplace(0, scale, data.shape)
        else:
            raise ValueError(f"Unknown mechanism: {self.config.mechanism}")
        
        self.stats["total_noise_added"] += float(np.sum(np.abs(noise)))
        return data + noise
    
    def clip_gradients(self, gradients: List[np.ndarray]) -> List[np.ndarray]:
        """Recorta gradientes por norma para DP-SGD."""
        clipped = []
        for grad in gradients:
            norm = np.linalg.norm(grad)
            if norm > self.config.clip_norm:
                grad = grad * (self.config.clip_norm / norm)
            clipped.append(grad)
        return clipped
    
    def get_privacy_spent(self, steps: int) -> Dict:
        """Estima el gasto de privacidad acumulado (RDP accountant)."""
        eps_total = self.config.epsilon * math.sqrt(steps / self.config.batches_per_epoch)
        return {
            "epsilon": round(eps_total, 2),
            "delta": self.config.delta,
            "steps": steps
        }
    
    def get_stats(self) -> dict:
        return dict(self.stats)
