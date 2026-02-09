"""
Training Generator - Generador de utilidades de entrenamiento
=============================================================

Genera módulos especializados para entrenamiento:
- Trainer con mixed precision
- Early stopping
- Learning rate schedulers
- Gradient utilities
"""

import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


class TrainingGenerator:
    """Generador de utilidades de entrenamiento"""
    
    def __init__(self):
        """Inicializa el generador de entrenamiento"""
        pass
    
    def generate(
        self,
        utils_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """
        Genera utilidades de entrenamiento.
        
        Args:
            utils_dir: Directorio donde generar las utilidades
            keywords: Keywords extraídos
            project_info: Información del proyecto
        """
        utils_dir.mkdir(parents=True, exist_ok=True)
        
        # Generar módulos de entrenamiento
        self._generate_trainer(utils_dir, keywords, project_info)
        self._generate_early_stopping(utils_dir, keywords, project_info)
        self._generate_schedulers(utils_dir, keywords, project_info)
        self._generate_experiment_tracker(utils_dir, keywords, project_info)
        self._generate_training_init(utils_dir, keywords)
    
    def _generate_training_init(
        self,
        utils_dir: Path,
        keywords: Dict[str, Any],
    ) -> None:
        """Genera __init__.py del módulo de entrenamiento"""
        
        init_content = '''"""
Training Utilities Module
==========================

Utilidades para entrenamiento de modelos de Deep Learning.
Incluye experiment tracking, multi-GPU, y mejores prácticas.
"""

from .trainer import Trainer
from .early_stopping import EarlyStopping
from .schedulers import get_scheduler
from .experiment_tracker import ExperimentTracker, get_tracker

__all__ = [
    "Trainer",
    "EarlyStopping",
    "get_scheduler",
    "ExperimentTracker",
    "get_tracker",
]
'''
        
        training_dir = utils_dir / "training"
        training_dir.mkdir(parents=True, exist_ok=True)
        (training_dir / "__init__.py").write_text(init_content, encoding="utf-8")
    
    def _generate_trainer(
        self,
        utils_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """Genera clase Trainer"""
        
        trainer_content = '''"""
Trainer - Clase para entrenamiento de modelos
==============================================

Implementa entrenamiento con mejores prácticas:
- Mixed precision training
- Gradient accumulation
- Gradient clipping
- Checkpointing
- Multi-GPU support (DataParallel/DistributedDataParallel)
- Experiment tracking
- NaN/Inf detection
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Optimizer
from torch.nn.parallel import DataParallel, DistributedDataParallel
from typing import Optional, Dict, Any
import logging
from pathlib import Path
import os

from .experiment_tracker import ExperimentTracker

logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer con mejores prácticas para PyTorch.
    
    Incluye:
    - Mixed precision training
    - Gradient accumulation
    - Learning rate scheduling
    - Early stopping
    - Checkpointing
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        device: str = "cuda",
        use_amp: bool = True,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        use_multi_gpu: bool = False,
        use_distributed: bool = False,
        tracker: Optional[ExperimentTracker] = None,
        detect_anomaly: bool = False,
    ):
        """
        Args:
            model: Modelo a entrenar
            optimizer: Optimizador
            criterion: Función de pérdida
            device: Dispositivo a usar
            use_amp: Si usar mixed precision
            gradient_accumulation_steps: Pasos de acumulación de gradiente
            max_grad_norm: Norma máxima de gradiente (para clipping)
            use_multi_gpu: Si usar DataParallel para multi-GPU
            use_distributed: Si usar DistributedDataParallel
            tracker: Tracker de experimentos (opcional)
            detect_anomaly: Si detectar anomalías en autograd (debug)
        """
        self.device = device
        self.use_amp = use_amp and torch.cuda.is_available()
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.tracker = tracker
        self.detect_anomaly = detect_anomaly
        
        # Configurar modelo para multi-GPU
        if use_distributed:
            self.model = DistributedDataParallel(
                model.to(device),
                device_ids=[device] if isinstance(device, int) else None,
                find_unused_parameters=False,  # Más rápido
            )
            logger.info("Usando DistributedDataParallel")
        elif use_multi_gpu and torch.cuda.device_count() > 1:
            self.model = DataParallel(
                model.to(device),
                device_ids=list(range(torch.cuda.device_count())),
            )
            logger.info(f"Usando DataParallel en {torch.cuda.device_count()} GPUs")
        else:
            self.model = model.to(device)
        
        # Optimizaciones adicionales de velocidad agresivas
        if device == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False  # Más rápido
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            # Habilitar flash attention
            try:
                torch.backends.cuda.enable_flash_sdp(True)
                logger.info("Flash attention habilitado para entrenamiento")
            except:
                pass
            # Optimizar allocator
            try:
                torch.cuda.set_per_process_memory_fraction(0.95)
            except:
                pass
        
        self.optimizer = optimizer
        self.criterion = criterion
        self.scaler = GradScaler() if self.use_amp else None
        self.global_step = 0
        
        # Habilitar detección de anomalías si se solicita
        if detect_anomaly:
            torch.autograd.set_detect_anomaly(True)
            logger.warning("Detección de anomalías habilitada (puede ser lento)")
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Un paso de entrenamiento.
        
        Args:
            batch: Batch de datos
            
        Returns:
            Diccionario con métricas
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass con mixed precision
        with autocast(enabled=self.use_amp):
            outputs = self.model(**batch)
            loss = self.criterion(outputs, batch.get("labels"))
            loss = loss / self.gradient_accumulation_steps
        
        # Backward pass
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient accumulation
        if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
            # Gradient clipping
            if self.use_amp:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )
                self.optimizer.step()
        
        self.global_step += 1
        
        # Verificar NaN/Inf
        loss_value = loss.item() * self.gradient_accumulation_steps
        if not torch.isfinite(torch.tensor(loss_value)):
            logger.error(f"Loss contiene NaN o Inf en step {self.global_step}")
            raise RuntimeError("Loss contiene valores inválidos")
        
        # Registrar métricas
        if self.tracker:
            self.tracker.log_metric("train/loss", loss_value, self.global_step)
        
        return {
            "loss": loss_value,
        }
    
    def save_checkpoint(
        self,
        path: Path,
        epoch: int,
        metrics: Dict[str, float],
    ):
        """Guarda un checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "global_step": self.global_step,
        }
        
        if self.scaler:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint guardado en {path}")
    
    def load_checkpoint(self, path: Path):
        """Carga un checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scaler and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        logger.info(f"Checkpoint cargado desde {path}")
        return checkpoint
'''
        
        training_dir = utils_dir / "training"
        (training_dir / "trainer.py").write_text(trainer_content, encoding="utf-8")
    
    def _generate_early_stopping(
        self,
        utils_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """Genera clase EarlyStopping"""
        
        early_stopping_content = '''"""
Early Stopping - Detención temprana para evitar overfitting
============================================================
"""

import logging

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping para evitar overfitting"""
    
    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0.0,
        mode: str = "min",
    ):
        """
        Args:
            patience: Número de épocas sin mejora antes de parar
            min_delta: Cambio mínimo para considerar mejora
            mode: "min" o "max" según si se minimiza o maximiza la métrica
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """Verifica si se debe parar el entrenamiento"""
        if self.best_score is None:
            self.best_score = score
        elif self._is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info(f"Early stopping activado después de {self.patience} épocas sin mejora")
        
        return self.early_stop
    
    def _is_better(self, current: float, best: float) -> bool:
        """Verifica si el score actual es mejor"""
        if self.mode == "min":
            return current < best - self.min_delta
        else:
            return current > best + self.min_delta
    
    def reset(self):
        """Reinicia el estado del early stopping"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
'''
        
        training_dir = utils_dir / "training"
        (training_dir / "early_stopping.py").write_text(early_stopping_content, encoding="utf-8")
    
    def _generate_schedulers(
        self,
        utils_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """Genera utilidades para learning rate schedulers"""
        
        schedulers_content = '''"""
Learning Rate Schedulers - Utilidades para schedulers
=====================================================
"""

from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    StepLR,
    ExponentialLR,
    ReduceLROnPlateau,
    _LRScheduler,
)
from torch.optim import Optimizer
from typing import Optional

def get_scheduler(
    optimizer: Optimizer,
    scheduler_type: str = "cosine",
    **kwargs,
) -> _LRScheduler:
    """
    Obtiene un scheduler según el tipo.
    
    Args:
        optimizer: Optimizador
        scheduler_type: Tipo de scheduler (cosine, step, exponential, plateau)
        **kwargs: Argumentos adicionales para el scheduler
        
    Returns:
        Scheduler configurado
    """
    if scheduler_type == "cosine":
        T_max = kwargs.get("T_max", 10)
        return CosineAnnealingLR(optimizer, T_max=T_max)
    
    elif scheduler_type == "step":
        step_size = kwargs.get("step_size", 10)
        gamma = kwargs.get("gamma", 0.1)
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    elif scheduler_type == "exponential":
        gamma = kwargs.get("gamma", 0.95)
        return ExponentialLR(optimizer, gamma=gamma)
    
    elif scheduler_type == "plateau":
        mode = kwargs.get("mode", "min")
        factor = kwargs.get("factor", 0.1)
        patience = kwargs.get("patience", 10)
        return ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
        )
    
    else:
        raise ValueError(f"Scheduler type {scheduler_type} no soportado")
'''
        
        training_dir = utils_dir / "training"
        (training_dir / "schedulers.py").write_text(schedulers_content, encoding="utf-8")

