"""
Optimizer Scheduler Manager - Gestor de optimizadores y schedulers
====================================================================
"""

import logging
import torch
import torch.optim as optim
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class OptimizerType(Enum):
    """Tipos de optimizadores"""
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"
    RMSPROP = "rmsprop"
    ADAGRAD = "adagrad"
    ADADELTA = "adadelta"


class SchedulerType(Enum):
    """Tipos de schedulers"""
    STEP = "step"
    EXPONENTIAL = "exponential"
    COSINE_ANNEALING = "cosine_annealing"
    COSINE_ANNEALING_WARM_RESTARTS = "cosine_annealing_warm_restarts"
    REDUCE_ON_PLATEAU = "reduce_on_plateau"
    ONE_CYCLE = "one_cycle"
    LAMBDA = "lambda"
    WARMUP = "warmup"


@dataclass
class OptimizerConfig:
    """Configuración de optimizador"""
    optimizer_type: OptimizerType = OptimizerType.ADAMW
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)
    momentum: float = 0.9
    eps: float = 1e-8


@dataclass
class SchedulerConfig:
    """Configuración de scheduler"""
    scheduler_type: SchedulerType = SchedulerType.COSINE_ANNEALING
    step_size: int = 10
    gamma: float = 0.1
    T_max: int = 10
    eta_min: float = 0.0
    factor: float = 0.1
    patience: int = 10
    warmup_steps: int = 0
    max_lr: float = 1e-3


class OptimizerSchedulerManager:
    """Gestor de optimizadores y schedulers"""
    
    def __init__(self):
        self.optimizers: Dict[str, optim.Optimizer] = {}
        self.schedulers: Dict[str, Any] = {}
    
    def create_optimizer(
        self,
        model: torch.nn.Module,
        config: OptimizerConfig,
        name: str = "default"
    ) -> optim.Optimizer:
        """Crea un optimizador"""
        params = model.parameters()
        
        if config.optimizer_type == OptimizerType.ADAM:
            optimizer = optim.Adam(
                params,
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                betas=config.betas,
                eps=config.eps
            )
        elif config.optimizer_type == OptimizerType.ADAMW:
            optimizer = optim.AdamW(
                params,
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                betas=config.betas,
                eps=config.eps
            )
        elif config.optimizer_type == OptimizerType.SGD:
            optimizer = optim.SGD(
                params,
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                momentum=config.momentum
            )
        elif config.optimizer_type == OptimizerType.RMSPROP:
            optimizer = optim.RMSprop(
                params,
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                momentum=config.momentum
            )
        elif config.optimizer_type == OptimizerType.ADAGRAD:
            optimizer = optim.Adagrad(
                params,
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        elif config.optimizer_type == OptimizerType.ADADELTA:
            optimizer = optim.Adadelta(
                params,
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        else:
            raise ValueError(f"Optimizador {config.optimizer_type} no soportado")
        
        self.optimizers[name] = optimizer
        logger.info(f"Optimizador {config.optimizer_type.value} creado: {name}")
        return optimizer
    
    def create_scheduler(
        self,
        optimizer: optim.Optimizer,
        config: SchedulerConfig,
        name: str = "default"
    ) -> Any:
        """Crea un scheduler"""
        if config.scheduler_type == SchedulerType.STEP:
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config.step_size,
                gamma=config.gamma
            )
        elif config.scheduler_type == SchedulerType.EXPONENTIAL:
            scheduler = optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=config.gamma
            )
        elif config.scheduler_type == SchedulerType.COSINE_ANNEALING:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.T_max,
                eta_min=config.eta_min
            )
        elif config.scheduler_type == SchedulerType.COSINE_ANNEALING_WARM_RESTARTS:
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=config.T_max,
                eta_min=config.eta_min
            )
        elif config.scheduler_type == SchedulerType.REDUCE_ON_PLATEAU:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=config.factor,
                patience=config.patience
            )
        elif config.scheduler_type == SchedulerType.ONE_CYCLE:
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=config.max_lr,
                total_steps=config.T_max
            )
        else:
            raise ValueError(f"Scheduler {config.scheduler_type} no soportado")
        
        self.schedulers[name] = scheduler
        logger.info(f"Scheduler {config.scheduler_type.value} creado: {name}")
        return scheduler
    
    def create_warmup_scheduler(
        self,
        optimizer: optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        initial_lr: float,
        target_lr: float
    ) -> Callable:
        """Crea un scheduler con warmup"""
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return initial_lr + (target_lr - initial_lr) * step / warmup_steps
            else:
                # Cosine annealing después de warmup
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return scheduler
    
    def get_optimizer(self, name: str = "default") -> Optional[optim.Optimizer]:
        """Obtiene un optimizador"""
        return self.optimizers.get(name)
    
    def get_scheduler(self, name: str = "default") -> Optional[Any]:
        """Obtiene un scheduler"""
        return self.schedulers.get(name)

