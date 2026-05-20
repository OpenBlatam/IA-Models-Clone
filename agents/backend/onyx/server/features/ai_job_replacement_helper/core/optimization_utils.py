"""
Optimization Utils - Utilidades de optimización
===============================================

Utilidades avanzadas para optimización de modelos y entrenamiento.
Sigue mejores prácticas de PyTorch optimization.
"""

import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OptimizerConfig:
    """Configuración de optimizador"""
    optimizer_type: str = "adamw"  # adam, adamw, sgd, rmsprop
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)  # For Adam/AdamW
    momentum: float = 0.9  # For SGD
    eps: float = 1e-8
    amsgrad: bool = False  # For Adam
    nesterov: bool = False  # For SGD


@dataclass
class SchedulerConfig:
    """Configuración de scheduler"""
    scheduler_type: str = "cosine"  # cosine, step, plateau, warmup_cosine
    # Cosine
    T_max: Optional[int] = None
    eta_min: float = 0.0
    # Step
    step_size: Optional[int] = None
    gamma: float = 0.1
    # Plateau
    mode: str = "min"
    factor: float = 0.1
    patience: int = 10
    # Warmup
    warmup_steps: Optional[int] = None


class OptimizationUtils:
    """Utilidades de optimización"""
    
    @staticmethod
    def create_optimizer(
        model: nn.Module,
        config: OptimizerConfig
    ) -> optim.Optimizer:
        """
        Crear optimizador con configuración.
        
        Args:
            model: Modelo
            config: Configuración del optimizador
        
        Returns:
            Optimizador configurado
        """
        params = model.parameters()
        
        optimizer_type = config.optimizer_type.lower()
        
        if optimizer_type == "adam":
            optimizer = optim.Adam(
                params,
                lr=config.learning_rate,
                betas=config.betas,
                weight_decay=config.weight_decay,
                eps=config.eps,
                amsgrad=config.amsgrad,
            )
        elif optimizer_type == "adamw":
            optimizer = optim.AdamW(
                params,
                lr=config.learning_rate,
                betas=config.betas,
                weight_decay=config.weight_decay,
                eps=config.eps,
                amsgrad=config.amsgrad,
            )
        elif optimizer_type == "sgd":
            optimizer = optim.SGD(
                params,
                lr=config.learning_rate,
                momentum=config.momentum,
                weight_decay=config.weight_decay,
                nesterov=config.nesterov,
            )
        elif optimizer_type == "rmsprop":
            optimizer = optim.RMSprop(
                params,
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                momentum=config.momentum,
                eps=config.eps,
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        
        logger.info(f"Created {optimizer_type} optimizer with lr={config.learning_rate}")
        return optimizer
    
    @staticmethod
    def create_scheduler(
        optimizer: optim.Optimizer,
        config: SchedulerConfig,
        num_training_steps: Optional[int] = None
    ) -> Optional[optim.lr_scheduler._LRScheduler]:
        """
        Crear learning rate scheduler.
        
        Args:
            optimizer: Optimizador
            config: Configuración del scheduler
            num_training_steps: Número total de steps (para warmup_cosine)
        
        Returns:
            Scheduler o None
        """
        scheduler_type = config.scheduler_type.lower()
        
        if scheduler_type == "cosine":
            T_max = config.T_max or num_training_steps or 1000
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=T_max,
                eta_min=config.eta_min,
            )
        elif scheduler_type == "step":
            step_size = config.step_size or (num_training_steps // 3 if num_training_steps else 100)
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=step_size,
                gamma=config.gamma,
            )
        elif scheduler_type == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=config.mode,
                factor=config.factor,
                patience=config.patience,
            )
        elif scheduler_type == "warmup_cosine":
            # Linear warmup + cosine annealing
            if num_training_steps is None:
                raise ValueError("num_training_steps required for warmup_cosine")
            
            warmup_steps = config.warmup_steps or (num_training_steps // 10)
            
            def lr_lambda(current_step: int) -> float:
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                progress = float(current_step - warmup_steps) / float(
                    max(1, num_training_steps - warmup_steps)
                )
                return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
            
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        else:
            logger.warning(f"Unknown scheduler type: {scheduler_type}, returning None")
            return None
        
        logger.info(f"Created {scheduler_type} scheduler")
        return scheduler
    
    @staticmethod
    def apply_weight_decay(
        model: nn.Module,
        weight_decay: float,
        exclude_bn: bool = True,
        exclude_bias: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Aplicar weight decay de forma selectiva.
        
        Args:
            model: Modelo
            weight_decay: Valor de weight decay
            exclude_bn: Excluir BatchNorm
            exclude_bias: Excluir bias
        
        Returns:
            Lista de grupos de parámetros
        """
        decay = []
        no_decay = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            # Check if BatchNorm
            if exclude_bn and "bn" in name.lower():
                no_decay.append(param)
            # Check if bias
            elif exclude_bias and "bias" in name.lower():
                no_decay.append(param)
            else:
                decay.append(param)
        
        param_groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        
        logger.info(
            f"Created parameter groups: {len(decay)} with decay, "
            f"{len(no_decay)} without decay"
        )
        
        return param_groups
    
    @staticmethod
    def get_learning_rate(optimizer: optim.Optimizer) -> float:
        """Obtener learning rate actual"""
        return optimizer.param_groups[0]["lr"]
    
    @staticmethod
    def set_learning_rate(optimizer: optim.Optimizer, lr: float) -> None:
        """Establecer learning rate"""
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        logger.info(f"Learning rate set to {lr}")
    
    @staticmethod
    def clip_gradients(
        model: nn.Module,
        max_norm: float = 1.0,
        norm_type: float = 2.0
    ) -> float:
        """
        Clippear gradientes.
        
        Args:
            model: Modelo
            max_norm: Norma máxima
            norm_type: Tipo de norma
        
        Returns:
            Norma de gradientes antes del clipping
        """
        parameters = [p for p in model.parameters() if p.grad is not None]
        if len(parameters) == 0:
            return 0.0
        
        total_norm = torch.nn.utils.clip_grad_norm_(
            parameters,
            max_norm=max_norm,
            norm_type=norm_type
        )
        
        return total_norm.item()
    
    @staticmethod
    def check_gradients(model: nn.Module) -> Dict[str, Any]:
        """
        Verificar estado de gradientes.
        
        Args:
            model: Modelo
        
        Returns:
            Información de gradientes
        """
        info = {
            "has_gradients": False,
            "num_params_with_grad": 0,
            "num_params_without_grad": 0,
            "grad_norms": [],
            "has_nan": False,
            "has_inf": False,
        }
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                info["has_gradients"] = True
                info["num_params_with_grad"] += 1
                
                grad_norm = param.grad.norm().item()
                info["grad_norms"].append(grad_norm)
                
                if torch.isnan(param.grad).any():
                    info["has_nan"] = True
                    logger.warning(f"NaN gradient detected in {name}")
                
                if torch.isinf(param.grad).any():
                    info["has_inf"] = True
                    logger.warning(f"Inf gradient detected in {name}")
            else:
                info["num_params_without_grad"] += 1
        
        if info["grad_norms"]:
            info["mean_grad_norm"] = sum(info["grad_norms"]) / len(info["grad_norms"])
            info["max_grad_norm"] = max(info["grad_norms"])
            info["min_grad_norm"] = min(info["grad_norms"])
        
        return info

