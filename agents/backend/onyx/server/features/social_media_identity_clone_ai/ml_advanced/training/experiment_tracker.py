"""
Sistema avanzado de experiment tracking con wandb y tensorboard

Mejoras:
- Tracking completo de métricas
- Logging de modelos y artefactos
- Hyperparameter tracking
- System metrics
- Visualizaciones avanzadas
"""

import logging
import torch
import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Tracker de experimentos (wandb o tensorboard)"""
    
    def __init__(
        self,
        tracker_type: str = "wandb",  # "wandb" or "tensorboard"
        project_name: str = "social-media-identity-clone",
        experiment_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.tracker_type = tracker_type
        self.tracker = None
        self._init_tracker(project_name, experiment_name, config)
    
    def _init_tracker(
        self,
        project_name: str,
        experiment_name: Optional[str],
        config: Optional[Dict[str, Any]]
    ):
        """Inicializa tracker"""
        try:
            if self.tracker_type == "wandb":
                try:
                    import wandb
                    wandb.init(
                        project=project_name,
                        name=experiment_name,
                        config=config or {}
                    )
                    self.tracker = wandb
                    logger.info("Wandb tracker inicializado")
                except ImportError:
                    logger.warning("wandb no instalado, usando TensorBoard")
                    self.tracker_type = "tensorboard"
                    self._init_tracker(project_name, experiment_name, config)
            elif self.tracker_type == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter
                log_dir = Path("./logs/tensorboard") / (experiment_name or "experiment")
                self.tracker = SummaryWriter(log_dir=str(log_dir))
                logger.info(f"TensorBoard tracker inicializado: {log_dir}")
            else:
                logger.warning(f"Tracker type {self.tracker_type} no soportado")
        except ImportError as e:
            logger.warning(f"No se pudo inicializar tracker: {e}")
            self.tracker = None
    
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log métricas"""
        if not self.tracker:
            return
        
        try:
            if self.tracker_type == "wandb":
                self.tracker.log(metrics, step=step)
            elif self.tracker_type == "tensorboard":
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        self.tracker.add_scalar(key, value, step or 0)
                    elif isinstance(value, torch.Tensor):
                        if value.numel() == 1:
                            self.tracker.add_scalar(key, value.item(), step or 0)
        except Exception as e:
            logger.error(f"Error logging metrics: {e}")
    
    def log_hyperparameters(self, hyperparameters: Dict[str, Any]):
        """Log hyperparameters"""
        if not self.tracker:
            return
        
        try:
            if self.tracker_type == "wandb":
                self.tracker.config.update(hyperparameters)
            elif self.tracker_type == "tensorboard":
                # TensorBoard no tiene método directo, usar text
                hparams_str = "\n".join(
                    f"{k}: {v}" for k, v in hyperparameters.items()
                )
                self.tracker.add_text("hyperparameters", hparams_str)
        except Exception as e:
            logger.error(f"Error logging hyperparameters: {e}")
    
    def log_model_weights(
        self,
        model: torch.nn.Module,
        step: Optional[int] = None
    ):
        """Log distribución de pesos del modelo"""
        if not self.tracker:
            return
        
        try:
            for name, param in model.named_parameters():
                if param.requires_grad and param.data.numel() > 0:
                    # Histograma de pesos
                    if self.tracker_type == "wandb":
                        try:
                            import wandb
                            self.tracker.log(
                                {f"weights/{name}": wandb.Histogram(param.data.cpu().numpy())},
                                step=step
                            )
                        except ImportError:
                            pass
                    elif self.tracker_type == "tensorboard":
                        self.tracker.add_histogram(
                            f"weights/{name}",
                            param.data,
                            step or 0
                        )
                    
                    # Gradientes si están disponibles
                    if param.grad is not None:
                        if self.tracker_type == "wandb":
                            try:
                                import wandb
                                self.tracker.log(
                                    {f"gradients/{name}": wandb.Histogram(param.grad.cpu().numpy())},
                                    step=step
                                )
                            except ImportError:
                                pass
                        elif self.tracker_type == "tensorboard":
                            self.tracker.add_histogram(
                                f"gradients/{name}",
                                param.grad,
                                step or 0
                            )
        except Exception as e:
            logger.error(f"Error logging model weights: {e}")
    
    def log_system_metrics(self, step: Optional[int] = None):
        """Log métricas del sistema (CPU, GPU, memoria)"""
        if not self.tracker:
            return
        
        try:
            import psutil
            
            # CPU
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()
            
            # Memoria
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_gb = memory.used / (1024 ** 3)
            memory_total_gb = memory.total / (1024 ** 3)
            
            metrics = {
                "system/cpu_percent": cpu_percent,
                "system/cpu_count": cpu_count,
                "system/memory_percent": memory_percent,
                "system/memory_used_gb": memory_used_gb,
                "system/memory_total_gb": memory_total_gb
            }
            
            # GPU si está disponible
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    metrics[f"system/gpu_{i}_memory_used_gb"] = (
                        torch.cuda.memory_allocated(i) / (1024 ** 3)
                    )
                    metrics[f"system/gpu_{i}_memory_reserved_gb"] = (
                        torch.cuda.memory_reserved(i) / (1024 ** 3)
                    )
                    metrics[f"system/gpu_{i}_utilization"] = (
                        torch.cuda.utilization(i) if hasattr(torch.cuda, 'utilization') else 0
                    )
            
            self.log(metrics, step=step)
            
        except ImportError:
            logger.warning("psutil no disponible para system metrics")
        except Exception as e:
            logger.error(f"Error logging system metrics: {e}")
    
    def log_learning_rate(
        self,
        optimizer: torch.optim.Optimizer,
        step: Optional[int] = None
    ):
        """Log learning rate del optimizer"""
        if not self.tracker:
            return
        
        try:
            lrs = {}
            for i, param_group in enumerate(optimizer.param_groups):
                lr = param_group.get('lr', 0)
                lrs[f"learning_rate/group_{i}"] = lr
            
            self.log(lrs, step=step)
        except Exception as e:
            logger.error(f"Error logging learning rate: {e}")
    
    def log_images(
        self,
        images: List[torch.Tensor],
        tag: str = "images",
        step: Optional[int] = None
    ):
        """Log imágenes"""
        if not self.tracker:
            return
        
        try:
            if self.tracker_type == "wandb":
                try:
                    import wandb
                    image_list = [
                        wandb.Image(img.cpu().numpy() if isinstance(img, torch.Tensor) else img)
                        for img in images
                    ]
                    self.tracker.log({tag: image_list}, step=step)
                except ImportError:
                    pass
            elif self.tracker_type == "tensorboard":
                # TensorBoard espera formato específico
                for i, img in enumerate(images):
                    if isinstance(img, torch.Tensor):
                        img = img.cpu().numpy()
                    self.tracker.add_image(
                        f"{tag}/{i}",
                        img,
                        step or 0,
                        dataformats="CHW" if len(img.shape) == 3 else "HWC"
                    )
        except Exception as e:
            logger.error(f"Error logging images: {e}")
    
    def log_text(
        self,
        texts: List[str],
        tag: str = "texts",
        step: Optional[int] = None
    ):
        """Log textos"""
        if not self.tracker:
            return
        
        try:
            if self.tracker_type == "wandb":
                try:
                    import wandb
                    text_table = wandb.Table(
                        columns=["text"],
                        data=[[text] for text in texts]
                    )
                    self.tracker.log({tag: text_table}, step=step)
                except ImportError:
                    pass
            elif self.tracker_type == "tensorboard":
                text_str = "\n\n".join(f"{i+1}. {text}" for i, text in enumerate(texts))
                self.tracker.add_text(tag, text_str, step or 0)
        except Exception as e:
            logger.error(f"Error logging text: {e}")
    
    def log_model(self, model, artifact_name: str = "model"):
        """Log modelo"""
        if not self.tracker:
            return
        
        try:
            if self.tracker_type == "wandb":
                self.tracker.log_model(artifact_name, model)
        except Exception as e:
            logger.error(f"Error logging model: {e}")
    
    def finish(self):
        """Finaliza tracking"""
        if not self.tracker:
            return
        
        try:
            if self.tracker_type == "wandb":
                self.tracker.finish()
            elif self.tracker_type == "tensorboard":
                self.tracker.close()
        except Exception as e:
            logger.error(f"Error finishing tracker: {e}")

