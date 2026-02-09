"""
Training Pipeline Module
=========================

Pipeline profesional para entrenamiento de modelos de deep learning.
Incluye optimizaciones avanzadas, experiment tracking y manejo robusto de errores.
"""

import logging
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from torch.cuda.amp import autocast, GradScaler
    from torch.nn.utils import clip_grad_norm_
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    DataLoader = None
    Dataset = None
    logging.warning("PyTorch not available. Training pipeline will be disabled.")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

from .pipelines_config import TrainingConfig, OptimizerType, SchedulerType, LossType
from .pipelines_datasets import create_dataloader

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping para prevenir overfitting (optimizado).
    
    Incluye validaciones, mejor logging, y soporte para restore de mejor modelo.
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-6,
        mode: str = "min",
        restore_best_weights: bool = True
    ):
        """
        Inicializar early stopping (optimizado).
        
        Args:
            patience: Número de épocas sin mejora antes de parar
            min_delta: Cambio mínimo para considerar mejora
            mode: "min" o "max" para minimizar o maximizar métrica
            restore_best_weights: Si restaurar los mejores pesos al finalizar
            
        Raises:
            ValueError: Si los parámetros son inválidos
        """
        # Guard clauses
        if patience <= 0:
            raise ValueError("patience must be positive")
        if min_delta < 0:
            raise ValueError("min_delta must be non-negative")
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")
        
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.best_score = float('inf') if mode == "min" else float('-inf')
        self.counter = 0
        self.early_stop = False
        self.best_weights = None
        self.best_epoch = 0
    
    def __call__(self, score: float, epoch: int = 0, model: Optional[nn.Module] = None) -> bool:
        """
        Verificar si se debe parar el entrenamiento (optimizado).
        
        Args:
            score: Métrica actual
            epoch: Época actual (opcional)
            model: Modelo para guardar mejores pesos (opcional)
            
        Returns:
            True si se debe parar
        """
        if not isinstance(score, (int, float)):
            raise ValueError("score must be a number")
        
        if self.mode == "min":
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            
            # Guardar mejores pesos si se solicita
            if self.restore_best_weights and model is not None:
                try:
                    self.best_weights = {
                        k: v.cpu().clone() for k, v in model.state_dict().items()
                    }
                    logger.debug(f"Saved best weights at epoch {epoch} with score {score:.6f}")
                except Exception as e:
                    logger.warning(f"Failed to save best weights: {e}")
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info(
                    f"Early stopping triggered at epoch {epoch}. "
                    f"Best score: {self.best_score:.6f} at epoch {self.best_epoch}"
                )
        
        return self.early_stop
    
    def restore_best_model(self, model: nn.Module) -> None:
        """
        Restaurar los mejores pesos al modelo (optimizado).
        
        Args:
            model: Modelo al cual restaurar los pesos
        """
        if self.best_weights is None:
            logger.warning("No best weights saved to restore")
            return
        
        try:
            model.load_state_dict(self.best_weights)
            logger.info(f"Restored best weights from epoch {self.best_epoch}")
        except Exception as e:
            logger.error(f"Failed to restore best weights: {e}", exc_info=True)
            raise


class TrainingPipeline:
    """
    Pipeline profesional para entrenamiento de modelos.
    
    Características:
    - Mixed precision training
    - Gradient accumulation y clipping
    - Early stopping
    - Learning rate scheduling
    - Experiment tracking (WandB, TensorBoard)
    - Checkpointing automático
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        custom_loss_fn: Optional[Callable] = None,
        metrics: Optional[List[Callable]] = None
    ):
        """
        Inicializar pipeline de entrenamiento (optimizado).
        
        Args:
            model: Modelo PyTorch
            config: Configuración de entrenamiento
            train_dataset: Dataset de entrenamiento
            val_dataset: Dataset de validación
            custom_loss_fn: Función de pérdida personalizada
            metrics: Lista de funciones de métricas adicionales
            
        Raises:
            ImportError: Si PyTorch no está disponible
            ValueError: Si los parámetros son inválidos
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for TrainingPipeline")
        
        # Guard clauses
        if model is None:
            raise ValueError("Model cannot be None")
        if config is None:
            raise ValueError("Config cannot be None")
        if train_dataset is None and val_dataset is None:
            raise ValueError("At least one dataset (train or val) must be provided")
        
        self.model = model
        self.config = config
        self.custom_loss_fn = custom_loss_fn
        self.metrics = metrics or []
        
        # Habilitar detección de anomalías si se solicita
        if hasattr(config, 'detect_anomaly') and config.detect_anomaly:
            torch.autograd.set_detect_anomaly(True)
            logger.info("Anomaly detection enabled")
        
        # Device setup
        self.device = torch.device(config.device)
        self.model.to(self.device)
        
        # Multi-GPU support
        if config.multi_gpu and torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
            logger.info(f"Using {torch.cuda.device_count()} GPUs")
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Scheduler
        self.scheduler = self._create_scheduler()
        
        # Mixed precision
        self.scaler = GradScaler() if config.mixed_precision else None
        
        # Loss function
        self.criterion = self._create_loss_fn()
        
        # Data loaders
        self.train_loader = None
        self.val_loader = None
        if train_dataset:
            self.train_loader = create_dataloader(
                train_dataset,
                batch_size=config.batch_size,
                shuffle=config.shuffle,
                num_workers=config.num_workers,
                pin_memory=config.pin_memory,
                prefetch_factor=config.prefetch_factor,
                persistent_workers=config.persistent_workers
            )
        if val_dataset:
            self.val_loader = create_dataloader(
                val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                pin_memory=config.pin_memory
            )
        
        # Early stopping
        self.early_stopping = None
        if config.early_stopping:
            self.early_stopping = EarlyStopping(
                patience=config.early_stopping_patience,
                min_delta=config.early_stopping_min_delta
            )
        
        # Experiment tracking
        self.wandb_run = None
        self.tb_writer = None
        self._setup_tracking()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.training_history: List[Dict[str, float]] = []
        
        logger.info(f"TrainingPipeline initialized on {self.device}")
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Crear optimizador según configuración."""
        opt_config = self.config.optimizer
        
        if opt_config.optimizer_type == OptimizerType.ADAM:
            return torch.optim.Adam(
                self.model.parameters(),
                lr=opt_config.learning_rate,
                betas=opt_config.betas,
                eps=opt_config.eps,
                weight_decay=opt_config.weight_decay
            )
        elif opt_config.optimizer_type == OptimizerType.ADAMW:
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=opt_config.learning_rate,
                betas=opt_config.betas,
                eps=opt_config.eps,
                weight_decay=opt_config.weight_decay
            )
        elif opt_config.optimizer_type == OptimizerType.SGD:
            return torch.optim.SGD(
                self.model.parameters(),
                lr=opt_config.learning_rate,
                momentum=opt_config.momentum,
                weight_decay=opt_config.weight_decay
            )
        elif opt_config.optimizer_type == OptimizerType.RMSPROP:
            return torch.optim.RMSprop(
                self.model.parameters(),
                lr=opt_config.learning_rate,
                weight_decay=opt_config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer type: {opt_config.optimizer_type}")
    
    def _create_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """Crear scheduler según configuración."""
        sched_config = self.config.scheduler
        
        if sched_config.scheduler_type == SchedulerType.COSINE:
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs,
                eta_min=sched_config.eta_min
            )
        elif sched_config.scheduler_type == SchedulerType.STEP:
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sched_config.step_size,
                gamma=sched_config.gamma
            )
        elif sched_config.scheduler_type == SchedulerType.EXPONENTIAL:
            return torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=sched_config.gamma
            )
        elif sched_config.scheduler_type == SchedulerType.PLATEAU:
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=sched_config.factor,
                patience=sched_config.patience
            )
        else:
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs
            )
    
    def _create_loss_fn(self) -> nn.Module:
        """Crear función de pérdida según configuración."""
        if self.custom_loss_fn:
            return self.custom_loss_fn
        
        if self.config.loss_type == LossType.MSE:
            return nn.MSELoss(reduction=self.config.loss_reduction)
        elif self.config.loss_type == LossType.MAE:
            return nn.L1Loss(reduction=self.config.loss_reduction)
        elif self.config.loss_type == LossType.HUBER:
            return nn.HuberLoss(reduction=self.config.loss_reduction)
        elif self.config.loss_type == LossType.SMOOTH_L1:
            return nn.SmoothL1Loss(reduction=self.config.loss_reduction)
        else:
            return nn.MSELoss()
    
    def _setup_tracking(self):
        """Configurar experiment tracking."""
        if self.config.use_wandb and WANDB_AVAILABLE:
            try:
                self.wandb_run = wandb.init(
                    project=self.config.wandb_project,
                    entity=self.config.wandb_entity,
                    config=self.config.to_dict()
                )
                logger.info("WandB tracking initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize WandB: {e}")
        
        if self.config.use_tensorboard and TENSORBOARD_AVAILABLE:
            try:
                log_dir = Path(self.config.tensorboard_log_dir)
                log_dir.mkdir(parents=True, exist_ok=True)
                self.tb_writer = SummaryWriter(log_dir=str(log_dir))
                logger.info("TensorBoard tracking initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize TensorBoard: {e}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Entrenar una época completa."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        metric_values = {f"train_{m.__name__}": 0.0 for m in self.metrics}
        
        for batch_idx, batch in enumerate(self.train_loader):
            inputs = batch["input"].to(self.device, non_blocking=True)
            targets = batch["target"].to(self.device, non_blocking=True)
            
            # Forward pass
            loss = self._forward_pass(inputs, targets)
            
            # Backward pass
            self._backward_pass(loss, batch_idx)
            
            # Metrics
            with torch.no_grad():
                if self.config.mixed_precision:
                    with autocast():
                        outputs = self.model(inputs)
                else:
                    outputs = self.model(inputs)
                
                for metric_fn in self.metrics:
                    metric_name = f"train_{metric_fn.__name__}"
                    metric_values[metric_name] += metric_fn(outputs, targets).item()
            
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Logging
            if self.global_step % self.config.logging_steps == 0:
                self._log_step(loss.item(), "train")
        
        # Average metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_metrics = {k: v / num_batches for k, v in metric_values.items()}
        
        return {"train_loss": avg_loss, **avg_metrics}
    
    def _forward_pass(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass con mixed precision (optimizado).
        
        Args:
            inputs: Tensor de entrada
            targets: Tensor de targets
            
        Returns:
            Tensor de pérdida
            
        Raises:
            RuntimeError: Si se detectan NaN o Inf en la pérdida
        """
        # Guard clauses
        if inputs is None or targets is None:
            raise ValueError("Inputs and targets cannot be None")
        
        try:
            if self.config.mixed_precision and self.scaler:
                with autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
            
            # Detectar NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(f"NaN/Inf detected in loss: {loss.item()}")
                raise RuntimeError(f"Invalid loss value: {loss.item()}")
            
            return loss
        except Exception as e:
            logger.error(f"Error in forward pass: {e}", exc_info=True)
            raise
    
    def _backward_pass(self, loss: torch.Tensor, batch_idx: int):
        """
        Backward pass con gradient accumulation (optimizado).
        
        Args:
            loss: Tensor de pérdida
            batch_idx: Índice del batch actual
            
        Raises:
            RuntimeError: Si hay error en el backward pass o gradientes inválidos
        """
        try:
            # Normalizar pérdida por gradient accumulation steps
            loss = loss / self.config.gradient_accumulation_steps
            
            if self.config.mixed_precision and self.scaler:
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    
                    # Detectar gradientes inválidos antes de clipping
                    if self.config.detect_anomaly:
                        for name, param in self.model.named_parameters():
                            if param.grad is not None:
                                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                    logger.warning(f"Invalid gradients detected in {name}")
                    
                    clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                loss.backward()
                
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    # Detectar gradientes inválidos
                    if self.config.detect_anomaly:
                        for name, param in self.model.named_parameters():
                            if param.grad is not None:
                                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                    logger.warning(f"Invalid gradients detected in {name}")
                    
                    clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
        except RuntimeError as e:
            logger.error(f"Error in backward pass: {e}", exc_info=True)
            # Limpiar gradientes en caso de error
            self.optimizer.zero_grad()
            raise
    
    def validate(self) -> Dict[str, float]:
        """Validar modelo."""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        metric_values = {f"val_{m.__name__}": 0.0 for m in self.metrics}
        
        with torch.no_grad():
            for batch in self.val_loader:
                inputs = batch["input"].to(self.device, non_blocking=True)
                targets = batch["target"].to(self.device, non_blocking=True)
                
                if self.config.mixed_precision:
                    with autocast():
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)
                else:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                
                # Metrics
                for metric_fn in self.metrics:
                    metric_name = f"val_{metric_fn.__name__}"
                    metric_values[metric_name] += metric_fn(outputs, targets).item()
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_metrics = {k: v / num_batches for k, v in metric_values.items()}
        
        return {"val_loss": avg_loss, **avg_metrics}
    
    def _log_step(self, loss: float, phase: str):
        """Logear métricas en step."""
        if self.wandb_run:
            self.wandb_run.log({f"{phase}/loss": loss, "step": self.global_step})
        
        if self.tb_writer:
            self.tb_writer.add_scalar(f"{phase}/loss", loss, self.global_step)
    
    def train(self) -> Dict[str, List[float]]:
        """
        Entrenar modelo completo.
        
        Returns:
            Historial de entrenamiento
        """
        logger.info("Starting training pipeline")
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update scheduler
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics.get("val_loss", float('inf')))
            else:
                self.scheduler.step()
            
            # Combine metrics
            metrics = {**train_metrics, **val_metrics}
            metrics["epoch"] = epoch
            metrics["learning_rate"] = self.optimizer.param_groups[0]['lr']
            self.training_history.append(metrics)
            
            # Logging
            if (epoch + 1) % self.config.logging_steps == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{self.config.num_epochs}: "
                    f"Train Loss: {train_metrics.get('train_loss', 0):.4f}, "
                    f"Val Loss: {val_metrics.get('val_loss', 0):.4f}, "
                    f"LR: {metrics['learning_rate']:.6f}"
                )
            
            # Experiment tracking
            if self.wandb_run:
                self.wandb_run.log(metrics)
            
            if self.tb_writer:
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        self.tb_writer.add_scalar(key, value, epoch)
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_steps == 0:
                self.save_checkpoint(epoch)
            
            # Early stopping (mejorado)
            val_loss = val_metrics.get("val_loss", float('inf'))
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, is_best=True)
            
            if self.early_stopping:
                # Pasar modelo para guardar mejores pesos
                model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
                should_stop = self.early_stopping(val_loss, epoch=epoch, model=model_to_save)
                
                if should_stop:
                    logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    # Restaurar mejores pesos si se configuró
                    if self.early_stopping.restore_best_weights:
                        self.early_stopping.restore_best_model(model_to_save)
                    break
        
        logger.info("Training completed")
        
        # Close tracking
        if self.wandb_run:
            self.wandb_run.finish()
        if self.tb_writer:
            self.tb_writer.close()
        
        return {
            "train_loss": [m.get("train_loss", 0) for m in self.training_history],
            "val_loss": [m.get("val_loss", 0) for m in self.training_history]
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Guardar checkpoint del modelo."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get model state (handle DataParallel)
        model_state = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config.to_dict(),
            "training_history": self.training_history
        }
        
        if self.scaler:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = output_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Clean old checkpoints
        if self.config.save_total_limit > 0:
            checkpoints = sorted(output_dir.glob("checkpoint_epoch_*.pt"), key=lambda x: int(x.stem.split('_')[-1]))
            if len(checkpoints) > self.config.save_total_limit:
                for old_checkpoint in checkpoints[:-self.config.save_total_limit]:
                    old_checkpoint.unlink()
        
        # Save best model
        if is_best:
            best_path = output_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")

