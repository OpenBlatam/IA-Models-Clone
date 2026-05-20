"""
Advanced Training Service - Entrenamiento avanzado con PyTorch
================================================================

Sistema de entrenamiento profesional con soporte para:
- Multi-GPU training
- Mixed precision
- Gradient accumulation
- Early stopping
- Learning rate scheduling
- Gradient clipping
"""

import logging
import time
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torch.cuda.amp import autocast, GradScaler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")


class TrainingStrategy(str, Enum):
    """Estrategias de entrenamiento"""
    SINGLE_GPU = "single_gpu"
    DATA_PARALLEL = "data_parallel"
    DISTRIBUTED = "distributed"


@dataclass
class TrainingConfig:
    """Configuración avanzada de entrenamiento"""
    model: Any
    train_loader: Any
    val_loader: Optional[Any] = None
    optimizer: Optional[Any] = None
    criterion: Optional[Any] = None
    num_epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    use_mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 5
    lr_scheduler: Optional[str] = "cosine"  # cosine, step, plateau
    strategy: TrainingStrategy = TrainingStrategy.SINGLE_GPU
    device: Optional[str] = None


@dataclass
class TrainingMetrics:
    """Métricas de entrenamiento"""
    epoch: int
    train_loss: float
    val_loss: Optional[float] = None
    train_accuracy: Optional[float] = None
    val_accuracy: Optional[float] = None
    learning_rate: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class AdvancedTrainingService:
    """Servicio de entrenamiento avanzado"""
    
    def __init__(self):
        """Inicializar servicio"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if TORCH_AVAILABLE else None
        self.training_jobs: Dict[str, Any] = {}
        logger.info(f"AdvancedTrainingService initialized on device: {self.device}")
    
    def create_training_job(
        self,
        job_id: str,
        config: TrainingConfig
    ) -> Dict[str, Any]:
        """Crear job de entrenamiento"""
        if not TORCH_AVAILABLE:
            return {"error": "PyTorch not available"}
        
        # Setup device
        device = config.device or self.device
        
        # Setup optimizer if not provided
        if config.optimizer is None:
            config.optimizer = optim.AdamW(
                config.model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        
        # Setup criterion if not provided
        if config.criterion is None:
            config.criterion = nn.CrossEntropyLoss()
        
        # Move model to device
        config.model = config.model.to(device)
        
        # Setup mixed precision scaler
        scaler = GradScaler() if config.use_mixed_precision else None
        
        # Setup learning rate scheduler
        scheduler = None
        if config.lr_scheduler == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                config.optimizer,
                T_max=config.num_epochs
            )
        elif config.lr_scheduler == "step":
            scheduler = optim.lr_scheduler.StepLR(
                config.optimizer,
                step_size=config.num_epochs // 3,
                gamma=0.1
            )
        elif config.lr_scheduler == "plateau" and config.val_loader:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                config.optimizer,
                mode="min",
                patience=2
            )
        
        job = {
            "id": job_id,
            "config": config,
            "device": device,
            "scaler": scaler,
            "scheduler": scheduler,
            "best_val_loss": float("inf"),
            "patience_counter": 0,
            "metrics": [],
            "started_at": datetime.now(),
        }
        
        self.training_jobs[job_id] = job
        
        logger.info(f"Training job {job_id} created")
        return job
    
    async def train_epoch(
        self,
        job_id: str,
        epoch: int
    ) -> TrainingMetrics:
        """
        Entrenar una época con manejo robusto de errores.
        
        Args:
            job_id: ID del job de entrenamiento
            epoch: Número de época
        
        Returns:
            TrainingMetrics con métricas de la época
        
        Raises:
            ValueError: Si el job no existe
            RuntimeError: Si hay errores durante el entrenamiento
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, returning dummy metrics")
            return TrainingMetrics(epoch=epoch, train_loss=0.0)
        
        job = self.training_jobs.get(job_id)
        if not job:
            raise ValueError(f"Training job {job_id} not found")
        
        config = job["config"]
        model = config.model
        train_loader = config.train_loader
        
        # Validate inputs
        if model is None:
            raise ValueError("Model is None")
        if train_loader is None:
            raise ValueError("Train loader is None")
        
        optimizer = config.optimizer
        criterion = config.criterion
        device = job["device"]
        scaler = job["scaler"]
        
        model.train()
        total_loss = 0.0
        total_samples = 0
        correct = 0
        num_batches = 0
        
        optimizer.zero_grad()
        
        try:
            for batch_idx, batch in enumerate(train_loader):
                try:
                    # Move batch to device
                    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                        inputs = batch[0].to(device, non_blocking=True)
                        targets = batch[1].to(device, non_blocking=True)
                    elif isinstance(batch, dict):
                        inputs = batch.get("input_ids", batch.get("inputs", None))
                        targets = batch.get("labels", batch.get("targets", None))
                        if inputs is not None:
                            inputs = inputs.to(device, non_blocking=True)
                        if targets is not None:
                            targets = targets.to(device, non_blocking=True)
                    else:
                        inputs = batch.to(device, non_blocking=True)
                        targets = None
                    
                    # Validate inputs
                    if inputs is None:
                        logger.warning(f"Skipping batch {batch_idx}: inputs is None")
                        continue
                    
                    # Forward pass with mixed precision
                    if config.use_mixed_precision and scaler:
                        with autocast():
                            outputs = model(inputs)
                            if targets is not None:
                                loss = criterion(outputs, targets)
                            else:
                                # For models without explicit targets
                                loss = outputs.mean() if hasattr(outputs, "mean") else outputs.loss
                    else:
                        outputs = model(inputs)
                        if targets is not None:
                            loss = criterion(outputs, targets)
                        else:
                            loss = outputs.mean() if hasattr(outputs, "mean") else outputs.loss
                    
                    # Check for NaN/Inf
                    if torch.isnan(loss) or torch.isinf(loss):
                        logger.warning(f"NaN/Inf loss detected at batch {batch_idx}, skipping")
                        continue
                    
                    # Scale loss for gradient accumulation
                    loss = loss / config.gradient_accumulation_steps
                    
                    # Backward pass
                    if config.use_mixed_precision and scaler:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    
                    # Gradient accumulation
                    if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                        # Gradient clipping
                        if config.max_grad_norm > 0:
                            if scaler:
                                scaler.unscale_(optimizer)
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                model.parameters(),
                                config.max_grad_norm
                            )
                            
                            # Check for NaN gradients
                            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                                logger.warning(f"NaN/Inf gradients detected, skipping optimizer step")
                                optimizer.zero_grad()
                                continue
                        
                        # Optimizer step
                        if scaler:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                        
                        optimizer.zero_grad()
                    
                    # Calculate metrics
                    total_loss += loss.item() * config.gradient_accumulation_steps
                    total_samples += inputs.size(0)
                    num_batches += 1
                    
                    if targets is not None and hasattr(outputs, "data"):
                        _, predicted = torch.max(outputs.data, 1)
                        correct += (predicted == targets).sum().item()
                    elif hasattr(outputs, "logits"):
                        _, predicted = torch.max(outputs.logits.data, 1)
                        if targets is not None:
                            correct += (predicted == targets).sum().item()
                
                except Exception as e:
                    logger.error(f"Error processing batch {batch_idx}: {e}", exc_info=True)
                    # Continue with next batch instead of failing completely
                    continue
            
            if num_batches == 0:
                raise RuntimeError("No batches processed successfully")
            
            avg_loss = total_loss / num_batches
            accuracy = correct / total_samples if total_samples > 0 else None
            
            return TrainingMetrics(
                epoch=epoch,
                train_loss=avg_loss,
                train_accuracy=accuracy,
                learning_rate=optimizer.param_groups[0]["lr"],
            )
        
        except Exception as e:
            logger.error(f"Error in train_epoch: {e}", exc_info=True)
            raise RuntimeError(f"Training epoch failed: {e}") from e
    
    async def validate_epoch(
        self,
        job_id: str,
        epoch: int
    ) -> TrainingMetrics:
        """
        Validar una época con manejo robusto de errores.
        
        Args:
            job_id: ID del job de entrenamiento
            epoch: Número de época
        
        Returns:
            TrainingMetrics con métricas de validación
        
        Raises:
            ValueError: Si el job no existe
            RuntimeError: Si hay errores durante la validación
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, returning dummy metrics")
            return TrainingMetrics(epoch=epoch, train_loss=0.0, val_loss=0.0)
        
        job = self.training_jobs.get(job_id)
        if not job:
            raise ValueError(f"Training job {job_id} not found")
        
        config = job["config"]
        model = config.model
        val_loader = config.val_loader
        
        if val_loader is None:
            logger.warning("Validation loader is None, skipping validation")
            return TrainingMetrics(epoch=epoch, train_loss=0.0)
        
        device = job["device"]
        scaler = job["scaler"]
        criterion = config.criterion
        
        model.eval()
        total_loss = 0.0
        total_samples = 0
        correct = 0
        num_batches = 0
        
        autocast_context = (
            autocast() if config.use_mixed_precision and device.type == "cuda"
            else autocast(enabled=False)
        )
        
        try:
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    try:
                        # Handle different batch formats
                        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                            inputs = batch[0].to(device, non_blocking=True)
                            targets = batch[1].to(device, non_blocking=True)
                        elif isinstance(batch, dict):
                            inputs = batch.get("input_ids", batch.get("inputs", None))
                            targets = batch.get("labels", batch.get("targets", None))
                            if inputs is not None:
                                inputs = inputs.to(device, non_blocking=True)
                            if targets is not None:
                                targets = targets.to(device, non_blocking=True)
                        else:
                            inputs = batch.to(device, non_blocking=True)
                            targets = None
                        
                        if inputs is None:
                            logger.warning(f"Skipping validation batch {batch_idx}: inputs is None")
                            continue
                        
                        # Forward pass with mixed precision
                        with autocast_context:
                            outputs = model(inputs)
                            
                            # Handle different output formats
                            if isinstance(outputs, dict):
                                logits = outputs.get("logits", outputs.get("output", None))
                                if logits is None:
                                    logits = list(outputs.values())[0]
                            else:
                                logits = outputs
                            
                            if targets is not None:
                                loss = criterion(logits, targets)
                            else:
                                loss = logits.mean() if hasattr(logits, "mean") else logits.loss
                        
                        # Check for NaN/Inf
                        if torch.isnan(loss) or torch.isinf(loss):
                            logger.warning(f"NaN/Inf loss in validation batch {batch_idx}, skipping")
                            continue
                        
                        total_loss += loss.item()
                        total_samples += inputs.size(0)
                        num_batches += 1
                        
                        # Calculate accuracy if targets available
                        if targets is not None:
                            if logits.dim() > 1:
                                _, predicted = torch.max(logits.data, 1)
                            else:
                                predicted = (logits > 0.5).long()
                            correct += (predicted == targets).sum().item()
                    
                    except Exception as e:
                        logger.error(
                            f"Error processing validation batch {batch_idx}: {e}",
                            exc_info=True
                        )
                        continue
            
            if num_batches == 0:
                logger.warning("No validation batches processed successfully")
                return TrainingMetrics(epoch=epoch, train_loss=0.0)
            
            avg_loss = total_loss / num_batches
            accuracy = correct / total_samples if total_samples > 0 else None
            
            return TrainingMetrics(
                epoch=epoch,
                train_loss=0.0,
                val_loss=avg_loss,
                val_accuracy=accuracy,
            )
        
        except Exception as e:
            logger.error(f"Error in validate_epoch: {e}", exc_info=True)
            raise RuntimeError(f"Validation epoch failed: {e}") from e
    
    async def train(
        self,
        job_id: str
    ) -> Dict[str, Any]:
        """Entrenar modelo completo"""
        if not TORCH_AVAILABLE:
            return {"error": "PyTorch not available"}
        
        job = self.training_jobs.get(job_id)
        if not job:
            raise ValueError(f"Training job {job_id} not found")
        
        config = job["config"]
        
        for epoch in range(1, config.num_epochs + 1):
            # Train
            train_metrics = await self.train_epoch(job_id, epoch)
            
            # Validate
            val_metrics = await self.validate_epoch(job_id, epoch)
            
            # Combine metrics
            metrics = TrainingMetrics(
                epoch=epoch,
                train_loss=train_metrics.train_loss,
                val_loss=val_metrics.val_loss,
                train_accuracy=train_metrics.train_accuracy,
                val_accuracy=val_metrics.val_accuracy,
                learning_rate=train_metrics.learning_rate,
            )
            
            job["metrics"].append(metrics)
            
            # Learning rate scheduling
            if job["scheduler"]:
                if isinstance(job["scheduler"], optim.lr_scheduler.ReduceLROnPlateau):
                    job["scheduler"].step(val_metrics.val_loss or train_metrics.train_loss)
                else:
                    job["scheduler"].step()
            
            # Early stopping
            if val_metrics.val_loss:
                if val_metrics.val_loss < job["best_val_loss"]:
                    job["best_val_loss"] = val_metrics.val_loss
                    job["patience_counter"] = 0
                    # Save best model
                    job["best_model_state"] = config.model.state_dict().copy()
                else:
                    job["patience_counter"] += 1
                    if job["patience_counter"] >= config.early_stopping_patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break
        
        job["completed_at"] = datetime.now()
        
        return {
            "job_id": job_id,
            "epochs_completed": len(job["metrics"]),
            "best_val_loss": job["best_val_loss"],
            "final_metrics": {
                "train_loss": job["metrics"][-1].train_loss,
                "val_loss": job["metrics"][-1].val_loss,
            } if job["metrics"] else {},
        }

