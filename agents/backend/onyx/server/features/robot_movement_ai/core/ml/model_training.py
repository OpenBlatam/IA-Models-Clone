"""
Model Training System
=====================

Sistema de entrenamiento y fine-tuning de modelos.
"""

import logging
import uuid
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    DataLoader = None
    ReduceLROnPlateau = None
    CosineAnnealingLR = None

logger = logging.getLogger(__name__)


class TrainingStrategy(Enum):
    """Estrategia de entrenamiento."""
    STANDARD = "standard"
    TRANSFER_LEARNING = "transfer_learning"
    FINE_TUNING = "fine_tuning"
    CONTINUAL_LEARNING = "continual_learning"
    FEW_SHOT = "few_shot"


@dataclass
class TrainingConfig:
    """Configuración de entrenamiento."""
    training_id: str
    model_id: str
    strategy: TrainingStrategy
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    gradient_clip: float = 1.0
    early_stopping_patience: int = 10
    validation_split: float = 0.2
    use_mixed_precision: bool = False
    use_gradient_accumulation: bool = False
    gradient_accumulation_steps: int = 1
    scheduler_type: str = "plateau"  # "plateau", "cosine", "none"
    checkpoint_frequency: int = 10
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingProgress:
    """Progreso de entrenamiento."""
    training_id: str
    epoch: int
    total_epochs: int
    train_loss: float
    val_loss: Optional[float]
    learning_rate: float
    elapsed_time: float
    remaining_time: Optional[float] = None
    status: str = "training"  # "training", "completed", "stopped", "error"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class TrainingResult:
    """Resultado de entrenamiento."""
    training_id: str
    model_id: str
    final_train_loss: float
    final_val_loss: Optional[float]
    best_val_loss: Optional[float]
    total_epochs: int
    completed_epochs: int
    training_time: float
    checkpoints: List[str] = field(default_factory=list)
    metrics_history: List[Dict[str, float]] = field(default_factory=list)
    status: str = "completed"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class ModelTrainer:
    """
    Entrenador de modelos.
    
    Maneja entrenamiento, fine-tuning y transfer learning.
    """
    
    def __init__(self):
        """Inicializar entrenador."""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available. Training features will be limited.")
        
        self.trainings: Dict[str, TrainingConfig] = {}
        self.progress: Dict[str, List[TrainingProgress]] = {}
        self.results: Dict[str, TrainingResult] = {}
        self.callbacks: Dict[str, List[Callable]] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if TORCH_AVAILABLE else None
    
    def start_training(
        self,
        model_id: str,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[TrainingConfig] = None,
        model: Optional[nn.Module] = None,
        loss_fn: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> str:
        """
        Iniciar entrenamiento.
        
        Args:
            model_id: ID del modelo
            train_loader: DataLoader de entrenamiento
            val_loader: DataLoader de validación (opcional)
            config: Configuración (opcional)
            model: Modelo (opcional, se obtiene del manager si no se proporciona)
            loss_fn: Función de pérdida (opcional)
            optimizer: Optimizador (opcional)
            
        Returns:
            ID de entrenamiento
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for training")
        
        training_id = str(uuid.uuid4())
        
        if config is None:
            config = TrainingConfig(
                training_id=training_id,
                model_id=model_id,
                strategy=TrainingStrategy.STANDARD
            )
        
        self.trainings[training_id] = config
        self.progress[training_id] = []
        
        # Obtener modelo si no se proporciona
        if model is None:
            from .deep_learning_models import get_dl_model_manager
            manager = get_dl_model_manager()
            if model_id not in manager.models:
                raise ValueError(f"Model not found: {model_id}")
            model = manager.models[model_id]
        
        model = model.to(self.device)
        
        # Loss y optimizer
        if loss_fn is None:
            loss_fn = nn.MSELoss()
        
        if optimizer is None:
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        
        # Scheduler
        scheduler = None
        if config.scheduler_type == "plateau" and val_loader:
            scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        elif config.scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=config.num_epochs)
        
        # Mixed precision
        scaler = None
        if config.use_mixed_precision and self.device.type == "cuda":
            scaler = torch.cuda.amp.GradScaler()
        
        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        
        import time
        start_time = time.time()
        
        # Entrenamiento
        try:
            for epoch in range(config.num_epochs):
                epoch_start = time.time()
                
                # Training
                model.train()
                train_loss = 0.0
                num_batches = 0
                
                optimizer.zero_grad()
                
                for batch_idx, (inputs, targets) in enumerate(train_loader):
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    
                    if scaler:
                        with torch.cuda.amp.autocast():
                            outputs = model(inputs)
                            loss = loss_fn(outputs, targets)
                            if config.use_gradient_accumulation:
                                loss = loss / config.gradient_accumulation_steps
                        
                        scaler.scale(loss).backward()
                        
                        if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                            if config.gradient_clip > 0:
                                scaler.unscale_(optimizer)
                                torch.nn.utils.clip_grad_norm_(
                                    model.parameters(),
                                    config.gradient_clip
                                )
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()
                    else:
                        outputs = model(inputs)
                        loss = loss_fn(outputs, targets)
                        if config.use_gradient_accumulation:
                            loss = loss / config.gradient_accumulation_steps
                        
                        loss.backward()
                        
                        if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                            if config.gradient_clip > 0:
                                torch.nn.utils.clip_grad_norm_(
                                    model.parameters(),
                                    config.gradient_clip
                                )
                            optimizer.step()
                            optimizer.zero_grad()
                    
                    train_loss += loss.item() * config.gradient_accumulation_steps
                    num_batches += 1
                
                train_loss /= num_batches
                
                # Validación
                val_loss = None
                if val_loader:
                    model.eval()
                    val_loss = 0.0
                    val_batches = 0
                    
                    with torch.no_grad():
                        for inputs, targets in val_loader:
                            inputs = inputs.to(self.device)
                            targets = targets.to(self.device)
                            outputs = model(inputs)
                            loss = loss_fn(outputs, targets)
                            val_loss += loss.item()
                            val_batches += 1
                    
                    val_loss /= val_batches
                    
                    # Scheduler step
                    if scheduler:
                        if isinstance(scheduler, ReduceLROnPlateau):
                            scheduler.step(val_loss)
                        else:
                            scheduler.step()
                    
                    # Early stopping
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= config.early_stopping_patience:
                            logger.info(f"Early stopping at epoch {epoch + 1}")
                            break
                else:
                    if scheduler and not isinstance(scheduler, ReduceLROnPlateau):
                        scheduler.step()
                
                # Guardar checkpoint
                if (epoch + 1) % config.checkpoint_frequency == 0:
                    from .deep_learning_models import get_dl_model_manager
                    manager = get_dl_model_manager()
                    manager.save_checkpoint(
                        model_id,
                        epoch + 1,
                        train_loss,
                        {"val_loss": val_loss} if val_loss else {},
                        optimizer
                    )
                
                # Progreso
                elapsed = time.time() - start_time
                epoch_time = time.time() - epoch_start
                remaining = epoch_time * (config.num_epochs - epoch - 1) if epoch > 0 else None
                
                progress = TrainingProgress(
                    training_id=training_id,
                    epoch=epoch + 1,
                    total_epochs=config.num_epochs,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    learning_rate=optimizer.param_groups[0]['lr'],
                    elapsed_time=elapsed,
                    remaining_time=remaining
                )
                
                self.progress[training_id].append(progress)
                
                # Callbacks
                for callback in self.callbacks.get(training_id, []):
                    try:
                        callback(progress)
                    except Exception as e:
                        logger.error(f"Error in callback: {e}")
                
                if (epoch + 1) % 10 == 0:
                    logger.info(
                        f"Epoch {epoch + 1}/{config.num_epochs} - "
                        f"Train Loss: {train_loss:.4f}" +
                        (f", Val Loss: {val_loss:.4f}" if val_loss else "")
                    )
            
            # Resultado final
            final_time = time.time() - start_time
            
            result = TrainingResult(
                training_id=training_id,
                model_id=model_id,
                final_train_loss=train_loss,
                final_val_loss=val_loss,
                best_val_loss=best_val_loss if val_loader else None,
                total_epochs=config.num_epochs,
                completed_epochs=epoch + 1,
                training_time=final_time,
                status="completed"
            )
            
            self.results[training_id] = result
            
            logger.info(f"Training {training_id} completed in {final_time:.2f}s")
            
            return training_id
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            result = TrainingResult(
                training_id=training_id,
                model_id=model_id,
                final_train_loss=0.0,
                final_val_loss=None,
                best_val_loss=None,
                total_epochs=config.num_epochs,
                completed_epochs=0,
                training_time=0.0,
                status="error"
            )
            self.results[training_id] = result
            raise
    
    def fine_tune_model(
        self,
        model_id: str,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        freeze_base: bool = True,
        learning_rate: float = 0.0001
    ) -> str:
        """
        Fine-tuning de modelo.
        
        Args:
            model_id: ID del modelo
            train_loader: DataLoader de entrenamiento
            val_loader: DataLoader de validación (opcional)
            freeze_base: Congelar capas base
            learning_rate: Learning rate para fine-tuning
            
        Returns:
            ID de entrenamiento
        """
        from .deep_learning_models import get_dl_model_manager
        manager = get_dl_model_manager()
        
        if model_id not in manager.models:
            raise ValueError(f"Model not found: {model_id}")
        
        model = manager.models[model_id]
        
        # Congelar capas base si es necesario
        if freeze_base:
            for param in model.parameters():
                param.requires_grad = False
            
            # Descongelar última capa
            if hasattr(model, 'network'):
                for param in model.network[-1].parameters():
                    param.requires_grad = True
            elif hasattr(model, 'fc'):
                for param in model.fc.parameters():
                    param.requires_grad = True
        
        config = TrainingConfig(
            training_id=str(uuid.uuid4()),
            model_id=model_id,
            strategy=TrainingStrategy.FINE_TUNING,
            learning_rate=learning_rate,
            num_epochs=50
        )
        
        return self.start_training(
            model_id,
            train_loader,
            val_loader,
            config,
            model=model
        )
    
    def add_callback(
        self,
        training_id: str,
        callback: Callable[[TrainingProgress], None]
    ):
        """
        Agregar callback de progreso.
        
        Args:
            training_id: ID de entrenamiento
            callback: Función callback
        """
        if training_id not in self.callbacks:
            self.callbacks[training_id] = []
        self.callbacks[training_id].append(callback)
    
    def get_progress(self, training_id: str) -> List[TrainingProgress]:
        """
        Obtener progreso de entrenamiento.
        
        Args:
            training_id: ID de entrenamiento
            
        Returns:
            Lista de progreso
        """
        return self.progress.get(training_id, [])
    
    def get_result(self, training_id: str) -> Optional[TrainingResult]:
        """
        Obtener resultado de entrenamiento.
        
        Args:
            training_id: ID de entrenamiento
            
        Returns:
            Resultado o None
        """
        return self.results.get(training_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        strategy_counts = {}
        for config in self.trainings.values():
            strategy_counts[config.strategy.value] = strategy_counts.get(config.strategy.value, 0) + 1
        
        return {
            "total_trainings": len(self.trainings),
            "completed_trainings": sum(1 for r in self.results.values() if r.status == "completed"),
            "strategy_counts": strategy_counts
        }


# Instancia global
_model_trainer: Optional[ModelTrainer] = None


def get_model_trainer() -> ModelTrainer:
    """Obtener instancia global del entrenador."""
    global _model_trainer
    if _model_trainer is None:
        _model_trainer = ModelTrainer()
    return _model_trainer




