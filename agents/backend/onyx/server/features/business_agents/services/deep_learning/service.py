"""
Deep Learning Service - Main Service Class
==========================================

Refactored service that integrates all modular components following best practices.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import logging
import asyncio
from datetime import datetime

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)

# Import modular components
from .config.config_loader import ConfigLoader, TrainingConfig, ModelConfig, DataConfig
from .models import (
    BaseModel, SimpleCNN, LSTMTextClassifier, TransformerEncoder
)
# Optional model imports
try:
    from .models.transformers_models import HuggingFaceModel, CLIPTextEncoder
    TRANSFORMERS_MODELS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_MODELS_AVAILABLE = False
    HuggingFaceModel = None
    CLIPTextEncoder = None

try:
    from .models.diffusion_models import DiffusionModel
    DIFFUSION_MODELS_AVAILABLE = True
except ImportError:
    DIFFUSION_MODELS_AVAILABLE = False
    DiffusionModel = None
from .data import SimpleDataset, TextDataset, ImageDataset, create_dataloader, split_dataset
from .training import TrainingManager, EarlyStopping, CheckpointManager, apply_lora, LoraConfig
from .evaluation import ModelEvaluator, compute_metrics
from .utils.distributed import setup_ddp, cleanup_ddp, wrap_model_ddp, is_distributed
from .utils.profiling import profile_training, profile_inference
# Optional optimization imports
try:
    from .utils.optimization import (
        ModelOptimizer, MemoryOptimizer, InferenceOptimizer,
        optimize_model_for_production
    )
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False
    ModelOptimizer = None
    MemoryOptimizer = None
    InferenceOptimizer = None
    optimize_model_for_production = None

# Experiment tracking
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class DeepLearningService:
    """
    Refactored Deep Learning Service with modular architecture.
    
    Features:
    - YAML-based configuration
    - Modular model architectures
    - Efficient data loading
    - Comprehensive training with mixed precision
    - Model evaluation
    - Experiment tracking
    - Checkpointing
    - Distributed training support
    - Gradio integration
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, config_path: Optional[str] = None):
        """
        Initialize deep learning service.
        
        Args:
            config: Configuration dictionary (optional)
            config_path: Path to YAML config file (optional)
        """
        # Load configuration
        if config_path:
            self.config_loader = ConfigLoader(config_path)
        else:
            self.config_loader = ConfigLoader()
            if config:
                self.config_loader.update_config(config)
        
        # Get configurations
        self.model_config = self.config_loader.get_model_config()
        self.data_config = self.config_loader.get_data_config()
        self.training_config = self.config_loader.get_training_config()
        
        # Initialize device
        self.device = self._initialize_device()
        
        # Initialize experiment tracking
        self.tensorboard_writer = None
        self._initialize_experiment_tracking()
        
        # Storage
        self.models = {}
        self.training_jobs = {}
        self.evaluations = {}
        
        logger.info("✅ Deep Learning Service initialized")
    
    def _initialize_device(self) -> torch.device:
        """Initialize and configure device."""
        if self.model_config.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.model_config.device)
        
        # Set random seeds
        seed = self.data_config.seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        # GPU optimizations
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        logger.info(f"✅ Device initialized: {device}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        
        return device
    
    def _initialize_experiment_tracking(self) -> None:
        """Initialize experiment tracking (TensorBoard/W&B)."""
        if self.training_config.use_tensorboard and TENSORBOARD_AVAILABLE:
            log_dir = Path(self.training_config.log_dir) / "tensorboard"
            log_dir.mkdir(parents=True, exist_ok=True)
            self.tensorboard_writer = SummaryWriter(log_dir=str(log_dir))
            logger.info(f"✅ TensorBoard initialized: {log_dir}")
        
        if self.training_config.use_wandb and WANDB_AVAILABLE:
            try:
                wandb.init(
                    project=self.training_config.experiment_name,
                    reinit=True
                )
                logger.info("✅ Weights & Biases initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize wandb: {e}")
    
    def create_model(
        self,
        model_type: str,
        model_id: Optional[str] = None,
        **kwargs
    ) -> BaseModel:
        """
        Create a model instance.
        
        Args:
            model_type: Type of model (cnn, lstm, transformer)
            model_id: Optional model ID for storage
            **kwargs: Model-specific arguments
        
        Returns:
            Model instance
        """
        # Merge config with kwargs
        arch_config = self.model_config.architecture.copy()
        arch_config.update(kwargs)
        
        if model_type.lower() == "cnn":
            model = SimpleCNN(device=self.device, **arch_config)
        elif model_type.lower() == "lstm":
            model = LSTMTextClassifier(device=self.device, **arch_config)
        elif model_type.lower() == "transformer":
            # Use optimized transformer if available
            try:
                from .models.optimized_transformer import OptimizedTransformerEncoder
                model = OptimizedTransformerEncoder(device=self.device, **arch_config)
            except ImportError:
                model = TransformerEncoder(device=self.device, **arch_config)
        elif model_type.lower() == "huggingface" and TRANSFORMERS_MODELS_AVAILABLE:
            model = HuggingFaceModel(
                model_name=arch_config.get("model_name", "bert-base-uncased"),
                task_type=arch_config.get("task_type", "classification"),
                num_labels=arch_config.get("num_labels"),
                device=self.device,
                **{k: v for k, v in arch_config.items() 
                   if k not in ["model_name", "task_type", "num_labels"]}
            )
        elif model_type.lower() == "diffusion" and DIFFUSION_MODELS_AVAILABLE:
            model = DiffusionModel(
                model_name=arch_config.get("model_name", "runwayml/stable-diffusion-v1-5"),
                model_type=arch_config.get("model_type", "stable-diffusion"),
                device=self.device,
                **{k: v for k, v in arch_config.items() 
                   if k not in ["model_name", "model_type"]}
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Apply LoRA if configured
        if self.training_config.use_lora:
            try:
                lora_config = LoraConfig(**self.training_config.lora_config)
                model = apply_lora(model, lora_config)
            except Exception as e:
                logger.warning(f"Failed to apply LoRA: {e}")
        
        # Store model
        if model_id:
            self.models[model_id] = model
        
        logger.info(f"✅ Model created: {model_type}, parameters: {model.count_parameters()}")
        
        return model
    
    def create_dataset(
        self,
        data: Any,
        labels: Optional[Any] = None,
        dataset_type: str = "simple"
    ):
        """
        Create dataset.
        
        Args:
            data: Input data
            labels: Labels (optional)
            dataset_type: Type of dataset (simple, text, image)
        
        Returns:
            Dataset instance
        """
        if dataset_type == "simple":
            return SimpleDataset(data, labels)
        elif dataset_type == "text":
            return TextDataset(data, labels)
        elif dataset_type == "image":
            return ImageDataset(data, labels)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    def train_model(
        self,
        model: nn.Module,
        train_loader,
        val_loader: Optional = None,
        model_id: Optional[str] = None,
        use_optimized: bool = True
    ) -> Dict[str, Any]:
        """
        Train a model with best practices.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            model_id: Optional model ID
            use_optimized: Whether to use optimized trainer
        
        Returns:
            Training history
        """
        # Wrap with DDP if configured
        if self.training_config.use_ddp and is_distributed():
            model = wrap_model_ddp(model, self.device.index or 0)
        
        # Choose trainer
        if use_optimized:
            try:
                from .training.optimized_trainer import OptimizedTrainingManager
                TrainerClass = OptimizedTrainingManager
            except ImportError:
                TrainerClass = TrainingManager
        else:
            TrainerClass = TrainingManager
        
        # Create trainer
        trainer = TrainerClass(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=self.training_config,
            device=self.device,
            experiment_tracker=self.tensorboard_writer or (wandb if WANDB_AVAILABLE else None)
        )
        
        # Train
        history = trainer.train()
        
        # Store model
        if model_id:
            self.models[model_id] = model
        
        logger.info("✅ Training completed")
        
        return history
    
    def evaluate_model(
        self,
        model: nn.Module,
        dataloader,
        compute_roc: bool = True,
        compute_pr: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate model.
        
        Args:
            model: Model to evaluate
            dataloader: Data loader
            compute_roc: Whether to compute ROC curve
            compute_pr: Whether to compute PR curve
        
        Returns:
            Evaluation metrics
        """
        evaluator = ModelEvaluator(model, self.device)
        metrics = evaluator.evaluate(dataloader, compute_roc=compute_roc, compute_pr=compute_pr)
        
        logger.info("✅ Evaluation completed")
        
        return metrics
    
    def save_model(
        self,
        model: nn.Module,
        path: Union[str, Path],
        include_optimizer: bool = False,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> None:
        """Save model checkpoint."""
        if isinstance(model, BaseModel):
            model.save(path, include_optimizer, optimizer)
        else:
            checkpoint_manager = CheckpointManager()
            checkpoint_manager.save_checkpoint(model, optimizer, filename=str(path))
    
    def load_model(
        self,
        model: nn.Module,
        path: Union[str, Path]
    ) -> None:
        """Load model checkpoint."""
        if isinstance(model, BaseModel):
            model.load(path)
        else:
            checkpoint_manager = CheckpointManager()
            checkpoint_manager.load_checkpoint(model, filename=str(path))
    
    def profile_model(
        self,
        model: nn.Module,
        input_data: torch.Tensor,
        mode: str = "inference"
    ) -> Dict[str, Any]:
        """
        Profile model performance.
        
        Args:
            model: Model to profile
            input_data: Sample input
            mode: Profiling mode (inference or training)
        
        Returns:
            Profiling results
        """
        if mode == "inference":
            return profile_inference(model, input_data, self.device)
        else:
            # For training profiling, need a dataloader
            logger.warning("Training profiling requires a dataloader")
            return {}
    
    def create_optimized_inference(
        self,
        model: nn.Module,
        batch_size: int = 32,
        use_cache: bool = True
    ):
        """
        Create optimized inference wrapper.
        
        Args:
            model: Model for inference
            batch_size: Batch size
            use_cache: Whether to use caching
        
        Returns:
            OptimizedInference wrapper
        """
        if OPTIMIZATION_AVAILABLE:
            try:
                from .utils.batch_optimization import OptimizedInference
                return OptimizedInference(
                    model=model,
                    device=self.device,
                    batch_size=batch_size,
                    use_cache=use_cache
                )
            except ImportError:
                logger.warning("Batch optimization not available")
                return None
        else:
            logger.warning("Optimization utilities not available")
            return None
    
    def optimize_model(
        self,
        model: nn.Module,
        for_inference: bool = True,
        compile_model: bool = True
    ) -> nn.Module:
        """
        Optimize model for production.
        
        Args:
            model: Model to optimize
            for_inference: Whether to optimize for inference
            compile_model: Whether to compile with torch.compile
        
        Returns:
            Optimized model
        """
        if OPTIMIZATION_AVAILABLE:
            if for_inference:
                model = ModelOptimizer.optimize_for_inference(model)
            
            if compile_model:
                model = ModelOptimizer.compile_model(model, mode="reduce-overhead")
            
            ModelOptimizer.enable_tf32(model)
            ModelOptimizer.enable_flash_attention(model)
            
            logger.info("✅ Model optimized")
        else:
            logger.warning("Optimization utilities not available")
        
        return model
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Get current GPU memory usage.
        
        Returns:
            Memory usage statistics
        """
        if OPTIMIZATION_AVAILABLE and MemoryOptimizer:
            return MemoryOptimizer.get_memory_usage()
        return {"available": False}
    
    async def get_service_status(self) -> Dict[str, Any]:
        """Get service status."""
        return {
            "status": "active",
            "device": str(self.device),
            "models_loaded": len(self.models),
            "training_jobs": len(self.training_jobs),
            "tensorboard_available": TENSORBOARD_AVAILABLE,
            "wandb_available": WANDB_AVAILABLE,
            "mixed_precision": self.training_config.use_mixed_precision,
            "distributed": self.training_config.use_ddp,
            "timestamp": datetime.utcnow().isoformat()
        }

