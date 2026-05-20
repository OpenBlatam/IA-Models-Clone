from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import asyncio
import time
import logging
import threading
import json
import pickle
import hashlib
from typing import Dict, Any, Optional, List, Union, Tuple, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from pathlib import Path
import warnings
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
from sklearn.metrics import (
from sklearn.model_selection import train_test_split, KFold
import optuna
from optuna.samplers import TPESampler
import mlflow
import wandb
from tqdm import tqdm
import torch.profiler
import cProfile
import pstats
import io
from .production_transformers import ProductionTransformersEngine, DeviceManager
from .diffusion_models import ProductionDiffusionEngine
from .llm_models import ProductionLLMEngine
from .efficient_data_loader import (
from .data_splitting_cv import (
from .early_stopping_lr_scheduling import (
from .evaluation_metrics import (
        from .efficient_data_loader import create_data_loader_manager
        from .data_splitting_cv import create_data_splitting_manager
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
                from peft import get_peft_model, LoraConfig
            import gc
            import multiprocessing as mp
            import psutil
from typing import Any, List, Dict, Optional
"""
🚀 Model Training & Evaluation System - Production Ready
========================================================

Enterprise-grade model training and evaluation system with distributed training,
hyperparameter optimization, advanced metrics, and production integration.

Logging: This module implements comprehensive, production-grade logging for all training progress and errors. Logs are structured, info-level logs are used for progress, and all exceptions are logged with stack traces for robust error monitoring and debugging.

Debugging: This module includes comprehensive PyTorch debugging tools including autograd.detect_anomaly(), gradient checking, memory profiling, and performance monitoring. These can be enabled via TrainingConfig debug flags for troubleshooting training issues.

Performance: This module includes enterprise-grade performance optimization features including GPU acceleration, memory optimization, batch processing optimization, and real-time performance monitoring for maximum training efficiency.

Multi-GPU: This module includes comprehensive multi-GPU training support using both DataParallel and DistributedDataParallel for scaling training across multiple GPUs with automatic device management and optimization.
"""


    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, classification_report, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)

# Import our production engines
    DataLoaderManager, DataLoaderConfig, DataFormat, CacheStrategy,
    OptimizedTextDataset, CachedDataset
)
    DataSplittingManager, SplitConfig, SplitStrategy, CrossValidationConfig,
    CrossValidationStrategy, SplitResult, CrossValidationResult
)
    TrainingMonitor, EarlyStoppingConfig, EarlyStoppingStrategy, EarlyStoppingMode,
    LRSchedulerConfig, LRSchedulerType, create_training_monitor,
    create_early_stopping_config, create_lr_scheduler_config
)
    EvaluationMetrics, MetricConfig, TaskType, MetricType, EvaluationResult,
    create_evaluation_metrics, create_metric_config
)

# Set up global logging configuration if not already present
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)

logger = logging.getLogger(__name__)

class TrainingMode(Enum):
    """Available training modes."""
    FINE_TUNE = "fine_tune"
    TRANSFER_LEARNING = "transfer_learning"
    FROM_SCRATCH = "from_scratch"
    LORA = "lora"
    P_TUNING = "p_tuning"

class ModelType(Enum):
    """Available model types."""
    TRANSFORMER = "transformer"
    DIFFUSION = "diffusion"
    LLM = "llm"
    CUSTOM = "custom"

@dataclass
class TrainingConfig:
    """Training configuration."""
    model_type: ModelType
    training_mode: TrainingMode
    model_name: str
    dataset_path: str
    output_dir: str = "models"
    
    # Training parameters
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 10
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    
    # Advanced training
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    effective_batch_size: int = -1  # Auto-calculate based on batch_size * gradient_accumulation_steps * num_gpus
    gradient_accumulation_scheduling: bool = False  # Dynamic gradient accumulation
    gradient_accumulation_warmup_steps: int = 0  # Gradual increase in accumulation steps
    gradient_accumulation_max_steps: int = 16  # Maximum accumulation steps
    early_stopping_patience: int = 5
    save_steps: int = 500
    eval_steps: int = 500
    
    # Multi-GPU training
    distributed: bool = False
    distributed_backend: str = "nccl"  # "nccl" for GPU, "gloo" for CPU
    distributed_init_method: str = "env://"
    world_size: int = -1
    rank: int = -1
    local_rank: int = -1
    num_gpus: int = 1
    use_data_parallel: bool = False
    use_distributed_data_parallel: bool = False
    find_unused_parameters: bool = False
    gradient_as_bucket_view: bool = True
    broadcast_buffers: bool = True
    bucket_cap_mb: int = 25
    static_graph: bool = False
    
    # Hyperparameter optimization
    enable_hpo: bool = False
    hpo_trials: int = 50
    
    # Evaluation
    eval_split: float = 0.2
    test_split: float = 0.1
    cross_validation_folds: int = 5
    
    # Logging
    log_to_tensorboard: bool = True
    log_to_wandb: bool = False
    log_to_mlflow: bool = False
    
    # Debugging and monitoring
    debug_mode: bool = False
    detect_anomaly: bool = False
    gradient_checking: bool = False
    memory_profiling: bool = False
    performance_profiling: bool = False
    
    # Performance optimization
    enable_gpu_optimization: bool = True
    enable_memory_optimization: bool = True
    enable_batch_optimization: bool = True
    enable_compilation: bool = False  # torch.compile() for PyTorch 2.0+
    enable_amp: bool = True  # Automatic Mixed Precision
    enable_gradient_checkpointing: bool = False
    enable_dynamic_batching: bool = True
    enable_pin_memory: bool = True
    enable_persistent_workers: bool = True
    num_workers: int = 4
    prefetch_factor: int = 2
    
    # Advanced performance
    enable_cudnn_benchmark: bool = True
    enable_cudnn_deterministic: bool = False
    enable_tf32: bool = True  # TensorFloat-32 for Ampere GPUs
    enable_channels_last: bool = False  # Memory format optimization
    enable_compile_mode: str = "default"  # "default", "reduce-overhead", "max-autotune"
    
    def __post_init__(self) -> Any:
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Auto-detect multi-GPU settings
        if self.num_gpus == -1:
            self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        
        # Auto-detect distributed settings
        if self.world_size == -1:
            self.world_size = self.num_gpus
        
        # Set distributed flags based on GPU count
        if self.num_gpus > 1:
            if not self.use_data_parallel and not self.use_distributed_data_parallel:
                # Default to DataParallel for simplicity
                self.use_data_parallel = True
        
        # Calculate effective batch size
        if self.effective_batch_size == -1:
            self.effective_batch_size = self.batch_size * self.gradient_accumulation_steps * self.num_gpus
        
        # Validate gradient accumulation settings
        if self.gradient_accumulation_steps < 1:
            self.gradient_accumulation_steps = 1
        if self.gradient_accumulation_max_steps < self.gradient_accumulation_steps:
            self.gradient_accumulation_max_steps = self.gradient_accumulation_steps
    
    def _calculate_effective_batch_size(self) -> Any:
        return self.batch_size * self.gradient_accumulation_steps * self.num_gpus

@dataclass
class TrainingMetrics:
    """Training metrics."""
    epoch: int
    train_loss: float
    val_loss: float
    train_accuracy: float
    val_accuracy: float
    learning_rate: float
    training_time: float
    
    # Advanced metrics
    train_f1: Optional[float] = None
    val_f1: Optional[float] = None
    train_precision: Optional[float] = None
    val_precision: Optional[float] = None
    train_recall: Optional[float] = None
    val_recall: Optional[float] = None

@dataclass
class EvaluationResult:
    """Evaluation results."""
    model_name: str
    test_accuracy: float
    test_f1: float
    test_precision: float
    test_recall: float
    confusion_matrix: np.ndarray
    classification_report: str
    roc_auc: Optional[float] = None
    mse: Optional[float] = None
    mae: Optional[float] = None
    r2_score: Optional[float] = None
    inference_time_ms: float = 0.0

class CustomDataset(Dataset):
    """Custom dataset for training."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer=None, max_length: int = 512):
        
    """__init__ function."""
self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> Any:
        return len(self.texts)
    
    def __getitem__(self, idx) -> Optional[Dict[str, Any]]:
        text = self.texts[idx]
        label = self.labels[idx]
        
        if self.tokenizer:
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)
            }
        else:
            return {
                'text': text,
                'labels': torch.tensor(label, dtype=torch.long)
            }

class ProfilerManager:
    """
    Manages profiling for data loading, preprocessing, and training.
    """
    def __init__(self, enabled: bool = False, output_dir: str = "./profile_logs"):
        
    """__init__ function."""
self.enabled = enabled
        self.output_dir = output_dir
        self.profiler = None
        self.profile_results = {}
        self.cpu_profiler = None
        self.cpu_stats = None
        os.makedirs(self.output_dir, exist_ok=True)

    def start_torch_profiler(self, activities=None, schedule=None, record_shapes=True, profile_memory=True, with_stack=True) -> Any:
        if not self.enabled:
            return
        self.profiler = torch.profiler.profile(
            activities=activities or [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=schedule or torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(self.output_dir),
            record_shapes=record_shapes,
            profile_memory=profile_memory,
            with_stack=with_stack
        )
        self.profiler.__enter__()

    def step_torch_profiler(self) -> Any:
        if self.profiler:
            self.profiler.step()

    def stop_torch_profiler(self) -> Any:
        if self.profiler:
            self.profiler.__exit__(None, None, None)
            self.profiler = None

    def start_cpu_profiler(self) -> Any:
        if not self.enabled:
            return
        self.cpu_profiler = cProfile.Profile()
        self.cpu_profiler.enable()

    def stop_cpu_profiler(self, label="profile") -> Any:
        if self.cpu_profiler:
            self.cpu_profiler.disable()
            s = io.StringIO()
            ps = pstats.Stats(self.cpu_profiler, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
            ps.print_stats(30)
            self.cpu_stats = s.getvalue()
            with open(os.path.join(self.output_dir, f"{label}_cpu_profile.txt"), "w") as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                f.write(self.cpu_stats)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            self.cpu_profiler = None

    def get_cpu_profile_summary(self) -> Optional[Dict[str, Any]]:
        return self.cpu_stats

    def log_profile_summary(self, logger, label="profile") -> Any:
        if self.cpu_stats:
            logger.info(f"CPU Profile Summary ({label}):\n{self.cpu_stats}")

class ModelTrainer:
    """Production-ready model trainer."""
    
    def __init__(self, device_manager: DeviceManager):
        
    """__init__ function."""
self.device_manager = device_manager
        self.device = device_manager.get_best_device()
        self.logger = logging.getLogger(f"{__name__}.ModelTrainer")
        self.writer = None
        self.best_model_path = None
        self.best_metric = float('inf')
        self.patience_counter = 0
        
        # Initialize data loader manager
        self.data_loader_manager = None
        self.data_splitting_manager = None
        self.training_monitor = None
        self.evaluation_metrics = None
        self._init_data_loader_manager()
        
        # Debug state
        self.debug_enabled = False
        self.anomaly_detection_enabled = False
        self.gradient_checking_enabled = False
        self.memory_profiling_enabled = False
        self.performance_profiling_enabled = False
        
        # Performance optimization state
        self.gpu_optimized = False
        self.memory_optimized = False
        self.batch_optimized = False
        self.compiled_model = None
        self.performance_metrics = {
            'gpu_memory_allocated': [],
            'gpu_memory_reserved': [],
            'batch_processing_times': [],
            'epoch_times': [],
            'throughput_samples_per_sec': [],
            'memory_efficiency': []
        }
        
        # Multi-GPU training state
        self.is_distributed = False
        self.is_data_parallel = False
        self.distributed_process_group = None
        self.distributed_rank = 0
        self.distributed_world_size = 1
        self.distributed_local_rank = 0
        self.multi_gpu_devices = []
        self.master_device = None
        self.current_epoch = 0 # Added for distributed sampler
        
        # Gradient accumulation state
        self.gradient_accumulation_enabled = False
        self.current_accumulation_step = 0
        self.accumulation_steps_history = []
        self.effective_batch_size_history = []
        self.gradient_accumulation_metrics = {
            'total_accumulation_steps': 0,
            'average_accumulation_steps': 0,
            'max_accumulation_steps': 0,
            'accumulation_efficiency': 0.0,
            'memory_savings_mb': 0.0
        }
        self.profiler_manager = ProfilerManager(enabled=self.performance_profiling, output_dir="profile_logs")
    
    async def _init_data_loader_manager(self) -> Any:
        """Initialize data loader manager."""
        
        self.data_loader_manager = await create_data_loader_manager(self.device_manager)
        self.data_splitting_manager = await create_data_splitting_manager(self.device_manager)
        self.training_monitor = await create_training_monitor(self.device_manager)
        self.evaluation_metrics = await create_evaluation_metrics(self.device_manager)
        
    def setup_logging(self, config: TrainingConfig):
        """Setup logging and experiment tracking."""
        if config.log_to_tensorboard:
            log_dir = config.output_dir / "logs" / "tensorboard"
            log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(log_dir))
        
        if config.log_to_wandb:
            wandb.init(
                project="blatam-ai-training",
                name=f"{config.model_name}_{config.training_mode.value}",
                config=vars(config)
            )
        
        if config.log_to_mlflow:
            mlflow.set_tracking_uri("sqlite:///mlflow.db")
            mlflow.start_run(run_name=f"{config.model_name}_{config.training_mode.value}")
    
    async def load_dataset(self, config: TrainingConfig) -> Tuple[Dataset, Dataset, Dataset]:
        """Load and split dataset with profiling."""
        if self.performance_profiling:
            self.profiler_manager.start_cpu_profiler()
        result = await self._load_dataset_impl(config)
        if self.performance_profiling:
            self.profiler_manager.stop_cpu_profiler(label="data_loading")
            self.profiler_manager.log_profile_summary(self.logger, label="data_loading")
        return result

    async def _load_dataset_impl(self, config: TrainingConfig) -> Tuple[Dataset, Dataset, Dataset]:
        """Load and split dataset using efficient data loading and proper splitting."""
        if self.data_loader_manager is None:
            await self._init_data_loader_manager()
        
        # Determine data format
        data_format = self._detect_data_format(config.dataset_path)
        
        # Create data loader config
        dataloader_config = DataLoaderConfig(
            batch_size=config.batch_size,
            num_workers=4,  # Optimal for most systems
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
            cache_strategy=CacheStrategy.MEMORY if config.enable_hpo else CacheStrategy.HYBRID,
            cache_dir=str(config.output_dir / "cache"),
            cache_size_gb=5.0
        )
        
        # Load full dataset
        full_dataset, _ = await self.data_loader_manager.load_dataset(
            config.dataset_path, data_format, dataloader_config
        )
        
        # Create split configuration
        split_config = SplitConfig(
            strategy=SplitStrategy.STRATIFIED,  # Use stratified splitting by default
            train_ratio=1 - config.eval_split - config.test_split,
            val_ratio=config.eval_split,
            test_ratio=config.test_split,
            random_state=config.random_state if hasattr(config, 'random_state') else 42,
            shuffle=True
        )
        
        # Split dataset using proper splitting strategy
        split_result = self.data_splitting_manager.splitter.split_dataset(full_dataset, split_config)
        
        # Analyze split quality
        split_quality = self.data_splitting_manager.analyze_split_quality(split_result)
        
        self.logger.info(f"Dataset loaded: {len(full_dataset)} total samples")
        self.logger.info(f"Split: {len(split_result.train_dataset)} train, {len(split_result.val_dataset)} val, {len(split_result.test_dataset)} test")
        self.logger.info(f"Split quality - distribution similarity: {split_quality['distribution_similarity']:.4f}")
        
        return split_result.train_dataset, split_result.val_dataset, split_result.test_dataset
    
    def _detect_data_format(self, dataset_path: str) -> DataFormat:
        """Detect data format from file extension."""
        path = Path(dataset_path)
        extension = path.suffix.lower()
        
        format_mapping = {
            '.csv': DataFormat.CSV,
            '.json': DataFormat.JSON,
            '.h5': DataFormat.HDF5,
            '.hdf5': DataFormat.HDF5,
            '.lmdb': DataFormat.LMDB,
            '.parquet': DataFormat.PARQUET,
            '.pkl': DataFormat.PICKLE,
            '.pickle': DataFormat.PICKLE,
            '.npy': DataFormat.NUMPY
        }
        
        return format_mapping.get(extension, DataFormat.CSV)
    
    def create_model(self, config: TrainingConfig, num_classes: int) -> nn.Module:
        """Create model based on configuration."""
        if config.model_type == ModelType.TRANSFORMER:
            
            model = AutoModelForSequenceClassification.from_pretrained(
                config.model_name,
                num_labels=num_classes
            )
            tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            
            # Apply training mode specific modifications
            if config.training_mode == TrainingMode.LORA:
                
                peft_config = LoraConfig(
                    task_type="SEQUENCE_CLASSIFICATION",
                    inference_mode=False,
                    r=8,
                    lora_alpha=32,
                    lora_dropout=0.1
                )
                model = get_peft_model(model, peft_config)
            
            return model, tokenizer
        
        elif config.model_type == ModelType.CUSTOM:
            # Implement custom model architecture
            model = self._create_custom_model(num_classes)
            return model, None
        
        else:
            raise ValueError(f"Unsupported model type: {config.model_type}")
    
    def _create_custom_model(self, num_classes: int) -> nn.Module:
        """Create custom model architecture."""
        class CustomTransformer(nn.Module):
            def __init__(self, num_classes: int, vocab_size: int = 30522, hidden_size: int = 768):
                
    """__init__ function."""
super().__init__()
                self.embedding = nn.Embedding(vocab_size, hidden_size)
                self.transformer = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=hidden_size,
                        nhead=12,
                        dim_feedforward=3072,
                        dropout=0.1
                    ),
                    num_layers=6
                )
                self.classifier = nn.Linear(hidden_size, num_classes)
                self.dropout = nn.Dropout(0.1)
            
            def forward(self, input_ids, attention_mask=None) -> Any:
                x = self.embedding(input_ids)
                if attention_mask is not None:
                    x = x * attention_mask.unsqueeze(-1)
                x = self.transformer(x)
                x = x.mean(dim=1)  # Global average pooling
                x = self.dropout(x)
                return self.classifier(x)
        
        return CustomTransformer(num_classes)
    
    def create_optimizer(self, model: nn.Module, config: TrainingConfig):
        """Create optimizer with advanced features."""
        # Separate parameters for different learning rates
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': config.weight_decay,
            },
            {
                'params': [p for n, p in model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            },
        ]
        
        optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            lr=config.learning_rate,
            eps=1e-8
        )
        
        return optimizer
    
    def create_scheduler(self, optimizer, config: TrainingConfig, num_training_steps: int):
        """Create learning rate scheduler with advanced scheduling."""
        # Create LR scheduler configuration
        lr_config = create_lr_scheduler_config(
            scheduler_type=LRSchedulerType.COSINE_ANNEALING,
            initial_lr=config.learning_rate,
            min_lr=config.learning_rate * 0.01,  # 1% of initial LR
            max_lr=config.learning_rate
        )
        
        # Setup scheduler through training monitor
        scheduler = self.training_monitor.setup_lr_scheduler(lr_config, optimizer, num_training_steps)
        
        return scheduler
    
    def calculate_metrics(self, predictions: np.ndarray, targets: np.ndarray, 
                         task_type: str = "classification", probabilities: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate comprehensive metrics using the evaluation metrics system."""
        # Determine task type
        if task_type == "classification":
            task_enum = TaskType.CLASSIFICATION
        elif task_type == "regression":
            task_enum = TaskType.REGRESSION
        elif task_type == "generation":
            task_enum = TaskType.GENERATION
        elif task_type == "translation":
            task_enum = TaskType.TRANSLATION
        elif task_type == "summarization":
            task_enum = TaskType.SUMMARIZATION
        elif task_type == "question_answering":
            task_enum = TaskType.QUESTION_ANSWERING
        elif task_type == "ner":
            task_enum = TaskType.NAMED_ENTITY_RECOGNITION
        elif task_type == "sentiment":
            task_enum = TaskType.SENTIMENT_ANALYSIS
        elif task_type == "text_classification":
            task_enum = TaskType.TEXT_CLASSIFICATION
        elif task_type == "image_classification":
            task_enum = TaskType.IMAGE_CLASSIFICATION
        elif task_type == "object_detection":
            task_enum = TaskType.OBJECT_DETECTION
        elif task_type == "segmentation":
            task_enum = TaskType.SEGMENTATION
        elif task_type == "multi_label":
            task_enum = TaskType.MULTI_LABEL
        elif task_type == "ranking":
            task_enum = TaskType.RANKING
        elif task_type == "recommendation":
            task_enum = TaskType.RECOMMENDATION
        elif task_type == "anomaly_detection":
            task_enum = TaskType.ANOMALY_DETECTION
        elif task_type == "clustering":
            task_enum = TaskType.CLUSTERING
        else:
            task_enum = TaskType.CLASSIFICATION  # Default
        
        # Create metric configuration
        config = create_metric_config(task_type=task_enum)
        
        # Evaluate using the evaluation metrics system
        result = self.evaluation_metrics.evaluate(config, targets, predictions, probabilities)
        
        return result.metrics
    
    def setup_debugging(self, config: TrainingConfig):
        """Setup PyTorch debugging tools and monitoring."""
        self.debug_enabled = config.debug_mode
        self.anomaly_detection_enabled = config.detect_anomaly
        self.gradient_checking_enabled = config.gradient_checking
        self.memory_profiling_enabled = config.memory_profiling
        self.performance_profiling_enabled = config.performance_profiling
        
        if self.debug_enabled:
            self.logger.info("🔧 Debug mode enabled")
            
            # Enable autograd anomaly detection
            if self.anomaly_detection_enabled:
                torch.autograd.set_detect_anomaly(True)
                self.logger.info("🔍 Autograd anomaly detection enabled")
            
            # Enable gradient checking
            if self.gradient_checking_enabled:
                torch.autograd.set_detect_anomaly(True)
                self.logger.info("🔍 Gradient checking enabled")
            
            # Setup memory profiling
            if self.memory_profiling_enabled:
                self.logger.info("📊 Memory profiling enabled")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    self.logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                    self.logger.info(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
            
            # Setup performance profiling
            if self.performance_profiling_enabled:
                self.logger.info("⚡ Performance profiling enabled")
        
        else:
            # Disable debugging features for production
            torch.autograd.set_detect_anomaly(False)
            self.logger.info("🚀 Production mode - debugging disabled")
    
    def setup_performance_optimization(self, config: TrainingConfig):
        """Setup comprehensive performance optimization."""
        self.logger.info("🚀 Setting up performance optimization")
        
        # GPU optimizations
        if config.enable_gpu_optimization and torch.cuda.is_available():
            self._setup_gpu_optimization(config)
        
        # Memory optimizations
        if config.enable_memory_optimization:
            self._setup_memory_optimization(config)
        
        # Batch optimizations
        if config.enable_batch_optimization:
            self._setup_batch_optimization(config)
        
        # PyTorch 2.0+ compilation
        if config.enable_compilation and hasattr(torch, 'compile'):
            self.logger.info("🔧 PyTorch compilation enabled")
        
        self.logger.info("✅ Performance optimization setup completed")
    
    def setup_multi_gpu_training(self, config: TrainingConfig):
        """Setup multi-GPU training configuration."""
        self.logger.info("🚀 Setting up multi-GPU training")
        
        # Detect available GPUs
        available_gpus = torch.cuda.device_count()
        self.logger.info(f"Available GPUs: {available_gpus}")
        
        if available_gpus == 0:
            self.logger.warning("No GPUs available, falling back to CPU training")
            config.num_gpus = 1
            config.use_data_parallel = False
            config.use_distributed_data_parallel = False
            return
        
        # Auto-detect number of GPUs if not specified
        if config.num_gpus == -1:
            config.num_gpus = available_gpus
            self.logger.info(f"Auto-detected {config.num_gpus} GPUs")
        
        # Validate GPU count
        if config.num_gpus > available_gpus:
            self.logger.warning(f"Requested {config.num_gpus} GPUs but only {available_gpus} available")
            config.num_gpus = available_gpus
        
        if config.num_gpus == 1:
            self.logger.info("Single GPU training - no parallelization needed")
            config.use_data_parallel = False
            config.use_distributed_data_parallel = False
            return
        
        # Setup multi-GPU devices
        self.multi_gpu_devices = list(range(config.num_gpus))
        self.master_device = torch.device(f"cuda:{self.multi_gpu_devices[0]}")
        
        self.logger.info(f"Multi-GPU setup: {config.num_gpus} GPUs, devices: {self.multi_gpu_devices}")
        self.logger.info(f"Master device: {self.master_device}")
        
        # Choose parallelization strategy
        if config.use_distributed_data_parallel:
            self._setup_distributed_training(config)
        elif config.use_data_parallel:
            self._setup_data_parallel_training(config)
        else:
            # Auto-choose based on GPU count
            if config.num_gpus <= 4:
                self._setup_data_parallel_training(config)
            else:
                self._setup_distributed_training(config)
    
    def _setup_data_parallel_training(self, config: TrainingConfig):
        """Setup DataParallel training."""
        try:
            self.is_data_parallel = True
            self.logger.info("📦 Setting up DataParallel training")
            
            # Set master device
            self.device = self.master_device
            
            # Log DataParallel configuration
            self.logger.info(f"DataParallel devices: {self.multi_gpu_devices}")
            self.logger.info(f"Master device: {self.master_device}")
            
            # Adjust batch size for DataParallel
            effective_batch_size = config.batch_size * config.num_gpus
            self.logger.info(f"Effective batch size: {effective_batch_size} (per GPU: {config.batch_size})")
            
        except Exception as e:
            self.logger.error(f"DataParallel setup failed: {e}", exc_info=True)
            raise
    
    def _setup_distributed_training(self, config: TrainingConfig):
        """Setup DistributedDataParallel training."""
        try:
            self.is_distributed = True
            self.logger.info("🌐 Setting up DistributedDataParallel training")
            
            # Initialize distributed process group
            if config.distributed_init_method == "env://":
                # Use environment variables
                if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
                    self.logger.warning("Distributed environment variables not set, using defaults")
                    os.environ["RANK"] = "0"
                    os.environ["WORLD_SIZE"] = str(config.num_gpus)
                    os.environ["MASTER_ADDR"] = "localhost"
                    os.environ["MASTER_PORT"] = "12355"
            
            # Initialize process group
            torch.distributed.init_process_group(
                backend=config.distributed_backend,
                init_method=config.distributed_init_method,
                world_size=config.world_size,
                rank=config.rank
            )
            
            # Set distributed rank and world size
            self.distributed_rank = torch.distributed.get_rank()
            self.distributed_world_size = torch.distributed.get_world_size()
            self.distributed_local_rank = config.local_rank if config.local_rank != -1 else 0
            
            # Set device for current process
            self.device = torch.device(f"cuda:{self.distributed_local_rank}")
            torch.cuda.set_device(self.device)
            
            self.logger.info(f"Distributed training initialized:")
            self.logger.info(f"  Rank: {self.distributed_rank}/{self.distributed_world_size}")
            self.logger.info(f"  Local rank: {self.distributed_local_rank}")
            self.logger.info(f"  Device: {self.device}")
            self.logger.info(f"  Backend: {config.distributed_backend}")
            
            # Store process group
            self.distributed_process_group = torch.distributed.group.WORLD
            
            # Adjust batch size for distributed training
            effective_batch_size = config.batch_size * self.distributed_world_size
            self.logger.info(f"Effective batch size: {effective_batch_size} (per GPU: {config.batch_size})")
            
        except Exception as e:
            self.logger.error(f"Distributed training setup failed: {e}", exc_info=True)
            raise
    
    def wrap_model_for_multi_gpu(self, model: nn.Module, config: TrainingConfig) -> nn.Module:
        """Wrap model for multi-GPU training."""
        try:
            if self.is_distributed:
                return self._wrap_model_for_distributed(model, config)
            elif self.is_data_parallel:
                return self._wrap_model_for_data_parallel(model, config)
            else:
                return model
        
        except Exception as e:
            self.logger.error(f"Model wrapping failed: {e}", exc_info=True)
            raise
    
    def _wrap_model_for_data_parallel(self, model: nn.Module, config: TrainingConfig) -> nn.Module:
        """Wrap model for DataParallel training."""
        try:
            self.logger.info("📦 Wrapping model for DataParallel")
            
            # Move model to master device first
            model = model.to(self.master_device)
            
            # Wrap with DataParallel
            model = nn.DataParallel(
                model,
                device_ids=self.multi_gpu_devices,
                output_device=self.master_device
            )
            
            self.logger.info(f"DataParallel model created with devices: {self.multi_gpu_devices}")
            return model
        
        except Exception as e:
            self.logger.error(f"DataParallel wrapping failed: {e}", exc_info=True)
            raise
    
    def _wrap_model_for_distributed(self, model: nn.Module, config: TrainingConfig) -> nn.Module:
        """Wrap model for DistributedDataParallel training."""
        try:
            self.logger.info("🌐 Wrapping model for DistributedDataParallel")
            
            # Move model to current device
            model = model.to(self.device)
            
            # Wrap with DistributedDataParallel
            model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.distributed_local_rank],
                output_device=self.distributed_local_rank,
                find_unused_parameters=config.find_unused_parameters,
                gradient_as_bucket_view=config.gradient_as_bucket_view,
                broadcast_buffers=config.broadcast_buffers,
                bucket_cap_mb=config.bucket_cap_mb,
                static_graph=config.static_graph
            )
            
            self.logger.info(f"DistributedDataParallel model created for rank {self.distributed_rank}")
            return model
        
        except Exception as e:
            self.logger.error(f"DistributedDataParallel wrapping failed: {e}", exc_info=True)
            raise
    
    def create_multi_gpu_dataloaders(self, train_dataset, val_dataset, test_dataset, config: TrainingConfig):
        """Create DataLoaders optimized for multi-GPU training."""
        try:
            self.logger.info("📊 Creating multi-GPU DataLoaders")
            
            # Adjust batch size for multi-GPU
            if self.is_distributed:
                effective_batch_size = config.batch_size
                # DistributedDataParallel handles batch splitting automatically
            elif self.is_data_parallel:
                effective_batch_size = config.batch_size
                # DataParallel handles batch splitting automatically
            else:
                effective_batch_size = config.batch_size
            
            # Create DataLoader configuration
            dataloader_config = DataLoaderConfig(
                batch_size=effective_batch_size,
                num_workers=config.num_workers,
                pin_memory=config.enable_pin_memory,
                persistent_workers=config.enable_persistent_workers,
                prefetch_factor=config.prefetch_factor,
                shuffle=True
            )
            
            # Add distributed sampler for DistributedDataParallel
            if self.is_distributed:
                train_sampler = torch.utils.data.distributed.DistributedSampler(
                    train_dataset,
                    num_replicas=self.distributed_world_size,
                    rank=self.distributed_rank,
                    shuffle=True
                )
                val_sampler = torch.utils.data.distributed.DistributedSampler(
                    val_dataset,
                    num_replicas=self.distributed_world_size,
                    rank=self.distributed_rank,
                    shuffle=False
                )
                test_sampler = torch.utils.data.distributed.DistributedSampler(
                    test_dataset,
                    num_replicas=self.distributed_world_size,
                    rank=self.distributed_rank,
                    shuffle=False
                )
                
                # Create DataLoaders with distributed samplers
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=effective_batch_size,
                    sampler=train_sampler,
                    num_workers=config.num_workers,
                    pin_memory=config.enable_pin_memory,
                    persistent_workers=config.enable_persistent_workers,
                    prefetch_factor=config.prefetch_factor
                )
                
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=effective_batch_size,
                    sampler=val_sampler,
                    num_workers=config.num_workers,
                    pin_memory=config.enable_pin_memory,
                    persistent_workers=config.enable_persistent_workers,
                    prefetch_factor=config.prefetch_factor
                )
                
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=effective_batch_size,
                    sampler=test_sampler,
                    num_workers=config.num_workers,
                    pin_memory=config.enable_pin_memory,
                    persistent_workers=config.enable_persistent_workers,
                    prefetch_factor=config.prefetch_factor
                )
                
                self.logger.info(f"Created distributed DataLoaders with samplers")
                
            else:
                # Use standard DataLoader creation
                train_loader, val_loader, test_loader = self.data_loader_manager.create_dataloaders(
                    train_dataset, val_dataset, test_dataset, dataloader_config
                )
            
            return train_loader, val_loader, test_loader
        
        except Exception as e:
            self.logger.error(f"Multi-GPU DataLoader creation failed: {e}", exc_info=True)
            raise
    
    def sync_metrics_across_gpus(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Synchronize metrics across all GPUs in distributed training."""
        try:
            if not self.is_distributed:
                return metrics
            
            self.logger.debug("🔄 Synchronizing metrics across GPUs")
            
            synced_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    # Convert to tensor and synchronize
                    tensor = torch.tensor(value, device=self.device, dtype=torch.float32)
                    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
                    synced_metrics[key] = tensor.item() / self.distributed_world_size
                else:
                    synced_metrics[key] = value
            
            return synced_metrics
        
        except Exception as e:
            self.logger.warning(f"Metrics synchronization failed: {e}")
            return metrics
    
    def get_multi_gpu_summary(self) -> Dict[str, Any]:
        """Get comprehensive multi-GPU training summary."""
        try:
            summary = {
                'multi_gpu_enabled': self.is_distributed or self.is_data_parallel,
                'training_type': 'distributed' if self.is_distributed else 'data_parallel' if self.is_data_parallel else 'single_gpu',
                'num_gpus': len(self.multi_gpu_devices) if self.multi_gpu_devices else 1,
                'devices': self.multi_gpu_devices,
                'master_device': str(self.master_device) if self.master_device else str(self.device)
            }
            
            if self.is_distributed:
                summary.update({
                    'distributed_rank': self.distributed_rank,
                    'distributed_world_size': self.distributed_world_size,
                    'distributed_local_rank': self.distributed_local_rank,
                    'distributed_backend': 'nccl'  # Assuming NCCL for GPU
                })
            
            # Add GPU information
            if torch.cuda.is_available():
                summary['gpu_info'] = {}
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    summary['gpu_info'][f'gpu_{i}'] = {
                        'name': props.name,
                        'memory_total_gb': props.total_memory / (1024**3),
                        'memory_allocated_mb': torch.cuda.memory_allocated(i) / (1024**2),
                        'memory_reserved_mb': torch.cuda.memory_reserved(i) / (1024**2)
                    }
            
            return summary
        
        except Exception as e:
            self.logger.error(f"Multi-GPU summary generation failed: {e}")
            return {'error': str(e)}
    
    def cleanup_multi_gpu(self) -> Any:
        """Cleanup multi-GPU resources."""
        try:
            if self.is_distributed:
                self.logger.info("🧹 Cleaning up distributed training")
                torch.distributed.destroy_process_group()
                self.is_distributed = False
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.logger.info("🧹 GPU cache cleared")
        
        except Exception as e:
            self.logger.warning(f"Multi-GPU cleanup failed: {e}")
    
    def setup_gradient_accumulation(self, config: TrainingConfig):
        """Setup gradient accumulation for large batch sizes."""
        try:
            if config.gradient_accumulation_steps > 1:
                self.gradient_accumulation_enabled = True
                self.logger.info(f"🔄 Setting up gradient accumulation with {config.gradient_accumulation_steps} steps")
                
                # Calculate effective batch size
                effective_batch_size = config.batch_size * config.gradient_accumulation_steps * config.num_gpus
                self.logger.info(f"Effective batch size: {effective_batch_size} (per step: {config.batch_size}, "
                               f"accumulation: {config.gradient_accumulation_steps}, GPUs: {config.num_gpus})")
                
                # Log memory savings
                if torch.cuda.is_available():
                    estimated_memory_savings = (config.gradient_accumulation_steps - 1) * config.batch_size * 0.1  # Rough estimate
                    self.gradient_accumulation_metrics['memory_savings_mb'] = estimated_memory_savings
                    self.logger.info(f"Estimated memory savings: {estimated_memory_savings:.1f} MB")
                
                # Setup dynamic scheduling if enabled
                if config.gradient_accumulation_scheduling:
                    self.logger.info("📈 Dynamic gradient accumulation scheduling enabled")
                    self.logger.info(f"Warmup steps: {config.gradient_accumulation_warmup_steps}")
                    self.logger.info(f"Max accumulation steps: {config.gradient_accumulation_max_steps}")
                
                self.logger.info("✅ Gradient accumulation setup completed")
            else:
                self.gradient_accumulation_enabled = False
                self.logger.info("⏭️ Gradient accumulation disabled (steps = 1)")
        
        except Exception as e:
            self.logger.error(f"Gradient accumulation setup failed: {e}", exc_info=True)
            raise
    
    def update_gradient_accumulation_schedule(self, config: TrainingConfig, current_step: int):
        """Update gradient accumulation schedule for dynamic accumulation."""
        try:
            if not config.gradient_accumulation_scheduling:
                return config.gradient_accumulation_steps
            
            # Calculate current accumulation steps based on warmup
            if current_step < config.gradient_accumulation_warmup_steps:
                # Gradual increase during warmup
                progress = current_step / config.gradient_accumulation_warmup_steps
                current_accumulation_steps = max(1, int(config.gradient_accumulation_steps * progress))
            else:
                # Use maximum accumulation steps after warmup
                current_accumulation_steps = config.gradient_accumulation_max_steps
            
            # Update configuration
            old_steps = config.gradient_accumulation_steps
            config.gradient_accumulation_steps = current_accumulation_steps
            
            # Recalculate effective batch size
            config.effective_batch_size = config.batch_size * config.gradient_accumulation_steps * config.num_gpus
            
            # Log if accumulation steps changed
            if old_steps != current_accumulation_steps:
                self.logger.info(f"🔄 Gradient accumulation steps updated: {old_steps} → {current_accumulation_steps}")
                self.logger.info(f"Effective batch size: {config.effective_batch_size}")
                
                # Track history
                self.accumulation_steps_history.append({
                    'step': current_step,
                    'accumulation_steps': current_accumulation_steps,
                    'effective_batch_size': config.effective_batch_size
                })
            
            return current_accumulation_steps
        
        except Exception as e:
            self.logger.warning(f"Gradient accumulation schedule update failed: {e}")
            return config.gradient_accumulation_steps
    
    def get_gradient_accumulation_summary(self) -> Dict[str, Any]:
        """Get comprehensive gradient accumulation summary."""
        try:
            summary = {
                'gradient_accumulation_enabled': self.gradient_accumulation_enabled,
                'current_accumulation_step': self.current_accumulation_step,
                'metrics': self.gradient_accumulation_metrics.copy()
            }
            
            # Add history if available
            if self.accumulation_steps_history:
                summary['history'] = {
                    'total_updates': len(self.accumulation_steps_history),
                    'average_accumulation_steps': np.mean([h['accumulation_steps'] for h in self.accumulation_steps_history]),
                    'max_accumulation_steps': max([h['accumulation_steps'] for h in self.accumulation_steps_history]),
                    'min_accumulation_steps': min([h['accumulation_steps'] for h in self.accumulation_steps_history])
                }
            
            # Add effective batch size history
            if self.effective_batch_size_history:
                summary['effective_batch_size'] = {
                    'current': self.effective_batch_size_history[-1] if self.effective_batch_size_history else None,
                    'average': np.mean(self.effective_batch_size_history),
                    'max': max(self.effective_batch_size_history),
                    'min': min(self.effective_batch_size_history)
                }
            
            return summary
        
        except Exception as e:
            self.logger.error(f"Gradient accumulation summary generation failed: {e}")
            return {'error': str(e)}
    
    def log_gradient_accumulation_metrics(self, config: TrainingConfig, batch_idx: int, loss: float):
        """Log gradient accumulation metrics during training."""
        try:
            if not self.gradient_accumulation_enabled:
                return
            
            # Update metrics
            self.gradient_accumulation_metrics['total_accumulation_steps'] += 1
            
            # Calculate accumulation efficiency
            if self.gradient_accumulation_metrics['total_accumulation_steps'] > 0:
                self.gradient_accumulation_metrics['average_accumulation_steps'] = (
                    self.gradient_accumulation_metrics['total_accumulation_steps'] / 
                    (batch_idx + 1)
                )
            
            # Track effective batch size
            current_effective_batch_size = config.batch_size * config.gradient_accumulation_steps * config.num_gpus
            self.effective_batch_size_history.append(current_effective_batch_size)
            
            # Log periodically
            if batch_idx % 100 == 0:
                self.logger.debug(f"Gradient accumulation - Step {batch_idx}, "
                                f"Accumulation: {config.gradient_accumulation_steps}, "
                                f"Effective batch: {current_effective_batch_size}, "
                                f"Loss: {loss:.4f}")
        
        except Exception as e:
            self.logger.warning(f"Gradient accumulation metrics logging failed: {e}")
    
    def _setup_gpu_optimization(self, config: TrainingConfig):
        """Setup GPU-specific optimizations."""
        try:
            # Enable cuDNN benchmark for optimal performance
            if config.enable_cudnn_benchmark:
                torch.backends.cudnn.benchmark = True
                self.logger.info("⚡ cuDNN benchmark enabled")
            
            # Enable deterministic mode if needed
            if config.enable_cudnn_deterministic:
                torch.backends.cudnn.deterministic = True
                self.logger.info("🎯 cuDNN deterministic mode enabled")
            
            # Enable TensorFloat-32 for Ampere GPUs
            if config.enable_tf32:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                self.logger.info("🔢 TensorFloat-32 enabled")
            
            # Set memory fraction
            if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                torch.cuda.set_per_process_memory_fraction(0.9)
                self.logger.info("💾 GPU memory fraction set to 90%")
            
            # Warm up GPU
            self._warmup_gpu()
            
            self.gpu_optimized = True
            self.logger.info("✅ GPU optimization completed")
        
        except Exception as e:
            self.logger.warning(f"GPU optimization failed: {e}")
    
    def _setup_memory_optimization(self, config: TrainingConfig):
        """Setup memory optimization."""
        try:
            # Enable gradient checkpointing for memory efficiency
            if config.enable_gradient_checkpointing:
                self.logger.info("💾 Gradient checkpointing enabled")
            
            # Set memory format optimization
            if config.enable_channels_last:
                self.logger.info("🔄 Channels last memory format enabled")
            
            # Configure garbage collection
            gc.set_threshold(700, 10, 10)  # More aggressive GC
            self.logger.info("🗑️ Aggressive garbage collection enabled")
            
            self.memory_optimized = True
            self.logger.info("✅ Memory optimization completed")
        
        except Exception as e:
            self.logger.warning(f"Memory optimization failed: {e}")
    
    def _setup_batch_optimization(self, config: TrainingConfig):
        """Setup batch processing optimization."""
        try:
            # Auto-detect optimal batch size
            if config.enable_dynamic_batching:
                optimal_batch_size = self._detect_optimal_batch_size()
                if optimal_batch_size != config.batch_size:
                    self.logger.info(f"🔄 Auto-detected optimal batch size: {optimal_batch_size}")
                    config.batch_size = optimal_batch_size
            
            # Auto-detect optimal number of workers
            if config.num_workers == -1:
                optimal_workers = self._detect_optimal_workers()
                config.num_workers = optimal_workers
                self.logger.info(f"🔄 Auto-detected optimal workers: {optimal_workers}")
            
            self.batch_optimized = True
            self.logger.info("✅ Batch optimization completed")
        
        except Exception as e:
            self.logger.warning(f"Batch optimization failed: {e}")
    
    def _warmup_gpu(self) -> Any:
        """Warm up GPU for optimal performance."""
        try:
            if torch.cuda.is_available():
                # Create dummy tensors and perform operations
                dummy_input = torch.randn(32, 3, 224, 224, device=self.device)
                dummy_model = torch.nn.Conv2d(3, 64, 3, padding=1).to(self.device)
                
                # Warm up with a few forward/backward passes
                for _ in range(3):
                    output = dummy_model(dummy_input)
                    loss = output.sum()
                    loss.backward()
                
                # Clear cache
                del dummy_input, dummy_model, output, loss
                torch.cuda.empty_cache()
                
                self.logger.info("🔥 GPU warmup completed")
        
        except Exception as e:
            self.logger.warning(f"GPU warmup failed: {e}")
    
    def _detect_optimal_batch_size(self) -> int:
        """Detect optimal batch size based on available memory."""
        try:
            if torch.cuda.is_available():
                # Get available GPU memory
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
                
                # Heuristic: 1GB per batch for typical models
                optimal_batch_size = max(1, int(gpu_memory * 0.8))
                
                # Cap at reasonable maximum
                optimal_batch_size = min(optimal_batch_size, 128)
                
                return optimal_batch_size
            else:
                # CPU: use smaller batches
                return 16
        
        except Exception as e:
            self.logger.warning(f"Batch size detection failed: {e}")
            return 16
    
    def _detect_optimal_workers(self) -> int:
        """Detect optimal number of workers."""
        try:
            
            cpu_count = mp.cpu_count()
            memory_gb = psutil.virtual_memory().total / (1024**3)
            
            # Heuristic: 1 worker per 2GB RAM, max 8 workers
            optimal_workers = min(cpu_count, int(memory_gb / 2), 8)
            return max(1, optimal_workers)
        
        except Exception as e:
            self.logger.warning(f"Worker detection failed: {e}")
            return 4
    
    def _compile_model(self, model: nn.Module, config: TrainingConfig) -> nn.Module:
        """Compile model for optimal performance (PyTorch 2.0+)."""
        try:
            if config.enable_compilation and hasattr(torch, 'compile'):
                compiled_model = torch.compile(
                    model,
                    mode=config.enable_compile_mode,
                    fullgraph=True
                )
                self.logger.info(f"🔧 Model compiled with mode: {config.enable_compile_mode}")
                return compiled_model
            else:
                return model
        
        except Exception as e:
            self.logger.warning(f"Model compilation failed: {e}")
            return model
    
    def _update_performance_metrics(self, stage: str, start_time: float, batch_size: int = None):
        """Update performance metrics."""
        try:
            elapsed = time.time() - start_time
            
            # GPU memory metrics
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024**2)  # MB
                reserved = torch.cuda.memory_reserved() / (1024**2)  # MB
                
                self.performance_metrics['gpu_memory_allocated'].append(allocated)
                self.performance_metrics['gpu_memory_reserved'].append(reserved)
            
            # Timing metrics
            if stage == 'batch' and batch_size:
                self.performance_metrics['batch_processing_times'].append(elapsed)
                throughput = batch_size / elapsed
                self.performance_metrics['throughput_samples_per_sec'].append(throughput)
            elif stage == 'epoch':
                self.performance_metrics['epoch_times'].append(elapsed)
            
            # Memory efficiency
            if torch.cuda.is_available():
                max_allocated = max(self.performance_metrics['gpu_memory_allocated'])
                current_allocated = self.performance_metrics['gpu_memory_allocated'][-1]
                efficiency = current_allocated / max_allocated if max_allocated > 0 else 1.0
                self.performance_metrics['memory_efficiency'].append(efficiency)
        
        except Exception as e:
            self.logger.warning(f"Performance metrics update failed: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        try:
            summary = {
                'optimization_status': {
                    'gpu_optimized': self.gpu_optimized,
                    'memory_optimized': self.memory_optimized,
                    'batch_optimized': self.batch_optimized,
                    'model_compiled': self.compiled_model is not None
                },
                'gpu_info': {}
            }
            
            # GPU information
            if torch.cuda.is_available():
                summary['gpu_info'] = {
                    'device_name': torch.cuda.get_device_name(0),
                    'memory_total_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3),
                    'memory_allocated_mb': torch.cuda.memory_allocated() / (1024**2),
                    'memory_reserved_mb': torch.cuda.memory_reserved() / (1024**2),
                    'cuda_version': torch.version.cuda
                }
            
            # Performance metrics
            if self.performance_metrics['batch_processing_times']:
                summary['performance'] = {
                    'avg_batch_time': np.mean(self.performance_metrics['batch_processing_times']),
                    'avg_throughput': np.mean(self.performance_metrics['throughput_samples_per_sec']),
                    'max_throughput': np.max(self.performance_metrics['throughput_samples_per_sec']),
                    'avg_epoch_time': np.mean(self.performance_metrics['epoch_times']) if self.performance_metrics['epoch_times'] else None,
                    'memory_efficiency': np.mean(self.performance_metrics['memory_efficiency']) if self.performance_metrics['memory_efficiency'] else None
                }
            
            return summary
        
        except Exception as e:
            self.logger.error(f"Performance summary generation failed: {e}")
            return {'error': str(e)}
    
    def _check_gradients(self, model: nn.Module, batch_idx: int):
        """Check gradients for anomalies and log issues."""
        if not self.gradient_checking_enabled:
            return
        
        try:
            total_norm = 0
            param_count = 0
            nan_count = 0
            inf_count = 0
            
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    param_count += 1
                    
                    # Check for NaN/Inf gradients
                    if torch.isnan(p.grad).any():
                        nan_count += 1
                        self.logger.warning(f"NaN gradient detected in parameter at batch {batch_idx}")
                    
                    if torch.isinf(p.grad).any():
                        inf_count += 1
                        self.logger.warning(f"Inf gradient detected in parameter at batch {batch_idx}")
            
            if param_count > 0:
                total_norm = total_norm ** (1. / 2)
                self.logger.debug(f"Batch {batch_idx}: Gradient norm: {total_norm:.6f}, "
                                f"NaN params: {nan_count}, Inf params: {inf_count}")
                
                if nan_count > 0 or inf_count > 0:
                    self.logger.error(f"Gradient anomalies detected at batch {batch_idx}: "
                                    f"{nan_count} NaN, {inf_count} Inf")
        
        except Exception as e:
            self.logger.error(f"Error checking gradients at batch {batch_idx}: {e}")
    
    def _log_memory_usage(self, stage: str, batch_idx: int = None):
        """Log memory usage for debugging."""
        if not self.memory_profiling_enabled:
            return
        
        try:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**2
                reserved = torch.cuda.memory_reserved() / 1024**2
                max_allocated = torch.cuda.max_memory_allocated() / 1024**2
                
                batch_info = f" batch {batch_idx}" if batch_idx is not None else ""
                self.logger.debug(f"Memory usage at {stage}{batch_info}: "
                                f"Allocated: {allocated:.2f}MB, "
                                f"Reserved: {reserved:.2f}MB, "
                                f"Max: {max_allocated:.2f}MB")
        
        except Exception as e:
            self.logger.error(f"Error logging memory usage: {e}")
    
    def _log_performance_metrics(self, stage: str, start_time: float, batch_idx: int = None):
        """Log performance metrics for debugging."""
        if not self.performance_profiling_enabled:
            return
        
        try:
            elapsed = time.time() - start_time
            batch_info = f" batch {batch_idx}" if batch_idx is not None else ""
            self.logger.debug(f"Performance at {stage}{batch_info}: {elapsed:.4f}s")
        
        except Exception as e:
            self.logger.error(f"Error logging performance metrics: {e}")
    
    async def train_epoch(self, model: nn.Module, dataloader: DataLoader, 
                         optimizer, scheduler, config: TrainingConfig, 
                         scaler=None) -> Dict[str, float]:
        """Train for one epoch with profiling."""
        if self.performance_profiling:
            self.profiler_manager.start_torch_profiler()
        self.logger.info("Starting training epoch")
        model.train()
        
        # Set epoch for distributed sampler
        if self.is_distributed and hasattr(dataloader.sampler, 'set_epoch'):
            dataloader.sampler.set_epoch(self.current_epoch)
        
        total_loss = 0
        all_predictions = []
        all_targets = []
        all_probabilities = []
        progress_bar = tqdm(dataloader, desc="Training")
        
        # Memory profiling
        self._log_memory_usage("epoch_start")
        epoch_start_time = time.time()
        
        # Gradient accumulation tracking
        accumulation_loss = 0.0
        accumulation_count = 0
        
        try:
            for batch_idx, batch in enumerate(progress_bar):
                batch_start_time = time.time()
                
                # Update gradient accumulation schedule if dynamic scheduling is enabled
                if config.gradient_accumulation_scheduling:
                    current_accumulation_steps = self.update_gradient_accumulation_schedule(config, batch_idx)
                else:
                    current_accumulation_steps = config.gradient_accumulation_steps
                
                # Memory profiling per batch
                self._log_memory_usage("batch_start", batch_idx)
                
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                try:
                    # Use Automatic Mixed Precision if enabled
                    if config.enable_amp:
                        with torch.cuda.amp.autocast():
                            outputs = model(**batch)
                            loss = outputs.loss
                    else:
                        outputs = model(**batch)
                        loss = outputs.loss
                    
                    # Check for NaN/Inf loss
                    if torch.isnan(loss) or torch.isinf(loss):
                        self.logger.error(f"Invalid loss detected at batch {batch_idx}: {loss.item()}")
                        continue
                    
                    # Scale loss for gradient accumulation
                    if self.gradient_accumulation_enabled and current_accumulation_steps > 1:
                        loss = loss / current_accumulation_steps
                    
                    # Accumulate loss for logging
                    accumulation_loss += loss.item()
                    accumulation_count += 1
                
                except Exception as e:
                    self.logger.error(f"Error in forward pass at batch {batch_idx}: {e}", exc_info=True)
                    continue
                
                try:
                    if config.enable_amp:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    
                    # Check if we should perform gradient update
                    should_update = (batch_idx + 1) % current_accumulation_steps == 0
                    
                    if should_update:
                        # Gradient clipping
                        if config.enable_amp:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                            
                            # Gradient checking
                            self._check_gradients(model, batch_idx)
                            
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                            
                            # Gradient checking
                            self._check_gradients(model, batch_idx)
                            
                            optimizer.step()
                        
                        # Update scheduler and zero gradients
                        scheduler.step()
                        optimizer.zero_grad()
                        
                        # Update accumulation step counter
                        self.current_accumulation_step += 1
                        
                        # Log gradient accumulation metrics
                        if self.gradient_accumulation_enabled:
                            self.log_gradient_accumulation_metrics(config, batch_idx, accumulation_loss)
                    
                except Exception as e:
                    self.logger.error(f"Error in backward pass at batch {batch_idx}: {e}", exc_info=True)
                    continue
                
                # Calculate average loss for this accumulation step
                avg_loss = accumulation_loss / accumulation_count if accumulation_count > 0 else 0.0
                total_loss += avg_loss * current_accumulation_steps
                
                predictions = torch.argmax(outputs.logits, dim=-1)
                probabilities = torch.softmax(outputs.logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(batch['labels'].cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
                # Performance profiling
                self._log_performance_metrics("batch_end", batch_start_time, batch_idx)
                self._update_performance_metrics("batch", batch_start_time, len(batch['labels']))
                self._log_memory_usage("batch_end", batch_idx)
                
                # Update progress bar with gradient accumulation info
                progress_info = {
                    'loss': f"{avg_loss:.4f}",
                    'lr': f"{scheduler.get_last_lr()[0]:.2e}"
                }
                
                if self.gradient_accumulation_enabled:
                    progress_info['acc'] = f"{current_accumulation_steps}"
                    progress_info['eff_batch'] = f"{config.effective_batch_size}"
                
                progress_bar.set_postfix(progress_info)
                
                # Reset accumulation counters after update
                if should_update:
                    accumulation_loss = 0.0
                    accumulation_count = 0
                if self.performance_profiling:
                    self.profiler_manager.step_torch_profiler()
        
        except Exception as e:
            self.logger.error(f"Exception during training epoch: {e}", exc_info=True)
        
        epoch_metrics = self.calculate_metrics(
            np.array(all_predictions), 
            np.array(all_targets),
            probabilities=np.array(all_probabilities)
        )
        epoch_metrics['loss'] = total_loss / len(dataloader)
        
        # Synchronize metrics across GPUs for distributed training
        if self.is_distributed:
            epoch_metrics = self.sync_metrics_across_gpus(epoch_metrics)
        
        self.logger.info(f"Finished training epoch. Loss: {epoch_metrics['loss']:.4f}")
        
        # Final memory profiling and performance metrics
        self._log_memory_usage("epoch_end")
        self._update_performance_metrics("epoch", epoch_start_time)
        if self.performance_profiling:
            self.profiler_manager.stop_torch_profiler()
        
        return epoch_metrics
    
    async def validate_epoch(self, model: nn.Module, dataloader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch with logging and error handling."""
        self.logger.info("Starting validation epoch")
        model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        # Memory profiling
        self._log_memory_usage("validation_start")
        
        try:
            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(dataloader, desc="Validation")):
                    batch_start_time = time.time()
                    
                    # Memory profiling per batch
                    self._log_memory_usage("validation_batch_start", batch_idx)
                    
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                    
                    try:
                        outputs = model(**batch)
                        loss = outputs.loss
                        
                        # Check for NaN/Inf loss
                        if torch.isnan(loss) or torch.isinf(loss):
                            self.logger.error(f"Invalid validation loss at batch {batch_idx}: {loss.item()}")
                            continue
                    
                    except Exception as e:
                        self.logger.error(f"Error in validation forward pass at batch {batch_idx}: {e}", exc_info=True)
                        continue
                    
                    total_loss += loss.item()
                    predictions = torch.argmax(outputs.logits, dim=-1)
                    probabilities = torch.softmax(outputs.logits, dim=-1)
                    all_predictions.extend(predictions.cpu().numpy())
                    all_targets.extend(batch['labels'].cpu().numpy())
                    all_probabilities.extend(probabilities.cpu().numpy())
                    
                    # Performance profiling
                    self._log_performance_metrics("validation_batch_end", batch_start_time, batch_idx)
                    self._log_memory_usage("validation_batch_end", batch_idx)
        
        except Exception as e:
            self.logger.error(f"Exception during validation epoch: {e}", exc_info=True)
        
        val_metrics = self.calculate_metrics(
            np.array(all_predictions), 
            np.array(all_targets),
            probabilities=np.array(all_probabilities)
        )
        val_metrics['loss'] = total_loss / len(dataloader)
        self.logger.info(f"Finished validation epoch. Loss: {val_metrics['loss']:.4f}")
        
        # Final memory profiling
        self._log_memory_usage("validation_end")
        
        return val_metrics
    
    async def train(self, config: TrainingConfig) -> Dict[str, Any]:
        """Main training loop with comprehensive logging and error handling."""
        self.logger.info(f"Starting training for {config.model_name}")
        
        try:
            # Setup multi-GPU training
            self.setup_multi_gpu_training(config)
            
            # Setup gradient accumulation
            self.setup_gradient_accumulation(config)
            
            # Setup performance optimization
            self.setup_performance_optimization(config)
            
            # Setup debugging tools
            self.setup_debugging(config)
            
            # Setup logging
            self.setup_logging(config)
            
            # Memory profiling at start
            self._log_memory_usage("training_start")
            
            train_dataset, val_dataset, test_dataset = await self.load_dataset(config)
            
            # Create multi-GPU DataLoaders
            train_loader, val_loader, test_loader = self.create_multi_gpu_dataloaders(
                train_dataset, val_dataset, test_dataset, config
            )
            
            num_classes = len(set(train_dataset.labels))
            model, tokenizer = self.create_model(config, num_classes)
            
            # Wrap model for multi-GPU training
            model = self.wrap_model_for_multi_gpu(model, config)
            
            # Compile model for optimal performance
            model = self._compile_model(model, config)
            self.compiled_model = model
            
            # Log model info in debug mode
            if self.debug_enabled:
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                self.logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
            
            optimizer = self.create_optimizer(model, config)
            num_training_steps = len(train_loader) * config.num_epochs
            scheduler = self.create_scheduler(optimizer, config, num_training_steps)
            scaler = torch.cuda.amp.GradScaler() if config.enable_amp else None
            
            es_config = create_early_stopping_config(
                enabled=True,
                strategy=EarlyStoppingStrategy.PATIENCE,
                mode=EarlyStoppingMode.MIN,
                patience=config.early_stopping_patience,
                monitor="val_loss",
                min_epochs=5
            )
            self.training_monitor.setup_early_stopping(es_config)
            
            training_history = []
            start_time = time.time()
            
            for epoch in range(config.num_epochs):
                self.current_epoch = epoch
                epoch_start_time = time.time()
                self.logger.info(f"Epoch {epoch+1}/{config.num_epochs} started")
                
                # Memory profiling per epoch
                self._log_memory_usage("epoch_start", epoch)
                
                train_metrics = await self.train_epoch(
                    model, train_loader, optimizer, scheduler, config, scaler
                )
                val_metrics = await self.validate_epoch(model, val_loader)
                
                epoch_time = time.time() - epoch_start_time
                metrics = TrainingMetrics(
                    epoch=epoch,
                    train_loss=train_metrics['loss'],
                    val_loss=val_metrics['loss'],
                    train_accuracy=train_metrics['accuracy'],
                    val_accuracy=val_metrics['accuracy'],
                    learning_rate=scheduler.get_last_lr()[0],
                    training_time=epoch_time,
                    train_f1=train_metrics.get('f1'),
                    val_f1=val_metrics.get('f1'),
                    train_precision=train_metrics.get('precision'),
                    val_precision=val_metrics.get('precision'),
                    train_recall=train_metrics.get('recall'),
                    val_recall=val_metrics.get('recall')
                )
                
                training_history.append(metrics)
                monitor_metrics = {
                    'train_loss': train_metrics['loss'],
                    'val_loss': val_metrics['loss'],
                    'train_accuracy': train_metrics['accuracy'],
                    'val_accuracy': val_metrics['accuracy'],
                    'learning_rate': scheduler.get_last_lr()[0]
                }
                
                should_stop = self.training_monitor.update(epoch, monitor_metrics, model)
                self.logger.info(f"Epoch {epoch+1} completed. Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_metrics['loss']:.4f}")
                
                # Performance profiling per epoch
                self._log_performance_metrics("epoch_end", epoch_start_time, epoch)
                self._log_memory_usage("epoch_end", epoch)
                
                if self.writer:
                    self.writer.add_scalar('Loss/Train', train_metrics['loss'], epoch)
                    self.writer.add_scalar('Loss/Val', val_metrics['loss'], epoch)
                    self.writer.add_scalar('Accuracy/Train', train_metrics['accuracy'], epoch)
                    self.writer.add_scalar('Accuracy/Val', val_metrics['accuracy'], epoch)
                    self.writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], epoch)
                
                if config.log_to_wandb:
                    wandb.log({
                        'epoch': epoch,
                        'train_loss': train_metrics['loss'],
                        'val_loss': val_metrics['loss'],
                        'train_accuracy': train_metrics['accuracy'],
                        'val_accuracy': val_metrics['accuracy'],
                        'learning_rate': scheduler.get_last_lr()[0]
                    })
                
                if should_stop:
                    self.logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
                
                if (epoch + 1) % config.save_steps == 0:
                    checkpoint_path = config.output_dir / f"{config.model_name}_epoch_{epoch+1}.pth"
                    try:
                        # Save model state (unwrap if needed)
                        model_to_save = model.module if hasattr(model, 'module') else model
                        torch.save({
                            'model_state_dict': model_to_save.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'config': config,
                            'epoch': epoch,
                            'metrics': metrics
                        }, checkpoint_path)
                        self.logger.info(f"Checkpoint saved at {checkpoint_path}")
                    except Exception as e:
                        self.logger.error(f"Error saving checkpoint at epoch {epoch+1}: {e}", exc_info=True)
            
            self.training_monitor.restore_best_model(model)
            evaluation_result = await self.evaluate_model(model, test_dataset, config)
            cv_result = None
            if config.cross_validation_folds > 1:
                self.logger.info(f"Performing {config.cross_validation_folds}-fold cross-validation")
                cv_result = await self.perform_cross_validation(train_dataset, config)
            
            history_path = config.output_dir / f"{config.model_name}_training_history.json"
            try:
                with open(history_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    json.dump([vars(m) for m in training_history], f, indent=2)
                self.logger.info(f"Training history saved at {history_path}")
            except Exception as e:
                self.logger.error(f"Error saving training history: {e}", exc_info=True)
            
            if cv_result:
                cv_path = config.output_dir / f"{config.model_name}_cross_validation.json"
                try:
                    with open(cv_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                        json.dump(vars(cv_result), f, indent=2)
                    self.logger.info(f"Cross-validation results saved at {cv_path}")
                except Exception as e:
                    self.logger.error(f"Error saving cross-validation results: {e}", exc_info=True)
            
            if self.writer:
                self.writer.close()
            if config.log_to_wandb:
                wandb.finish()
            if config.log_to_mlflow:
                mlflow.end_run()
            
            total_time = time.time() - start_time
            self.logger.info(f"Training completed in {total_time:.2f} seconds")
            
            # Final memory profiling
            self._log_memory_usage("training_end")
            
            # Get performance summary
            performance_summary = self.get_performance_summary()
            self.logger.info(f"Performance summary: {performance_summary}")
            
            # Get multi-GPU summary
            multi_gpu_summary = self.get_multi_gpu_summary()
            self.logger.info(f"Multi-GPU summary: {multi_gpu_summary}")
            
            # Get gradient accumulation summary
            gradient_accumulation_summary = self.get_gradient_accumulation_summary()
            self.logger.info(f"Gradient accumulation summary: {gradient_accumulation_summary}")
            
            training_summary = self.training_monitor.get_training_summary()
            curves_path = config.output_dir / f"{config.model_name}_training_curves.png"
            try:
                self.training_monitor.plot_training_curves(str(curves_path))
                self.logger.info(f"Training curves saved at {curves_path}")
            except Exception as e:
                self.logger.error(f"Error saving training curves: {e}", exc_info=True)
            
            return {
                'training_history': training_history,
                'evaluation_result': evaluation_result,
                'cross_validation_result': cv_result,
                'training_summary': training_summary,
                'performance_summary': performance_summary,
                'multi_gpu_summary': multi_gpu_summary,
                'gradient_accumulation_summary': gradient_accumulation_summary,
                'best_model_path': str(self.best_model_path),
                'total_training_time': total_time
            }
        
        except Exception as e:
            self.logger.error(f"Exception in main training loop: {e}", exc_info=True)
            raise
        finally:
            # Cleanup multi-GPU resources
            self.cleanup_multi_gpu()
            
            # Cleanup debugging tools
            if self.anomaly_detection_enabled:
                torch.autograd.set_detect_anomaly(False)
                self.logger.info("🔧 Autograd anomaly detection disabled")
            
            if self.memory_profiling_enabled and torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.logger.info("🧹 GPU memory cache cleared")
    
    async def evaluate_model(self, model: nn.Module, test_dataset: Dataset, 
                           config: TrainingConfig) -> EvaluationResult:
        """Evaluate model on test dataset using the evaluation metrics system."""
        self.logger.info("Evaluating model on test dataset")
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=config.batch_size, 
            shuffle=False,
            num_workers=4
        )
        
        model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        inference_times = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                start_time = time.time()
                outputs = model(**batch)
                inference_time = (time.time() - start_time) * 1000  # Convert to ms
                inference_times.append(inference_time)
                
                predictions = torch.argmax(outputs.logits, dim=-1)
                probabilities = torch.softmax(outputs.logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(batch['labels'].cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Determine task type based on config
        task_type = getattr(config, 'task_type', 'classification')
        
        # Create metric configuration
        metric_config = create_metric_config(
            task_type=TaskType.CLASSIFICATION if task_type == 'classification' else TaskType.REGRESSION,
            metric_types=[
                MetricType.ACCURACY, MetricType.PRECISION, MetricType.RECALL,
                MetricType.F1, MetricType.ROC_AUC, MetricType.CONFUSION_MATRIX,
                MetricType.CLASSIFICATION_REPORT
            ]
        )
        
        # Evaluate using the evaluation metrics system
        test_predictions = np.array(all_predictions)
        test_targets = np.array(all_targets)
        test_probabilities = np.array(all_probabilities)
        
        evaluation_result = self.evaluation_metrics.evaluate(
            metric_config, test_targets, test_predictions, test_probabilities
        )
        
        avg_inference_time = np.mean(inference_times)
        
        # Create legacy EvaluationResult for compatibility
        result = EvaluationResult(
            model_name=config.model_name,
            test_accuracy=evaluation_result.metrics.get('accuracy', 0.0),
            test_f1=evaluation_result.metrics.get('f1', 0.0),
            test_precision=evaluation_result.metrics.get('precision', 0.0),
            test_recall=evaluation_result.metrics.get('recall', 0.0),
            confusion_matrix=evaluation_result.confusion_matrix,
            classification_report=evaluation_result.classification_report,
            roc_auc=evaluation_result.metrics.get('roc_auc'),
            inference_time_ms=avg_inference_time
        )
        
        self.logger.info(f"Evaluation completed: Accuracy={result.test_accuracy:.4f}, F1={result.test_f1:.4f}")
        
        return result
    
    async def perform_cross_validation(self, train_dataset: Dataset, config: TrainingConfig) -> CrossValidationResult:
        """Perform cross-validation on training dataset."""
        self.logger.info(f"Starting {config.cross_validation_folds}-fold cross-validation")
        
        # Create cross-validation configuration
        cv_config = CrossValidationConfig(
            strategy=CrossValidationStrategy.STRATIFIED_K_FOLD,
            n_splits=config.cross_validation_folds,
            random_state=config.random_state if hasattr(config, 'random_state') else 42,
            shuffle=True
        )
        
        # Perform cross-validation
        cv_result = await self.data_splitting_manager.cross_validator.cross_validate(
            train_dataset, cv_config, type(self), config, self.data_loader_manager
        )
        
        self.logger.info(f"Cross-validation completed. Mean F1: {cv_result.mean_scores.get('val_f1_score', 0):.4f}")
        
        return cv_result
        )
        
        # Save evaluation results
        eval_path = config.output_dir / f"{config.model_name}_evaluation.json"
        with open(eval_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump({
                'model_name': result.model_name,
                'test_accuracy': result.test_accuracy,
                'test_f1': result.test_f1,
                'test_precision': result.test_precision,
                'test_recall': result.test_recall,
                'roc_auc': result.roc_auc,
                'inference_time_ms': result.inference_time_ms,
                'confusion_matrix': result.confusion_matrix.tolist(),
                'classification_report': result.classification_report
            }, f, indent=2)
        
        self.logger.info(f"Evaluation completed. Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        return result

class HyperparameterOptimizer:
    """Hyperparameter optimization using Optuna."""
    
    def __init__(self, trainer: ModelTrainer):
        
    """__init__ function."""
self.trainer = trainer
        self.logger = logging.getLogger(f"{__name__}.HyperparameterOptimizer")
    
    def objective(self, trial: optuna.Trial, config: TrainingConfig) -> float:
        """Objective function for hyperparameter optimization."""
        # Suggest hyperparameters
        config.learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-3, log=True)
        config.batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64])
        config.weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-1, log=True)
        config.warmup_steps = trial.suggest_int('warmup_steps', 50, 500)
        
        # Train model with suggested hyperparameters
        try:
            result = await self.trainer.train(config)
            return result['evaluation_result'].test_f1  # Optimize for F1 score
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            return 0.0
    
    async def optimize(self, config: TrainingConfig) -> Dict[str, Any]:
        """Run hyperparameter optimization."""
        self.logger.info(f"Starting hyperparameter optimization with {config.hpo_trials} trials")
        
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42)
        )
        
        # Create objective function with config
        objective_with_config = lambda trial: self.objective(trial, config)
        
        study.optimize(objective_with_config, n_trials=config.hpo_trials)
        
        # Get best parameters
        best_params = study.best_params
        best_value = study.best_value
        
        self.logger.info(f"Best parameters: {best_params}")
        self.logger.info(f"Best F1 score: {best_value:.4f}")
        
        # Update config with best parameters
        for param, value in best_params.items():
            setattr(config, param, value)
        
        # Train final model with best parameters
        final_result = await self.trainer.train(config)
        
        return {
            'best_parameters': best_params,
            'best_f1_score': best_value,
            'study': study,
            'final_training_result': final_result
        }

# Factory functions
async def create_model_trainer(device_manager: DeviceManager) -> ModelTrainer:
    """Create a model trainer instance."""
    return ModelTrainer(device_manager)

async def create_hyperparameter_optimizer(trainer: ModelTrainer) -> HyperparameterOptimizer:
    """Create a hyperparameter optimizer instance."""
    return HyperparameterOptimizer(trainer)

# Quick training functions
async def quick_train_transformer(
    model_name: str,
    dataset_path: str,
    output_dir: str = "models",
    num_epochs: int = 5
) -> Dict[str, Any]:
    """Quick training for transformer models."""
    device_manager = DeviceManager()
    trainer = await create_model_trainer(device_manager)
    
    config = TrainingConfig(
        model_type=ModelType.TRANSFORMER,
        training_mode=TrainingMode.FINE_TUNE,
        model_name=model_name,
        dataset_path=dataset_path,
        output_dir=output_dir,
        num_epochs=num_epochs
    )
    
    return await trainer.train(config)

async def quick_hyperparameter_optimization(
    model_name: str,
    dataset_path: str,
    output_dir: str = "models",
    trials: int = 20
) -> Dict[str, Any]:
    """Quick hyperparameter optimization."""
    device_manager = DeviceManager()
    trainer = await create_model_trainer(device_manager)
    optimizer = await create_hyperparameter_optimizer(trainer)
    
    config = TrainingConfig(
        model_type=ModelType.TRANSFORMER,
        training_mode=TrainingMode.FINE_TUNE,
        model_name=model_name,
        dataset_path=dataset_path,
        output_dir=output_dir,
        enable_hpo=True,
        hpo_trials=trials,
        num_epochs=3  # Shorter epochs for HPO
    )
    
    return await optimizer.optimize(config)

def setup_comprehensive_debugging(config: TrainingConfig) -> TrainingConfig:
    """
    Setup comprehensive debugging for training.
    
    This function enables all debugging features for troubleshooting training issues.
    Use this when you encounter gradient problems, memory issues, or performance bottlenecks.
    
    Args:
        config: Training configuration
        
    Returns:
        Updated configuration with all debugging enabled
    """
    config.debug_mode = True
    config.detect_anomaly = True
    config.gradient_checking = True
    config.memory_profiling = True
    config.performance_profiling = True
    
    return config

def setup_gradient_debugging(config: TrainingConfig) -> TrainingConfig:
    """
    Setup gradient-specific debugging.
    
    Use this when you encounter gradient explosion, vanishing gradients, or NaN/Inf issues.
    
    Args:
        config: Training configuration
        
    Returns:
        Updated configuration with gradient debugging enabled
    """
    config.debug_mode = True
    config.detect_anomaly = True
    config.gradient_checking = True
    
    return config

def setup_memory_debugging(config: TrainingConfig) -> TrainingConfig:
    """
    Setup memory-specific debugging.
    
    Use this when you encounter out-of-memory errors or want to optimize memory usage.
    
    Args:
        config: Training configuration
        
    Returns:
        Updated configuration with memory debugging enabled
    """
    config.debug_mode = True
    config.memory_profiling = True
    
    return config

def setup_performance_debugging(config: TrainingConfig) -> TrainingConfig:
    """
    Setup performance-specific debugging.
    
    Use this when you want to profile training performance and identify bottlenecks.
    
    Args:
        config: Training configuration
        
    Returns:
        Updated configuration with performance debugging enabled
    """
    config.debug_mode = True
    config.performance_profiling = True
    
    return config

def get_debug_summary(trainer: ModelTrainer) -> Dict[str, Any]:
    """
    Get a summary of debugging information.
    
    Args:
        trainer: Model trainer instance
        
    Returns:
        Dictionary containing debugging summary
    """
    summary = {
        'debug_enabled': trainer.debug_enabled,
        'anomaly_detection_enabled': trainer.anomaly_detection_enabled,
        'gradient_checking_enabled': trainer.gradient_checking_enabled,
        'memory_profiling_enabled': trainer.memory_profiling_enabled,
        'performance_profiling_enabled': trainer.performance_profiling_enabled,
    }
    
    if torch.cuda.is_available():
        summary['gpu_memory'] = {
            'allocated_mb': torch.cuda.memory_allocated() / 1024**2,
            'reserved_mb': torch.cuda.memory_reserved() / 1024**2,
            'max_allocated_mb': torch.cuda.max_memory_allocated() / 1024**2,
            'device_count': torch.cuda.device_count(),
            'current_device': torch.cuda.current_device()
        }
    
    return summary

# Example usage functions with debugging
async def debug_train_transformer(
    model_name: str,
    dataset_path: str,
    output_dir: str = "models",
    num_epochs: int = 5,
    debug_type: str = "comprehensive"
) -> Dict[str, Any]:
    """
    Quick training with debugging enabled.
    
    Args:
        model_name: Name of the model to train
        dataset_path: Path to the dataset
        output_dir: Output directory for models
        num_epochs: Number of training epochs
        debug_type: Type of debugging ("comprehensive", "gradient", "memory", "performance")
        
    Returns:
        Training results with debugging information
    """
    device_manager = DeviceManager()
    trainer = await create_model_trainer(device_manager)
    
    config = TrainingConfig(
        model_type=ModelType.TRANSFORMER,
        training_mode=TrainingMode.FINE_TUNE,
        model_name=model_name,
        dataset_path=dataset_path,
        output_dir=output_dir,
        num_epochs=num_epochs
    )
    
    # Setup debugging based on type
    if debug_type == "comprehensive":
        config = setup_comprehensive_debugging(config)
    elif debug_type == "gradient":
        config = setup_gradient_debugging(config)
    elif debug_type == "memory":
        config = setup_memory_debugging(config)
    elif debug_type == "performance":
        config = setup_performance_debugging(config)
    
    # Run training
    result = await trainer.train(config)
    
    # Add debugging summary
    result['debug_summary'] = get_debug_summary(trainer)
    
    return result

def setup_performance_optimization_config(config: TrainingConfig) -> TrainingConfig:
    """
    Setup comprehensive performance optimization configuration.
    
    This function enables all performance optimization features for maximum training efficiency.
    Use this when you want to achieve the best possible training performance.
    
    Args:
        config: Training configuration
        
    Returns:
        Updated configuration with all performance optimizations enabled
    """
    config.enable_gpu_optimization = True
    config.enable_memory_optimization = True
    config.enable_batch_optimization = True
    config.enable_compilation = True
    config.enable_amp = True
    config.enable_gradient_checkpointing = True
    config.enable_dynamic_batching = True
    config.enable_pin_memory = True
    config.enable_persistent_workers = True
    config.enable_cudnn_benchmark = True
    config.enable_tf32 = True
    config.enable_channels_last = True
    config.enable_compile_mode = "max-autotune"
    
    return config

def setup_gpu_optimization_config(config: TrainingConfig) -> TrainingConfig:
    """
    Setup GPU-specific performance optimization.
    
    Use this when you have a powerful GPU and want to maximize GPU utilization.
    
    Args:
        config: Training configuration
        
    Returns:
        Updated configuration with GPU optimizations enabled
    """
    config.enable_gpu_optimization = True
    config.enable_amp = True
    config.enable_cudnn_benchmark = True
    config.enable_tf32 = True
    config.enable_pin_memory = True
    config.enable_compilation = True
    config.enable_compile_mode = "max-autotune"
    
    return config

def setup_memory_optimization_config(config: TrainingConfig) -> TrainingConfig:
    """
    Setup memory-specific performance optimization.
    
    Use this when you have limited memory or want to train larger models.
    
    Args:
        config: Training configuration
        
    Returns:
        Updated configuration with memory optimizations enabled
    """
    config.enable_memory_optimization = True
    config.enable_gradient_checkpointing = True
    config.enable_channels_last = True
    config.enable_amp = True
    config.batch_size = max(1, config.batch_size // 2)  # Reduce batch size
    
    return config

def setup_batch_optimization_config(config: TrainingConfig) -> TrainingConfig:
    """
    Setup batch processing optimization.
    
    Use this when you want to optimize data loading and batch processing efficiency.
    
    Args:
        config: Training configuration
        
    Returns:
        Updated configuration with batch optimizations enabled
    """
    config.enable_batch_optimization = True
    config.enable_dynamic_batching = True
    config.enable_pin_memory = True
    config.enable_persistent_workers = True
    config.num_workers = -1  # Auto-detect
    config.prefetch_factor = 4  # Increase prefetch
    
    return config

def get_performance_optimization_summary(trainer: ModelTrainer) -> Dict[str, Any]:
    """
    Get a summary of performance optimization status and metrics.
    
    Args:
        trainer: Model trainer instance
        
    Returns:
        Dictionary containing performance optimization summary
    """
    summary = {
        'optimization_status': {
            'gpu_optimized': trainer.gpu_optimized,
            'memory_optimized': trainer.memory_optimized,
            'batch_optimized': trainer.batch_optimized,
            'model_compiled': trainer.compiled_model is not None
        },
        'performance_metrics': trainer.performance_metrics.copy()
    }
    
    # Add GPU information if available
    if torch.cuda.is_available():
        summary['gpu_info'] = {
            'device_name': torch.cuda.get_device_name(0),
            'memory_total_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3),
            'memory_allocated_mb': torch.cuda.memory_allocated() / (1024**2),
            'memory_reserved_mb': torch.cuda.memory_reserved() / (1024**2),
            'cuda_version': torch.version.cuda
        }
    
    return summary

# Quick training functions with performance optimization
async def optimized_train_transformer(
    model_name: str,
    dataset_path: str,
    output_dir: str = "models",
    num_epochs: int = 5,
    optimization_type: str = "comprehensive"
) -> Dict[str, Any]:
    """
    Quick training with performance optimization enabled.
    
    Args:
        model_name: Name of the model to train
        dataset_path: Path to the dataset
        output_dir: Output directory for models
        num_epochs: Number of training epochs
        optimization_type: Type of optimization ("comprehensive", "gpu", "memory", "batch")
        
    Returns:
        Training results with performance optimization information
    """
    device_manager = DeviceManager()
    trainer = await create_model_trainer(device_manager)
    
    config = TrainingConfig(
        model_type=ModelType.TRANSFORMER,
        training_mode=TrainingMode.FINE_TUNE,
        model_name=model_name,
        dataset_path=dataset_path,
        output_dir=output_dir,
        num_epochs=num_epochs
    )
    
    # Setup performance optimization based on type
    if optimization_type == "comprehensive":
        config = setup_performance_optimization_config(config)
    elif optimization_type == "gpu":
        config = setup_gpu_optimization_config(config)
    elif optimization_type == "memory":
        config = setup_memory_optimization_config(config)
    elif optimization_type == "batch":
        config = setup_batch_optimization_config(config)
    
    # Run training
    result = await trainer.train(config)
    
    # Add performance optimization summary
    result['performance_optimization_summary'] = get_performance_optimization_summary(trainer)
    
    return result

async def ultra_optimized_train_transformer(
    model_name: str,
    dataset_path: str,
    output_dir: str = "models",
    num_epochs: int = 5
) -> Dict[str, Any]:
    """
    Ultra-optimized training with all performance features enabled.
    
    This function enables all possible performance optimizations for maximum training speed.
    
    Args:
        model_name: Name of the model to train
        dataset_path: Path to the dataset
        output_dir: Output directory for models
        num_epochs: Number of training epochs
        
    Returns:
        Training results with comprehensive performance information
    """
    device_manager = DeviceManager()
    trainer = await create_model_trainer(device_manager)
    
    config = TrainingConfig(
        model_type=ModelType.TRANSFORMER,
        training_mode=TrainingMode.FINE_TUNE,
        model_name=model_name,
        dataset_path=dataset_path,
        output_dir=output_dir,
        num_epochs=num_epochs
    )
    
    # Enable all optimizations
    config = setup_performance_optimization_config(config)
    
    # Additional ultra-optimizations
    config.enable_compilation = True
    config.enable_compile_mode = "max-autotune"
    config.enable_gradient_checkpointing = True
    config.enable_channels_last = True
    config.enable_tf32 = True
    config.enable_cudnn_benchmark = True
    config.enable_amp = True
    config.enable_dynamic_batching = True
    config.enable_pin_memory = True
    config.enable_persistent_workers = True
    config.num_workers = -1  # Auto-detect optimal
    config.prefetch_factor = 4  # Maximum prefetch
    
    # Run training
    result = await trainer.train(config)
    
    # Add comprehensive performance summary
    result['ultra_optimization_summary'] = get_performance_optimization_summary(trainer)
    result['performance_summary'] = trainer.get_performance_summary()
    
    return result

# Quick training functions with multi-GPU support
async def multi_gpu_train_transformer(
    model_name: str,
    dataset_path: str,
    output_dir: str = "models",
    num_epochs: int = 5,
    multi_gpu_type: str = "auto"
) -> Dict[str, Any]:
    """
    Quick training with multi-GPU support enabled.
    
    Args:
        model_name: Name of the model to train
        dataset_path: Path to the dataset
        output_dir: Output directory for models
        num_epochs: Number of training epochs
        multi_gpu_type: Type of multi-GPU training ("auto", "data_parallel", "distributed")
        
    Returns:
        Training results with multi-GPU information
    """
    device_manager = DeviceManager()
    trainer = await create_model_trainer(device_manager)
    
    config = TrainingConfig(
        model_type=ModelType.TRANSFORMER,
        training_mode=TrainingMode.FINE_TUNE,
        model_name=model_name,
        dataset_path=dataset_path,
        output_dir=output_dir,
        num_epochs=num_epochs
    )
    
    # Setup multi-GPU training based on type
    if multi_gpu_type == "data_parallel":
        config = setup_data_parallel_config(config)
    elif multi_gpu_type == "distributed":
        config = setup_distributed_config(config)
    else:  # auto
        config = setup_auto_multi_gpu_config(config)
    
    # Run training
    result = await trainer.train(config)
    
    # Add multi-GPU summary
    result['multi_gpu_summary'] = trainer.get_multi_gpu_summary()
    
    return result

async def data_parallel_train_transformer(
    model_name: str,
    dataset_path: str,
    output_dir: str = "models",
    num_epochs: int = 5
) -> Dict[str, Any]:
    """
    Quick training with DataParallel (simpler multi-GPU).
    
    Args:
        model_name: Name of the model to train
        dataset_path: Path to the dataset
        output_dir: Output directory for models
        num_epochs: Number of training epochs
        
    Returns:
        Training results with DataParallel information
    """
    return await multi_gpu_train_transformer(
        model_name=model_name,
        dataset_path=dataset_path,
        output_dir=output_dir,
        num_epochs=num_epochs,
        multi_gpu_type="data_parallel"
    )

async def distributed_train_transformer(
    model_name: str,
    dataset_path: str,
    output_dir: str = "models",
    num_epochs: int = 5
) -> Dict[str, Any]:
    """
    Quick training with DistributedDataParallel (advanced multi-GPU).
    
    Args:
        model_name: Name of the model to train
        dataset_path: Path to the dataset
        output_dir: Output directory for models
        num_epochs: Number of training epochs
        
    Returns:
        Training results with DistributedDataParallel information
    """
    return await multi_gpu_train_transformer(
        model_name=model_name,
        dataset_path=dataset_path,
        output_dir=output_dir,
        num_epochs=num_epochs,
        multi_gpu_type="distributed"
    )

def setup_auto_multi_gpu_config(config: TrainingConfig) -> TrainingConfig:
    """
    Setup automatic multi-GPU configuration.
    
    This function automatically chooses the best multi-GPU strategy based on
    available hardware and automatically configures all necessary settings.
    
    Args:
        config: Training configuration
        
    Returns:
        Updated configuration with automatic multi-GPU settings
    """
    # Auto-detect number of GPUs
    available_gpus = torch.cuda.device_count()
    config.num_gpus = available_gpus
    
    if available_gpus <= 1:
        # Single GPU - no parallelization needed
        config.use_data_parallel = False
        config.use_distributed_data_parallel = False
        return config
    
    # Choose strategy based on GPU count
    if available_gpus <= 4:
        # Use DataParallel for smaller setups (simpler)
        config.use_data_parallel = True
        config.use_distributed_data_parallel = False
    else:
        # Use DistributedDataParallel for larger setups (more efficient)
        config.use_data_parallel = False
        config.use_distributed_data_parallel = True
    
    # Auto-configure distributed settings
    if config.use_distributed_data_parallel:
        config.world_size = available_gpus
        config.rank = 0
        config.local_rank = 0
        config.distributed_backend = "nccl"
        config.distributed_init_method = "env://"
    
    return config

def setup_data_parallel_config(config: TrainingConfig) -> TrainingConfig:
    """
    Setup DataParallel configuration.
    
    Use this for simpler multi-GPU training with 2-4 GPUs.
    
    Args:
        config: Training configuration
        
    Returns:
        Updated configuration with DataParallel settings
    """
    config.use_data_parallel = True
    config.use_distributed_data_parallel = False
    
    # Auto-detect GPUs if not specified
    if config.num_gpus == -1:
        config.num_gpus = torch.cuda.device_count()
    
    # Validate GPU count
    available_gpus = torch.cuda.device_count()
    if config.num_gpus > available_gpus:
        config.num_gpus = available_gpus
    
    return config

def setup_distributed_config(config: TrainingConfig) -> TrainingConfig:
    """
    Setup DistributedDataParallel configuration.
    
    Use this for advanced multi-GPU training with 4+ GPUs or multi-node setups.
    
    Args:
        config: Training configuration
        
    Returns:
        Updated configuration with DistributedDataParallel settings
    """
    config.use_data_parallel = False
    config.use_distributed_data_parallel = True
    
    # Auto-detect GPUs if not specified
    if config.num_gpus == -1:
        config.num_gpus = torch.cuda.device_count()
    
    # Validate GPU count
    available_gpus = torch.cuda.device_count()
    if config.num_gpus > available_gpus:
        config.num_gpus = available_gpus
    
    # Configure distributed settings
    config.world_size = config.num_gpus
    config.rank = 0
    config.local_rank = 0
    config.distributed_backend = "nccl"
    config.distributed_init_method = "env://"
    
    # Optimize distributed settings
    config.find_unused_parameters = False
    config.gradient_as_bucket_view = True
    config.broadcast_buffers = True
    config.bucket_cap_mb = 25
    config.static_graph = False
    
    return config

def get_multi_gpu_training_summary(trainer: ModelTrainer) -> Dict[str, Any]:
    """
    Get a summary of multi-GPU training status and configuration.
    
    Args:
        trainer: Model trainer instance
        
    Returns:
        Dictionary containing multi-GPU training summary
    """
    summary = trainer.get_multi_gpu_summary()
    
    # Add training efficiency metrics
    if summary.get('multi_gpu_enabled', False):
        summary['efficiency'] = {
            'gpu_utilization': 'high' if summary.get('num_gpus', 1) > 1 else 'single',
            'scaling_factor': summary.get('num_gpus', 1),
            'training_type': summary.get('training_type', 'single_gpu')
        }
    
    return summary

# Multi-GPU training with performance optimization
async def optimized_multi_gpu_train_transformer(
    model_name: str,
    dataset_path: str,
    output_dir: str = "models",
    num_epochs: int = 5,
    multi_gpu_type: str = "auto",
    optimization_type: str = "comprehensive"
) -> Dict[str, Any]:
    """
    Quick training with both multi-GPU and performance optimization.
    
    Args:
        model_name: Name of the model to train
        dataset_path: Path to the dataset
        output_dir: Output directory for models
        num_epochs: Number of training epochs
        multi_gpu_type: Type of multi-GPU training ("auto", "data_parallel", "distributed")
        optimization_type: Type of optimization ("comprehensive", "gpu", "memory", "batch")
        
    Returns:
        Training results with multi-GPU and performance optimization information
    """
    device_manager = DeviceManager()
    trainer = await create_model_trainer(device_manager)
    
    config = TrainingConfig(
        model_type=ModelType.TRANSFORMER,
        training_mode=TrainingMode.FINE_TUNE,
        model_name=model_name,
        dataset_path=dataset_path,
        output_dir=output_dir,
        num_epochs=num_epochs
    )
    
    # Setup multi-GPU training
    if multi_gpu_type == "data_parallel":
        config = setup_data_parallel_config(config)
    elif multi_gpu_type == "distributed":
        config = setup_distributed_config(config)
    else:  # auto
        config = setup_auto_multi_gpu_config(config)
    
    # Setup performance optimization
    if optimization_type == "comprehensive":
        config = setup_performance_optimization_config(config)
    elif optimization_type == "gpu":
        config = setup_gpu_optimization_config(config)
    elif optimization_type == "memory":
        config = setup_memory_optimization_config(config)
    elif optimization_type == "batch":
        config = setup_batch_optimization_config(config)
    
    # Run training
    result = await trainer.train(config)
    
    # Add comprehensive summaries
    result['multi_gpu_summary'] = trainer.get_multi_gpu_summary()
    result['performance_optimization_summary'] = get_performance_optimization_summary(trainer)
    result['performance_summary'] = trainer.get_performance_summary()
    
    return result

# Ultra-optimized multi-GPU training
async def ultra_optimized_multi_gpu_train_transformer(
    model_name: str,
    dataset_path: str,
    output_dir: str = "models",
    num_epochs: int = 5
) -> Dict[str, Any]:
    """
    Ultra-optimized training with maximum multi-GPU and performance optimization.
    
    This function enables all possible optimizations for maximum training speed
    across multiple GPUs.
    
    Args:
        model_name: Name of the model to train
        dataset_path: Path to the dataset
        output_dir: Output directory for models
        num_epochs: Number of training epochs
        
    Returns:
        Training results with comprehensive optimization information
    """
    device_manager = DeviceManager()
    trainer = await create_model_trainer(device_manager)
    
    config = TrainingConfig(
        model_type=ModelType.TRANSFORMER,
        training_mode=TrainingMode.FINE_TUNE,
        model_name=model_name,
        dataset_path=dataset_path,
        output_dir=output_dir,
        num_epochs=num_epochs
    )
    
    # Enable all multi-GPU optimizations
    config = setup_auto_multi_gpu_config(config)
    
    # Enable all performance optimizations
    config = setup_performance_optimization_config(config)
    
    # Additional ultra-optimizations
    config.enable_compilation = True
    config.enable_compile_mode = "max-autotune"
    config.enable_gradient_checkpointing = True
    config.enable_channels_last = True
    config.enable_tf32 = True
    config.enable_cudnn_benchmark = True
    config.enable_amp = True
    config.enable_dynamic_batching = True
    config.enable_pin_memory = True
    config.enable_persistent_workers = True
    config.num_workers = -1  # Auto-detect optimal
    config.prefetch_factor = 4  # Maximum prefetch
    
    # Run training
    result = await trainer.train(config)
    
    # Add comprehensive summaries
    result['ultra_optimization_summary'] = get_performance_optimization_summary(trainer)
    result['multi_gpu_summary'] = trainer.get_multi_gpu_summary()
    result['performance_summary'] = trainer.get_performance_summary()
    
    return result

# Example usage
if __name__ == "__main__":
    async def demo():
        
    """demo function."""
# Quick training example
        result = await quick_train_transformer(
            model_name="distilbert-base-uncased",
            dataset_path="data/sentiment_dataset.csv",
            num_epochs=3
        )
        print(f"Training completed: {result}")
        
        # Hyperparameter optimization example
        hpo_result = await quick_hyperparameter_optimization(
            model_name="distilbert-base-uncased",
            dataset_path="data/sentiment_dataset.csv",
            trials=10
        )
        print(f"HPO completed: {hpo_result}")
        
        # Debugging examples
        print("\n=== Debugging Examples ===")
        
        # Comprehensive debugging
        debug_result = await debug_train_transformer(
            model_name="distilbert-base-uncased",
            dataset_path="data/sentiment_dataset.csv",
            debug_type="comprehensive"
        )
        print(f"Comprehensive debugging: {debug_result['debug_summary']}")
        
        # Performance optimization examples
        print("\n=== Performance Optimization Examples ===")
        
        # Comprehensive optimization
        optimized_result = await optimized_train_transformer(
            model_name="distilbert-base-uncased",
            dataset_path="data/sentiment_dataset.csv",
            optimization_type="comprehensive"
        )
        print(f"Comprehensive optimization: {optimized_result['performance_optimization_summary']}")
        
        # GPU optimization
        gpu_result = await optimized_train_transformer(
            model_name="distilbert-base-uncased",
            dataset_path="data/sentiment_dataset.csv",
            optimization_type="gpu"
        )
        print(f"GPU optimization: {gpu_result['performance_optimization_summary']}")
        
        # Memory optimization
        memory_result = await optimized_train_transformer(
            model_name="distilbert-base-uncased",
            dataset_path="data/sentiment_dataset.csv",
            optimization_type="memory"
        )
        print(f"Memory optimization: {memory_result['performance_optimization_summary']}")
        
        # Ultra optimization
        ultra_result = await ultra_optimized_train_transformer(
            model_name="distilbert-base-uncased",
            dataset_path="data/sentiment_dataset.csv"
        )
        print(f"Ultra optimization: {ultra_result['performance_summary']}")
        
        # Multi-GPU training examples
        print("\n=== Multi-GPU Training Examples ===")
        
        # Auto multi-GPU training
        multi_gpu_result = await multi_gpu_train_transformer(
            model_name="distilbert-base-uncased",
            dataset_path="data/sentiment_dataset.csv",
            multi_gpu_type="auto"
        )
        print(f"Auto multi-GPU: {multi_gpu_result['multi_gpu_summary']}")
        
        # DataParallel training
        data_parallel_result = await data_parallel_train_transformer(
            model_name="distilbert-base-uncased",
            dataset_path="data/sentiment_dataset.csv"
        )
        print(f"DataParallel: {data_parallel_result['multi_gpu_summary']}")
        
        # DistributedDataParallel training
        distributed_result = await distributed_train_transformer(
            model_name="distilbert-base-uncased",
            dataset_path="data/sentiment_dataset.csv"
        )
        print(f"DistributedDataParallel: {distributed_result['multi_gpu_summary']}")
        
        # Combined multi-GPU + performance optimization
        print("\n=== Combined Multi-GPU + Performance Optimization ===")
        
        # Optimized multi-GPU training
        optimized_multi_gpu_result = await optimized_multi_gpu_train_transformer(
            model_name="distilbert-base-uncased",
            dataset_path="data/sentiment_dataset.csv",
            multi_gpu_type="auto",
            optimization_type="comprehensive"
        )
        print(f"Optimized multi-GPU: {optimized_multi_gpu_result['multi_gpu_summary']}")
        print(f"Performance: {optimized_multi_gpu_result['performance_summary']}")
        
        # Ultra-optimized multi-GPU training
        ultra_multi_gpu_result = await ultra_optimized_multi_gpu_train_transformer(
            model_name="distilbert-base-uncased",
            dataset_path="data/sentiment_dataset.csv"
        )
        print(f"Ultra-optimized multi-GPU: {ultra_multi_gpu_result['multi_gpu_summary']}")
        print(f"Performance: {ultra_multi_gpu_result['performance_summary']}")
        
        print("\n=== Performance Comparison ===")
        print("Standard training vs Multi-GPU vs Ultra-optimized Multi-GPU:")
        print(f"Standard: {result.get('total_training_time', 'N/A')} seconds")
        print(f"Multi-GPU: {multi_gpu_result.get('total_training_time', 'N/A')} seconds")
        print(f"Ultra-optimized Multi-GPU: {ultra_multi_gpu_result.get('total_training_time', 'N/A')} seconds")
        
        # Compare GPU utilization
        if 'multi_gpu_summary' in multi_gpu_result:
            mgpu = multi_gpu_result['multi_gpu_summary']
            print(f"GPU utilization: {mgpu.get('num_gpus', 1)} GPUs")
            print(f"Training type: {mgpu.get('training_type', 'single_gpu')}")
        
        if 'performance_summary' in ultra_multi_gpu_result:
            perf = ultra_multi_gpu_result['performance_summary']
            if 'performance' in perf:
                print(f"Average throughput: {perf['performance'].get('avg_throughput', 'N/A')} samples/sec")
                print(f"Max throughput: {perf['performance'].get('max_throughput', 'N/A')} samples/sec")
                print(f"Memory efficiency: {perf['performance'].get('memory_efficiency', 'N/A')}")
    
    # Run demo
    asyncio.run(demo()) 