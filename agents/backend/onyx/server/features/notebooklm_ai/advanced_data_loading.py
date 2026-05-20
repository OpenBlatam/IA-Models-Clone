from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import torch
import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn as nn
import torch.optim as optim.nn as nn
import torch
import torch.nn as nn
import torch.optim as optim.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split, Subset, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW, Adam, SGD
from torch.optim.lr_scheduler import (
from torch.cuda.amp import GradScaler, autocast
import torch
import torch.nn as nn
import torch.optim as optim.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from diffusers import (
from transformers import CLIPTextModel, CLIPTokenizer, CLIPProcessor, CLIPModel
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import asyncio
import time
import gc
import logging
import json
import os
import pickle
import hashlib
from typing import Dict, Any, List, Optional, Union, Tuple, Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from functools import lru_cache, partial
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
from contextlib import contextmanager
import warnings
from collections import defaultdict, deque
import random
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
    from pytorch_fid import fid_score
    import lpips
    from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
    from prometheus_client import Counter, Histogram, Gauge
        import shutil
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Advanced Data Loading and Evaluation System
==========================================

Comprehensive data loading system with:
- Efficient PyTorch DataLoader implementation
- Proper train/validation/test splits
- Cross-validation support
- Advanced early stopping strategies
- Sophisticated learning rate scheduling
- Task-specific evaluation metrics

Features: Async data loading, memory optimization, caching,
distributed training support, and production-ready evaluation.
"""

    CosineAnnealingLR, StepLR, ReduceLROnPlateau, 
    CosineAnnealingWarmRestarts, OneCycleLR, ExponentialLR,
    MultiStepLR, LambdaLR, ChainedScheduler
)

    StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL,
    DDIMScheduler, DDPMScheduler, PNDMScheduler, EulerDiscreteScheduler
)


# Evaluation metrics
try:
    FID_AVAILABLE = True
except ImportError:
    FID_AVAILABLE = False

try:
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False

try:
    TORCHMETRICS_AVAILABLE = True
except ImportError:
    TORCHMETRICS_AVAILABLE = False

# Performance monitoring
try:
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
if PROMETHEUS_AVAILABLE:
    DATA_LOADING_TIME = Histogram('data_loading_duration_seconds', 'Data loading time')
    BATCH_PROCESSING_TIME = Histogram('batch_processing_duration_seconds', 'Batch processing time')
    EVALUATION_METRICS = Histogram('evaluation_metrics', 'Evaluation metric values', ['metric_name', 'task_type'])
    EARLY_STOPPING_EVENTS = Counter('early_stopping_events_total', 'Early stopping events', ['reason'])


@dataclass
class DataConfig:
    """Configuration for data loading and processing."""
    # Data paths
    data_dir: str = "data"
    train_dir: str = "data/train"
    val_dir: str = "data/validation"
    test_dir: str = "data/test"
    cache_dir: str = "cache"
    
    # Data splits
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42
    
    # Cross-validation
    use_cross_validation: bool = False
    n_folds: int = 5
    cv_strategy: str = "stratified"  # stratified, kfold, time_series
    
    # Data loading
    batch_size: int = 8
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    drop_last: bool = False
    
    # Image processing
    image_size: int = 512
    center_crop: bool = False
    random_flip: bool = True
    normalize: bool = True
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    
    # Augmentation
    use_augmentation: bool = True
    augmentation_strength: float = 0.5
    color_jitter: bool = True
    random_rotation: bool = True
    random_scale: bool = True
    
    # Caching
    use_cache: bool = True
    cache_size: int = 1000
    cache_ttl: int = 3600  # seconds
    
    # Memory optimization
    use_memory_mapping: bool = False
    memory_map_size: int = 1024 * 1024 * 1024  # 1GB
    use_shared_memory: bool = True
    
    # Distributed training
    distributed: bool = False
    world_size: int = 1
    rank: int = 0


@dataclass
class TrainingConfig:
    """Enhanced training configuration with advanced features."""
    # Model configuration
    model_name: str = "runwayml/stable-diffusion-v1-5"
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 100
    max_steps: Optional[int] = None
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Optimizer
    optimizer_type: str = "adamw"  # adamw, adam, sgd, lion
    optimizer_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # Learning rate scheduling
    lr_scheduler_type: str = "cosine"  # cosine, linear, step, reduce_lr_on_plateau, onecycle, exponential
    warmup_steps: int = 500
    warmup_ratio: float = 0.1
    min_lr: float = 1e-6
    max_lr: float = 1e-3
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    early_stopping_mode: str = "min"  # min, max
    early_stopping_metric: str = "val_loss"
    restore_best_weights: bool = True
    
    # Advanced early stopping
    use_plateau_detection: bool = True
    plateau_patience: int = 5
    plateau_threshold: float = 0.01
    use_overfitting_detection: bool = True
    overfitting_threshold: float = 0.1
    
    # Mixed precision
    mixed_precision: str = "fp16"  # fp16, fp32, bf16
    gradient_checkpointing: bool = True
    
    # Monitoring
    log_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500
    save_total_limit: Optional[int] = None
    
    # Output
    output_dir: str = "outputs"
    logging_dir: str = "logs"
    run_name: Optional[str] = None


@dataclass
class EvaluationConfig:
    """Comprehensive evaluation configuration."""
    # Task-specific metrics
    task_type: str = "image_generation"  # image_generation, image_classification, image_segmentation, text_to_image
    
    # Image generation metrics
    compute_fid: bool = True
    compute_lpips: bool = True
    compute_clip_score: bool = True
    compute_psnr: bool = True
    compute_ssim: bool = True
    compute_inception_score: bool = False
    
    # Classification metrics
    compute_accuracy: bool = False
    compute_precision_recall: bool = False
    compute_confusion_matrix: bool = False
    compute_roc_auc: bool = False
    
    # Segmentation metrics
    compute_iou: bool = False
    compute_dice: bool = False
    compute_pixel_accuracy: bool = False
    
    # Generation parameters
    num_eval_samples: int = 100
    eval_prompt: str = "A beautiful landscape"
    eval_negative_prompt: str = "blurry, low quality"
    eval_guidance_scale: float = 7.5
    eval_num_inference_steps: int = 50
    
    # Evaluation settings
    eval_batch_size: int = 8
    save_eval_results: bool = True
    generate_eval_report: bool = True
    eval_output_dir: str = "evaluation_results"


class AdvancedDiffusionDataset(Dataset):
    """Advanced dataset with efficient loading and caching."""
    
    def __init__(self, data_dir: str, tokenizer: CLIPTokenizer, config: DataConfig,
                 split: str = "train", transform: Optional[Callable] = None):
        
    """__init__ function."""
self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.config = config
        self.split = split
        self.transform = transform
        
        # Load data files
        self.data_files = self._load_data_files()
        
        # Setup caching
        if config.use_cache:
            self.cache = {}
            self.cache_file = Path(config.cache_dir) / f"cache_{split}_{hashlib.md5(str(data_dir).encode()).hexdigest()[:8]}.pkl"
            self._load_cache()
        
        # Setup memory mapping if enabled
        if config.use_memory_mapping:
            self._setup_memory_mapping()
        
        logger.info(f"Loaded {len(self.data_files)} samples for {split} split")
    
    def _load_data_files(self) -> List[Dict[str, Any]]:
        """Load data files with metadata."""
        data_files = []
        
        # Load images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(self.data_dir.glob(f"**/*{ext}"))
            image_files.extend(self.data_dir.glob(f"**/*{ext.upper()}"))
        
        # Load captions if available
        caption_file = self.data_dir / "captions.json"
        captions = {}
        if caption_file.exists():
            with open(caption_file, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                captions = json.load(f)
        
        # Create data entries
        for img_path in image_files:
            rel_path = img_path.relative_to(self.data_dir)
            caption = captions.get(str(rel_path), f"Image from {rel_path.stem}")
            
            data_files.append({
                'image_path': str(img_path),
                'caption': caption,
                'rel_path': str(rel_path)
            })
        
        return data_files
    
    def _load_cache(self) -> Any:
        """Load cached data."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    self.cache = pickle.load(f)
                logger.info(f"Loaded cache with {len(self.cache)} entries")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                self.cache = {}
    
    def _save_cache(self) -> Any:
        """Save cache to disk."""
        try:
            self.cache_file.parent.mkdir(exist_ok=True)
            with open(self.cache_file, 'wb') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                pickle.dump(self.cache, f)
            logger.info(f"Saved cache with {len(self.cache)} entries")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def _setup_memory_mapping(self) -> Any:
        """Setup memory mapping for large datasets."""
        # Implementation for memory mapping
        pass
    
    def __len__(self) -> Any:
        return len(self.data_files)
    
    def __getitem__(self, idx) -> Optional[Dict[str, Any]]:
        data_entry = self.data_files[idx]
        
        # Check cache first
        cache_key = f"{idx}_{data_entry['rel_path']}"
        if self.config.use_cache and cache_key in self.cache:
            return self.cache[cache_key]
        
        # Load image
        image = self._load_image(data_entry['image_path'])
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        # Tokenize caption
        inputs = self.tokenizer(
            data_entry['caption'],
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Prepare output
        output = {
            "pixel_values": image,
            "input_ids": inputs.input_ids.squeeze(),
            "attention_mask": inputs.attention_mask.squeeze(),
            "caption": data_entry['caption'],
            "image_path": data_entry['image_path']
        }
        
        # Cache result
        if self.config.use_cache and len(self.cache) < self.config.cache_size:
            self.cache[cache_key] = output
        
        return output
    
    def _load_image(self, image_path: str) -> Image.Image:
        """Load and preprocess image."""
        try:
            image = Image.open(image_path).convert("RGB")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            # Apply basic preprocessing
            if self.config.center_crop:
                image = self._center_crop(image)
            
            image = image.resize((self.config.image_size, self.config.image_size), Image.LANCZOS)
            
            if self.config.random_flip and self.split == "train" and random.random() > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            
            return image
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            # Return a blank image as fallback
            return Image.new('RGB', (self.config.image_size, self.config.image_size), (128, 128, 128))
    
    def _center_crop(self, image: Image.Image) -> Image.Image:
        """Center crop image to square."""
        width, height = image.size
        size = min(width, height)
        left = (width - size) // 2
        top = (height - size) // 2
        right = left + size
        bottom = top + size
        return image.crop((left, top, right, bottom))
    
    def __del__(self) -> Any:
        """Cleanup cache on deletion."""
        if hasattr(self, 'config') and self.config.use_cache:
            self._save_cache()


class AdvancedDataLoader:
    """Advanced data loader with efficient loading and splitting."""
    
    def __init__(self, config: DataConfig, tokenizer: CLIPTokenizer):
        
    """__init__ function."""
self.config = config
        self.tokenizer = tokenizer
        self.transforms = self._create_transforms()
    
    def _create_transforms(self) -> Dict[str, Callable]:
        """Create transforms for different splits."""
        # Base transforms
        base_transforms = A.Compose([
            A.Resize(self.config.image_size, self.config.image_size),
            A.Normalize(mean=self.config.mean, std=self.config.std),
            ToTensorV2()
        ])
        
        # Training transforms with augmentation
        if self.config.use_augmentation:
            train_transforms = A.Compose([
                A.Resize(self.config.image_size, self.config.image_size),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
                A.OneOf([
                    A.RandomBrightnessContrast(p=1),
                    A.RandomGamma(p=1),
                    A.HueSaturationValue(p=1),
                ], p=0.5),
                A.Normalize(mean=self.config.mean, std=self.config.std),
                ToTensorV2()
            ])
        else:
            train_transforms = base_transforms
        
        return {
            'train': train_transforms,
            'val': base_transforms,
            'test': base_transforms
        }
    
    def create_datasets(self, data_dir: str) -> Dict[str, AdvancedDiffusionDataset]:
        """Create train/val/test datasets."""
        # Load full dataset
        full_dataset = AdvancedDiffusionDataset(
            data_dir, self.tokenizer, self.config, split="full"
        )
        
        # Calculate split sizes
        total_size = len(full_dataset)
        train_size = int(self.config.train_ratio * total_size)
        val_size = int(self.config.val_ratio * total_size)
        test_size = total_size - train_size - val_size
        
        # Set random seed for reproducible splits
        torch.manual_seed(self.config.random_seed)
        
        # Split dataset
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(self.config.random_seed)
        )
        
        # Create datasets with appropriate transforms
        datasets = {
            'train': AdvancedDiffusionDataset(
                data_dir, self.tokenizer, self.config, split="train",
                transform=self.transforms['train']
            ),
            'val': AdvancedDiffusionDataset(
                data_dir, self.tokenizer, self.config, split="val",
                transform=self.transforms['val']
            ),
            'test': AdvancedDiffusionDataset(
                data_dir, self.tokenizer, self.config, split="test",
                transform=self.transforms['test']
            )
        }
        
        # Apply splits
        datasets['train'] = Subset(datasets['train'], train_dataset.indices)
        datasets['val'] = Subset(datasets['val'], val_dataset.indices)
        datasets['test'] = Subset(datasets['test'], test_dataset.indices)
        
        return datasets
    
    def create_cross_validation_splits(self, data_dir: str) -> List[Dict[str, AdvancedDiffusionDataset]]:
        """Create cross-validation splits."""
        full_dataset = AdvancedDiffusionDataset(
            data_dir, self.tokenizer, self.config, split="full"
        )
        
        # Create cross-validation splits
        if self.config.cv_strategy == "stratified":
            # For stratified CV, we need labels - using image paths as proxy
            labels = [hash(d['image_path']) % 10 for d in full_dataset.data_files]
            kfold = StratifiedKFold(n_splits=self.config.n_folds, shuffle=True, random_state=self.config.random_seed)
        else:
            kfold = KFold(n_splits=self.config.n_folds, shuffle=True, random_state=self.config.random_seed)
            labels = None
        
        cv_splits = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(range(len(full_dataset)), labels)):
            # Create datasets for this fold
            train_dataset = AdvancedDiffusionDataset(
                data_dir, self.tokenizer, self.config, split=f"train_fold_{fold}",
                transform=self.transforms['train']
            )
            val_dataset = AdvancedDiffusionDataset(
                data_dir, self.tokenizer, self.config, split=f"val_fold_{fold}",
                transform=self.transforms['val']
            )
            
            # Apply splits
            train_dataset = Subset(train_dataset, train_idx)
            val_dataset = Subset(val_dataset, val_idx)
            
            cv_splits.append({
                'fold': fold,
                'train': train_dataset,
                'val': val_dataset
            })
        
        return cv_splits
    
    def create_data_loaders(self, datasets: Dict[str, AdvancedDiffusionDataset]) -> Dict[str, DataLoader]:
        """Create data loaders for all splits."""
        loaders = {}
        
        for split, dataset in datasets.items():
            # Create sampler
            if self.config.distributed:
                sampler = DistributedSampler(
                    dataset,
                    num_replicas=self.config.world_size,
                    rank=self.config.rank,
                    shuffle=(split == 'train')
                )
            else:
                sampler = None
            
            # Create data loader
            loader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=(split == 'train' and sampler is None),
                sampler=sampler,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
                persistent_workers=self.config.persistent_workers,
                prefetch_factor=self.config.prefetch_factor,
                drop_last=self.config.drop_last,
                collate_fn=self._collate_fn
            )
            
            loaders[split] = loader
        
        return loaders
    
    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Custom collate function for batching."""
        pixel_values = torch.stack([item['pixel_values'] for item in batch])
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        
        return {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'captions': [item['caption'] for item in batch],
            'image_paths': [item['image_path'] for item in batch]
        }


class AdvancedLearningRateScheduler:
    """Advanced learning rate scheduler with multiple strategies."""
    
    def __init__(self, optimizer: torch.optim.Optimizer, config: TrainingConfig, num_training_steps: int):
        
    """__init__ function."""
self.optimizer = optimizer
        self.config = config
        self.num_training_steps = num_training_steps
        self.scheduler = self._create_scheduler()
    
    def _create_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler based on configuration."""
        if self.config.lr_scheduler_type == "cosine":
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.num_training_steps,
                eta_min=self.config.min_lr
            )
        
        elif self.config.lr_scheduler_type == "cosine_warmup":
            return get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.warmup_steps,
                num_training_steps=self.num_training_steps
            )
        
        elif self.config.lr_scheduler_type == "linear":
            return get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.warmup_steps,
                num_training_steps=self.num_training_steps
            )
        
        elif self.config.lr_scheduler_type == "step":
            return StepLR(
                self.optimizer,
                step_size=self.config.warmup_steps,
                gamma=0.1
            )
        
        elif self.config.lr_scheduler_type == "reduce_lr_on_plateau":
            return ReduceLROnPlateau(
                self.optimizer,
                mode=self.config.early_stopping_mode,
                factor=0.5,
                patience=self.config.plateau_patience,
                min_lr=self.config.min_lr,
                verbose=True
            )
        
        elif self.config.lr_scheduler_type == "onecycle":
            return OneCycleLR(
                self.optimizer,
                max_lr=self.config.max_lr,
                total_steps=self.num_training_steps,
                pct_start=self.config.warmup_ratio,
                anneal_strategy='cos'
            )
        
        elif self.config.lr_scheduler_type == "exponential":
            return ExponentialLR(
                self.optimizer,
                gamma=0.95
            )
        
        elif self.config.lr_scheduler_type == "cosine_restarts":
            return CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.config.warmup_steps,
                T_mult=2,
                eta_min=self.config.min_lr
            )
        
        else:
            raise ValueError(f"Unknown scheduler type: {self.config.lr_scheduler_type}")
    
    def step(self, metrics: Optional[float] = None):
        """Step the scheduler."""
        if isinstance(self.scheduler, ReduceLROnPlateau):
            if metrics is not None:
                self.scheduler.step(metrics)
        else:
            self.scheduler.step()
    
    def get_last_lr(self) -> List[float]:
        """Get current learning rates."""
        return self.scheduler.get_last_lr()
    
    def state_dict(self) -> Dict[str, Any]:
        """Get scheduler state."""
        return self.scheduler.state_dict()
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load scheduler state."""
        self.scheduler.load_state_dict(state_dict)


class AdvancedEarlyStopping:
    """Advanced early stopping with multiple strategies."""
    
    def __init__(self, config: TrainingConfig):
        
    """__init__ function."""
self.config = config
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.should_stop = False
        self.stopping_reason = None
        
        # History for plateau detection
        self.metric_history = deque(maxlen=20)
        self.loss_history = deque(maxlen=20)
        
        # Overfitting detection
        self.train_loss_history = deque(maxlen=20)
        self.val_loss_history = deque(maxlen=20)
    
    def __call__(self, val_metric: float, train_metric: Optional[float] = None, 
                 model: Optional[nn.Module] = None) -> bool:
        """Check if training should stop."""
        self.metric_history.append(val_metric)
        
        if train_metric is not None:
            self.train_loss_history.append(train_metric)
            self.val_loss_history.append(val_metric)
        
        # Check for improvement
        if self.best_score is None:
            self.best_score = val_metric
            if model is not None and self.config.restore_best_weights:
                self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            return False
        
        # Determine if metric improved
        if self.config.early_stopping_mode == "min":
            improved = val_metric < self.best_score - self.config.early_stopping_min_delta
        else:
            improved = val_metric > self.best_score + self.config.early_stopping_min_delta
        
        if improved:
            self.best_score = val_metric
            self.counter = 0
            if model is not None and self.config.restore_best_weights:
                self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        
        # Check early stopping conditions
        if self.counter >= self.config.early_stopping_patience:
            self.should_stop = True
            self.stopping_reason = "patience_exceeded"
            if PROMETHEUS_AVAILABLE:
                EARLY_STOPPING_EVENTS.labels(reason="patience_exceeded").inc()
        
        # Plateau detection
        if self.config.use_plateau_detection and len(self.metric_history) >= 10:
            recent_metrics = list(self.metric_history)[-10:]
            if self._is_plateau(recent_metrics):
                self.should_stop = True
                self.stopping_reason = "plateau_detected"
                if PROMETHEUS_AVAILABLE:
                    EARLY_STOPPING_EVENTS.labels(reason="plateau_detected").inc()
        
        # Overfitting detection
        if self.config.use_overfitting_detection and len(self.train_loss_history) >= 10:
            if self._is_overfitting():
                self.should_stop = True
                self.stopping_reason = "overfitting_detected"
                if PROMETHEUS_AVAILABLE:
                    EARLY_STOPPING_EVENTS.labels(reason="overfitting_detected").inc()
        
        return self.should_stop
    
    def _is_plateau(self, metrics: List[float]) -> bool:
        """Check if metrics have plateaued."""
        if len(metrics) < 10:
            return False
        
        # Calculate variance of recent metrics
        recent_variance = np.var(metrics[-5:])
        overall_variance = np.var(metrics)
        
        # Check if recent variance is very low compared to overall variance
        return recent_variance < overall_variance * 0.1
    
    def _is_overfitting(self) -> bool:
        """Check if model is overfitting."""
        if len(self.train_loss_history) < 10 or len(self.val_loss_history) < 10:
            return False
        
        # Calculate recent trends
        recent_train_trend = np.mean(list(self.train_loss_history)[-5:]) - np.mean(list(self.train_loss_history)[-10:-5])
        recent_val_trend = np.mean(list(self.val_loss_history)[-5:]) - np.mean(list(self.val_loss_history)[-10:-5])
        
        # Check if validation loss is increasing while training loss is decreasing
        return recent_val_trend > self.config.overfitting_threshold and recent_train_trend < -self.config.overfitting_threshold
    
    def restore_best_weights(self, model: nn.Module):
        """Restore best weights to model."""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            logger.info("Restored best weights from early stopping")
    
    def get_stopping_reason(self) -> Optional[str]:
        """Get the reason for stopping."""
        return self.stopping_reason


class TaskSpecificEvaluator:
    """Task-specific evaluation metrics."""
    
    def __init__(self, config: EvaluationConfig):
        
    """__init__ function."""
self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available()  # AI: Device optimization else "cpu")
        self._setup_metrics()
    
    def _setup_metrics(self) -> Any:
        """Setup evaluation metrics based on task type."""
        if self.config.task_type == "image_generation":
            self._setup_generation_metrics()
        elif self.config.task_type == "image_classification":
            self._setup_classification_metrics()
        elif self.config.task_type == "image_segmentation":
            self._setup_segmentation_metrics()
        elif self.config.task_type == "text_to_image":
            self._setup_text_to_image_metrics()
    
    def _setup_generation_metrics(self) -> Any:
        """Setup image generation metrics."""
        if self.config.compute_lpips and LPIPS_AVAILABLE:
            self.lpips_model = lpips.LPIPS(net='alex', spatial=False)
            self.lpips_model.to(self.device)
        
        if self.config.compute_clip_score:
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model.to(self.device)
        
        if self.config.compute_psnr and TORCHMETRICS_AVAILABLE:
            self.psnr_metric = PeakSignalNoiseRatio()
        
        if self.config.compute_ssim and TORCHMETRICS_AVAILABLE:
            self.ssim_metric = StructuralSimilarityIndexMeasure()
    
    def _setup_classification_metrics(self) -> Any:
        """Setup classification metrics."""
        pass  # Implement classification-specific metrics
    
    def _setup_segmentation_metrics(self) -> Any:
        """Setup segmentation metrics."""
        pass  # Implement segmentation-specific metrics
    
    def _setup_text_to_image_metrics(self) -> Any:
        """Setup text-to-image metrics."""
        # Similar to generation metrics but with text conditioning
        self._setup_generation_metrics()
    
    async def evaluate_generation(self, generated_images: List[Image.Image], 
                                reference_images: Optional[List[Image.Image]] = None,
                                prompts: Optional[List[str]] = None) -> Dict[str, float]:
        """Evaluate image generation quality."""
        metrics = {}
        
        # FID Score
        if self.config.compute_fid and FID_AVAILABLE and reference_images:
            fid_score = await self._compute_fid(generated_images, reference_images)
            metrics['fid'] = fid_score
        
        # LPIPS Score
        if self.config.compute_lpips and LPIPS_AVAILABLE and reference_images:
            lpips_score = await self._compute_lpips(generated_images, reference_images)
            metrics['lpips'] = lpips_score
        
        # CLIP Score
        if self.config.compute_clip_score and prompts:
            clip_score = await self._compute_clip_score(generated_images, prompts)
            metrics['clip_score'] = clip_score
        
        # PSNR and SSIM
        if self.config.compute_psnr and reference_images:
            psnr_score = await self._compute_psnr(generated_images, reference_images)
            metrics['psnr'] = psnr_score
        
        if self.config.compute_ssim and reference_images:
            ssim_score = await self._compute_ssim(generated_images, reference_images)
            metrics['ssim'] = ssim_score
        
        # Update Prometheus metrics
        if PROMETHEUS_AVAILABLE:
            for metric_name, value in metrics.items():
                EVALUATION_METRICS.labels(metric_name=metric_name, task_type=self.config.task_type).observe(value)
        
        return metrics
    
    async def _compute_fid(self, generated_images: List[Image.Image], 
                          reference_images: List[Image.Image]) -> float:
        """Compute FID score."""
        # Save images temporarily
        temp_gen_dir = Path("temp_generated")
        temp_ref_dir = Path("temp_reference")
        temp_gen_dir.mkdir(exist_ok=True)
        temp_ref_dir.mkdir(exist_ok=True)
        
        for i, img in enumerate(generated_images):
            img.save(temp_gen_dir / f"gen_{i:04d}.png")
        
        for i, img in enumerate(reference_images):
            img.save(temp_ref_dir / f"ref_{i:04d}.png")
        
        # Compute FID
        def _compute_fid_score():
            
    """_compute_fid_score function."""
return fid_score.calculate_fid_given_paths(
                [str(temp_ref_dir), str(temp_gen_dir)],
                batch_size=50,
                device=self.device
            )
        
        fid_score_value = await asyncio.get_event_loop().run_in_executor(None, _compute_fid_score)
        
        # Cleanup
        shutil.rmtree(temp_gen_dir)
        shutil.rmtree(temp_ref_dir)
        
        return fid_score_value
    
    async def _compute_lpips(self, generated_images: List[Image.Image], 
                           reference_images: List[Image.Image]) -> float:
        """Compute LPIPS score."""
        total_lpips = 0.0
        num_pairs = 0
        
        for gen_img, ref_img in zip(generated_images, reference_images):
            # Convert to tensors
            gen_tensor = torch.from_numpy(np.array(gen_img)).permute(2, 0, 1).float() / 255.0
            ref_tensor = torch.from_numpy(np.array(ref_img)).permute(2, 0, 1).float() / 255.0
            
            gen_tensor = gen_tensor.unsqueeze(0).to(self.device)
            ref_tensor = ref_tensor.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                lpips_score = self.lpips_model(gen_tensor, ref_tensor).item()
            
            total_lpips += lpips_score
            num_pairs += 1
        
        return total_lpips / num_pairs if num_pairs > 0 else 0.0
    
    async def _compute_clip_score(self, generated_images: List[Image.Image], 
                                prompts: List[str]) -> float:
        """Compute CLIP score."""
        inputs = self.clip_processor(
            images=generated_images,
            text=prompts,
            return_tensors="pt",
            padding=True
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=-1)
            clip_score = probs.diagonal().mean().item()
        
        return clip_score
    
    async def _compute_psnr(self, generated_images: List[Image.Image], 
                           reference_images: List[Image.Image]) -> float:
        """Compute PSNR score."""
        total_psnr = 0.0
        
        for gen_img, ref_img in zip(generated_images, reference_images):
            gen_tensor = torch.from_numpy(np.array(gen_img)).permute(2, 0, 1).float() / 255.0
            ref_tensor = torch.from_numpy(np.array(ref_img)).permute(2, 0, 1).float() / 255.0
            
            gen_tensor = gen_tensor.unsqueeze(0)
            ref_tensor = ref_tensor.unsqueeze(0)
            
            psnr_score = self.psnr_metric(gen_tensor, ref_tensor)
            total_psnr += psnr_score.item()
        
        return total_psnr / len(generated_images)
    
    async def _compute_ssim(self, generated_images: List[Image.Image], 
                           reference_images: List[Image.Image]) -> float:
        """Compute SSIM score."""
        total_ssim = 0.0
        
        for gen_img, ref_img in zip(generated_images, reference_images):
            gen_tensor = torch.from_numpy(np.array(gen_img)).permute(2, 0, 1).float() / 255.0
            ref_tensor = torch.from_numpy(np.array(ref_img)).permute(2, 0, 1).float() / 255.0
            
            gen_tensor = gen_tensor.unsqueeze(0)
            ref_tensor = ref_tensor.unsqueeze(0)
            
            ssim_score = self.ssim_metric(gen_tensor, ref_tensor)
            total_ssim += ssim_score.item()
        
        return total_ssim / len(generated_images)


async def main():
    """Example usage of advanced data loading and evaluation."""
    # Configuration
    data_config = DataConfig(
        data_dir="data",
        batch_size=4,
        num_workers=4,
        use_cache=True,
        use_augmentation=True
    )
    
    training_config = TrainingConfig(
        learning_rate=1e-4,
        lr_scheduler_type="cosine_warmup",
        early_stopping_patience=10,
        use_plateau_detection=True,
        use_overfitting_detection=True
    )
    
    eval_config = EvaluationConfig(
        task_type="image_generation",
        compute_fid=True,
        compute_lpips=True,
        compute_clip_score=True,
        compute_psnr=True,
        compute_ssim=True
    )
    
    # Initialize components
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    data_loader = AdvancedDataLoader(data_config, tokenizer)
    
    # Create datasets and data loaders
    datasets = data_loader.create_datasets("data")
    loaders = data_loader.create_data_loaders(datasets)
    
    # Create cross-validation splits
    cv_splits = data_loader.create_cross_validation_splits("data")
    
    # Initialize evaluator
    evaluator = TaskSpecificEvaluator(eval_config)
    
    logger.info("Advanced data loading and evaluation system initialized successfully!")
    logger.info(f"Created {len(loaders)} data loaders")
    logger.info(f"Created {len(cv_splits)} cross-validation splits")


match __name__:
    case "__main__":
    asyncio.run(main()) 