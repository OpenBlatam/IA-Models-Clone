from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass
import math
import random
import time
from pathlib import Path
import pickle
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import (
from torch.utils.data.dataloader import default_collate
import numpy as np
import pandas as pd
from PIL import Image
import h5py
import lmdb
import msgpack
from tqdm import tqdm
from transformers import (
from ..models.sequence import EmailSequence, SequenceStep
from ..models.subscriber import Subscriber
from ..models.template import EmailTemplate
from typing import Any, List, Dict, Optional
"""
Efficient Data Loading for Email Sequence System

Advanced data loading implementation using PyTorch's DataLoader with
memory optimization, caching, prefetching, and multi-processing support.
"""


    DataLoader, 
    Dataset, 
    Sampler, 
    BatchSampler,
    RandomSampler,
    SequentialSampler,
    WeightedRandomSampler,
    SubsetRandomSampler,
    DistributedSampler
)

    CLIPTokenizer,
    AutoTokenizer,
    CLIPProcessor
)


logger = logging.getLogger(__name__)


@dataclass
class DataLoaderConfig:
    """Configuration for efficient data loading"""
    # Basic parameters
    batch_size: int = 32
    shuffle: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = False
    
    # Advanced parameters
    prefetch_factor: int = 2
    persistent_workers: bool = True
    timeout: int = 0
    
    # Memory optimization
    memory_efficient: bool = True
    cache_size: int = 1000
    use_memory_mapping: bool = True
    
    # Caching
    enable_caching: bool = True
    cache_dir: Optional[str] = "./cache"
    cache_format: str = "h5"  # h5, lmdb, pickle
    
    # Multi-processing
    multiprocessing_context: str = "spawn"  # spawn, fork, forkserver
    
    # Device configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Data augmentation
    enable_augmentation: bool = True
    augmentation_probability: float = 0.5


class EfficientEmailDataset(Dataset):
    """Efficient dataset for email sequences with caching and optimization"""
    
    def __init__(
        self,
        sequences: List[EmailSequence],
        subscribers: List[Subscriber],
        templates: List[EmailTemplate],
        tokenizer,
        config: DataLoaderConfig,
        max_length: int = 77
    ):
        
    """__init__ function."""
self.sequences = sequences
        self.subscribers = subscribers
        self.templates = templates
        self.tokenizer = tokenizer
        self.config = config
        self.max_length = max_length
        
        # Initialize caching
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        if config.enable_caching:
            self._setup_caching()
        
        # Preprocess and cache data
        self.processed_data = self._preprocess_data()
        
        logger.info(f"Efficient Email Dataset initialized with {len(self.processed_data)} samples")
    
    def _setup_caching(self) -> Any:
        """Setup caching system"""
        if self.config.cache_dir:
            cache_path = Path(self.config.cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)
            
            if self.config.cache_format == "h5":
                self.cache_file = cache_path / "email_data.h5"
            elif self.config.cache_format == "lmdb":
                self.cache_file = cache_path / "email_data.lmdb"
            else:
                self.cache_file = cache_path / "email_data.pkl"
    
    def _preprocess_data(self) -> List[Dict[str, Any]]:
        """Preprocess data with caching support"""
        
        # Check if cached data exists
        if self.config.enable_caching and self._load_cached_data():
            logger.info("Loaded data from cache")
            return self._load_from_cache()
        
        # Process data
        processed_data = []
        
        for sequence in tqdm(self.sequences, desc="Processing sequences"):
            for step in sequence.steps:
                # Create multiple samples for each step for data augmentation
                num_samples = 3 if self.config.enable_augmentation else 1
                
                for _ in range(num_samples):
                    sample = self._create_sample(sequence, step)
                    processed_data.append(sample)
        
        # Cache processed data
        if self.config.enable_caching:
            self._cache_data(processed_data)
        
        return processed_data
    
    def _create_sample(self, sequence: EmailSequence, step: SequenceStep) -> Dict[str, Any]:
        """Create a single sample with augmentation"""
        
        # Get random subscriber and template
        subscriber = random.choice(self.subscribers)
        template = random.choice(self.templates)
        
        # Create context
        context = self._create_context(sequence, step, subscriber, template)
        
        # Apply augmentation if enabled
        if self.config.enable_augmentation and random.random() < self.config.augmentation_probability:
            context = self._augment_context(context, subscriber, template)
        
        # Tokenize
        tokens = self.tokenizer(
            context,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": tokens["input_ids"].squeeze(),
            "attention_mask": tokens["attention_mask"].squeeze(),
            "sequence_id": sequence.id,
            "step_order": step.order,
            "subscriber_id": subscriber.id,
            "template_id": template.id,
            "content_length": len(step.content or ""),
            "delay_hours": step.delay_hours or 0,
            "context": context
        }
    
    def _create_context(
        self,
        sequence: EmailSequence,
        step: SequenceStep,
        subscriber: Subscriber,
        template: EmailTemplate
    ) -> str:
        """Create context string"""
        
        context_parts = [
            f"Sequence: {sequence.name}",
            f"Step: {step.order}",
            f"Subscriber: {subscriber.first_name} {subscriber.last_name}",
            f"Company: {subscriber.company}",
            f"Interests: {', '.join(subscriber.interests)}",
            f"Template: {template.name}",
            f"Category: {template.category}",
            f"Content: {step.content or 'No content'}"
        ]
        
        return " | ".join(context_parts)
    
    def _augment_context(
        self,
        context: str,
        subscriber: Subscriber,
        template: EmailTemplate
    ) -> str:
        """Apply data augmentation to context"""
        
        augmentations = [
            self._add_urgency,
            self._add_personalization,
            self._add_benefits,
            self._add_social_proof,
            self._add_call_to_action
        ]
        
        # Apply random augmentation
        augmentation = random.choice(augmentations)
        return augmentation(context, subscriber, template)
    
    def _add_urgency(self, context: str, subscriber: Subscriber, template: EmailTemplate) -> str:
        """Add urgency to context"""
        urgency_phrases = [
            "Limited time offer",
            "Act now",
            "Don't miss out",
            "Exclusive opportunity",
            "Time-sensitive"
        ]
        return f"{random.choice(urgency_phrases)} | {context}"
    
    def _add_personalization(self, context: str, subscriber: Subscriber, template: EmailTemplate) -> str:
        """Add personalization to context"""
        personal_phrases = [
            f"Specially for {subscriber.first_name}",
            f"Your exclusive access",
            f"Personal invitation",
            f"Customized for you"
        ]
        return f"{random.choice(personal_phrases)} | {context}"
    
    def _add_benefits(self, context: str, subscriber: Subscriber, template: EmailTemplate) -> str:
        """Add benefits to context"""
        benefit_phrases = [
            "Save time and money",
            "Boost your productivity",
            "Achieve better results",
            "Gain competitive advantage"
        ]
        return f"{context} | {random.choice(benefit_phrases)}"
    
    def _add_social_proof(self, context: str, subscriber: Subscriber, template: EmailTemplate) -> str:
        """Add social proof to context"""
        social_phrases = [
            "Join thousands of satisfied customers",
            "Trusted by industry leaders",
            "Proven results",
            "Customer favorite"
        ]
        return f"{context} | {random.choice(social_phrases)}"
    
    def _add_call_to_action(self, context: str, subscriber: Subscriber, template: EmailTemplate) -> str:
        """Add call to action to context"""
        cta_phrases = [
            "Get started today",
            "Learn more now",
            "Sign up today",
            "Download now"
        ]
        return f"{context} | {random.choice(cta_phrases)}"
    
    def _load_cached_data(self) -> bool:
        """Load cached data if available"""
        try:
            if self.config.cache_format == "h5":
                return self.cache_file.exists()
            elif self.config.cache_format == "lmdb":
                return self.cache_file.exists()
            else:
                return self.cache_file.exists()
        except Exception:
            return False
    
    def _load_from_cache(self) -> List[Dict[str, Any]]:
        """Load data from cache"""
        try:
            if self.config.cache_format == "h5":
                return self._load_h5_cache()
            elif self.config.cache_format == "lmdb":
                return self._load_lmdb_cache()
            else:
                return self._load_pickle_cache()
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return []
    
    def _load_h5_cache(self) -> List[Dict[str, Any]]:
        """Load data from HDF5 cache"""
        data = []
        with h5py.File(self.cache_file, 'r') as f:
            for i in range(len(f['input_ids'])):
                sample = {
                    "input_ids": torch.tensor(f['input_ids'][i]),
                    "attention_mask": torch.tensor(f['attention_mask'][i]),
                    "sequence_id": f['sequence_id'][i],
                    "step_order": f['step_order'][i],
                    "subscriber_id": f['subscriber_id'][i],
                    "template_id": f['template_id'][i],
                    "content_length": f['content_length'][i],
                    "delay_hours": f['delay_hours'][i]
                }
                data.append(sample)
        return data
    
    def _load_lmdb_cache(self) -> List[Dict[str, Any]]:
        """Load data from LMDB cache"""
        data = []
        env = lmdb.open(str(self.cache_file), readonly=True)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        with env.begin() as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                sample = msgpack.unpackb(value)
                data.append(sample)
        env.close()
        return data
    
    def _load_pickle_cache(self) -> List[Dict[str, Any]]:
        """Load data from pickle cache"""
        with open(self.cache_file, 'rb') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            return pickle.load(f)
    
    def _cache_data(self, data: List[Dict[str, Any]]):
        """Cache processed data"""
        try:
            if self.config.cache_format == "h5":
                self._save_h5_cache(data)
            elif self.config.cache_format == "lmdb":
                self._save_lmdb_cache(data)
            else:
                self._save_pickle_cache(data)
            logger.info(f"Data cached to {self.cache_file}")
        except Exception as e:
            logger.warning(f"Failed to cache data: {e}")
    
    def _save_h5_cache(self, data: List[Dict[str, Any]]):
        """Save data to HDF5 cache"""
        with h5py.File(self.cache_file, 'w') as f:
            f.create_dataset('input_ids', data=np.array([d['input_ids'].numpy() for d in data]))
            f.create_dataset('attention_mask', data=np.array([d['attention_mask'].numpy() for d in data]))
            f.create_dataset('sequence_id', data=np.array([d['sequence_id'] for d in data]))
            f.create_dataset('step_order', data=np.array([d['step_order'] for d in data]))
            f.create_dataset('subscriber_id', data=np.array([d['subscriber_id'] for d in data]))
            f.create_dataset('template_id', data=np.array([d['template_id'] for d in data]))
            f.create_dataset('content_length', data=np.array([d['content_length'] for d in data]))
            f.create_dataset('delay_hours', data=np.array([d['delay_hours'] for d in data]))
    
    def _save_lmdb_cache(self, data: List[Dict[str, Any]]):
        """Save data to LMDB cache"""
        env = lmdb.open(str(self.cache_file), map_size=int(1e12))
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        with env.begin(write=True) as txn:
            for i, sample in enumerate(data):
                key = f"sample_{i}".encode()
                value = msgpack.packb(sample)
                txn.put(key, value)
        env.close()
    
    def _save_pickle_cache(self, data: List[Dict[str, Any]]):
        """Save data to pickle cache"""
        with open(self.cache_file, 'wb') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            pickle.dump(data, f)
    
    def __len__(self) -> Any:
        return len(self.processed_data)
    
    def __getitem__(self, idx) -> Optional[Dict[str, Any]]:
        # Check cache first
        if idx in self.cache:
            self.cache_hits += 1
            return self.cache[idx]
        
        self.cache_misses += 1
        sample = self.processed_data[idx]
        
        # Cache sample if cache is not full
        if len(self.cache) < self.config.cache_size:
            self.cache[idx] = sample
        
        return sample


class SmartBatchSampler(Sampler):
    """Smart batch sampler for efficient data loading"""
    
    def __init__(
        self,
        dataset: EfficientEmailDataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False
    ):
        
    """__init__ function."""
self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        # Group samples by length for efficient batching
        self.length_groups = self._group_by_length()
    
    def _group_by_length(self) -> Dict[int, List[int]]:
        """Group samples by sequence length"""
        length_groups = {}
        
        for i, sample in enumerate(self.dataset.processed_data):
            length = sample['input_ids'].shape[0]
            if length not in length_groups:
                length_groups[length] = []
            length_groups[length].append(i)
        
        return length_groups
    
    def __iter__(self) -> Any:
        if self.shuffle:
            # Shuffle within each length group
            for length in self.length_groups:
                random.shuffle(self.length_groups[length])
        
        # Create batches from each length group
        batches = []
        for length, indices in self.length_groups.items():
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                if len(batch_indices) == self.batch_size or not self.drop_last:
                    batches.extend(batch_indices)
        
        if self.shuffle:
            random.shuffle(batches)
        
        return iter(batches)
    
    def __len__(self) -> Any:
        total_samples = sum(len(indices) for indices in self.length_groups.values())
        if self.drop_last:
            return total_samples // self.batch_size
        else:
            return (total_samples + self.batch_size - 1) // self.batch_size


class CustomCollateFn:
    """Custom collate function for efficient batching"""
    
    def __init__(self, config: DataLoaderConfig):
        
    """__init__ function."""
self.config = config
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Custom collate function"""
        
        # Separate different types of data
        input_ids = [item['input_ids'] for item in batch]
        attention_masks = [item['attention_mask'] for item in batch]
        
        # Pad sequences to max length in batch
        max_length = max(len(ids) for ids in input_ids)
        
        # Pad input_ids
        padded_input_ids = []
        for ids in input_ids:
            padding_length = max_length - len(ids)
            padded_ids = torch.cat([ids, torch.zeros(padding_length, dtype=ids.dtype)])
            padded_input_ids.append(padded_ids)
        
        # Pad attention masks
        padded_attention_masks = []
        for mask in attention_masks:
            padding_length = max_length - len(mask)
            padded_mask = torch.cat([mask, torch.zeros(padding_length, dtype=mask.dtype)])
            padded_attention_masks.append(padded_mask)
        
        # Stack tensors
        batch_input_ids = torch.stack(padded_input_ids)
        batch_attention_masks = torch.stack(padded_attention_masks)
        
        # Create metadata
        metadata = {
            'sequence_ids': [item['sequence_id'] for item in batch],
            'step_orders': [item['step_order'] for item in batch],
            'subscriber_ids': [item['subscriber_id'] for item in batch],
            'template_ids': [item['template_id'] for item in batch],
            'content_lengths': [item['content_length'] for item in batch],
            'delay_hours': [item['delay_hours'] for item in batch]
        }
        
        return {
            'input_ids': batch_input_ids,
            'attention_mask': batch_attention_masks,
            'metadata': metadata
        }


class DataLoaderManager:
    """Manager for efficient data loading"""
    
    def __init__(self, config: DataLoaderConfig):
        
    """__init__ function."""
self.config = config
        self.device = torch.device(config.device)
        
        # Performance tracking
        self.loading_stats = defaultdict(int)
        
        logger.info("Data Loader Manager initialized")
    
    def create_dataloader(
        self,
        dataset: EfficientEmailDataset,
        sampler_type: str = "smart",
        **kwargs
    ) -> DataLoader:
        """Create optimized DataLoader"""
        
        # Setup sampler
        if sampler_type == "smart":
            sampler = SmartBatchSampler(
                dataset,
                self.config.batch_size,
                shuffle=self.config.shuffle,
                drop_last=self.config.drop_last
            )
            batch_sampler = None
        else:
            sampler = self._create_sampler(dataset, sampler_type)
            batch_sampler = None
        
        # Setup collate function
        collate_fn = CustomCollateFn(self.config)
        
        # Create DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size if batch_sampler is None else 1,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=self.config.drop_last,
            collate_fn=collate_fn,
            prefetch_factor=self.config.prefetch_factor,
            persistent_workers=self.config.persistent_workers,
            timeout=self.config.timeout,
            multiprocessing_context=self.config.multiprocessing_context
        )
        
        self.loading_stats["dataloaders_created"] += 1
        
        return dataloader
    
    def _create_sampler(self, dataset: EfficientEmailDataset, sampler_type: str) -> Sampler:
        """Create sampler based on type"""
        
        if sampler_type == "random":
            return RandomSampler(dataset)
        elif sampler_type == "sequential":
            return SequentialSampler(dataset)
        elif sampler_type == "weighted":
            weights = self._calculate_weights(dataset)
            return WeightedRandomSampler(weights, len(weights))
        elif sampler_type == "distributed":
            return DistributedSampler(dataset)
        else:
            raise ValueError(f"Unknown sampler type: {sampler_type}")
    
    def _calculate_weights(self, dataset: EfficientEmailDataset) -> List[float]:
        """Calculate sample weights for weighted sampling"""
        
        # Calculate weights based on content length (longer content gets higher weight)
        weights = []
        for sample in dataset.processed_data:
            weight = 1.0 + (sample['content_length'] / 1000.0)  # Normalize
            weights.append(weight)
        
        return weights
    
    async def benchmark_dataloader(
        self,
        dataloader: DataLoader,
        num_batches: int = 10
    ) -> Dict[str, float]:
        """Benchmark dataloader performance"""
        
        start_time = time.time()
        batch_times = []
        
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            
            batch_start = time.time()
            
            # Move to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
        
        total_time = time.time() - start_time
        
        return {
            "total_time": total_time,
            "avg_batch_time": np.mean(batch_times),
            "std_batch_time": np.std(batch_times),
            "batches_per_second": num_batches / total_time,
            "samples_per_second": (num_batches * self.config.batch_size) / total_time
        }
    
    async def get_loading_report(self) -> Dict[str, Any]:
        """Generate comprehensive loading report"""
        
        return {
            "loading_stats": dict(self.loading_stats),
            "config": {
                "batch_size": self.config.batch_size,
                "num_workers": self.config.num_workers,
                "pin_memory": self.config.pin_memory,
                "prefetch_factor": self.config.prefetch_factor,
                "persistent_workers": self.config.persistent_workers
            },
            "performance_metrics": {
                "memory_usage": self._get_memory_usage(),
                "cache_efficiency": self._calculate_cache_efficiency()
            }
        }
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage information"""
        if torch.cuda.is_available():
            return {
                "gpu_memory_allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
                "gpu_memory_reserved": torch.cuda.memory_reserved() / 1024**3,  # GB
                "gpu_memory_free": (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**3  # GB
            }
        else:
            return {"cpu_memory": "N/A"}
    
    def _calculate_cache_efficiency(self) -> Dict[str, float]:
        """Calculate cache efficiency metrics"""
        # This would be calculated from dataset cache statistics
        return {
            "cache_hit_rate": 0.0,  # Placeholder
            "cache_miss_rate": 0.0,  # Placeholder
            "cache_size": self.config.cache_size
        } 