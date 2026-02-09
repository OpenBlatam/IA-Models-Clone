"""
Optimized Data Loading System for Video-OpusClip

High-performance data loading using PyTorch's DataLoader with video-specific optimizations,
memory management, and parallel processing capabilities.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image
import os
import json
import pickle
import gzip
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import asyncio
import aiofiles
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import structlog
from dataclasses import dataclass
import time

logger = structlog.get_logger()

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class VideoSample:
    """Video sample data structure."""
    video_path: str
    frames: torch.Tensor
    audio: Optional[torch.Tensor] = None
    metadata: Optional[Dict[str, Any]] = None
    labels: Optional[Dict[str, Any]] = None
    duration: Optional[float] = None
    fps: Optional[float] = None
    resolution: Optional[Tuple[int, int]] = None

@dataclass
class BatchData:
    """Batch data structure for efficient processing."""
    video_frames: torch.Tensor
    audio_features: Optional[torch.Tensor] = None
    metadata: Optional[Dict[str, Any]] = None
    labels: Optional[Dict[str, Any]] = None
    video_paths: List[str] = None
    frame_indices: Optional[torch.Tensor] = None

# =============================================================================
# OPTIMIZED VIDEO DATASET
# =============================================================================

class OptimizedVideoDataset(Dataset):
    """High-performance video dataset with caching and memory optimization."""
    
    def __init__(
        self,
        data_root: str,
        video_extensions: List[str] = ['.mp4', '.avi', '.mov', '.mkv'],
        max_frames: int = 300,
        target_size: Tuple[int, int] = (224, 224),
        cache_dir: Optional[str] = None,
        enable_caching: bool = True,
        preload_frames: bool = False,
        num_workers: int = 4,
        device: str = "cpu"
    ):
        self.data_root = Path(data_root)
        self.video_extensions = video_extensions
        self.max_frames = max_frames
        self.target_size = target_size
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.enable_caching = enable_caching
        self.preload_frames = preload_frames
        self.num_workers = num_workers
        self.device = device
        
        # Video file discovery
        self.video_files = self._discover_videos()
        logger.info(f"Found {len(self.video_files)} video files")
        
        # Setup caching
        if self.enable_caching and self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._setup_cache()
        
        # Preload frames if requested
        if self.preload_frames:
            self._preload_frames()
        
        # Setup transforms
        self.transforms = self._setup_transforms()
        
        # Memory pool for efficient tensor reuse
        self.memory_pool = {}
    
    def _discover_videos(self) -> List[Path]:
        """Discover video files in data root."""
        video_files = []
        for ext in self.video_extensions:
            video_files.extend(self.data_root.rglob(f"*{ext}"))
        return sorted(video_files)
    
    def _setup_cache(self):
        """Setup caching system."""
        self.cache_metadata_file = self.cache_dir / "cache_metadata.json"
        self.cache_metadata = self._load_cache_metadata()
    
    def _load_cache_metadata(self) -> Dict[str, Any]:
        """Load cache metadata."""
        if self.cache_metadata_file.exists():
            try:
                with open(self.cache_metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache metadata: {e}")
        return {}
    
    def _save_cache_metadata(self):
        """Save cache metadata."""
        try:
            with open(self.cache_metadata_file, 'w') as f:
                json.dump(self.cache_metadata, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache metadata: {e}")
    
    def _get_cache_key(self, video_path: Path) -> str:
        """Generate cache key for video."""
        import hashlib
        key_data = f"{video_path}_{self.max_frames}_{self.target_size}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path."""
        return self.cache_dir / f"{cache_key}.pkl.gz"
    
    def _load_from_cache(self, video_path: Path) -> Optional[VideoSample]:
        """Load video from cache."""
        if not self.enable_caching or not self.cache_dir:
            return None
        
        cache_key = self._get_cache_key(video_path)
        cache_path = self._get_cache_path(cache_key)
        
        if cache_path.exists():
            try:
                with gzip.open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                
                # Verify cache validity
                if cached_data.get('video_path') == str(video_path):
                    logger.debug(f"Loaded from cache: {video_path}")
                    return VideoSample(**cached_data['data'])
            except Exception as e:
                logger.warning(f"Failed to load from cache: {e}")
        
        return None
    
    def _save_to_cache(self, video_path: Path, video_sample: VideoSample):
        """Save video to cache."""
        if not self.enable_caching or not self.cache_dir:
            return
        
        cache_key = self._get_cache_key(video_path)
        cache_path = self._get_cache_path(cache_key)
        
        try:
            cached_data = {
                'video_path': str(video_path),
                'data': {
                    'video_path': video_sample.video_path,
                    'frames': video_sample.frames,
                    'audio': video_sample.audio,
                    'metadata': video_sample.metadata,
                    'labels': video_sample.labels,
                    'duration': video_sample.duration,
                    'fps': video_sample.fps,
                    'resolution': video_sample.resolution
                }
            }
            
            with gzip.open(cache_path, 'wb') as f:
                pickle.dump(cached_data, f)
            
            # Update metadata
            self.cache_metadata[cache_key] = {
                'video_path': str(video_path),
                'cache_path': str(cache_path),
                'timestamp': time.time()
            }
            self._save_cache_metadata()
            
            logger.debug(f"Saved to cache: {video_path}")
        except Exception as e:
            logger.warning(f"Failed to save to cache: {e}")
    
    def _load_video_frames(self, video_path: Path) -> VideoSample:
        """Load video frames efficiently."""
        # Try cache first
        cached_sample = self._load_from_cache(video_path)
        if cached_sample:
            return cached_sample
        
        # Load from disk
        try:
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps if fps > 0 else 0
            
            # Calculate frame indices
            if total_frames <= self.max_frames:
                frame_indices = list(range(total_frames))
            else:
                # Uniform sampling
                step = total_frames / self.max_frames
                frame_indices = [int(i * step) for i in range(self.max_frames)]
            
            # Load frames
            frames = []
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
                else:
                    # Use last valid frame
                    if frames:
                        frames.append(frames[-1])
                    else:
                        # Create black frame
                        black_frame = np.zeros((height, width, 3), dtype=np.uint8)
                        frames.append(black_frame)
            
            cap.release()
            
            # Convert to tensor
            frames_array = np.array(frames)
            frames_tensor = torch.from_numpy(frames_array).float()
            
            # Apply transforms
            if self.transforms:
                frames_tensor = self.transforms(frames_tensor)
            
            # Create video sample
            video_sample = VideoSample(
                video_path=str(video_path),
                frames=frames_tensor,
                metadata={
                    'fps': fps,
                    'total_frames': total_frames,
                    'original_resolution': (width, height),
                    'loaded_frames': len(frames)
                },
                duration=duration,
                fps=fps,
                resolution=(width, height)
            )
            
            # Save to cache
            self._save_to_cache(video_path, video_sample)
            
            return video_sample
            
        except Exception as e:
            logger.error(f"Failed to load video {video_path}: {e}")
            # Return empty sample
            empty_frames = torch.zeros((self.max_frames, 3, *self.target_size))
            return VideoSample(
                video_path=str(video_path),
                frames=empty_frames,
                metadata={'error': str(e)}
            )
    
    def _setup_transforms(self):
        """Setup video transforms."""
        return transforms.Compose([
            transforms.Lambda(lambda x: x / 255.0),  # Normalize to [0, 1]
            transforms.Lambda(lambda x: x.permute(0, 3, 1, 2)),  # NHWC to NCHW
            transforms.Lambda(lambda x: torch.nn.functional.interpolate(
                x, size=self.target_size, mode='bilinear', align_corners=False
            )),
        ])
    
    def _preload_frames(self):
        """Preload all frames into memory."""
        logger.info("Preloading frames into memory...")
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for video_path in self.video_files:
                future = executor.submit(self._load_video_frames, video_path)
                futures.append((video_path, future))
            
            self.preloaded_samples = {}
            for video_path, future in futures:
                try:
                    sample = future.result()
                    self.preloaded_samples[str(video_path)] = sample
                except Exception as e:
                    logger.error(f"Failed to preload {video_path}: {e}")
        
        logger.info(f"Preloaded {len(self.preloaded_samples)} videos")
    
    def __len__(self) -> int:
        return len(self.video_files)
    
    def __getitem__(self, idx: int) -> VideoSample:
        video_path = self.video_files[idx]
        
        # Use preloaded data if available
        if self.preload_frames and str(video_path) in self.preloaded_samples:
            return self.preloaded_samples[str(video_path)]
        
        # Load from disk or cache
        return self._load_video_frames(video_path)

# =============================================================================
# OPTIMIZED DATA LOADER
# =============================================================================

class OptimizedDataLoader:
    """High-performance data loader with memory optimization and parallel processing."""
    
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = False,
        persistent_workers: bool = True,
        prefetch_factor: int = 2,
        device: str = "cuda",
        enable_memory_pool: bool = True,
        max_memory_gb: float = 8.0
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        self.enable_memory_pool = enable_memory_pool
        self.max_memory_gb = max_memory_gb
        
        # Memory pool for tensor reuse
        self.memory_pool = {}
        
        # Setup data loader
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            collate_fn=self._collate_fn
        )
        
        logger.info(f"Initialized optimized data loader: {len(dataset)} samples, "
                   f"batch_size={batch_size}, workers={num_workers}")
    
    def _collate_fn(self, batch: List[VideoSample]) -> BatchData:
        """Custom collate function for efficient batching."""
        # Extract components
        video_frames = [sample.frames for sample in batch]
        audio_features = [sample.audio for sample in batch if sample.audio is not None]
        metadata = [sample.metadata for sample in batch]
        labels = [sample.labels for sample in batch if sample.labels is not None]
        video_paths = [sample.video_path for sample in batch]
        
        # Stack video frames
        video_frames_tensor = torch.stack(video_frames, dim=0)
        
        # Stack audio features if available
        audio_tensor = None
        if audio_features:
            audio_tensor = torch.stack(audio_features, dim=0)
        
        # Create batch data
        batch_data = BatchData(
            video_frames=video_frames_tensor,
            audio_features=audio_tensor,
            metadata=metadata,
            labels=labels,
            video_paths=video_paths
        )
        
        return batch_data
    
    def __iter__(self):
        """Iterate over batches with memory optimization."""
        for batch in self.dataloader:
            # Move to device
            if self.device != "cpu":
                batch.video_frames = batch.video_frames.to(self.device, non_blocking=True)
                if batch.audio_features is not None:
                    batch.audio_features = batch.audio_features.to(self.device, non_blocking=True)
            
            # Memory optimization
            if self.enable_memory_pool:
                self._optimize_memory_usage()
            
            yield batch
    
    def _optimize_memory_usage(self):
        """Optimize memory usage during iteration."""
        if torch.cuda.is_available():
            # Clear cache if memory usage is high
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            if memory_allocated > self.max_memory_gb:
                torch.cuda.empty_cache()
                logger.debug(f"Cleared GPU cache, memory usage: {memory_allocated:.2f}GB")

# =============================================================================
# DISTRIBUTED DATA LOADING
# =============================================================================

class DistributedVideoDataset(Dataset):
    """Distributed video dataset for multi-GPU training."""
    
    def __init__(
        self,
        data_root: str,
        world_size: int,
        rank: int,
        **kwargs
    ):
        super().__init__()
        self.world_size = world_size
        self.rank = rank
        
        # Create base dataset
        self.base_dataset = OptimizedVideoDataset(data_root, **kwargs)
        
        # Split data across ranks
        self.indices = self._split_indices()
        
        logger.info(f"Rank {rank}: {len(self.indices)} samples")
    
    def _split_indices(self) -> List[int]:
        """Split dataset indices across ranks."""
        total_samples = len(self.base_dataset)
        indices = list(range(total_samples))
        
        # Shuffle for better distribution
        np.random.shuffle(indices)
        
        # Split across ranks
        samples_per_rank = total_samples // self.world_size
        start_idx = self.rank * samples_per_rank
        end_idx = start_idx + samples_per_rank if self.rank < self.world_size - 1 else total_samples
        
        return indices[start_idx:end_idx]
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> VideoSample:
        actual_idx = self.indices[idx]
        return self.base_dataset[actual_idx]

def create_distributed_dataloader(
    data_root: str,
    world_size: int,
    rank: int,
    batch_size: int = 32,
    **kwargs
) -> DataLoader:
    """Create distributed data loader for multi-GPU training."""
    
    # Create distributed dataset
    dataset = DistributedVideoDataset(data_root, world_size, rank, **kwargs)
    
    # Create distributed sampler
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    # Create data loader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=kwargs.get('num_workers', 4),
        pin_memory=kwargs.get('pin_memory', True),
        persistent_workers=kwargs.get('persistent_workers', True),
        prefetch_factor=kwargs.get('prefetch_factor', 2),
        collate_fn=lambda x: OptimizedDataLoader._collate_fn(None, x)
    )
    
    return dataloader

# =============================================================================
# MEMORY-EFFICIENT DATA LOADING
# =============================================================================

class MemoryEfficientVideoDataset(Dataset):
    """Memory-efficient video dataset with streaming and lazy loading."""
    
    def __init__(
        self,
        data_root: str,
        max_memory_gb: float = 4.0,
        **kwargs
    ):
        super().__init__()
        self.max_memory_gb = max_memory_gb
        self.base_dataset = OptimizedVideoDataset(data_root, **kwargs)
        
        # Memory tracking
        self.current_memory_usage = 0.0
        self.loaded_samples = {}
        
        logger.info(f"Memory-efficient dataset initialized, max memory: {max_memory_gb}GB")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in GB."""
        import psutil
        return psutil.virtual_memory().used / 1024**3
    
    def _load_sample_if_needed(self, idx: int) -> VideoSample:
        """Load sample only if not in memory and memory allows."""
        if idx in self.loaded_samples:
            return self.loaded_samples[idx]
        
        # Check memory usage
        current_memory = self._get_memory_usage()
        if current_memory > self.max_memory_gb:
            # Remove oldest samples
            self._evict_oldest_samples()
        
        # Load sample
        sample = self.base_dataset[idx]
        self.loaded_samples[idx] = sample
        
        # Update memory usage
        sample_memory = sample.frames.element_size() * sample.frames.nelement() / 1024**3
        self.current_memory_usage += sample_memory
        
        return sample
    
    def _evict_oldest_samples(self):
        """Evict oldest samples from memory."""
        if len(self.loaded_samples) > 10:  # Keep at least 10 samples
            # Remove oldest 20% of samples
            num_to_remove = max(1, len(self.loaded_samples) // 5)
            oldest_keys = list(self.loaded_samples.keys())[:num_to_remove]
            
            for key in oldest_keys:
                sample = self.loaded_samples.pop(key)
                sample_memory = sample.frames.element_size() * sample.frames.nelement() / 1024**3
                self.current_memory_usage -= sample_memory
            
            # Force garbage collection
            import gc
            gc.collect()
    
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> VideoSample:
        return self._load_sample_if_needed(idx)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_optimized_dataloader(
    data_root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    device: str = "cuda",
    enable_caching: bool = True,
    preload_frames: bool = False,
    **kwargs
) -> OptimizedDataLoader:
    """Create optimized data loader with default settings."""
    
    # Create dataset
    dataset = OptimizedVideoDataset(
        data_root=data_root,
        enable_caching=enable_caching,
        preload_frames=preload_frames,
        num_workers=num_workers,
        device=device,
        **kwargs
    )
    
    # Create data loader
    dataloader = OptimizedDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        **kwargs
    )
    
    return dataloader

def get_optimal_batch_size(
    dataset: Dataset,
    max_memory_gb: float = 8.0,
    device: str = "cuda"
) -> int:
    """Calculate optimal batch size based on available memory."""
    
    if device == "cuda" and torch.cuda.is_available():
        # Get GPU memory
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        available_memory = min(total_memory * 0.8, max_memory_gb)
        
        # Estimate memory per sample
        sample = dataset[0]
        sample_memory = sample.frames.element_size() * sample.frames.nelement() / 1024**3
        
        optimal_batch_size = int(available_memory / sample_memory)
        return max(1, min(optimal_batch_size, 128))  # Between 1 and 128
    
    return 32  # Default for CPU

def benchmark_dataloader(
    dataloader: DataLoader,
    num_batches: int = 10
) -> Dict[str, float]:
    """Benchmark data loader performance."""
    
    logger.info(f"Benchmarking data loader for {num_batches} batches...")
    
    start_time = time.time()
    total_samples = 0
    batch_times = []
    
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        
        batch_start = time.time()
        total_samples += batch.video_frames.size(0)
        batch_times.append(time.time() - batch_start)
    
    total_time = time.time() - start_time
    
    # Calculate metrics
    avg_batch_time = np.mean(batch_times)
    throughput = total_samples / total_time
    
    metrics = {
        'total_time': total_time,
        'avg_batch_time': avg_batch_time,
        'throughput_samples_per_sec': throughput,
        'total_samples': total_samples,
        'num_batches': len(batch_times)
    }
    
    logger.info(f"Benchmark results: {throughput:.2f} samples/sec, "
               f"avg batch time: {avg_batch_time:.3f}s")
    
    return metrics

# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_video_dataloader_factory(
    data_root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    device: str = "cuda",
    **kwargs
):
    """Factory function for creating video data loaders."""
    
    def create_dataloader(**override_kwargs):
        """Create data loader with overridden parameters."""
        params = {
            'data_root': data_root,
            'batch_size': batch_size,
            'num_workers': num_workers,
            'device': device,
            **kwargs,
            **override_kwargs
        }
        
        return create_optimized_dataloader(**params)
    
    return create_dataloader

# Global factory instance
video_dataloader_factory = None

def get_video_dataloader_factory(data_root: str, **kwargs):
    """Get global video data loader factory."""
    global video_dataloader_factory
    if video_dataloader_factory is None:
        video_dataloader_factory = create_video_dataloader_factory(data_root, **kwargs)
    return video_dataloader_factory 