"""
Advanced Memory Management System for HeyGen AI Enterprise

This module provides cutting-edge memory management capabilities:
- Intelligent memory allocation and optimization
- Virtual memory management with smart paging
- Memory prefetching and intelligent caching
- Memory pool management and garbage collection
- Memory leak detection and prevention
- Cross-platform memory optimization
- Real-time memory monitoring and analytics
"""

import logging
import os
import time
import gc
import warnings
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
from contextlib import contextmanager
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import json
import pickle
from collections import defaultdict, deque
from datetime import datetime, timedelta
import statistics
import weakref

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

# Memory monitoring libraries
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    warnings.warn("psutil not available. Install for system memory monitoring.")

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    warnings.warn("pynvml not available. Install for GPU memory monitoring.")

logger = logging.getLogger(__name__)


@dataclass
class MemoryConfig:
    """Configuration for advanced memory management system."""

    # Memory allocation settings
    enable_intelligent_allocation: bool = True
    enable_virtual_memory: bool = True
    enable_memory_pooling: bool = True
    enable_prefetching: bool = True

    # Memory pool settings
    pool_initial_size_mb: int = 1024  # 1GB
    pool_max_size_mb: int = 8192      # 8GB
    pool_growth_factor: float = 1.5
    pool_cleanup_threshold: float = 0.8

    # Virtual memory settings
    virtual_memory_size_mb: int = 16384  # 16GB
    page_size_mb: int = 64              # 64MB pages
    swap_threshold: float = 0.9

    # Prefetching settings
    prefetch_enabled: bool = True
    prefetch_window_size: int = 10
    prefetch_confidence_threshold: float = 0.7

    # Monitoring settings
    monitoring_interval: float = 1.0  # 1 second
    memory_history_size: int = 1000
    enable_memory_profiling: bool = True

    # Optimization settings
    enable_automatic_cleanup: bool = True
    cleanup_interval: float = 30.0  # 30 seconds
    memory_compression: bool = True
    compression_ratio: float = 0.8


class MemoryBlock:
    """Represents a memory block with metadata."""

    def __init__(self, block_id: str, size_bytes: int, block_type: str = "general"):
        self.block_id = block_id
        self.size_bytes = size_bytes
        self.block_type = block_type
        self.creation_time = time.time()
        self.last_access_time = time.time()
        self.access_count = 0
        self.is_pinned = False
        self.is_compressed = False
        self.compression_ratio = 1.0
        self.metadata = {}

    def access(self):
        """Record memory block access."""
        self.last_access_time = time.time()
        self.access_count += 1

    def get_age(self) -> float:
        """Get age of memory block in seconds."""
        return time.time() - self.creation_time

    def get_access_frequency(self) -> float:
        """Get access frequency (accesses per second)."""
        age = self.get_age()
        return self.access_count / max(age, 1.0)

    def to_dict(self) -> Dict[str, Any]:
        """Convert memory block to dictionary."""
        return {
            "block_id": self.block_id,
            "size_bytes": self.size_bytes,
            "block_type": self.block_type,
            "creation_time": self.creation_time,
            "last_access_time": self.last_access_time,
            "access_count": self.access_count,
            "is_pinned": self.is_pinned,
            "is_compressed": self.is_compressed,
            "compression_ratio": self.compression_ratio,
            "metadata": self.metadata
        }


class MemoryPool:
    """Intelligent memory pool for efficient allocation."""

    def __init__(self, config: MemoryConfig):
        self.config = config
        self.pools = defaultdict(list)  # size -> list of blocks
        self.allocated_blocks = {}      # block_id -> MemoryBlock
        self.total_allocated = 0
        self.total_pooled = 0
        self.pool_stats = defaultdict(int)
        
        # Initialize pools
        self._initialize_pools()
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self.cleanup_thread.start()

    def _initialize_pools(self):
        """Initialize memory pools with different sizes."""
        try:
            # Create pools for common block sizes
            sizes = [64, 128, 256, 512, 1024, 2048, 4096, 8192]  # KB
            
            for size_kb in sizes:
                size_bytes = size_kb * 1024
                pool_size = max(1, self.config.pool_initial_size_mb * 1024 * 1024 // size_bytes)
                
                # Pre-allocate some blocks
                for _ in range(min(pool_size, 10)):  # Start with max 10 blocks
                    block = MemoryBlock(
                        block_id=f"pool_{size_kb}kb_{_}_{int(time.time())}",
                        size_bytes=size_bytes,
                        block_type="pooled"
                    )
                    self.pools[size_bytes].append(block)
                    self.total_pooled += size_bytes

            logger.info(f"Memory pools initialized with {self.total_pooled / (1024*1024):.1f}MB total")

        except Exception as e:
            logger.error(f"Memory pool initialization failed: {e}")

    def allocate(self, size_bytes: int, block_type: str = "general") -> MemoryBlock:
        """Allocate memory from pool or create new block."""
        try:
            # Try to find a suitable pool
            if size_bytes in self.pools and self.pools[size_bytes]:
                block = self.pools[size_bytes].pop()
                block.block_type = block_type
                block.creation_time = time.time()
                block.last_access_time = time.time()
                block.access_count = 0
                self.total_pooled -= size_bytes
                self.total_allocated += size_bytes
                self.allocated_blocks[block.block_id] = block
                return block

            # Create new block if no pool available
            block = MemoryBlock(
                block_id=f"{block_type}_{int(time.time())}_{size_bytes}",
                size_bytes=size_bytes,
                block_type=block_type
            )
            
            self.total_allocated += size_bytes
            self.allocated_blocks[block.block_id] = block
            self.pool_stats[block_type] += 1
            
            return block

        except Exception as e:
            logger.error(f"Memory allocation failed: {e}")
            raise

    def deallocate(self, block: MemoryBlock):
        """Return memory block to pool or free it."""
        try:
            if block.block_id in self.allocated_blocks:
                del self.allocated_blocks[block.block_id]
                self.total_allocated -= block.size_bytes

                # Try to return to pool if size matches
                if block.size_bytes in self.pools:
                    if len(self.pools[block.size_bytes]) < 20:  # Limit pool size
                        block.block_type = "pooled"
                        self.pools[block.size_bytes].append(block)
                        self.total_pooled += block.size_bytes
                        return

                # Free memory if not pooled
                del block

        except Exception as e:
            logger.error(f"Memory deallocation failed: {e}")

    def get_pool_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics."""
        try:
            stats = {
                "total_allocated_mb": self.total_allocated / (1024 * 1024),
                "total_pooled_mb": self.total_pooled / (1024 * 1024),
                "allocated_blocks_count": len(self.allocated_blocks),
                "pool_counts": {size: len(blocks) for size, blocks in self.pools.items()},
                "block_type_stats": dict(self.pool_stats),
                "utilization": self.total_allocated / max(self.total_pooled + self.total_allocated, 1)
            }
            return stats

        except Exception as e:
            logger.error(f"Failed to get pool stats: {e}")
            return {}

    def _cleanup_worker(self):
        """Background cleanup worker."""
        while True:
            try:
                time.sleep(self.config.cleanup_interval)
                self._cleanup_pools()
            except Exception as e:
                logger.error(f"Pool cleanup worker failed: {e}")

    def _cleanup_pools(self):
        """Clean up unused memory pools."""
        try:
            for size, blocks in self.pools.items():
                # Remove old unused blocks
                current_time = time.time()
                blocks[:] = [
                    block for block in blocks
                    if current_time - block.last_access_time < 300  # 5 minutes
                ]

            # Update total pooled
            self.total_pooled = sum(
                sum(block.size_bytes for block in blocks)
                for blocks in self.pools.values()
            )

        except Exception as e:
            logger.error(f"Pool cleanup failed: {e}")


class VirtualMemoryManager:
    """Virtual memory management with intelligent paging."""

    def __init__(self, config: MemoryConfig):
        self.config = config
        self.virtual_pages = {}  # page_id -> page_data
        self.page_table = {}     # virtual_address -> page_id
        self.swap_file = None
        self.swap_file_path = None
        self.total_virtual_memory = config.virtual_memory_size_mb * 1024 * 1024
        self.used_virtual_memory = 0
        self.page_size = config.page_size_mb * 1024 * 1024
        
        # Initialize swap file
        self._initialize_swap_file()

    def _initialize_swap_file(self):
        """Initialize swap file for virtual memory."""
        try:
            swap_dir = Path("memory_swap")
            swap_dir.mkdir(exist_ok=True)
            
            self.swap_file_path = swap_dir / f"swap_{int(time.time())}.dat"
            
            # Create swap file
            with open(self.swap_file_path, 'wb') as f:
                f.write(b'\x00' * self.total_virtual_memory)
            
            logger.info(f"Swap file initialized: {self.swap_file_path}")

        except Exception as e:
            logger.error(f"Swap file initialization failed: {e}")

    def allocate_virtual_memory(self, size_bytes: int) -> int:
        """Allocate virtual memory address space."""
        try:
            # Calculate number of pages needed
            pages_needed = (size_bytes + self.page_size - 1) // self.page_size
            
            # Find available virtual address
            virtual_address = self._find_free_virtual_address(pages_needed)
            
            # Create pages
            for i in range(pages_needed):
                page_id = f"page_{virtual_address}_{i}_{int(time.time())}"
                page_data = {
                    "virtual_address": virtual_address + i * self.page_size,
                    "size": self.page_size,
                    "is_loaded": False,
                    "is_dirty": False,
                    "last_access": time.time(),
                    "access_count": 0
                }
                
                self.virtual_pages[page_id] = page_data
                self.page_table[virtual_address + i * self.page_size] = page_id
            
            self.used_virtual_memory += pages_needed * self.page_size
            
            return virtual_address

        except Exception as e:
            logger.error(f"Virtual memory allocation failed: {e}")
            raise

    def _find_free_virtual_address(self, pages_needed: int) -> int:
        """Find free virtual address space."""
        try:
            # Simple linear search for free space
            address = 0
            while address < self.total_virtual_memory:
                if self._is_address_free(address, pages_needed):
                    return address
                address += self.page_size
            
            # If no free space, trigger swap
            self._trigger_swap()
            return 0  # Return first available address after swap

        except Exception as e:
            logger.error(f"Free address search failed: {e}")
            return 0

    def _is_address_free(self, address: int, pages_needed: int) -> bool:
        """Check if address range is free."""
        try:
            for i in range(pages_needed):
                if address + i * self.page_size in self.page_table:
                    return False
            return True

        except Exception as e:
            logger.error(f"Address free check failed: {e}")
            return False

    def _trigger_swap(self):
        """Trigger memory swapping to free space."""
        try:
            logger.info("Triggering memory swap...")
            
            # Find least recently used pages
            lru_pages = sorted(
                self.virtual_pages.items(),
                key=lambda x: x[1]["last_access"]
            )
            
            # Swap out oldest pages
            pages_to_swap = len(lru_pages) // 4  # Swap out 25%
            
            for i in range(pages_to_swap):
                page_id, page_data = lru_pages[i]
                if page_data["is_loaded"]:
                    self._swap_out_page(page_id, page_data)
            
            logger.info(f"Swapped out {pages_to_swap} pages")

        except Exception as e:
            logger.error(f"Memory swap failed: {e}")

    def _swap_out_page(self, page_id: str, page_data: Dict[str, Any]):
        """Swap out a page to disk."""
        try:
            # Mark page as swapped out
            page_data["is_loaded"] = False
            
            # Write to swap file (simulated)
            # In real implementation, this would write actual page data
            
            logger.debug(f"Page {page_id} swapped out")

        except Exception as e:
            logger.error(f"Page swap out failed: {e}")

    def get_virtual_memory_stats(self) -> Dict[str, Any]:
        """Get virtual memory statistics."""
        try:
            loaded_pages = sum(1 for page in self.virtual_pages.values() if page["is_loaded"])
            total_pages = len(self.virtual_pages)
            
            stats = {
                "total_virtual_memory_mb": self.total_virtual_memory / (1024 * 1024),
                "used_virtual_memory_mb": self.used_virtual_memory / (1024 * 1024),
                "total_pages": total_pages,
                "loaded_pages": loaded_pages,
                "swapped_pages": total_pages - loaded_pages,
                "utilization": self.used_virtual_memory / self.total_virtual_memory
            }
            return stats

        except Exception as e:
            logger.error(f"Failed to get virtual memory stats: {e}")
            return {}


class MemoryPrefetcher:
    """Intelligent memory prefetching system."""

    def __init__(self, config: MemoryConfig):
        self.config = config
        self.access_patterns = defaultdict(list)
        self.prefetch_predictions = {}
        self.prefetch_cache = {}
        self.prefetch_stats = {
            "hits": 0,
            "misses": 0,
            "predictions": 0
        }

    def record_access(self, block_id: str, access_type: str = "read"):
        """Record memory access pattern."""
        try:
            current_time = time.time()
            
            # Record access
            self.access_patterns[block_id].append({
                "time": current_time,
                "type": access_type
            })
            
            # Keep only recent accesses
            if len(self.access_patterns[block_id]) > self.config.prefetch_window_size:
                self.access_patterns[block_id] = self.access_patterns[block_id][-self.config.prefetch_window_size:]
            
            # Update prefetch predictions
            self._update_predictions(block_id)

        except Exception as e:
            logger.error(f"Access recording failed: {e}")

    def _update_predictions(self, block_id: str):
        """Update prefetch predictions based on access patterns."""
        try:
            if block_id not in self.access_patterns:
                return
            
            accesses = self.access_patterns[block_id]
            if len(accesses) < 3:
                return
            
            # Analyze access pattern
            intervals = []
            for i in range(1, len(accesses)):
                interval = accesses[i]["time"] - accesses[i-1]["time"]
                intervals.append(interval)
            
            if not intervals:
                return
            
            # Calculate prediction
            avg_interval = statistics.mean(intervals)
            next_access_time = accesses[-1]["time"] + avg_interval
            
            # Calculate confidence based on pattern consistency
            variance = statistics.variance(intervals) if len(intervals) > 1 else 0
            confidence = max(0, 1 - (variance / max(avg_interval, 1)))
            
            if confidence >= self.config.prefetch_confidence_threshold:
                self.prefetch_predictions[block_id] = {
                    "next_access_time": next_access_time,
                    "confidence": confidence,
                    "prediction_time": time.time()
                }
                
                # Trigger prefetch
                self._trigger_prefetch(block_id)

        except Exception as e:
            logger.error(f"Prediction update failed: {e}")

    def _trigger_prefetch(self, block_id: str):
        """Trigger memory prefetching."""
        try:
            if block_id in self.prefetch_cache:
                self.prefetch_stats["hits"] += 1
                return
            
            # Simulate prefetching
            self.prefetch_cache[block_id] = {
                "prefetch_time": time.time(),
                "status": "prefetched"
            }
            
            self.prefetch_stats["predictions"] += 1
            logger.debug(f"Prefetching block {block_id}")

        except Exception as e:
            logger.error(f"Prefetch trigger failed: {e}")

    def get_prefetch_stats(self) -> Dict[str, Any]:
        """Get prefetching statistics."""
        try:
            total_requests = self.prefetch_stats["hits"] + self.prefetch_stats["misses"]
            hit_rate = self.prefetch_stats["hits"] / max(total_requests, 1)
            
            stats = {
                "prefetch_hits": self.prefetch_stats["hits"],
                "prefetch_misses": self.prefetch_stats["misses"],
                "hit_rate": hit_rate,
                "total_predictions": self.prefetch_stats["predictions"],
                "active_predictions": len(self.prefetch_predictions),
                "prefetch_cache_size": len(self.prefetch_cache)
            }
            return stats

        except Exception as e:
            logger.error(f"Failed to get prefetch stats: {e}")
            return {}


class MemoryLeakDetector:
    """Advanced memory leak detection and prevention."""

    def __init__(self, config: MemoryConfig):
        self.config = config
        self.memory_snapshots = deque(maxlen=100)
        self.leak_suspicions = []
        self.leak_threshold = 0.1  # 10% growth threshold
        self.detection_interval = 60  # 1 minute

    def take_snapshot(self):
        """Take a memory usage snapshot."""
        try:
            snapshot = {
                "timestamp": time.time(),
                "total_memory": self._get_total_memory_usage(),
                "allocated_blocks": len(self._get_allocated_blocks()),
                "memory_by_type": self._get_memory_by_type()
            }
            
            self.memory_snapshots.append(snapshot)
            
            # Check for leaks
            self._detect_leaks()

        except Exception as e:
            logger.error(f"Memory snapshot failed: {e}")

    def _get_total_memory_usage(self) -> int:
        """Get total memory usage in bytes."""
        try:
            if PSUTIL_AVAILABLE:
                process = psutil.Process()
                return process.memory_info().rss
            
            # Fallback to torch memory stats
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated()
            
            return 0

        except Exception as e:
            logger.error(f"Total memory usage check failed: {e}")
            return 0

    def _get_allocated_blocks(self) -> List[str]:
        """Get list of allocated memory blocks."""
        try:
            # This would return actual allocated block IDs
            # For now, return empty list
            return []

        except Exception as e:
            logger.error(f"Allocated blocks check failed: {e}")
            return []

    def _get_memory_by_type(self) -> Dict[str, int]:
        """Get memory usage by type."""
        try:
            memory_by_type = {}
            
            if torch.cuda.is_available():
                memory_by_type["gpu_allocated"] = torch.cuda.memory_allocated()
                memory_by_type["gpu_reserved"] = torch.cuda.memory_reserved()
            
            if PSUTIL_AVAILABLE:
                process = psutil.Process()
                memory_by_type["system_rss"] = process.memory_info().rss
                memory_by_type["system_vms"] = process.memory_info().vms
            
            return memory_by_type

        except Exception as e:
            logger.error(f"Memory by type check failed: {e}")
            return {}

    def _detect_leaks(self):
        """Detect potential memory leaks."""
        try:
            if len(self.memory_snapshots) < 3:
                return
            
            # Compare recent snapshots
            recent = self.memory_snapshots[-1]
            previous = self.memory_snapshots[-2]
            
            # Check for memory growth
            memory_growth = (recent["total_memory"] - previous["total_memory"]) / max(previous["total_memory"], 1)
            
            if memory_growth > self.leak_threshold:
                suspicion = {
                    "timestamp": recent["timestamp"],
                    "memory_growth": memory_growth,
                    "growth_amount": recent["total_memory"] - previous["total_memory"],
                    "severity": "high" if memory_growth > 0.5 else "medium"
                }
                
                self.leak_suspicions.append(suspicion)
                logger.warning(f"Potential memory leak detected: {memory_growth:.2%} growth")

        except Exception as e:
            logger.error(f"Leak detection failed: {e}")

    def get_leak_report(self) -> Dict[str, Any]:
        """Get memory leak detection report."""
        try:
            if not self.memory_snapshots:
                return {"status": "no_data"}
            
            latest = self.memory_snapshots[-1]
            
            report = {
                "current_memory_mb": latest["total_memory"] / (1024 * 1024),
                "total_snapshots": len(self.memory_snapshots),
                "leak_suspicions": len(self.leak_suspicions),
                "recent_suspicions": self.leak_suspicions[-5:] if self.leak_suspicions else [],
                "memory_trend": self._calculate_memory_trend()
            }
            
            return report

        except Exception as e:
            logger.error(f"Leak report generation failed: {e}")
            return {"error": str(e)}

    def _calculate_memory_trend(self) -> str:
        """Calculate memory usage trend."""
        try:
            if len(self.memory_snapshots) < 2:
                return "insufficient_data"
            
            first = self.memory_snapshots[0]
            last = self.memory_snapshots[-1]
            
            growth = (last["total_memory"] - first["total_memory"]) / max(first["total_memory"], 1)
            
            if growth > 0.1:
                return "increasing"
            elif growth < -0.1:
                return "decreasing"
            else:
                return "stable"

        except Exception as e:
            logger.error(f"Memory trend calculation failed: {e}")
            return "unknown"


class AdvancedMemoryManagementSystem:
    """Main memory management system orchestrating all components."""

    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()
        self.logger = logging.getLogger(f"{__name__}.system")

        # Initialize components
        self.memory_pool = MemoryPool(self.config)
        self.virtual_memory = VirtualMemoryManager(self.config)
        self.prefetcher = MemoryPrefetcher(self.config)
        self.leak_detector = MemoryLeakDetector(self.config)

        # System state
        self.is_monitoring = False
        self.monitoring_thread = None
        self.memory_history = deque(maxlen=self.config.memory_history_size)

    def start_monitoring(self):
        """Start memory monitoring."""
        try:
            if self.is_monitoring:
                return
            
            self.is_monitoring = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_worker, daemon=True)
            self.monitoring_thread.start()
            
            self.logger.info("Memory monitoring started")

        except Exception as e:
            self.logger.error(f"Failed to start monitoring: {e}")

    def stop_monitoring(self):
        """Stop memory monitoring."""
        try:
            self.is_monitoring = False
            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=5)
            
            self.logger.info("Memory monitoring stopped")

        except Exception as e:
            self.logger.error(f"Failed to stop monitoring: {e}")

    def _monitoring_worker(self):
        """Background monitoring worker."""
        while self.is_monitoring:
            try:
                # Take memory snapshot
                self.leak_detector.take_snapshot()
                
                # Collect memory metrics
                metrics = self._collect_memory_metrics()
                self.memory_history.append(metrics)
                
                # Sleep
                time.sleep(self.config.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Monitoring worker failed: {e}")
                time.sleep(5)

    def _collect_memory_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive memory metrics."""
        try:
            metrics = {
                "timestamp": time.time(),
                "memory_pool": self.memory_pool.get_pool_stats(),
                "virtual_memory": self.virtual_memory.get_virtual_memory_stats(),
                "prefetch": self.prefetcher.get_prefetch_stats(),
                "leak_detection": self.leak_detector.get_leak_report()
            }
            
            # Add system memory info
            if PSUTIL_AVAILABLE:
                process = psutil.Process()
                metrics["system"] = {
                    "rss_mb": process.memory_info().rss / (1024 * 1024),
                    "vms_mb": process.memory_info().vms / (1024 * 1024),
                    "percent": process.memory_percent()
                }
            
            # Add GPU memory info
            if PYNVML_AVAILABLE and torch.cuda.is_available():
                try:
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    
                    metrics["gpu"] = {
                        "total_mb": info.total / (1024 * 1024),
                        "free_mb": info.free / (1024 * 1024),
                        "used_mb": info.used / (1024 * 1024),
                        "utilization": info.used / info.total
                    }
                except Exception as e:
                    logger.debug(f"GPU memory info collection failed: {e}")
            
            return metrics

        except Exception as e:
            self.logger.error(f"Memory metrics collection failed: {e}")
            return {"error": str(e)}

    def allocate_memory(self, size_bytes: int, block_type: str = "general") -> MemoryBlock:
        """Allocate memory using the memory pool."""
        try:
            # Record access pattern for prefetching
            self.prefetcher.record_access(f"alloc_{block_type}", "allocate")
            
            # Allocate from pool
            block = self.memory_pool.allocate(size_bytes, block_type)
            
            self.logger.debug(f"Allocated {size_bytes} bytes for {block_type}")
            return block

        except Exception as e:
            self.logger.error(f"Memory allocation failed: {e}")
            raise

    def deallocate_memory(self, block: MemoryBlock):
        """Deallocate memory using the memory pool."""
        try:
            self.memory_pool.deallocate(block)
            self.logger.debug(f"Deallocated block {block.block_id}")

        except Exception as e:
            self.logger.error(f"Memory deallocation failed: {e}")

    def allocate_virtual_memory(self, size_bytes: int) -> int:
        """Allocate virtual memory address space."""
        try:
            address = self.virtual_memory.allocate_virtual_memory(size_bytes)
            self.logger.debug(f"Allocated {size_bytes} bytes of virtual memory at {address}")
            return address

        except Exception as e:
            self.logger.error(f"Virtual memory allocation failed: {e}")
            raise

    def get_memory_summary(self) -> Dict[str, Any]:
        """Get comprehensive memory system summary."""
        try:
            summary = {
                "timestamp": time.time(),
                "memory_pool": self.memory_pool.get_pool_stats(),
                "virtual_memory": self.virtual_memory.get_virtual_memory_stats(),
                "prefetch": self.prefetcher.get_prefetch_stats(),
                "leak_detection": self.leak_detector.get_leak_report(),
                "monitoring": {
                    "is_active": self.is_monitoring,
                    "history_size": len(self.memory_history)
                }
            }
            
            # Add recent memory history
            if self.memory_history:
                summary["recent_history"] = list(self.memory_history)[-10:]
            
            return summary

        except Exception as e:
            self.logger.error(f"Memory summary generation failed: {e}")
            return {"error": str(e)}

    def optimize_memory(self) -> Dict[str, Any]:
        """Run memory optimization procedures."""
        try:
            optimization_results = {}
            
            # Force garbage collection
            gc.collect()
            optimization_results["garbage_collection"] = "completed"
            
            # Clean up memory pools
            self.memory_pool._cleanup_pools()
            optimization_results["pool_cleanup"] = "completed"
            
            # Trigger virtual memory swap if needed
            if self.virtual_memory.used_virtual_memory > self.virtual_memory.total_virtual_memory * self.config.swap_threshold:
                self.virtual_memory._trigger_swap()
                optimization_results["memory_swap"] = "triggered"
            
            # Clear prefetch cache if too large
            if len(self.prefetcher.prefetch_cache) > 1000:
                self.prefetcher.prefetch_cache.clear()
                optimization_results["prefetch_cache_clear"] = "completed"
            
            self.logger.info("Memory optimization completed")
            return optimization_results

        except Exception as e:
            self.logger.error(f"Memory optimization failed: {e}")
            return {"error": str(e)}

    def get_memory_recommendations(self) -> List[str]:
        """Get memory optimization recommendations."""
        try:
            recommendations = []
            summary = self.get_memory_summary()
            
            # Check memory pool utilization
            pool_stats = summary.get("memory_pool", {})
            utilization = pool_stats.get("utilization", 0)
            
            if utilization > 0.9:
                recommendations.append("High memory pool utilization detected. Consider increasing pool size.")
            
            # Check virtual memory usage
            vm_stats = summary.get("virtual_memory", {})
            vm_utilization = vm_stats.get("utilization", 0)
            
            if vm_utilization > 0.8:
                recommendations.append("High virtual memory usage. Consider memory cleanup or swap optimization.")
            
            # Check prefetch hit rate
            prefetch_stats = summary.get("prefetch", {})
            hit_rate = prefetch_stats.get("hit_rate", 0)
            
            if hit_rate < 0.5:
                recommendations.append("Low prefetch hit rate. Consider adjusting prefetch confidence threshold.")
            
            # Check for memory leaks
            leak_report = summary.get("leak_detection", {})
            leak_suspicions = leak_report.get("leak_suspicions", 0)
            
            if leak_suspicions > 5:
                recommendations.append("Multiple memory leak suspicions detected. Investigate memory usage patterns.")
            
            return recommendations

        except Exception as e:
            self.logger.error(f"Recommendation generation failed: {e}")
            return ["Unable to generate recommendations due to error"]


# Factory functions
def create_advanced_memory_management_system(config: Optional[MemoryConfig] = None) -> AdvancedMemoryManagementSystem:
    """Create an advanced memory management system."""
    if config is None:
        config = MemoryConfig()
    
    return AdvancedMemoryManagementSystem(config)


def create_memory_config_for_performance() -> MemoryConfig:
    """Create memory configuration optimized for performance."""
    return MemoryConfig(
        enable_intelligent_allocation=True,
        enable_virtual_memory=True,
        enable_memory_pooling=True,
        enable_prefetching=True,
        pool_initial_size_mb=2048,  # 2GB
        pool_max_size_mb=16384,     # 16GB
        prefetch_confidence_threshold=0.6,
        monitoring_interval=0.5,    # 500ms
        cleanup_interval=15.0       # 15 seconds
    )


def create_memory_config_for_memory_efficiency() -> MemoryConfig:
    """Create memory configuration optimized for memory efficiency."""
    return MemoryConfig(
        enable_intelligent_allocation=True,
        enable_virtual_memory=True,
        enable_memory_pooling=True,
        enable_prefetching=False,  # Disable prefetching to save memory
        pool_initial_size_mb=512,   # 512MB
        pool_max_size_mb=4096,      # 4GB
        virtual_memory_size_mb=8192, # 8GB
        memory_compression=True,
        compression_ratio=0.7,
        cleanup_interval=10.0       # 10 seconds
    )


def create_memory_config_for_balanced_usage() -> MemoryConfig:
    """Create memory configuration for balanced performance and efficiency."""
    return MemoryConfig(
        enable_intelligent_allocation=True,
        enable_virtual_memory=True,
        enable_memory_pooling=True,
        enable_prefetching=True,
        pool_initial_size_mb=1024,  # 1GB
        pool_max_size_mb=8192,      # 8GB
        prefetch_confidence_threshold=0.7,
        monitoring_interval=1.0,    # 1 second
        cleanup_interval=30.0       # 30 seconds
    )


if __name__ == "__main__":
    # Test the advanced memory management system
    config = create_memory_config_for_performance()
    memory_system = create_advanced_memory_management_system(config)
    
    # Start monitoring
    memory_system.start_monitoring()
    
    # Test memory allocation
    try:
        # Allocate some memory
        block1 = memory_system.allocate_memory(1024 * 1024, "test_block_1")
        block2 = memory_system.allocate_memory(2048 * 1024, "test_block_2")
        
        # Wait a bit
        time.sleep(2)
        
        # Get summary
        summary = memory_system.get_memory_summary()
        print(f"Memory summary: {json.dumps(summary, indent=2, default=str)}")
        
        # Get recommendations
        recommendations = memory_system.get_memory_recommendations()
        print(f"Recommendations: {recommendations}")
        
        # Deallocate memory
        memory_system.deallocate_memory(block1)
        memory_system.deallocate_memory(block2)
        
        # Run optimization
        optimization_results = memory_system.optimize_memory()
        print(f"Optimization results: {optimization_results}")
        
    except Exception as e:
        print(f"Test failed: {e}")
    
    finally:
        # Stop monitoring
        memory_system.stop_monitoring()
        print("Memory management system test completed")
