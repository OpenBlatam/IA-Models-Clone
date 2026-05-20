"""
Cache documentation generator.

Generates comprehensive documentation for cache instances.
"""
from __future__ import annotations

import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class CacheDocumentationGenerator:
    """
    Generates comprehensive documentation for cache instances.
    
    Creates detailed documentation including configuration, stats, and usage.
    """
    
    def __init__(self, cache: Any):
        """
        Initialize documentation generator.
        
        Args:
            cache: Cache instance
        """
        self.cache = cache
    
    def generate_config_doc(self) -> str:
        """
        Generate configuration documentation.
        
        Returns:
            Configuration documentation string
        """
        config = self.cache.config
        
        doc = f"""
# Cache Configuration Documentation

Generated: {datetime.now().isoformat()}

## Configuration

- **Max Tokens**: {config.max_tokens}
- **Number of Heads**: {config.num_heads}
- **Head Dimension**: {config.head_dim}
- **Cache Strategy**: {config.cache_strategy.value}
- **Cache Mode**: {config.cache_mode.value}
- **Use Quantization**: {config.use_quantization}
- **Quantization Bits**: {config.quantization_bits if config.use_quantization else 'N/A'}
- **Use Compression**: {config.use_compression}
- **Compression Ratio**: {config.compression_ratio if config.use_compression else 'N/A'}
- **Device**: {config.device}
- **Dtype**: {config.dtype}

## Performance Settings

- **Enable GC**: {config.enable_gc}
- **Enable Profiling**: {config.enable_profiling}
- **Pin Memory**: {config.pin_memory}
"""
        return doc
    
    def generate_stats_doc(self) -> str:
        """
        Generate statistics documentation.
        
        Returns:
            Statistics documentation string
        """
        stats = self.cache.get_stats(include_history=True)
        
        doc = f"""
# Cache Statistics Documentation

Generated: {datetime.now().isoformat()}

## Performance Metrics

- **Hit Rate**: {stats.get('hit_rate', 0.0):.2%}
- **Miss Rate**: {stats.get('miss_rate', 0.0):.2%}
- **Total Hits**: {stats.get('hits', 0)}
- **Total Misses**: {stats.get('misses', 0)}
- **Total Evictions**: {stats.get('evictions', 0)}

## Cache State

- **Number of Entries**: {stats.get('num_entries', 0)}
- **Max Tokens**: {stats.get('max_tokens', 0)}
- **Memory Usage (MB)**: {stats.get('storage_memory_mb', 0.0):.2f}
- **Utilization**: {(stats.get('num_entries', 0) / max(stats.get('max_tokens', 1), 1)) * 100:.2f}%

## Memory Statistics

{self._format_memory_stats(stats.get('memory_stats', {}))}
"""
        return doc
    
    def _format_memory_stats(self, memory_stats: Dict[str, Any]) -> str:
        """Format memory statistics."""
        if not memory_stats:
            return "No memory statistics available"
        
        return f"""
- **GPU Memory Reserved (MB)**: {memory_stats.get('gpu_reserved_mb', 0.0):.2f}
- **GPU Memory Allocated (MB)**: {memory_stats.get('gpu_allocated_mb', 0.0):.2f}
- **CPU Memory (MB)**: {memory_stats.get('cpu_memory_mb', 0.0):.2f}
"""
    
    def generate_usage_doc(self) -> str:
        """
        Generate usage documentation.
        
        Returns:
            Usage documentation string
        """
        doc = f"""
# Cache Usage Documentation

Generated: {datetime.now().isoformat()}

## Basic Usage

```python
from kv_cache import BaseKVCache, KVCacheConfig

# Create cache
config = KVCacheConfig(max_tokens=4096)
cache = BaseKVCache(config)

# Use cache
key, value = cache.forward(key, value)
result = cache.get(position)
cache.put(position, key, value)
```

## Advanced Features

- **Async Operations**: Use `AsyncCacheOperations` for async support
- **Batch Processing**: Use `BatchProcessor` for batch operations
- **Health Monitoring**: Use `CacheHealthMonitor` for health checks
- **Benchmarking**: Use `CacheBenchmark` for performance testing
- **ML Optimization**: Use `CacheMLOptimizer` for ML-based optimization
"""
        return doc
    
    def generate_full_documentation(self) -> str:
        """
        Generate full documentation.
        
        Returns:
            Complete documentation string
        """
        return f"""
{self.generate_config_doc()}

{self.generate_stats_doc()}

{self.generate_usage_doc()}
"""

