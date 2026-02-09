"""
Intelligent Prefetcher
======================

Predictive prefetching based on access patterns.
"""

import logging
import asyncio
from typing import Dict, Any, List, Callable, Optional
from collections import defaultdict
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class IntelligentPrefetcher:
    """Intelligent prefetcher with pattern recognition."""
    
    def __init__(self):
        self._patterns: Dict[str, List[str]] = defaultdict(list)
        self._access_sequences: List[List[str]] = []
        self._prefetch_tasks: Dict[str, asyncio.Task] = {}
        self._confidence_threshold = 0.6
    
    def record_access_sequence(self, sequence: List[str]):
        """Record access sequence for pattern learning."""
        self._access_sequences.append(sequence)
        
        # Learn patterns
        for i in range(len(sequence) - 1):
            current = sequence[i]
            next_item = sequence[i + 1]
            
            if next_item not in self._patterns[current]:
                self._patterns[current].append(next_item)
        
        # Keep only recent sequences
        if len(self._access_sequences) > 1000:
            self._access_sequences = self._access_sequences[-500:]
    
    def predict_next(self, current: str, top_n: int = 3) -> List[str]:
        """Predict next items based on patterns."""
        if current not in self._patterns:
            return []
        
        candidates = self._patterns[current]
        
        # Calculate confidence (simple frequency-based)
        frequencies = defaultdict(int)
        for sequence in self._access_sequences:
            for i in range(len(sequence) - 1):
                if sequence[i] == current:
                    frequencies[sequence[i + 1]] += 1
        
        # Sort by frequency
        sorted_candidates = sorted(
            frequencies.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [item for item, freq in sorted_candidates[:top_n]]
    
    async def prefetch(self, current: str, loader: Callable, top_n: int = 3):
        """Prefetch predicted items."""
        predicted = self.predict_next(current, top_n)
        
        for item in predicted:
            if item in self._prefetch_tasks:
                continue
            
            async def prefetch_item(key: str):
                try:
                    if asyncio.iscoroutinefunction(loader):
                        await loader(key)
                    else:
                        loader(key)
                    logger.debug(f"Prefetched: {key}")
                except Exception as e:
                    logger.error(f"Prefetch failed for {key}: {e}")
                finally:
                    if key in self._prefetch_tasks:
                        del self._prefetch_tasks[key]
            
            task = asyncio.create_task(prefetch_item(item))
            self._prefetch_tasks[item] = task
    
    def get_pattern_stats(self) -> Dict[str, Any]:
        """Get pattern statistics."""
        return {
            "learned_patterns": len(self._patterns),
            "total_sequences": len(self._access_sequences),
            "active_prefetches": len(self._prefetch_tasks),
            "top_patterns": sorted(
                [
                    (key, len(patterns))
                    for key, patterns in self._patterns.items()
                ],
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }















