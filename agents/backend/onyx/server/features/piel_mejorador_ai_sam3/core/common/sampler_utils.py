"""
Sampler Utilities for Piel Mejorador AI SAM3
============================================

Unified sampling pattern utilities.
"""

import random
import logging
from typing import TypeVar, List, Callable, Optional, Any
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

T = TypeVar('T')


class Sampler(ABC):
    """Base sampler interface."""
    
    @abstractmethod
    def sample(self, items: List[T], count: int) -> List[T]:
        """Sample items from list."""
        pass


class RandomSampler(Sampler):
    """Random sampler."""
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize random sampler.
        
        Args:
            seed: Optional random seed
        """
        if seed is not None:
            random.seed(seed)
    
    def sample(self, items: List[T], count: int) -> List[T]:
        """
        Sample random items.
        
        Args:
            items: Items to sample from
            count: Number of items to sample
            
        Returns:
            Sampled items
        """
        if count >= len(items):
            return items.copy()
        return random.sample(items, count)


class SystematicSampler(Sampler):
    """Systematic sampler (every Nth item)."""
    
    def __init__(self, step: int = 1, start: int = 0):
        """
        Initialize systematic sampler.
        
        Args:
            step: Step size
            start: Starting index
        """
        self.step = step
        self.start = start
    
    def sample(self, items: List[T], count: Optional[int] = None) -> List[T]:
        """
        Sample systematic items.
        
        Args:
            items: Items to sample from
            count: Optional maximum count (ignored for systematic)
            
        Returns:
            Sampled items
        """
        return items[self.start::self.step]


class StratifiedSampler(Sampler):
    """Stratified sampler (proportional sampling from groups)."""
    
    def __init__(self, key_func: Callable[[T], Any]):
        """
        Initialize stratified sampler.
        
        Args:
            key_func: Function to extract group key
        """
        self._key_func = key_func
    
    def sample(self, items: List[T], count: int) -> List[T]:
        """
        Sample stratified items.
        
        Args:
            items: Items to sample from
            count: Number of items to sample
            
        Returns:
            Sampled items
        """
        # Group items by key
        groups: dict = {}
        for item in items:
            key = self._key_func(item)
            if key not in groups:
                groups[key] = []
            groups[key].append(item)
        
        # Calculate samples per group
        total_items = len(items)
        sampled = []
        
        for group_items in groups.values():
            group_count = max(1, int(len(group_items) * count / total_items))
            sampled.extend(random.sample(group_items, min(group_count, len(group_items))))
        
        # If we need more, randomly sample from all
        if len(sampled) < count:
            remaining = [item for item in items if item not in sampled]
            needed = count - len(sampled)
            if remaining:
                sampled.extend(random.sample(remaining, min(needed, len(remaining))))
        
        return sampled[:count]


class WeightedSampler(Sampler):
    """Weighted sampler."""
    
    def __init__(self, weight_func: Callable[[T], float]):
        """
        Initialize weighted sampler.
        
        Args:
            weight_func: Function to get weight for each item
        """
        self._weight_func = weight_func
    
    def sample(self, items: List[T], count: int) -> List[T]:
        """
        Sample weighted items.
        
        Args:
            items: Items to sample from
            count: Number of items to sample
            
        Returns:
            Sampled items
        """
        if count >= len(items):
            return items.copy()
        
        # Calculate weights
        weights = [self._weight_func(item) for item in items]
        total_weight = sum(weights)
        
        if total_weight == 0:
            return random.sample(items, count)
        
        # Normalize weights
        normalized = [w / total_weight for w in weights]
        
        # Sample with replacement based on weights
        sampled = random.choices(items, weights=normalized, k=count)
        
        # Remove duplicates while preserving order
        seen = set()
        result = []
        for item in sampled:
            if item not in seen:
                seen.add(item)
                result.append(item)
        
        # If we need more, randomly sample from remaining
        if len(result) < count:
            remaining = [item for item in items if item not in seen]
            needed = count - len(result)
            if remaining:
                result.extend(random.sample(remaining, min(needed, len(remaining))))
        
        return result


class SamplerUtils:
    """Unified sampler utilities."""
    
    @staticmethod
    def create_random_sampler(seed: Optional[int] = None) -> RandomSampler:
        """
        Create random sampler.
        
        Args:
            seed: Optional random seed
            
        Returns:
            RandomSampler
        """
        return RandomSampler(seed)
    
    @staticmethod
    def create_systematic_sampler(step: int = 1, start: int = 0) -> SystematicSampler:
        """
        Create systematic sampler.
        
        Args:
            step: Step size
            start: Starting index
            
        Returns:
            SystematicSampler
        """
        return SystematicSampler(step, start)
    
    @staticmethod
    def create_stratified_sampler(key_func: Callable[[T], Any]) -> StratifiedSampler:
        """
        Create stratified sampler.
        
        Args:
            key_func: Function to extract group key
            
        Returns:
            StratifiedSampler
        """
        return StratifiedSampler(key_func)
    
    @staticmethod
    def create_weighted_sampler(weight_func: Callable[[T], float]) -> WeightedSampler:
        """
        Create weighted sampler.
        
        Args:
            weight_func: Function to get weight
            
        Returns:
            WeightedSampler
        """
        return WeightedSampler(weight_func)
    
    @staticmethod
    def sample_random(items: List[T], count: int, seed: Optional[int] = None) -> List[T]:
        """
        Sample random items.
        
        Args:
            items: Items to sample from
            count: Number of items to sample
            seed: Optional random seed
            
        Returns:
            Sampled items
        """
        sampler = RandomSampler(seed)
        return sampler.sample(items, count)


# Convenience functions
def create_random_sampler(seed: Optional[int] = None) -> RandomSampler:
    """Create random sampler."""
    return SamplerUtils.create_random_sampler(seed)


def create_systematic_sampler(step: int = 1, **kwargs) -> SystematicSampler:
    """Create systematic sampler."""
    return SamplerUtils.create_systematic_sampler(step, **kwargs)


def create_stratified_sampler(key_func: Callable[[T], Any]) -> StratifiedSampler:
    """Create stratified sampler."""
    return SamplerUtils.create_stratified_sampler(key_func)




