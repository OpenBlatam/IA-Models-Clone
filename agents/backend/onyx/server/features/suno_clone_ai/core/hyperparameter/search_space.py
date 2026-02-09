"""
Search Space

Utilities for defining hyperparameter search spaces.
"""

import logging
from typing import Dict, List, Any, Union
import random

logger = logging.getLogger(__name__)


class SearchSpace:
    """Define hyperparameter search space."""
    
    def __init__(self):
        """Initialize search space."""
        self.space: Dict[str, List[Any]] = {}
    
    def add_categorical(
        self,
        name: str,
        choices: List[Any]
    ) -> 'SearchSpace':
        """
        Add categorical parameter.
        
        Args:
            name: Parameter name
            choices: List of choices
            
        Returns:
            Self for chaining
        """
        self.space[name] = choices
        return self
    
    def add_int(
        self,
        name: str,
        min_val: int,
        max_val: int,
        step: int = 1
    ) -> 'SearchSpace':
        """
        Add integer parameter.
        
        Args:
            name: Parameter name
            min_val: Minimum value
            max_val: Maximum value
            step: Step size
            
        Returns:
            Self for chaining
        """
        self.space[name] = list(range(min_val, max_val + 1, step))
        return self
    
    def add_float(
        self,
        name: str,
        min_val: float,
        max_val: float,
        num_samples: int = 10
    ) -> 'SearchSpace':
        """
        Add float parameter.
        
        Args:
            name: Parameter name
            min_val: Minimum value
            max_val: Maximum value
            num_samples: Number of samples
            
        Returns:
            Self for chaining
        """
        import numpy as np
        self.space[name] = np.linspace(min_val, max_val, num_samples).tolist()
        return self
    
    def sample(self) -> Dict[str, Any]:
        """
        Sample random parameters from space.
        
        Returns:
            Sampled parameters
        """
        params = {}
        for key, values in self.space.items():
            params[key] = random.choice(values)
        return params
    
    def get_space(self) -> Dict[str, List[Any]]:
        """Get search space dictionary."""
        return self.space


def create_search_space() -> SearchSpace:
    """Create new search space."""
    return SearchSpace()


def sample_from_space(space: Dict[str, List[Any]]) -> Dict[str, Any]:
    """Sample from search space."""
    params = {}
    for key, values in space.items():
        params[key] = random.choice(values)
    return params



