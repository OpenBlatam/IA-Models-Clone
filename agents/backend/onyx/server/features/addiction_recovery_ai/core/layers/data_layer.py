"""
Data Layer - Ultra Modular Data Processing
Handles all data-related operations with clear separation of concerns
"""

from typing import Optional, Dict, Any, List, Callable, Iterator
from torch.utils.data import Dataset, DataLoader
import torch
import logging
from abc import ABC, abstractmethod

from .interfaces import IDataProcessor, IDataLoader, BaseDataProcessor

logger = logging.getLogger(__name__)


# ============================================================================
# Data Processors - Modular transformation components
# ============================================================================

class DataProcessor(BaseDataProcessor):
    """
    Base data processor with chaining support
    """
    
    def __init__(self, name: str = "DataProcessor"):
        self.name = name
        self.next_processor: Optional['DataProcessor'] = None
    
    def set_next(self, processor: 'DataProcessor') -> 'DataProcessor':
        """Chain processors together"""
        self.next_processor = processor
        return processor
    
    def process(self, data: Any) -> Any:
        """Process data and pass to next processor if available"""
        processed = self._process(data)
        if self.next_processor:
            return self.next_processor.process(processed)
        return processed
    
    @abstractmethod
    def _process(self, data: Any) -> Any:
        """Internal processing logic"""
        pass


class NormalizationProcessor(DataProcessor):
    """Normalize numerical data"""
    
    def __init__(self, mean: Optional[float] = None, std: Optional[float] = None):
        super().__init__("NormalizationProcessor")
        self.mean = mean
        self.std = std
    
    def _process(self, data: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            self.mean = data.mean()
            self.std = data.std() + 1e-8
        return (data - self.mean) / self.std


class TokenizationProcessor(DataProcessor):
    """Tokenize text data"""
    
    def __init__(self, tokenizer: Callable):
        super().__init__("TokenizationProcessor")
        self.tokenizer = tokenizer
    
    def _process(self, data: str) -> Dict[str, torch.Tensor]:
        return self.tokenizer(data, return_tensors="pt")


class PaddingProcessor(DataProcessor):
    """Pad sequences to fixed length"""
    
    def __init__(self, max_length: int, pad_value: int = 0):
        super().__init__("PaddingProcessor")
        self.max_length = max_length
        self.pad_value = pad_value
    
    def _process(self, data: torch.Tensor) -> torch.Tensor:
        if data.size(0) >= self.max_length:
            return data[:self.max_length]
        padding = torch.full((self.max_length - data.size(0),), self.pad_value)
        return torch.cat([data, padding])


# ============================================================================
# Data Validators
# ============================================================================

class DataValidator:
    """Validate data before processing"""
    
    @staticmethod
    def validate_tensor(tensor: torch.Tensor, shape: Optional[tuple] = None) -> bool:
        """Validate tensor shape and values"""
        if not isinstance(tensor, torch.Tensor):
            return False
        if shape and tensor.shape != shape:
            return False
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            return False
        return True
    
    @staticmethod
    def validate_text(text: str, max_length: Optional[int] = None) -> bool:
        """Validate text data"""
        if not isinstance(text, str):
            return False
        if max_length and len(text) > max_length:
            return False
        return True
    
    @staticmethod
    def validate_batch(batch: List[Any], expected_size: Optional[int] = None) -> bool:
        """Validate batch consistency"""
        if not batch:
            return False
        if expected_size and len(batch) != expected_size:
            return False
        return True


# ============================================================================
# Dataset Factory - Modular dataset creation
# ============================================================================

class DatasetFactory:
    """Factory for creating datasets with different configurations"""
    
    _registry: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str, dataset_class: type):
        """Register a dataset class"""
        cls._registry[name] = dataset_class
        logger.info(f"Registered dataset: {name}")
    
    @classmethod
    def create(
        cls,
        dataset_type: str,
        data: Any,
        config: Optional[Dict[str, Any]] = None
    ) -> Dataset:
        """Create dataset instance"""
        if dataset_type not in cls._registry:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        config = config or {}
        dataset_class = cls._registry[dataset_type]
        return dataset_class(data, **config)
    
    @classmethod
    def list_types(cls) -> List[str]:
        """List available dataset types"""
        return list(cls._registry.keys())


# ============================================================================
# Data Pipeline - Composable data processing pipeline
# ============================================================================

class DataPipeline:
    """
    Composable data processing pipeline
    Chains multiple processors together
    """
    
    def __init__(self, name: str = "DataPipeline"):
        self.name = name
        self.processors: List[DataProcessor] = []
        self.validator = DataValidator()
    
    def add_processor(self, processor: DataProcessor) -> 'DataPipeline':
        """Add processor to pipeline"""
        self.processors.append(processor)
        return self
    
    def process(self, data: Any, validate: bool = True) -> Any:
        """Process data through pipeline"""
        # Validate input
        if validate and not self.validator.validate_tensor(data) if isinstance(data, torch.Tensor) else True:
            raise ValueError(f"Invalid input data in pipeline {self.name}")
        
        # Process through all processors
        result = data
        for processor in self.processors:
            result = processor.process(result)
        
        return result
    
    def process_batch(self, batch: List[Any], validate: bool = True) -> List[Any]:
        """Process batch through pipeline"""
        if validate:
            self.validator.validate_batch(batch)
        return [self.process(item, validate=False) for item in batch]
    
    def clear(self) -> 'DataPipeline':
        """Clear all processors"""
        self.processors = []
        return self


# ============================================================================
# Data Loader Factory - Modular data loading
# ============================================================================

class DataLoaderFactory:
    """Factory for creating data loaders with different configurations"""
    
    @staticmethod
    def create(
        dataset: Dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        **kwargs
    ) -> DataLoader:
        """Create optimized data loader"""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            **kwargs
        )
    
    @staticmethod
    def create_for_training(
        dataset: Dataset,
        batch_size: int = 32,
        num_workers: int = 4
    ) -> DataLoader:
        """Create data loader optimized for training"""
        return DataLoaderFactory.create(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False
        )
    
    @staticmethod
    def create_for_inference(
        dataset: Dataset,
        batch_size: int = 64,
        num_workers: int = 2
    ) -> DataLoader:
        """Create data loader optimized for inference"""
        return DataLoaderFactory.create(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False
        )


# Export main components
__all__ = [
    "DataProcessor",
    "NormalizationProcessor",
    "TokenizationProcessor",
    "PaddingProcessor",
    "DataValidator",
    "DatasetFactory",
    "DataPipeline",
    "DataLoaderFactory",
]



