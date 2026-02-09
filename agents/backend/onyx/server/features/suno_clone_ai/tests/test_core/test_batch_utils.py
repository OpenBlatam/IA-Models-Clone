"""
Tests para utilidades de batch processing
"""

import pytest
import torch
from torch.utils.data import Dataset, DataLoader
from unittest.mock import Mock

from core.utils.batch_utils import (
    BatchProcessor,
    create_batch_processor,
    process_in_batches
)


class SimpleDataset(Dataset):
    """Dataset simple para tests"""
    
    def __init__(self, size=100):
        self.data = [torch.randn(10) for _ in range(size)]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


@pytest.fixture
def batch_processor():
    """Fixture para BatchProcessor"""
    return BatchProcessor(batch_size=32)


@pytest.fixture
def sample_dataset():
    """Dataset de prueba"""
    return SimpleDataset(size=100)


@pytest.mark.unit
@pytest.mark.core
class TestBatchProcessor:
    """Tests para BatchProcessor"""
    
    def test_init_default(self):
        """Test de inicialización por defecto"""
        processor = BatchProcessor()
        
        assert processor.batch_size == 32
        assert processor.collate_fn is not None
    
    def test_init_custom(self):
        """Test de inicialización personalizada"""
        custom_collate = Mock()
        processor = BatchProcessor(batch_size=64, collate_fn=custom_collate)
        
        assert processor.batch_size == 64
        assert processor.collate_fn == custom_collate
    
    def test_default_collate_tensors(self):
        """Test de collate por defecto con tensores"""
        batch = [torch.randn(10) for _ in range(5)]
        result = BatchProcessor._default_collate(batch)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == 5
        assert result.shape[1] == 10
    
    def test_default_collate_dicts(self):
        """Test de collate por defecto con diccionarios"""
        batch = [
            {"a": torch.randn(5), "b": torch.randn(3)}
            for _ in range(4)
        ]
        result = BatchProcessor._default_collate(batch)
        
        assert isinstance(result, dict)
        assert "a" in result
        assert "b" in result
        assert result["a"].shape[0] == 4
        assert result["b"].shape[0] == 4
    
    def test_default_collate_other(self):
        """Test de collate por defecto con otros tipos"""
        batch = [1, 2, 3, 4, 5]
        result = BatchProcessor._default_collate(batch)
        
        assert result == batch
    
    def test_create_dataloader(self, batch_processor, sample_dataset):
        """Test de creación de DataLoader"""
        dataloader = batch_processor.create_dataloader(sample_dataset)
        
        assert isinstance(dataloader, DataLoader)
        assert dataloader.batch_size == 32
    
    def test_create_dataloader_custom_params(self, batch_processor, sample_dataset):
        """Test de creación con parámetros personalizados"""
        dataloader = batch_processor.create_dataloader(
            sample_dataset,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        assert isinstance(dataloader, DataLoader)
        assert dataloader.shuffle is False
        assert dataloader.num_workers == 2
        assert dataloader.pin_memory is True
    
    def test_process_batches_list_result(self, batch_processor):
        """Test de procesamiento con resultado de lista"""
        data = list(range(100))
        
        def process_fn(batch):
            return [x * 2 for x in batch]
        
        results = batch_processor.process_batches(data, process_fn)
        
        assert len(results) == 100
        assert results[0] == 0
        assert results[1] == 2
        assert results[50] == 100
    
    def test_process_batches_single_result(self, batch_processor):
        """Test de procesamiento con resultado único"""
        data = list(range(100))
        
        def process_fn(batch):
            return sum(batch)
        
        results = batch_processor.process_batches(data, process_fn)
        
        assert len(results) == 4  # 100 / 32 = 4 batches (último más pequeño)
        assert all(isinstance(r, int) for r in results)
    
    def test_process_batches_custom_batch_size(self):
        """Test de procesamiento con batch_size personalizado"""
        processor = BatchProcessor(batch_size=10)
        data = list(range(50))
        
        def process_fn(batch):
            return [x * 2 for x in batch]
        
        results = processor.process_batches(data, process_fn)
        
        assert len(results) == 50
        assert results[0] == 0
        assert results[49] == 98
    
    def test_process_batches_empty(self, batch_processor):
        """Test de procesamiento con lista vacía"""
        data = []
        
        def process_fn(batch):
            return []
        
        results = batch_processor.process_batches(data, process_fn)
        
        assert results == []
    
    def test_process_batches_smaller_than_batch(self, batch_processor):
        """Test de procesamiento con menos items que batch_size"""
        data = list(range(10))
        
        def process_fn(batch):
            return [x * 2 for x in batch]
        
        results = batch_processor.process_batches(data, process_fn)
        
        assert len(results) == 10


@pytest.mark.unit
@pytest.mark.core
class TestHelperFunctions:
    """Tests para funciones helper"""
    
    def test_create_batch_processor(self):
        """Test de creación de BatchProcessor"""
        processor = create_batch_processor(batch_size=64)
        
        assert isinstance(processor, BatchProcessor)
        assert processor.batch_size == 64
    
    def test_process_in_batches(self):
        """Test de procesamiento en batches"""
        data = list(range(50))
        
        def process_fn(batch):
            return [x * 2 for x in batch]
        
        results = process_in_batches(data, process_fn, batch_size=10)
        
        assert len(results) == 50
        assert results[0] == 0
        assert results[49] == 98
    
    def test_process_in_batches_custom(self):
        """Test de procesamiento con función personalizada"""
        data = [torch.randn(5) for _ in range(20)]
        
        def process_fn(batch):
            return torch.stack(batch).mean(dim=0)
        
        results = process_in_batches(data, process_fn, batch_size=5)
        
        assert len(results) == 4  # 20 / 5 = 4 batches
        assert all(isinstance(r, torch.Tensor) for r in results)



