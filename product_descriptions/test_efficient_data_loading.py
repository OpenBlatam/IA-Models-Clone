from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import asyncio
import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.datasets import make_classification, make_regression
from efficient_data_loading import (
            from efficient_data_loading import BaseCybersecurityDataset
        import shutil
        from transformers import AutoTokenizer
        import shutil
        import shutil
        import shutil
        import shutil
        import shutil
        import shutil
        import psutil
            import shutil
            import shutil
            import shutil
from typing import Any, List, Dict, Optional
import logging
"""
Comprehensive Tests for Efficient Data Loading System

This test suite covers:
- Unit tests for individual components
- Integration tests for complete workflows
- Performance tests and benchmarking
- Memory usage and optimization tests
- Caching and async loading tests
- Edge case handling and error scenarios
"""



    DataLoaderConfig, ThreatDetectionDataset, AnomalyDetectionDataset,
    NetworkTrafficDataset, MalwareDataset, CachedDataset, DataAugmentation,
    DataLoaderFactory, DataLoaderMonitor, MemoryOptimizedDataLoader,
    AsyncDataLoader, DataLoaderBenchmark, create_balanced_sampler,
    split_dataset, get_dataset_info, optimize_dataloader_config,
    CustomCollateFn
)


class TestDataLoaderConfig:
    """Test DataLoaderConfig dataclass."""
    
    def test_data_loader_config_creation(self) -> Any:
        """Test creating DataLoaderConfig with default values."""
        config = DataLoaderConfig()
        
        assert config.batch_size == 32
        assert config.num_workers == 4
        assert config.pin_memory is True
        assert config.persistent_workers is True
        assert config.prefetch_factor == 2
        assert config.shuffle is True
        assert config.enable_caching is True
    
    def test_data_loader_config_custom_values(self) -> Any:
        """Test creating DataLoaderConfig with custom values."""
        config = DataLoaderConfig(
            batch_size=64,
            num_workers=8,
            pin_memory=False,
            enable_caching=False,
            cache_size=500
        )
        
        assert config.batch_size == 64
        assert config.num_workers == 8
        assert config.pin_memory is False
        assert config.enable_caching is False
        assert config.cache_size == 500


class TestBaseCybersecurityDataset:
    """Test BaseCybersecurityDataset abstract class."""
    
    def test_base_dataset_abstract_methods(self) -> Any:
        """Test that BaseCybersecurityDataset cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseCybersecurityDataset("test.csv", DataLoaderConfig())


class TestThreatDetectionDataset:
    """Test ThreatDetectionDataset."""
    
    def setup_method(self) -> Any:
        """Setup test data."""
        self.temp_dir = tempfile.mkdtemp()
        self.dataset_path = os.path.join(self.temp_dir, "threat_data.csv")
        
        # Create test data
        data = {
            'text': [
                "Normal network traffic",
                "Suspicious activity detected",
                "Malicious payload identified",
                "Legitimate user request"
            ],
            'label': [0, 1, 1, 0]
        }
        df = pd.DataFrame(data)
        df.to_csv(self.dataset_path, index=False)
    
    def teardown_method(self) -> Any:
        """Cleanup test data."""
        shutil.rmtree(self.temp_dir)
    
    def test_threat_detection_dataset_creation(self) -> Any:
        """Test creating ThreatDetectionDataset."""
        config = DataLoaderConfig()
        dataset = ThreatDetectionDataset(self.dataset_path, config)
        
        assert len(dataset) == 4
        assert len(dataset.data) == 4
        assert len(dataset.labels) == 4
        assert dataset.labels == [0, 1, 1, 0]
    
    def test_threat_detection_dataset_with_tokenizer(self) -> Any:
        """Test ThreatDetectionDataset with tokenizer."""
        
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        config = DataLoaderConfig()
        dataset = ThreatDetectionDataset(self.dataset_path, config, tokenizer)
        
        assert len(dataset) == 4
        assert isinstance(dataset.data[0], dict)
        assert 'input_ids' in dataset.data[0]
        assert 'attention_mask' in dataset.data[0]
    
    def test_threat_detection_dataset_getitem(self) -> Optional[Dict[str, Any]]:
        """Test ThreatDetectionDataset __getitem__ method."""
        config = DataLoaderConfig()
        dataset = ThreatDetectionDataset(self.dataset_path, config)
        
        item = dataset[0]
        assert isinstance(item, tuple)
        assert len(item) == 2
        assert item[1] == 0  # label
    
    def test_threat_detection_dataset_validation(self) -> Any:
        """Test data validation in ThreatDetectionDataset."""
        config = DataLoaderConfig(validate_data=True, sanitize_inputs=True)
        
        # Test with malicious content
        malicious_data = {
            'text': [
                "Normal traffic",
                "<script>alert('xss')</script>",  # Malicious
                "Legitimate request"
            ],
            'label': [0, 1, 0]
        }
        df = pd.DataFrame(malicious_data)
        df.to_csv(self.dataset_path, index=False)
        
        dataset = ThreatDetectionDataset(self.dataset_path, config)
        
        # Should filter out malicious content
        assert len(dataset) == 2  # Only 2 valid items
    
    def test_threat_detection_dataset_invalid_path(self) -> Any:
        """Test ThreatDetectionDataset with invalid path."""
        config = DataLoaderConfig()
        
        with pytest.raises(Exception):
            ThreatDetectionDataset("nonexistent.csv", config)


class TestAnomalyDetectionDataset:
    """Test AnomalyDetectionDataset."""
    
    def setup_method(self) -> Any:
        """Setup test data."""
        self.temp_dir = tempfile.mkdtemp()
        self.dataset_path = os.path.join(self.temp_dir, "anomaly_data.csv")
        
        # Create test data
        features = [
            json.dumps([0.1, 0.2, 0.3, 0.4, 0.5]),
            json.dumps([0.2, 0.3, 0.4, 0.5, 0.6]),
            json.dumps([0.3, 0.4, 0.5, 0.6, 0.7]),
            json.dumps([0.4, 0.5, 0.6, 0.7, 0.8])
        ]
        
        data = {
            'features': features,
            'label': [0, 0, 1, 1]
        }
        df = pd.DataFrame(data)
        df.to_csv(self.dataset_path, index=False)
    
    def teardown_method(self) -> Any:
        """Cleanup test data."""
        shutil.rmtree(self.temp_dir)
    
    def test_anomaly_detection_dataset_creation(self) -> Any:
        """Test creating AnomalyDetectionDataset."""
        config = DataLoaderConfig()
        dataset = AnomalyDetectionDataset(self.dataset_path, config)
        
        assert len(dataset) == 4
        assert isinstance(dataset.data, torch.Tensor)
        assert isinstance(dataset.labels, torch.Tensor)
        assert dataset.data.shape[1] == 5  # 5 features
        assert dataset.labels.shape[0] == 4
    
    def test_anomaly_detection_dataset_getitem(self) -> Optional[Dict[str, Any]]:
        """Test AnomalyDetectionDataset __getitem__ method."""
        config = DataLoaderConfig()
        dataset = AnomalyDetectionDataset(self.dataset_path, config)
        
        item = dataset[0]
        assert isinstance(item, tuple)
        assert len(item) == 2
        assert isinstance(item[0], torch.Tensor)
        assert isinstance(item[1], torch.Tensor)
    
    def test_anomaly_detection_dataset_metadata(self) -> Any:
        """Test metadata extraction."""
        config = DataLoaderConfig()
        dataset = AnomalyDetectionDataset(self.dataset_path, config)
        
        metadata = dataset.get_metadata()
        assert "feature_dim" in metadata
        assert "num_samples" in metadata
        assert "anomaly_ratio" in metadata
        assert metadata["feature_dim"] == 5
        assert metadata["num_samples"] == 4


class TestNetworkTrafficDataset:
    """Test NetworkTrafficDataset."""
    
    def setup_method(self) -> Any:
        """Setup test data."""
        self.temp_dir = tempfile.mkdtemp()
        self.dataset_path = os.path.join(self.temp_dir, "network_data.csv")
        
        # Create test data
        features = np.random.random((10, 5))
        data = {
            'feature_0': features[:, 0],
            'feature_1': features[:, 1],
            'feature_2': features[:, 2],
            'feature_3': features[:, 3],
            'feature_4': features[:, 4],
            'label': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        }
        df = pd.DataFrame(data)
        df.to_csv(self.dataset_path, index=False)
    
    def teardown_method(self) -> Any:
        """Cleanup test data."""
        shutil.rmtree(self.temp_dir)
    
    def test_network_traffic_dataset_creation(self) -> Any:
        """Test creating NetworkTrafficDataset."""
        config = DataLoaderConfig()
        dataset = NetworkTrafficDataset(self.dataset_path, config)
        
        assert len(dataset) == 10
        assert isinstance(dataset.data, torch.Tensor)
        assert isinstance(dataset.labels, torch.Tensor)
        assert dataset.data.shape[1] == 5  # 5 features
        assert dataset.labels.shape[0] == 10
    
    def test_network_traffic_dataset_scaler(self) -> Any:
        """Test that scaler is stored in metadata."""
        config = DataLoaderConfig()
        dataset = NetworkTrafficDataset(self.dataset_path, config)
        
        metadata = dataset.get_metadata()
        assert "scaler" in metadata
        assert metadata["scaler"] is not None


class TestMalwareDataset:
    """Test MalwareDataset."""
    
    def setup_method(self) -> Any:
        """Setup test data."""
        self.temp_dir = tempfile.mkdtemp()
        self.dataset_path = os.path.join(self.temp_dir, "malware_data.csv")
        
        # Create test data
        binary_features = [
            json.dumps([0.1, 0.2, 0.3, 0.4, 0.5]),
            json.dumps([0.2, 0.3, 0.4, 0.5, 0.6]),
            json.dumps([0.3, 0.4, 0.5, 0.6, 0.7])
        ]
        
        api_sequences = [
            json.dumps(["CreateFile", "ReadFile", "WriteFile"]),
            json.dumps(["RegCreateKey", "RegSetValue"]),
            json.dumps(["CreateProcess", "CreateThread"])
        ]
        
        data = {
            'binary_features': binary_features,
            'api_calls': api_sequences,
            'label': [0, 1, 2]
        }
        df = pd.DataFrame(data)
        df.to_csv(self.dataset_path, index=False)
    
    def teardown_method(self) -> Any:
        """Cleanup test data."""
        shutil.rmtree(self.temp_dir)
    
    def test_malware_dataset_creation(self) -> Any:
        """Test creating MalwareDataset."""
        config = DataLoaderConfig()
        dataset = MalwareDataset(self.dataset_path, config)
        
        assert len(dataset) == 3
        assert isinstance(dataset.data, dict)
        assert 'binary_features' in dataset.data
        assert 'api_sequences' in dataset.data
        assert isinstance(dataset.labels, torch.Tensor)
    
    def test_malware_dataset_getitem(self) -> Optional[Dict[str, Any]]:
        """Test MalwareDataset __getitem__ method."""
        config = DataLoaderConfig()
        dataset = MalwareDataset(self.dataset_path, config)
        
        item = dataset[0]
        assert isinstance(item, tuple)
        assert len(item) == 2
        assert isinstance(item[0], dict)
        assert isinstance(item[1], torch.Tensor)


class TestCachedDataset:
    """Test CachedDataset."""
    
    def setup_method(self) -> Any:
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = os.path.join(self.temp_dir, "cache")
    
    def teardown_method(self) -> Any:
        """Cleanup test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_cached_dataset_creation(self) -> Any:
        """Test creating CachedDataset."""
        # Create mock dataset
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=10)
        mock_dataset.__getitem__ = MagicMock(return_value=("data", "label"))
        
        cached_dataset = CachedDataset(mock_dataset, self.cache_dir, cache_size=5)
        
        assert len(cached_dataset) == 10
        assert cached_dataset.cache_size == 5
    
    def test_cached_dataset_getitem(self) -> Optional[Dict[str, Any]]:
        """Test CachedDataset __getitem__ method."""
        # Create mock dataset
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=10)
        mock_dataset.__getitem__ = MagicMock(return_value=("data", "label"))
        
        cached_dataset = CachedDataset(mock_dataset, self.cache_dir, cache_size=5)
        
        # First access (cache miss)
        item = cached_dataset[0]
        assert item == ("data", "label")
        assert cached_dataset.cache_misses == 1
        assert cached_dataset.cache_hits == 0
        
        # Second access (cache hit)
        item = cached_dataset[0]
        assert item == ("data", "label")
        assert cached_dataset.cache_hits == 1
    
    def test_cached_dataset_cache_stats(self) -> Any:
        """Test cache statistics."""
        # Create mock dataset
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=10)
        mock_dataset.__getitem__ = MagicMock(return_value=("data", "label"))
        
        cached_dataset = CachedDataset(mock_dataset, self.cache_dir, cache_size=5)
        
        # Access some items
        cached_dataset[0]
        cached_dataset[1]
        cached_dataset[0]  # Cache hit
        
        stats = cached_dataset.get_cache_stats()
        assert "cache_size" in stats
        assert "cache_hits" in stats
        assert "cache_misses" in stats
        assert "hit_rate" in stats
        assert stats["cache_hits"] == 1
        assert stats["cache_misses"] == 2


class TestDataAugmentation:
    """Test DataAugmentation."""
    
    def test_text_augmentation(self) -> Any:
        """Test text augmentation."""
        original_text = "Suspicious network activity detected"
        
        # Test with low probability (should return original)
        augmented = DataAugmentation.augment_text(original_text, augmentation_prob=0.0)
        assert augmented == original_text
        
        # Test with high probability
        augmented = DataAugmentation.augment_text(original_text, augmentation_prob=1.0)
        assert isinstance(augmented, str)
        assert len(augmented) > 0
    
    def test_feature_augmentation(self) -> Any:
        """Test feature augmentation."""
        original_features = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        
        # Test with low probability (should return original)
        augmented = DataAugmentation.augment_features(original_features, noise_factor=0.0)
        np.testing.assert_array_almost_equal(augmented, original_features)
        
        # Test with noise
        augmented = DataAugmentation.augment_features(original_features, noise_factor=0.1)
        assert isinstance(augmented, np.ndarray)
        assert augmented.shape == original_features.shape


class TestCustomCollateFn:
    """Test CustomCollateFn."""
    
    def test_threat_detection_collate(self) -> Any:
        """Test threat detection collate function."""
        # Test with tokenized data
        batch = [
            ({'input_ids': torch.tensor([1, 2, 3]), 'attention_mask': torch.tensor([1, 1, 1])}, 0),
            ({'input_ids': torch.tensor([4, 5, 6]), 'attention_mask': torch.tensor([1, 1, 1])}, 1)
        ]
        
        result = CustomCollateFn.threat_detection_collate(batch)
        
        assert 'input_ids' in result
        assert 'attention_mask' in result
        assert 'labels' in result
        assert result['input_ids'].shape[0] == 2
        assert result['labels'].shape[0] == 2
    
    def test_anomaly_detection_collate(self) -> Any:
        """Test anomaly detection collate function."""
        batch = [
            (torch.tensor([0.1, 0.2, 0.3]), 0),
            (torch.tensor([0.4, 0.5, 0.6]), 1)
        ]
        
        features, labels = CustomCollateFn.anomaly_detection_collate(batch)
        
        assert isinstance(features, torch.Tensor)
        assert isinstance(labels, torch.Tensor)
        assert features.shape[0] == 2
        assert labels.shape[0] == 2
    
    def test_malware_collate(self) -> Any:
        """Test malware collate function."""
        batch = [
            ({'binary_features': torch.tensor([0.1, 0.2]), 'api_sequences': ['a', 'b']}, 0),
            ({'binary_features': torch.tensor([0.3, 0.4]), 'api_sequences': ['c', 'd']}, 1)
        ]
        
        result = CustomCollateFn.malware_collate(batch)
        
        assert 'binary_features' in result
        assert 'api_sequences' in result
        assert 'labels' in result
        assert result['binary_features'].shape[0] == 2
        assert len(result['api_sequences']) == 2


class TestDataLoaderFactory:
    """Test DataLoaderFactory."""
    
    def test_create_dataloader(self) -> Any:
        """Test creating DataLoader with factory."""
        # Create mock dataset
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=100)
        mock_dataset.__getitem__ = MagicMock(return_value=("data", "label"))
        
        config = DataLoaderConfig(batch_size=32, num_workers=2)
        
        dataloader = DataLoaderFactory.create_dataloader(mock_dataset, config, "threat_detection")
        
        assert isinstance(dataloader, torch.utils.data.DataLoader)
        assert dataloader.batch_size == 32
        assert dataloader.num_workers == 2
    
    def test_create_dataloader_with_caching(self) -> Any:
        """Test creating DataLoader with caching."""
        # Create mock dataset
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=100)
        mock_dataset.__getitem__ = MagicMock(return_value=("data", "label"))
        
        config = DataLoaderConfig(
            batch_size=32,
            num_workers=2,
            enable_caching=True,
            cache_dir="./cache"
        )
        
        dataloader = DataLoaderFactory.create_dataloader(mock_dataset, config, "threat_detection")
        
        assert isinstance(dataloader.dataset, CachedDataset)


class TestDataLoaderMonitor:
    """Test DataLoaderMonitor."""
    
    def test_data_loader_monitor_creation(self) -> Any:
        """Test creating DataLoaderMonitor."""
        # Create mock DataLoader
        mock_dataloader = MagicMock()
        config = DataLoaderConfig()
        
        monitor = DataLoaderMonitor(mock_dataloader, config)
        
        assert monitor.dataloader == mock_dataloader
        assert monitor.config == config
        assert monitor.metrics["load_times"] == []
    
    def test_record_batch_load(self) -> Any:
        """Test recording batch load metrics."""
        mock_dataloader = MagicMock()
        config = DataLoaderConfig()
        
        monitor = DataLoaderMonitor(mock_dataloader, config)
        
        monitor.record_batch_load(0.1, 32, 0.5)
        
        assert len(monitor.metrics["load_times"]) == 1
        assert monitor.metrics["load_times"][0] == 0.1
        assert monitor.metrics["batch_sizes"][0] == 32
        assert monitor.metrics["memory_usage"][0] == 0.5
    
    def test_record_error(self) -> Any:
        """Test recording errors."""
        mock_dataloader = MagicMock()
        config = DataLoaderConfig()
        
        monitor = DataLoaderMonitor(mock_dataloader, config)
        
        error = Exception("Test error")
        monitor.record_error(error)
        
        assert len(monitor.metrics["errors"]) == 1
        assert monitor.metrics["errors"][0]["error"] == "Test error"
    
    def test_get_performance_report(self) -> Optional[Dict[str, Any]]:
        """Test getting performance report."""
        mock_dataloader = MagicMock()
        config = DataLoaderConfig()
        
        monitor = DataLoaderMonitor(mock_dataloader, config)
        
        # Record some metrics
        monitor.record_batch_load(0.1, 32, 0.5)
        monitor.record_batch_load(0.2, 32, 0.6)
        
        report = monitor.get_performance_report()
        
        assert "total_batches" in report
        assert "avg_load_time_ms" in report
        assert "avg_memory_usage_mb" in report
        assert report["total_batches"] == 2


class TestMemoryOptimizedDataLoader:
    """Test MemoryOptimizedDataLoader."""
    
    def test_memory_optimized_dataloader_creation(self) -> Any:
        """Test creating MemoryOptimizedDataLoader."""
        # Create mock DataLoader
        mock_dataloader = MagicMock()
        mock_dataloader.__iter__ = MagicMock(return_value=iter([("batch1",), ("batch2",)]))
        mock_dataloader.__len__ = MagicMock(return_value=2)
        
        memory_dataloader = MemoryOptimizedDataLoader(mock_dataloader, max_memory_usage=0.8)
        
        assert memory_dataloader.dataloader == mock_dataloader
        assert memory_dataloader.max_memory_usage == 0.8
    
    def test_memory_optimized_dataloader_iteration(self) -> Any:
        """Test MemoryOptimizedDataLoader iteration."""
        # Create mock DataLoader
        mock_dataloader = MagicMock()
        mock_dataloader.__iter__ = MagicMock(return_value=iter([("batch1",), ("batch2",)]))
        mock_dataloader.__len__ = MagicMock(return_value=2)
        
        memory_dataloader = MemoryOptimizedDataLoader(mock_dataloader, max_memory_usage=0.8)
        
        batches = list(memory_dataloader)
        assert len(batches) == 2
        assert batches[0] == ("batch1",)
        assert batches[1] == ("batch2",)


class TestAsyncDataLoader:
    """Test AsyncDataLoader."""
    
    @pytest.mark.asyncio
    async def test_async_dataloader_creation(self) -> Any:
        """Test creating AsyncDataLoader."""
        # Create mock DataLoader
        mock_dataloader = MagicMock()
        mock_dataloader.__iter__ = MagicMock(return_value=iter([("batch1",), ("batch2",)]))
        
        async_dataloader = AsyncDataLoader(mock_dataloader, max_queue_size=5)
        
        assert async_dataloader.dataloader == mock_dataloader
        assert async_dataloader.max_queue_size == 5
    
    @pytest.mark.asyncio
    async def test_async_dataloader_iteration(self) -> Any:
        """Test AsyncDataLoader iteration."""
        # Create mock DataLoader
        mock_dataloader = MagicMock()
        mock_dataloader.__iter__ = MagicMock(return_value=iter([("batch1",), ("batch2",)]))
        
        async_dataloader = AsyncDataLoader(mock_dataloader, max_queue_size=5)
        
        batches = []
        async for batch in async_dataloader:
            batches.append(batch)
        
        assert len(batches) == 2
        assert batches[0] == ("batch1",)
        assert batches[1] == ("batch2",)


class TestDataLoaderBenchmark:
    """Test DataLoaderBenchmark."""
    
    def test_benchmark_dataloader(self) -> Any:
        """Test DataLoaderBenchmark.benchmark_dataloader."""
        # Create mock DataLoader
        mock_dataloader = MagicMock()
        mock_dataloader.__iter__ = MagicMock(return_value=iter([("batch1",), ("batch2",)]))
        
        results = DataLoaderBenchmark.benchmark_dataloader(
            mock_dataloader, num_batches=2, warmup_batches=0
        )
        
        assert "total_batches" in results
        assert "total_time_seconds" in results
        assert "avg_batch_time_ms" in results
        assert "throughput_batches_per_sec" in results


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_create_balanced_sampler(self) -> Any:
        """Test create_balanced_sampler."""
        labels = [0, 0, 0, 1, 1, 2, 2, 2, 2]  # Imbalanced labels
        
        sampler = create_balanced_sampler(MagicMock(), labels)
        
        assert isinstance(sampler, torch.utils.data.WeightedRandomSampler)
        assert sampler.num_samples == len(labels)
    
    def test_split_dataset(self) -> Any:
        """Test split_dataset."""
        # Create mock dataset
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=100)
        
        train_dataset, val_dataset, test_dataset = split_dataset(
            mock_dataset, train_ratio=0.7, val_ratio=0.15
        )
        
        assert isinstance(train_dataset, torch.utils.data.Subset)
        assert isinstance(val_dataset, torch.utils.data.Subset)
        assert isinstance(test_dataset, torch.utils.data.Subset)
        
        # Check sizes
        assert len(train_dataset) == 70
        assert len(val_dataset) == 15
        assert len(test_dataset) == 15
    
    def test_get_dataset_info(self) -> Optional[Dict[str, Any]]:
        """Test get_dataset_info."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("col1,col2\n1,2\n3,4\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            temp_path = f.name
        
        try:
            info = get_dataset_info(temp_path)
            
            assert "size" in info
            assert "columns" in info
            assert "memory_usage_mb" in info
            assert "dtypes" in info
            assert info["size"] == 2
            assert "col1" in info["columns"]
            assert "col2" in info["columns"]
        finally:
            os.unlink(temp_path)
    
    def test_optimize_dataloader_config(self) -> Any:
        """Test optimize_dataloader_config."""
        config = optimize_dataloader_config(
            dataset_size=10000,
            available_memory_gb=16.0,
            num_cpus=8
        )
        
        assert isinstance(config, DataLoaderConfig)
        assert config.batch_size > 0
        assert config.num_workers > 0
        assert config.num_workers <= 8


class TestIntegrationTests:
    """Integration tests for complete workflows."""
    
    def setup_method(self) -> Any:
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test dataset
        self.dataset_path = os.path.join(self.temp_dir, "test_data.csv")
        
        # Generate synthetic data
        X, y = make_classification(n_samples=100, n_features=20, n_classes=2, random_state=42)
        texts = [f"Sample {i}: {', '.join([f'f{j}={v:.3f}' for j, v in enumerate(features[:5])])}" 
                for i, features in enumerate(X)]
        
        df = pd.DataFrame({'text': texts, 'label': y})
        df.to_csv(self.dataset_path, index=False)
    
    def teardown_method(self) -> Any:
        """Cleanup test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_complete_data_loading_workflow(self) -> Any:
        """Test complete data loading workflow."""
        # Create configuration
        config = DataLoaderConfig(
            batch_size=16,
            num_workers=2,
            pin_memory=True,
            enable_caching=True,
            cache_dir=self.temp_dir
        )
        
        # Create dataset
        dataset = ThreatDetectionDataset(self.dataset_path, config)
        
        # Create DataLoader
        dataloader = DataLoaderFactory.create_dataloader(
            dataset, config, "threat_detection"
        )
        
        # Test loading
        batch_count = 0
        for batch in dataloader:
            batch_count += 1
            if batch_count >= 3:
                break
        
        assert batch_count > 0
        assert len(dataset) == 100
    
    @pytest.mark.asyncio
    async def test_async_data_loading_workflow(self) -> Any:
        """Test async data loading workflow."""
        # Create configuration
        config = DataLoaderConfig(batch_size=16, num_workers=2)
        
        # Create dataset
        dataset = ThreatDetectionDataset(self.dataset_path, config)
        
        # Create DataLoader
        dataloader = DataLoaderFactory.create_dataloader(
            dataset, config, "threat_detection"
        )
        
        # Create async DataLoader
        async_dataloader = AsyncDataLoader(dataloader, max_queue_size=5)
        
        # Test async loading
        batch_count = 0
        async for batch in async_dataloader:
            batch_count += 1
            if batch_count >= 3:
                break
        
        assert batch_count > 0
    
    def test_memory_optimized_workflow(self) -> Any:
        """Test memory-optimized workflow."""
        # Create configuration
        config = DataLoaderConfig(batch_size=16, num_workers=2)
        
        # Create dataset
        dataset = ThreatDetectionDataset(self.dataset_path, config)
        
        # Create DataLoader
        dataloader = DataLoaderFactory.create_dataloader(
            dataset, config, "threat_detection"
        )
        
        # Create memory-optimized DataLoader
        memory_dataloader = MemoryOptimizedDataLoader(dataloader, max_memory_usage=0.8)
        
        # Test loading
        batch_count = 0
        for batch in memory_dataloader:
            batch_count += 1
            if batch_count >= 3:
                break
        
        assert batch_count > 0
        
        # Get performance report
        report = memory_dataloader.get_performance_report()
        assert "total_batches" in report


class TestPerformanceTests:
    """Performance tests."""
    
    def setup_method(self) -> Any:
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create large test dataset
        self.dataset_path = os.path.join(self.temp_dir, "large_test_data.csv")
        
        # Generate larger synthetic data
        X, y = make_classification(n_samples=1000, n_features=50, n_classes=2, random_state=42)
        texts = [f"Sample {i}: {', '.join([f'f{j}={v:.3f}' for j, v in enumerate(features[:10])])}" 
                for i, features in enumerate(X)]
        
        df = pd.DataFrame({'text': texts, 'label': y})
        df.to_csv(self.dataset_path, index=False)
    
    def teardown_method(self) -> Any:
        """Cleanup test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_large_dataset_performance(self) -> Any:
        """Test performance with large dataset."""
        config = DataLoaderConfig(
            batch_size=64,
            num_workers=4,
            pin_memory=True
        )
        
        dataset = ThreatDetectionDataset(self.dataset_path, config)
        dataloader = DataLoaderFactory.create_dataloader(
            dataset, config, "threat_detection"
        )
        
        # Benchmark performance
        start_time = time.time()
        batch_count = 0
        
        for batch in dataloader:
            batch_count += 1
            if batch_count >= 20:
                break
        
        total_time = time.time() - start_time
        
        # Performance assertions
        assert total_time < 30.0  # Should complete within 30 seconds
        assert batch_count == 20
        
        throughput = batch_count / total_time
        assert throughput > 0.5  # At least 0.5 batches per second
    
    def test_memory_usage_performance(self) -> Any:
        """Test memory usage performance."""
        
        config = DataLoaderConfig(
            batch_size=32,
            num_workers=2,
            pin_memory=True
        )
        
        dataset = ThreatDetectionDataset(self.dataset_path, config)
        dataloader = DataLoaderFactory.create_dataloader(
            dataset, config, "threat_detection"
        )
        
        # Monitor memory usage
        process = psutil.Process()
        memory_before = process.memory_info().rss / (1024 * 1024)  # MB
        
        batch_count = 0
        for batch in dataloader:
            batch_count += 1
            if batch_count >= 10:
                break
        
        memory_after = process.memory_info().rss / (1024 * 1024)  # MB
        memory_increase = memory_after - memory_before
        
        # Memory usage assertions
        assert memory_increase < 500  # Should not increase by more than 500MB
        assert batch_count == 10


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_dataset_path(self) -> Any:
        """Test handling of invalid dataset paths."""
        config = DataLoaderConfig()
        
        with pytest.raises(Exception):
            ThreatDetectionDataset("nonexistent.csv", config)
    
    def test_empty_dataset(self) -> Any:
        """Test handling of empty datasets."""
        temp_dir = tempfile.mkdtemp()
        try:
            empty_dataset_path = os.path.join(temp_dir, "empty.csv")
            
            # Create empty dataset
            df = pd.DataFrame(columns=['text', 'label'])
            df.to_csv(empty_dataset_path, index=False)
            
            config = DataLoaderConfig()
            
            with pytest.raises(ValueError, match="Dataset is empty"):
                ThreatDetectionDataset(empty_dataset_path, config)
        finally:
            shutil.rmtree(temp_dir)
    
    def test_malformed_dataset(self) -> Any:
        """Test handling of malformed datasets."""
        temp_dir = tempfile.mkdtemp()
        try:
            malformed_dataset_path = os.path.join(temp_dir, "malformed.csv")
            
            # Create malformed dataset
            with open(malformed_dataset_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                f.write("invalid,csv,format\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                f.write("no,proper,columns\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            config = DataLoaderConfig()
            
            with pytest.raises(Exception):
                ThreatDetectionDataset(malformed_dataset_path, config)
        finally:
            shutil.rmtree(temp_dir)
    
    def test_invalid_config(self) -> Any:
        """Test handling of invalid configurations."""
        with pytest.raises(Exception):
            DataLoaderConfig(batch_size=-1)  # Invalid batch size
    
    def test_cache_directory_creation(self) -> Any:
        """Test cache directory creation."""
        temp_dir = tempfile.mkdtemp()
        try:
            cache_dir = os.path.join(temp_dir, "new_cache")
            
            # Create mock dataset
            mock_dataset = MagicMock()
            mock_dataset.__len__ = MagicMock(return_value=10)
            mock_dataset.__getitem__ = MagicMock(return_value=("data", "label"))
            
            cached_dataset = CachedDataset(mock_dataset, cache_dir, cache_size=5)
            
            # Check that cache directory was created
            assert os.path.exists(cache_dir)
        finally:
            shutil.rmtree(temp_dir)


match __name__:
    case "__main__":
    pytest.main([__file__, "-v"]) 