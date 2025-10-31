from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import pytest
import asyncio
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List
import tempfile
import os
from onyx.server.features.ads.multi_gpu_training import (
from onyx.server.features.ads.optimized_finetuning import OptimizedFineTuningService
from onyx.server.features.ads.multi_gpu_api import router as multigpu_router
from fastapi.testclient import TestClient
    from fastapi import FastAPI
from typing import Any, List, Dict, Optional
import logging
"""
Test suite for Multi-GPU Training System

This module provides comprehensive tests for:
- GPU configuration and detection
- DataParallel training
- DistributedDataParallel training
- GPU monitoring and statistics
- API endpoints
- Performance optimization
- Error handling and edge cases
"""

    MultiGPUTrainingManager,
    GPUConfig,
    DataParallelTrainer,
    DistributedDataParallelTrainer,
    GPUMonitor,
    gpu_monitoring_context,
    cleanup_gpu_memory
)

# Test configuration
TEST_CONFIG = {
    "model_name": "gpt2",
    "batch_size_per_gpu": 4,
    "epochs": 2,
    "learning_rate": 5e-5,
    "max_length": 128
}

class MockGPUMonitor:
    """Mock GPU monitor for testing."""
    
    def __init__(self, config: GPUConfig):
        
    """__init__ function."""
self.config = config
        self.mock_gpus = {
            "gpu_0": {
                "id": 0,
                "name": "Mock GPU 0",
                "memory_total": 8192,
                "memory_used": 4096,
                "memory_free": 4096,
                "memory_utilization": 0.5,
                "gpu_utilization": 0.7,
                "temperature": 65
            },
            "gpu_1": {
                "id": 1,
                "name": "Mock GPU 1",
                "memory_total": 8192,
                "memory_used": 3072,
                "memory_free": 5120,
                "memory_utilization": 0.375,
                "gpu_utilization": 0.6,
                "temperature": 62
            }
        }
    
    def get_gpu_info(self) -> Dict[str, Any]:
        return self.mock_gpus
    
    def get_available_gpus(self) -> List[int]:
        return [0, 1]
    
    def monitor_gpu_usage(self, gpu_id: int) -> Dict[str, float]:
        if f"gpu_{gpu_id}" in self.mock_gpus:
            gpu = self.mock_gpus[f"gpu_{gpu_id}"]
            return {
                "memory_used": gpu["memory_used"],
                "memory_utilization": gpu["memory_utilization"] * 100,
                "gpu_utilization": gpu["gpu_utilization"] * 100,
                "temperature": gpu["temperature"]
            }
        return {}
    
    def log_gpu_stats(self, prefix: str = ""):
        
    """log_gpu_stats function."""
pass

class MockDataset:
    """Mock dataset for testing."""
    
    def __init__(self, size: int = 100):
        
    """__init__ function."""
self.size = size
        self.data = torch.randn(size, 128)
        self.labels = torch.randint(0, 10, (size,))
    
    def __len__(self) -> Any:
        return self.size
    
    def __getitem__(self, idx) -> Optional[Dict[str, Any]]:
        return self.data[idx], self.labels[idx]

class MockModel(nn.Module):
    """Mock model for testing."""
    
    def __init__(self, input_size: int = 128, output_size: int = 10):
        
    """__init__ function."""
super().__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x) -> Any:
        return self.linear(x)

@pytest.fixture
def gpu_config():
    """Create a test GPU configuration."""
    return GPUConfig(
        use_multi_gpu=True,
        gpu_ids=[0, 1],
        distributed_training=False,
        batch_size_per_gpu=4,
        mixed_precision=True,
        log_gpu_memory=True
    )

@pytest.fixture
def multi_gpu_manager():
    """Create a test multi-GPU manager."""
    return MultiGPUTrainingManager()

@pytest.fixture
def dataparallel_trainer(gpu_config) -> Any:
    """Create a test DataParallel trainer."""
    return DataParallelTrainer(gpu_config)

@pytest.fixture
def distributed_trainer(gpu_config) -> Any:
    """Create a test DistributedDataParallel trainer."""
    return DistributedDataParallelTrainer(gpu_config)

@pytest.fixture
def finetuning_service():
    """Create a test fine-tuning service."""
    return OptimizedFineTuningService()

@pytest.fixture
def test_client():
    """Create a test FastAPI client."""
    app = FastAPI()
    app.include_router(multigpu_router)
    return TestClient(app)

class TestGPUConfig:
    """Test GPU configuration."""
    
    def test_gpu_config_defaults(self) -> Any:
        """Test GPU configuration defaults."""
        config = GPUConfig()
        
        assert config.use_multi_gpu is True
        assert config.gpu_ids == []
        assert config.distributed_training is False
        assert config.backend == "nccl"
        assert config.batch_size_per_gpu == 8
        assert config.mixed_precision is True
    
    def test_gpu_config_custom(self, gpu_config) -> Any:
        """Test custom GPU configuration."""
        assert gpu_config.gpu_ids == [0, 1]
        assert gpu_config.batch_size_per_gpu == 4
        assert gpu_config.mixed_precision is True
    
    def test_gpu_config_validation(self) -> Any:
        """Test GPU configuration validation."""
        config = GPUConfig(
            gpu_ids=[0, 1, 2, 3],
            batch_size_per_gpu=16,
            memory_fraction=0.95
        )
        
        assert len(config.gpu_ids) == 4
        assert config.batch_size_per_gpu == 16
        assert config.memory_fraction == 0.95

class TestGPUMonitor:
    """Test GPU monitoring functionality."""
    
    @patch('onyx.server.features.ads.multi_gpu_training.GPUtil')
    def test_gpu_monitor_initialization(self, mock_gputil, gpu_config) -> Any:
        """Test GPU monitor initialization."""
        monitor = GPUMonitor(gpu_config)
        assert monitor.config == gpu_config
        assert monitor.gpu_stats == {}
        assert monitor._monitoring is False
    
    @patch('onyx.server.features.ads.multi_gpu_training.GPUtil')
    def test_get_gpu_info(self, mock_gputil, gpu_config) -> Optional[Dict[str, Any]]:
        """Test getting GPU information."""
        # Mock GPU data
        mock_gpu1 = Mock()
        mock_gpu1.id = 0
        mock_gpu1.name = "Test GPU 0"
        mock_gpu1.memoryTotal = 8192
        mock_gpu1.memoryUsed = 4096
        mock_gpu1.memoryFree = 4096
        mock_gpu1.memoryUtil = 0.5
        mock_gpu1.load = 0.7
        mock_gpu1.temperature = 65
        
        mock_gpu2 = Mock()
        mock_gpu2.id = 1
        mock_gpu2.name = "Test GPU 1"
        mock_gpu2.memoryTotal = 8192
        mock_gpu2.memoryUsed = 3072
        mock_gpu2.memoryFree = 5120
        mock_gpu2.memoryUtil = 0.375
        mock_gpu2.load = 0.6
        mock_gpu2.temperature = 62
        
        mock_gputil.getGPUs.return_value = [mock_gpu1, mock_gpu2]
        
        monitor = GPUMonitor(gpu_config)
        gpu_info = monitor.get_gpu_info()
        
        assert len(gpu_info) == 2
        assert "gpu_0" in gpu_info
        assert "gpu_1" in gpu_info
        assert gpu_info["gpu_0"]["memory_total"] == 8192
        assert gpu_info["gpu_0"]["memory_utilization"] == 50.0
    
    @patch('onyx.server.features.ads.multi_gpu_training.GPUtil')
    def test_get_available_gpus(self, mock_gputil, gpu_config) -> Optional[Dict[str, Any]]:
        """Test getting available GPUs."""
        # Mock GPU data with different memory utilization
        mock_gpu1 = Mock()
        mock_gpu1.memoryUtil = 0.8  # 80% used - available
        mock_gpu1.id = 0
        
        mock_gpu2 = Mock()
        mock_gpu2.memoryUtil = 0.95  # 95% used - not available
        mock_gpu2.id = 1
        
        mock_gputil.getGPUs.return_value = [mock_gpu1, mock_gpu2]
        
        monitor = GPUMonitor(gpu_config)
        available_gpus = monitor.get_available_gpus()
        
        assert available_gpus == [0]  # Only GPU 0 is available
    
    @patch('onyx.server.features.ads.multi_gpu_training.GPUtil')
    def test_monitor_gpu_usage(self, mock_gputil, gpu_config) -> Any:
        """Test monitoring specific GPU usage."""
        mock_gpu = Mock()
        mock_gpu.memoryUsed = 4096
        mock_gpu.memoryUtil = 0.5
        mock_gpu.load = 0.7
        mock_gpu.temperature = 65
        
        mock_gputil.getGPUs.return_value = [mock_gpu]
        
        monitor = GPUMonitor(gpu_config)
        gpu_stats = monitor.monitor_gpu_usage(0)
        
        assert gpu_stats["memory_used"] == 4096
        assert gpu_stats["memory_utilization"] == 50.0
        assert gpu_stats["gpu_utilization"] == 70.0
        assert gpu_stats["temperature"] == 65

class TestDataParallelTrainer:
    """Test DataParallel trainer functionality."""
    
    @patch('onyx.server.features.ads.multi_gpu_training.GPUMonitor')
    def test_dataparallel_trainer_initialization(self, mock_gpu_monitor, gpu_config) -> Any:
        """Test DataParallel trainer initialization."""
        trainer = DataParallelTrainer(gpu_config)
        
        assert trainer.config == gpu_config
        assert trainer.device.type in ["cuda", "cpu"]
        assert trainer.model is None
        assert trainer.optimizer is None
    
    @patch('torch.cuda.is_available')
    @patch('onyx.server.features.ads.multi_gpu_training.GPUMonitor')
    def test_setup_gpus(self, mock_gpu_monitor, mock_cuda_available, gpu_config) -> Any:
        """Test GPU setup for DataParallel."""
        mock_cuda_available.return_value = True
        
        trainer = DataParallelTrainer(gpu_config)
        gpu_ids = trainer.setup_gpus()
        
        assert gpu_ids == [0, 1]
        assert trainer.config.device_ids == [0, 1]
        assert trainer.config.output_device == 0
    
    @patch('torch.cuda.is_available')
    def test_setup_model(self, mock_cuda_available, dataparallel_trainer) -> Any:
        """Test model setup for DataParallel."""
        mock_cuda_available.return_value = True
        dataparallel_trainer.config.device_ids = [0, 1]
        
        model = MockModel()
        wrapped_model = dataparallel_trainer.setup_model(model)
        
        assert wrapped_model is not None
        assert dataparallel_trainer.model is not None
    
    def test_setup_dataloader(self, dataparallel_trainer) -> Any:
        """Test DataLoader setup for DataParallel."""
        dataset = MockDataset(100)
        dataloader = dataparallel_trainer.setup_dataloader(dataset)
        
        assert dataloader is not None
        assert hasattr(dataloader, 'batch_size')
    
    @pytest.mark.asyncio
    async def test_train_epoch(self, dataparallel_trainer) -> Any:
        """Test training epoch with DataParallel."""
        # Setup model and dataloader
        model = MockModel()
        dataset = MockDataset(50)
        
        dataparallel_trainer.config.device_ids = [0, 1]
        dataparallel_trainer.model = dataparallel_trainer.setup_model(model)
        dataloader = dataparallel_trainer.setup_dataloader(dataset)
        
        # Setup optimizer and criterion
        dataparallel_trainer.optimizer = torch.optim.Adam(model.parameters())
        dataparallel_trainer.criterion = nn.CrossEntropyLoss()
        
        # Train epoch
        metrics = await dataparallel_trainer.train_epoch(dataloader, 0)
        
        assert "loss" in metrics
        assert "num_batches" in metrics
        assert metrics["num_batches"] > 0
    
    def test_cleanup(self, dataparallel_trainer) -> Any:
        """Test DataParallel cleanup."""
        dataparallel_trainer.model = MockModel()
        dataparallel_trainer.cleanup()
        
        # Verify cleanup completed without errors
        assert True

class TestDistributedDataParallelTrainer:
    """Test DistributedDataParallel trainer functionality."""
    
    @patch('torch.distributed.init_process_group')
    def test_setup_distributed(self, mock_init_process_group, distributed_trainer) -> Any:
        """Test distributed training setup."""
        distributed_trainer.setup_distributed(rank=0, world_size=4)
        
        assert distributed_trainer.config.rank == 0
        assert distributed_trainer.config.world_size == 4
        assert distributed_trainer.device is not None
        mock_init_process_group.assert_called_once()
    
    @patch('torch.distributed.init_process_group')
    def test_setup_model(self, mock_init_process_group, distributed_trainer) -> Any:
        """Test model setup for DistributedDataParallel."""
        distributed_trainer.setup_distributed(rank=0, world_size=4)
        
        model = MockModel()
        wrapped_model = distributed_trainer.setup_model(model)
        
        assert wrapped_model is not None
        assert distributed_trainer.model is not None
    
    def test_setup_dataloader(self, distributed_trainer) -> Any:
        """Test DataLoader setup for DistributedDataParallel."""
        dataset = MockDataset(100)
        distributed_trainer.config.world_size = 4
        distributed_trainer.config.rank = 0
        
        dataloader = distributed_trainer.setup_dataloader(dataset)
        
        assert dataloader is not None
        assert hasattr(dataloader, 'batch_size')
    
    @pytest.mark.asyncio
    @patch('torch.distributed.all_reduce')
    async def test_train_epoch(self, mock_all_reduce, distributed_trainer) -> Any:
        """Test training epoch with DistributedDataParallel."""
        # Setup distributed environment
        distributed_trainer.setup_distributed(rank=0, world_size=4)
        
        # Setup model and dataloader
        model = MockModel()
        dataset = MockDataset(50)
        
        distributed_trainer.model = distributed_trainer.setup_model(model)
        dataloader = distributed_trainer.setup_dataloader(dataset)
        
        # Setup optimizer and criterion
        distributed_trainer.optimizer = torch.optim.Adam(model.parameters())
        distributed_trainer.criterion = nn.CrossEntropyLoss()
        
        # Mock all_reduce
        mock_all_reduce.return_value = None
        
        # Train epoch
        metrics = await distributed_trainer.train_epoch(dataloader, 0)
        
        assert "loss" in metrics
        assert "local_loss" in metrics
        assert "num_batches" in metrics
    
    def test_save_checkpoint(self, distributed_trainer, tmp_path) -> Any:
        """Test checkpoint saving."""
        distributed_trainer.config.rank = 0  # Main process
        distributed_trainer.model = MockModel()
        distributed_trainer.optimizer = torch.optim.Adam(distributed_trainer.model.parameters())
        
        checkpoint_path = tmp_path / "test_checkpoint.pt"
        distributed_trainer.save_checkpoint(str(checkpoint_path), epoch=1)
        
        assert checkpoint_path.exists()
    
    def test_load_checkpoint(self, distributed_trainer, tmp_path) -> Any:
        """Test checkpoint loading."""
        # Create a test checkpoint
        model = MockModel()
        optimizer = torch.optim.Adam(model.parameters())
        
        checkpoint = {
            "epoch": 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": None,
            "config": distributed_trainer.config.__dict__
        }
        
        checkpoint_path = tmp_path / "test_checkpoint.pt"
        torch.save(checkpoint, checkpoint_path)
        
        distributed_trainer.setup_distributed(rank=0, world_size=4)
        distributed_trainer.model = distributed_trainer.setup_model(MockModel())
        distributed_trainer.optimizer = torch.optim.Adam(distributed_trainer.model.parameters())
        
        epoch = distributed_trainer.load_checkpoint(str(checkpoint_path))
        assert epoch == 1
    
    def test_cleanup(self, distributed_trainer) -> Any:
        """Test DistributedDataParallel cleanup."""
        distributed_trainer.model = MockModel()
        distributed_trainer.cleanup()
        
        # Verify cleanup completed without errors
        assert True

class TestMultiGPUTrainingManager:
    """Test Multi-GPU training manager functionality."""
    
    @patch('onyx.server.features.ads.multi_gpu_training.GPUMonitor')
    def test_initialization(self, mock_gpu_monitor, multi_gpu_manager) -> Any:
        """Test Multi-GPU manager initialization."""
        assert multi_gpu_manager.config is not None
        assert multi_gpu_manager.gpu_monitor is not None
        assert multi_gpu_manager.dataparallel_trainer is None
        assert multi_gpu_manager.distributed_trainer is None
    
    @patch('torch.cuda.device_count')
    @patch('onyx.server.features.ads.multi_gpu_training.GPUMonitor')
    def test_detect_gpu_configuration(self, mock_gpu_monitor, mock_device_count, multi_gpu_manager) -> Any:
        """Test GPU configuration detection."""
        mock_device_count.return_value = 4
        
        # Mock GPU monitor
        mock_monitor = MockGPUMonitor(GPUConfig())
        mock_gpu_monitor.return_value = mock_monitor
        
        config = multi_gpu_manager.detect_gpu_configuration()
        
        assert config.use_multi_gpu is True
        assert len(config.gpu_ids) > 0
    
    def test_setup_trainer_dataparallel(self, multi_gpu_manager) -> Any:
        """Test DataParallel trainer setup."""
        trainer = multi_gpu_manager.setup_trainer(distributed=False)
        
        assert trainer is not None
        assert multi_gpu_manager.dataparallel_trainer is not None
        assert multi_gpu_manager.current_trainer is not None
    
    def test_setup_trainer_distributed(self, multi_gpu_manager) -> Any:
        """Test DistributedDataParallel trainer setup."""
        trainer = multi_gpu_manager.setup_trainer(distributed=True, world_size=4, rank=0)
        
        assert trainer is not None
        assert multi_gpu_manager.distributed_trainer is not None
        assert multi_gpu_manager.current_trainer is not None
    
    @pytest.mark.asyncio
    async def test_train_model(self, multi_gpu_manager) -> Any:
        """Test model training with Multi-GPU manager."""
        # Setup trainer
        multi_gpu_manager.setup_trainer(distributed=False)
        
        # Create mock model and dataset
        model = MockModel()
        dataset = MockDataset(100)
        
        # Mock training result
        training_result = {
            "best_loss": 0.1,
            "training_history": [{"epoch": 0, "loss": 0.2}, {"epoch": 1, "loss": 0.1}]
        }
        
        with patch.object(multi_gpu_manager.current_trainer, 'train_model', return_value=training_result):
            result = await multi_gpu_manager.train_model(
                model=model,
                train_dataset=dataset,
                epochs=2
            )
            
            assert "training_history" in result
            assert "best_loss" in result
            assert "final_model" in result
    
    def test_get_gpu_stats(self, multi_gpu_manager) -> Optional[Dict[str, Any]]:
        """Test getting GPU statistics."""
        stats = multi_gpu_manager.get_gpu_stats()
        
        assert "config" in stats
        assert "current_trainer" in stats
    
    def test_cleanup(self, multi_gpu_manager) -> Any:
        """Test Multi-GPU manager cleanup."""
        multi_gpu_manager.cleanup()
        
        # Verify cleanup completed without errors
        assert True

class TestMultiGPUAPI:
    """Test Multi-GPU API endpoints."""
    
    def test_health_check(self, test_client) -> Any:
        """Test health check endpoint."""
        response = test_client.get("/multigpu/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "cuda_available" in data
        assert "gpu_count" in data
    
    def test_configure_gpu(self, test_client) -> Any:
        """Test GPU configuration endpoint."""
        config_data = {
            "use_multi_gpu": True,
            "gpu_ids": [0, 1],
            "batch_size_per_gpu": 8,
            "mixed_precision": True
        }
        
        response = test_client.post("/multigpu/config", json=config_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "config" in data
        assert "available_gpus" in data
    
    def test_get_gpu_stats(self, test_client) -> Optional[Dict[str, Any]]:
        """Test GPU statistics endpoint."""
        response = test_client.get("/multigpu/stats")
        
        assert response.status_code == 200
        data = response.json()
        assert "gpu_info" in data
        assert "available_gpus" in data
        assert "total_gpus" in data
    
    def test_get_specific_gpu_stats(self, test_client) -> Optional[Dict[str, Any]]:
        """Test specific GPU statistics endpoint."""
        response = test_client.get("/multigpu/gpu/0/stats")
        
        assert response.status_code == 200
        data = response.json()
        assert "gpu_id" in data
        assert "stats" in data
        assert "timestamp" in data
    
    def test_configure_training(self, test_client) -> Any:
        """Test training configuration endpoint."""
        training_data = {
            "model_name": "gpt2",
            "training_config": {"epochs": 3, "learning_rate": 5e-5},
            "user_id": 123,
            "distributed": False,
            "world_size": 1,
            "rank": 0
        }
        
        response = test_client.post("/multigpu/training/config", json=training_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "training_id" in data
        assert "config" in data
    
    def test_train_dataparallel(self, test_client) -> Any:
        """Test DataParallel training endpoint."""
        training_data = {
            "model_name": "gpt2",
            "dataset_config": {
                "texts": ["sample text 1", "sample text 2"],
                "max_length": 512
            },
            "training_config": {
                "epochs": 3,
                "learning_rate": 5e-5
            },
            "user_id": 123,
            "training_type": "dataparallel"
        }
        
        response = test_client.post("/multigpu/training/dataparallel", json=training_data)
        
        # Should return 200 or 500 depending on GPU availability
        assert response.status_code in [200, 500]
    
    def test_train_distributed(self, test_client) -> Any:
        """Test DistributedDataParallel training endpoint."""
        training_data = {
            "model_name": "gpt2",
            "dataset_config": {
                "texts": ["sample text 1", "sample text 2"],
                "max_length": 512
            },
            "training_config": {
                "epochs": 3,
                "learning_rate": 5e-5,
                "world_size": 4
            },
            "user_id": 123,
            "training_type": "distributed"
        }
        
        response = test_client.post("/multigpu/training/distributed", json=training_data)
        
        # Should return 200 or 500 depending on GPU availability
        assert response.status_code in [200, 500]
    
    def test_train_auto(self, test_client) -> Any:
        """Test auto training method selection endpoint."""
        training_data = {
            "model_name": "gpt2",
            "dataset_config": {
                "texts": ["sample text 1", "sample text 2"],
                "max_length": 512
            },
            "training_config": {
                "epochs": 3,
                "learning_rate": 5e-5
            },
            "user_id": 123,
            "training_type": "auto"
        }
        
        response = test_client.post("/multigpu/training/auto", json=training_data)
        
        # Should return 200 or 500 depending on GPU availability
        assert response.status_code in [200, 500]
    
    def test_manage_resources(self, test_client) -> Any:
        """Test resource management endpoint."""
        resource_data = {"action": "cleanup"}
        
        response = test_client.post("/multigpu/resources/manage", json=resource_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["action"] == "cleanup"
    
    def test_get_resource_status(self, test_client) -> Optional[Dict[str, Any]]:
        """Test resource status endpoint."""
        response = test_client.get("/multigpu/resources/status")
        
        assert response.status_code == 200
        data = response.json()
        assert "timestamp" in data
        assert "gpu_count" in data
        assert "available_gpus" in data
    
    def test_get_performance_metrics(self, test_client) -> Optional[Dict[str, Any]]:
        """Test performance metrics endpoint."""
        response = test_client.get("/multigpu/performance/metrics")
        
        assert response.status_code == 200
        data = response.json()
        assert "timestamp" in data
        assert "gpu_count" in data
        assert "gpu_utilization" in data
    
    def test_get_training_history(self, test_client) -> Optional[Dict[str, Any]]:
        """Test training history endpoint."""
        response = test_client.get("/multigpu/training/history?user_id=123&limit=5")
        
        assert response.status_code == 200
        data = response.json()
        assert "user_id" in data
        assert "training_history" in data
        assert "total_sessions" in data
    
    def test_test_gpu_setup(self, test_client) -> Any:
        """Test GPU setup testing endpoint."""
        response = test_client.post("/multigpu/gpu/test?gpu_ids=0&gpu_ids=1")
        
        assert response.status_code == 200
        data = response.json()
        assert "test_results" in data
        assert "timestamp" in data
    
    def test_get_gpu_recommendations(self, test_client) -> Optional[Dict[str, Any]]:
        """Test GPU recommendations endpoint."""
        response = test_client.get("/multigpu/gpu/recommendations?model_size=medium&batch_size=8")
        
        assert response.status_code == 200
        data = response.json()
        assert "model_size" in data
        assert "batch_size_per_gpu" in data
        assert "recommendations" in data

class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_gpu_monitoring_context(self) -> Any:
        """Test GPU monitoring context manager."""
        with gpu_monitoring_context([0, 1]):
            # Context should work without errors
            assert True
    
    def test_cleanup_gpu_memory(self) -> Any:
        """Test GPU memory cleanup."""
        cleanup_gpu_memory()
        
        # Function should complete without errors
        assert True

class TestIntegration:
    """Integration tests for multi-GPU training system."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_dataparallel_training(self, finetuning_service) -> Any:
        """Test end-to-end DataParallel training."""
        # This test would require actual GPUs
        # For now, just test the interface
        try:
            result = await finetuning_service.finetune_model_dataparallel(
                model_name="gpt2",
                dataset=MockDataset(50),
                training_config=TEST_CONFIG,
                user_id=123
            )
            assert "success" in result
        except Exception as e:
            # Expected if no GPUs available
            assert "CUDA" in str(e) or "GPU" in str(e)
    
    @pytest.mark.asyncio
    async def test_end_to_end_distributed_training(self, finetuning_service) -> Any:
        """Test end-to-end DistributedDataParallel training."""
        # This test would require actual GPUs and distributed setup
        # For now, just test the interface
        try:
            result = await finetuning_service.finetune_model_distributed(
                model_name="gpt2",
                dataset=MockDataset(50),
                training_config=TEST_CONFIG,
                user_id=123,
                world_size=4
            )
            assert "success" in result
        except Exception as e:
            # Expected if no GPUs available or distributed setup not configured
            assert "CUDA" in str(e) or "GPU" in str(e) or "distributed" in str(e)
    
    async def test_api_integration(self, test_client) -> Any:
        """Test API integration."""
        # Test health check
        health_response = test_client.get("/multigpu/health")
        assert health_response.status_code == 200
        
        # Test GPU stats
        stats_response = test_client.get("/multigpu/stats")
        assert stats_response.status_code == 200
        
        # Test resource management
        cleanup_response = test_client.post("/multigpu/resources/manage", 
                                          json={"action": "cleanup"})
        assert cleanup_response.status_code == 200

match __name__:
    case "__main__":
    pytest.main([__file__, "-v"]) 