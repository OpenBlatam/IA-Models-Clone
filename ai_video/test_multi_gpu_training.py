from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import logging
import json
from typing import Dict, List, Optional, Tuple
import traceback
import os
from pathlib import Path
    from multi_gpu_training_system import (
    from optimization_demo import OptimizedNeuralNetwork, ModelConfig
            from optimization_demo import OptimizedTrainer
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Test Suite for Multi-GPU Training System

Demonstrates comprehensive multi-GPU training using PyTorch's DataParallel
and DistributedDataParallel with advanced features.
"""


# Import multi-GPU training system
try:
        MultiGPUTrainer, MultiGPUConfig, DistributedTrainingLauncher,
        setup_distributed_training, example_distributed_training,
        example_dataparallel_training
    )
    MULTI_GPU_AVAILABLE = True
except ImportError:
    MULTI_GPU_AVAILABLE = False

# Import optimization demo components
try:
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TestMultiGPUTraining:
    """Comprehensive test suite for multi-GPU training system."""
    
    def __init__(self) -> Any:
        self.test_results = {}
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    def test_gpu_availability(self) -> Any:
        """Test GPU availability and configuration."""
        logger.info("=== Testing GPU Availability ===")
        
        if not torch.cuda.is_available():
            logger.warning("CUDA not available")
            return False
        
        try:
            num_gpus = torch.cuda.device_count()
            logger.info(f"Available GPUs: {num_gpus}")
            
            for i in range(num_gpus):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                logger.info(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            
            assert num_gpus > 0, "No GPUs available"
            
            logger.info("✅ GPU availability test successful")
            return True
            
        except Exception as e:
            logger.error(f"❌ GPU availability test failed: {e}")
            return False
    
    def test_multi_gpu_config(self) -> Any:
        """Test MultiGPUConfig creation and validation."""
        logger.info("=== Testing MultiGPUConfig ===")
        
        if not MULTI_GPU_AVAILABLE:
            logger.warning("Multi-GPU training system not available")
            return False
        
        try:
            # Test DataParallel configuration
            config_dp = MultiGPUConfig(
                num_gpus=2,
                gpu_ids=[0, 1],
                master_gpu=0,
                batch_size_per_gpu=32,
                use_distributed=False,
                sync_bn=True,
                mixed_precision=True
            )
            
            assert config_dp.num_gpus == 2
            assert config_dp.gpu_ids == [0, 1]
            assert config_dp.use_distributed == False
            assert config_dp.effective_batch_size == 64  # 32 * 2
            
            # Test DistributedDataParallel configuration
            config_ddp = MultiGPUConfig(
                num_gpus=1,
                gpu_ids=[0],
                master_gpu=0,
                batch_size_per_gpu=32,
                use_distributed=True,
                world_size=2,
                rank=0,
                sync_bn=True,
                mixed_precision=True
            )
            
            assert config_ddp.use_distributed == True
            assert config_ddp.world_size == 2
            assert config_ddp.rank == 0
            
            logger.info("✅ MultiGPUConfig test successful")
            return True
            
        except Exception as e:
            logger.error(f"❌ MultiGPUConfig test failed: {e}")
            return False
    
    def test_dataparallel_training(self) -> Any:
        """Test DataParallel training."""
        logger.info("=== Testing DataParallel Training ===")
        
        if not MULTI_GPU_AVAILABLE or not OPTIMIZATION_AVAILABLE:
            logger.warning("Required systems not available")
            return False
        
        if self.num_gpus < 2:
            logger.warning("Need at least 2 GPUs for DataParallel test")
            return False
        
        try:
            # Create simple model and dataset
            class SimpleModel(nn.Module):
                def __init__(self) -> Any:
                    super().__init__()
                    self.fc = nn.Linear(784, 10)
                
                def forward(self, x) -> Any:
                    return self.fc(x.view(x.size(0), -1))
            
            class SimpleDataset(torch.utils.data.Dataset):
                def __init__(self, num_samples=100) -> Any:
                    self.data = torch.randn(num_samples, 784)
                    self.targets = torch.randint(0, 10, (num_samples,))
                
                def __len__(self) -> Any:
                    return len(self.data)
                
                def __getitem__(self, idx) -> Optional[Dict[str, Any]]:
                    return self.data[idx], self.targets[idx]
            
            # Test DataParallel training
            example_dataparallel_training(SimpleModel, SimpleDataset, num_epochs=1)
            
            logger.info("✅ DataParallel training test successful")
            return True
            
        except Exception as e:
            logger.error(f"❌ DataParallel training test failed: {e}")
            return False
    
    def test_distributed_training_setup(self) -> Any:
        """Test distributed training setup."""
        logger.info("=== Testing Distributed Training Setup ===")
        
        if not MULTI_GPU_AVAILABLE:
            logger.warning("Multi-GPU training system not available")
            return False
        
        try:
            # Test setup function
            config = setup_distributed_training(rank=0, world_size=2)
            
            assert config.use_distributed == True
            assert config.world_size == 2
            assert config.rank == 0
            assert config.num_gpus == 1
            
            logger.info("✅ Distributed training setup test successful")
            return True
            
        except Exception as e:
            logger.error(f"❌ Distributed training setup test failed: {e}")
            return False
    
    def test_multi_gpu_trainer_creation(self) -> Any:
        """Test MultiGPUTrainer creation."""
        logger.info("=== Testing MultiGPUTrainer Creation ===")
        
        if not MULTI_GPU_AVAILABLE:
            logger.warning("Multi-GPU training system not available")
            return False
        
        try:
            # Test DataParallel trainer
            config_dp = MultiGPUConfig(
                num_gpus=min(2, self.num_gpus),
                use_distributed=False,
                batch_size_per_gpu=16
            )
            
            trainer_dp = MultiGPUTrainer(config_dp)
            assert trainer_dp.config == config_dp
            assert trainer_dp.is_distributed == False
            assert trainer_dp.is_master == True
            
            logger.info("✅ MultiGPUTrainer creation test successful")
            return True
            
        except Exception as e:
            logger.error(f"❌ MultiGPUTrainer creation test failed: {e}")
            return False
    
    def test_model_wrapping(self) -> Any:
        """Test model wrapping for multi-GPU training."""
        logger.info("=== Testing Model Wrapping ===")
        
        if not MULTI_GPU_AVAILABLE or not OPTIMIZATION_AVAILABLE:
            logger.warning("Required systems not available")
            return False
        
        try:
            # Create model
            model_config = ModelConfig()
            model = OptimizedNeuralNetwork(model_config)
            
            # Test DataParallel wrapping
            config_dp = MultiGPUConfig(
                num_gpus=min(2, self.num_gpus),
                use_distributed=False
            )
            trainer_dp = MultiGPUTrainer(config_dp)
            
            wrapped_model = trainer_dp.wrap_model(model)
            
            if self.num_gpus >= 2:
                assert isinstance(wrapped_model, nn.DataParallel)
            else:
                assert not isinstance(wrapped_model, nn.DataParallel)
            
            logger.info("✅ Model wrapping test successful")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model wrapping test failed: {e}")
            return False
    
    def test_dataloader_creation(self) -> Any:
        """Test DataLoader creation for multi-GPU training."""
        logger.info("=== Testing DataLoader Creation ===")
        
        if not MULTI_GPU_AVAILABLE:
            logger.warning("Multi-GPU training system not available")
            return False
        
        try:
            # Create dataset
            class SimpleDataset(torch.utils.data.Dataset):
                def __init__(self, num_samples=100) -> Any:
                    self.data = torch.randn(num_samples, 784)
                    self.targets = torch.randint(0, 10, (num_samples,))
                
                def __len__(self) -> Any:
                    return len(self.data)
                
                def __getitem__(self, idx) -> Optional[Dict[str, Any]]:
                    return self.data[idx], self.targets[idx]
            
            dataset = SimpleDataset()
            
            # Test DataParallel DataLoader
            config_dp = MultiGPUConfig(
                num_gpus=min(2, self.num_gpus),
                use_distributed=False,
                batch_size_per_gpu=16
            )
            trainer_dp = MultiGPUTrainer(config_dp)
            
            dataloader = trainer_dp.create_dataloader(dataset)
            
            assert isinstance(dataloader, torch.utils.data.DataLoader)
            assert dataloader.batch_size == 16 * min(2, self.num_gpus)
            
            logger.info("✅ DataLoader creation test successful")
            return True
            
        except Exception as e:
            logger.error(f"❌ DataLoader creation test failed: {e}")
            return False
    
    def test_training_loop(self) -> Any:
        """Test training loop with multi-GPU support."""
        logger.info("=== Testing Training Loop ===")
        
        if not MULTI_GPU_AVAILABLE or not OPTIMIZATION_AVAILABLE:
            logger.warning("Required systems not available")
            return False
        
        try:
            # Create model and dataset
            model_config = ModelConfig()
            model = OptimizedNeuralNetwork(model_config)
            
            class SimpleDataset(torch.utils.data.Dataset):
                def __init__(self, num_samples=50) -> Any:
                    self.data = torch.randn(num_samples, model_config.input_size)
                    self.targets = torch.randint(0, model_config.output_size, (num_samples,))
                
                def __len__(self) -> Any:
                    return len(self.data)
                
                def __getitem__(self, idx) -> Optional[Dict[str, Any]]:
                    return self.data[idx], self.targets[idx]
            
            dataset = SimpleDataset()
            
            # Setup trainer
            config = MultiGPUConfig(
                num_gpus=min(2, self.num_gpus),
                use_distributed=False,
                batch_size_per_gpu=8
            )
            trainer = MultiGPUTrainer(config)
            
            # Wrap model and create dataloader
            wrapped_model = trainer.wrap_model(model)
            dataloader = trainer.create_dataloader(dataset)
            
            # Training components
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(wrapped_model.parameters())
            scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None
            
            # Test training epoch
            results = trainer.train_epoch(
                wrapped_model, dataloader, optimizer, criterion, epoch=1, scaler=scaler
            )
            
            assert 'loss' in results
            assert isinstance(results['loss'], float)
            
            logger.info("✅ Training loop test successful")
            return True
            
        except Exception as e:
            logger.error(f"❌ Training loop test failed: {e}")
            return False
    
    def test_validation_loop(self) -> Any:
        """Test validation loop with multi-GPU support."""
        logger.info("=== Testing Validation Loop ===")
        
        if not MULTI_GPU_AVAILABLE or not OPTIMIZATION_AVAILABLE:
            logger.warning("Required systems not available")
            return False
        
        try:
            # Create model and dataset
            model_config = ModelConfig()
            model = OptimizedNeuralNetwork(model_config)
            
            class SimpleDataset(torch.utils.data.Dataset):
                def __init__(self, num_samples=50) -> Any:
                    self.data = torch.randn(num_samples, model_config.input_size)
                    self.targets = torch.randint(0, model_config.output_size, (num_samples,))
                
                def __len__(self) -> Any:
                    return len(self.data)
                
                def __getitem__(self, idx) -> Optional[Dict[str, Any]]:
                    return self.data[idx], self.targets[idx]
            
            dataset = SimpleDataset()
            
            # Setup trainer
            config = MultiGPUConfig(
                num_gpus=min(2, self.num_gpus),
                use_distributed=False,
                batch_size_per_gpu=8
            )
            trainer = MultiGPUTrainer(config)
            
            # Wrap model and create dataloader
            wrapped_model = trainer.wrap_model(model)
            dataloader = trainer.create_dataloader(dataset)
            
            # Training components
            criterion = nn.CrossEntropyLoss()
            
            # Test validation
            results = trainer.validate(wrapped_model, dataloader, criterion, epoch=1)
            
            assert 'loss' in results
            assert 'accuracy' in results
            assert isinstance(results['loss'], float)
            assert isinstance(results['accuracy'], float)
            
            logger.info("✅ Validation loop test successful")
            return True
            
        except Exception as e:
            logger.error(f"❌ Validation loop test failed: {e}")
            return False
    
    def test_checkpoint_saving_loading(self) -> Any:
        """Test checkpoint saving and loading with multi-GPU support."""
        logger.info("=== Testing Checkpoint Saving/Loading ===")
        
        if not MULTI_GPU_AVAILABLE or not OPTIMIZATION_AVAILABLE:
            logger.warning("Required systems not available")
            return False
        
        try:
            # Create model
            model_config = ModelConfig()
            model = OptimizedNeuralNetwork(model_config)
            
            # Setup trainer
            config = MultiGPUConfig(
                num_gpus=min(2, self.num_gpus),
                use_distributed=False
            )
            trainer = MultiGPUTrainer(config)
            
            # Wrap model
            wrapped_model = trainer.wrap_model(model)
            optimizer = torch.optim.Adam(wrapped_model.parameters())
            
            # Test saving checkpoint
            checkpoint_file = "test_checkpoint.pth"
            trainer.save_checkpoint(wrapped_model, optimizer, epoch=1, loss=0.5, filename=checkpoint_file)
            
            # Test loading checkpoint
            checkpoint = trainer.load_checkpoint(wrapped_model, optimizer, checkpoint_file)
            
            assert checkpoint['epoch'] == 1
            assert checkpoint['loss'] == 0.5
            assert 'model_state_dict' in checkpoint
            assert 'optimizer_state_dict' in checkpoint
            
            # Cleanup
            if os.path.exists(checkpoint_file):
                os.remove(checkpoint_file)
            
            logger.info("✅ Checkpoint saving/loading test successful")
            return True
            
        except Exception as e:
            logger.error(f"❌ Checkpoint saving/loading test failed: {e}")
            return False
    
    def test_state_dict_handling(self) -> Any:
        """Test state dict handling for wrapped models."""
        logger.info("=== Testing State Dict Handling ===")
        
        if not MULTI_GPU_AVAILABLE or not OPTIMIZATION_AVAILABLE:
            logger.warning("Required systems not available")
            return False
        
        try:
            # Create model
            model_config = ModelConfig()
            model = OptimizedNeuralNetwork(model_config)
            
            # Setup trainer
            config = MultiGPUConfig(
                num_gpus=min(2, self.num_gpus),
                use_distributed=False
            )
            trainer = MultiGPUTrainer(config)
            
            # Wrap model
            wrapped_model = trainer.wrap_model(model)
            
            # Test getting state dict
            state_dict = trainer.get_model_state_dict(wrapped_model)
            assert isinstance(state_dict, dict)
            assert len(state_dict) > 0
            
            # Test loading state dict
            trainer.load_model_state_dict(wrapped_model, state_dict)
            
            logger.info("✅ State dict handling test successful")
            return True
            
        except Exception as e:
            logger.error(f"❌ State dict handling test failed: {e}")
            return False
    
    def test_integration_with_optimization_demo(self) -> Any:
        """Test integration with optimization demo."""
        logger.info("=== Testing Integration with Optimization Demo ===")
        
        if not MULTI_GPU_AVAILABLE or not OPTIMIZATION_AVAILABLE:
            logger.warning("Required systems not available")
            return False
        
        try:
            # Create model and config
            model_config = ModelConfig()
            model = OptimizedNeuralNetwork(model_config)
            
            # Create multi-GPU trainer
            multi_gpu_config = MultiGPUConfig(
                num_gpus=min(2, self.num_gpus),
                use_distributed=False,
                batch_size_per_gpu=16,
                mixed_precision=True
            )
            multi_gpu_trainer = MultiGPUTrainer(multi_gpu_config)
            
            # Create dataset
            class SimpleDataset(torch.utils.data.Dataset):
                def __init__(self, num_samples=100) -> Any:
                    self.data = torch.randn(num_samples, model_config.input_size)
                    self.targets = torch.randint(0, model_config.output_size, (num_samples,))
                
                def __len__(self) -> Any:
                    return len(self.data)
                
                def __getitem__(self, idx) -> Optional[Dict[str, Any]]:
                    return self.data[idx], self.targets[idx]
            
            dataset = SimpleDataset()
            
            # Test integration
            
            trainer = OptimizedTrainer(
                model, model_config, 
                multi_gpu_trainer=multi_gpu_trainer
            )
            
            # Create dataloader
            dataloader = trainer.create_dataloader(dataset)
            
            # Test training
            train_results = trainer.train_epoch(dataloader, epoch=1, total_epochs=1)
            assert 'loss' in train_results
            
            # Test validation
            val_results = trainer.validate(dataloader, epoch=1)
            assert 'loss' in val_results
            assert 'accuracy' in val_results
            
            logger.info("✅ Integration test successful")
            return True
            
        except Exception as e:
            logger.error(f"❌ Integration test failed: {e}")
            return False
    
    def test_distributed_training_launcher(self) -> Any:
        """Test distributed training launcher."""
        logger.info("=== Testing Distributed Training Launcher ===")
        
        if not MULTI_GPU_AVAILABLE:
            logger.warning("Multi-GPU training system not available")
            return False
        
        try:
            # Create launcher
            launcher = DistributedTrainingLauncher(world_size=2)
            assert launcher.world_size == 2
            
            # Test launcher creation
            logger.info("✅ Distributed training launcher test successful")
            return True
            
        except Exception as e:
            logger.error(f"❌ Distributed training launcher test failed: {e}")
            return False
    
    def run_all_tests(self) -> Any:
        """Run all multi-GPU training tests."""
        logger.info("Starting comprehensive multi-GPU training tests")
        
        tests = [
            ("GPU Availability", self.test_gpu_availability),
            ("MultiGPUConfig", self.test_multi_gpu_config),
            ("DataParallel Training", self.test_dataparallel_training),
            ("Distributed Training Setup", self.test_distributed_training_setup),
            ("MultiGPUTrainer Creation", self.test_multi_gpu_trainer_creation),
            ("Model Wrapping", self.test_model_wrapping),
            ("DataLoader Creation", self.test_dataloader_creation),
            ("Training Loop", self.test_training_loop),
            ("Validation Loop", self.test_validation_loop),
            ("Checkpoint Saving/Loading", self.test_checkpoint_saving_loading),
            ("State Dict Handling", self.test_state_dict_handling),
            ("Integration Test", self.test_integration_with_optimization_demo),
            ("Distributed Training Launcher", self.test_distributed_training_launcher)
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"Running: {test_name}")
                logger.info(f"{'='*60}")
                
                result = test_func()
                results[test_name] = result
                
                if result:
                    logger.info(f"✅ {test_name}: PASSED")
                else:
                    logger.error(f"❌ {test_name}: FAILED")
                    
            except Exception as e:
                logger.error(f"❌ {test_name}: ERROR - {e}")
                results[test_name] = False
        
        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info("TEST SUMMARY")
        logger.info(f"{'='*60}")
        
        passed = sum(1 for result in results.values() if result)
        total = len(results)
        
        for test_name, result in results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"{test_name}: {status}")
        
        logger.info(f"\nOverall: {passed}/{total} tests passed")
        
        if passed == total:
            logger.info("🎉 All multi-GPU training tests passed!")
        else:
            logger.warning(f"⚠️ {total - passed} tests failed")
        
        return results

def main():
    """Main test function."""
    logger.info("=== Multi-GPU Training System Test Suite ===")
    
    if not MULTI_GPU_AVAILABLE:
        logger.error("Multi-GPU training system not available. Please install required dependencies.")
        return
    
    # Create test suite
    test_suite = TestMultiGPUTraining()
    
    # Run all tests
    results = test_suite.run_all_tests()
    
    # Print final summary
    logger.info(f"\n{'='*60}")
    logger.info("FINAL SUMMARY")
    logger.info(f"{'='*60}")
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    logger.info(f"Tests Passed: {passed}/{total}")
    logger.info(f"Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        logger.info("🎉 All multi-GPU training tests completed successfully!")
    else:
        logger.warning(f"⚠️ {total - passed} tests failed. Check logs for details.")
    
    logger.info("=== Test Suite Completed ===")

match __name__:
    case "__main__":
    main() 