from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import torch
import torch.nn as nn
import torch.cuda.amp as amp
import torch.nn.functional as F
import numpy as np
import time
import logging
import json
from typing import Dict, List, Optional, Tuple
import traceback
import os
from pathlib import Path
    from mixed_precision_system import (
    from optimization_demo import OptimizedNeuralNetwork, ModelConfig
            from optimization_demo import OptimizedTrainer
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Test Suite for Mixed Precision Training System

Demonstrates comprehensive mixed precision training using torch.cuda.amp
with advanced features for optimal performance and memory efficiency.
"""


# Import mixed precision system
try:
        MixedPrecisionManager, MixedPrecisionConfig, AdaptiveMixedPrecisionManager,
        MixedPrecisionTrainer
    )
    MIXED_PRECISION_AVAILABLE = True
except ImportError:
    MIXED_PRECISION_AVAILABLE = False

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

class TestMixedPrecision:
    """Comprehensive test suite for mixed precision training system."""
    
    def __init__(self) -> Any:
        self.test_results = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def test_mixed_precision_config(self) -> Any:
        """Test MixedPrecisionConfig creation and validation."""
        logger.info("=== Testing MixedPrecisionConfig ===")
        
        if not MIXED_PRECISION_AVAILABLE:
            logger.warning("Mixed precision system not available")
            return False
        
        try:
            # Test basic configuration
            config = MixedPrecisionConfig(
                enabled=True,
                dtype=torch.float16,
                autocast_enabled=True,
                grad_scaler_enabled=True,
                memory_efficient=True,
                dynamic_scaling=True
            )
            
            assert config.enabled == True
            assert config.dtype == torch.float16
            assert config.autocast_enabled == True
            assert config.grad_scaler_enabled == True
            assert config.memory_efficient == True
            assert config.dynamic_scaling == True
            
            # Test CPU fallback
            if not torch.cuda.is_available():
                config_cpu = MixedPrecisionConfig()
                assert config_cpu.enabled == False
                assert config_cpu.device_type == "cpu"
            
            # Test advanced settings
            config_advanced = MixedPrecisionConfig(
                growth_factor=3.0,
                backoff_factor=0.3,
                growth_interval=1000,
                max_scale=2.0**20,
                min_scale=2.0**(-20)
            )
            
            assert config_advanced.growth_factor == 3.0
            assert config_advanced.backoff_factor == 0.3
            assert config_advanced.growth_interval == 1000
            
            logger.info("✅ MixedPrecisionConfig test successful")
            return True
            
        except Exception as e:
            logger.error(f"❌ MixedPrecisionConfig test failed: {e}")
            return False
    
    def test_mixed_precision_manager_creation(self) -> Any:
        """Test MixedPrecisionManager creation."""
        logger.info("=== Testing MixedPrecisionManager Creation ===")
        
        if not MIXED_PRECISION_AVAILABLE:
            logger.warning("Mixed precision system not available")
            return False
        
        try:
            config = MixedPrecisionConfig(
                enabled=True,
                dtype=torch.float16,
                autocast_enabled=True,
                grad_scaler_enabled=True,
                memory_efficient=True,
                track_performance=True,
                log_scaling=True
            )
            
            manager = MixedPrecisionManager(config)
            
            assert manager.config == config
            assert manager.scaler is not None
            assert manager.autocast_context is not None
            
            logger.info("✅ MixedPrecisionManager creation test successful")
            return True
            
        except Exception as e:
            logger.error(f"❌ MixedPrecisionManager creation test failed: {e}")
            return False
    
    def test_adaptive_mixed_precision_manager(self) -> Any:
        """Test AdaptiveMixedPrecisionManager creation and functionality."""
        logger.info("=== Testing AdaptiveMixedPrecisionManager ===")
        
        if not MIXED_PRECISION_AVAILABLE:
            logger.warning("Mixed precision system not available")
            return False
        
        try:
            config = MixedPrecisionConfig(
                enabled=True,
                dtype=torch.float16,
                autocast_enabled=True,
                grad_scaler_enabled=True,
                dynamic_scaling=True,
                memory_efficient=True,
                track_performance=True
            )
            
            manager = AdaptiveMixedPrecisionManager(config)
            
            assert isinstance(manager, AdaptiveMixedPrecisionManager)
            assert manager.config.dynamic_scaling == True
            assert manager.performance_threshold == 0.8
            
            logger.info("✅ AdaptiveMixedPrecisionManager test successful")
            return True
            
        except Exception as e:
            logger.error(f"❌ AdaptiveMixedPrecisionManager test failed: {e}")
            return False
    
    def test_basic_mixed_precision_training(self) -> Any:
        """Test basic mixed precision training functionality."""
        logger.info("=== Testing Basic Mixed Precision Training ===")
        
        if not MIXED_PRECISION_AVAILABLE:
            logger.warning("Mixed precision system not available")
            return False
        
        try:
            # Create simple model
            model = nn.Sequential(
                nn.Linear(784, 512),
                nn.ReLU(),
                nn.Linear(512, 10)
            ).to(self.device)
            
            # Create configuration
            config = MixedPrecisionConfig(
                enabled=True,
                dtype=torch.float16,
                autocast_enabled=True,
                grad_scaler_enabled=True,
                memory_efficient=True,
                track_performance=True,
                log_scaling=True
            )
            
            manager = MixedPrecisionManager(config)
            optimizer = torch.optim.Adam(model.parameters())
            criterion = nn.CrossEntropyLoss()
            
            # Create dummy data
            data = torch.randn(8, 784).to(self.device)
            targets = torch.randint(0, 10, (8,)).to(self.device)
            
            # Test autocast context
            with manager.autocast_context():
                # Forward pass
                outputs = model(data)
                loss = criterion(outputs, targets)
                
                # Scale loss
                scaled_loss = manager.scale_loss(loss)
                
                # Backward pass
                scaled_loss.backward()
                
                # Unscale optimizer
                manager.unscale_optimizer(optimizer)
                
                # Step optimizer
                manager.step_optimizer(optimizer)
                
                # Update scaler
                manager.update_scaler()
                
                # Get scale
                scale = manager.get_scale()
                
                assert scale > 0
                assert manager.is_enabled() == True
            
            logger.info("✅ Basic mixed precision training test successful")
            return True
            
        except Exception as e:
            logger.error(f"❌ Basic mixed precision training test failed: {e}")
            return False
    
    def test_mixed_precision_with_memory_tracking(self) -> Any:
        """Test mixed precision with memory tracking."""
        logger.info("=== Testing Mixed Precision with Memory Tracking ===")
        
        if not MIXED_PRECISION_AVAILABLE:
            logger.warning("Mixed precision system not available")
            return False
        
        try:
            # Create simple model
            model = nn.Sequential(
                nn.Linear(784, 512),
                nn.ReLU(),
                nn.Linear(512, 10)
            ).to(self.device)
            
            # Create configuration with memory tracking
            config = MixedPrecisionConfig(
                enabled=True,
                dtype=torch.float16,
                autocast_enabled=True,
                grad_scaler_enabled=True,
                memory_efficient=True,
                clear_cache=True,
                optimize_memory=True,
                profile_memory=True,
                track_performance=True
            )
            
            manager = MixedPrecisionManager(config)
            optimizer = torch.optim.Adam(model.parameters())
            criterion = nn.CrossEntropyLoss()
            
            # Create dummy data
            data = torch.randn(8, 784).to(self.device)
            targets = torch.randint(0, 10, (8,)).to(self.device)
            
            # Test mixed precision with memory tracking
            with manager.autocast_context():
                outputs = model(data)
                loss = criterion(outputs, targets)
                
                scaled_loss = manager.scale_loss(loss)
                scaled_loss.backward()
                
                manager.unscale_optimizer(optimizer)
                manager.step_optimizer(optimizer)
                manager.update_scaler()
            
            # Optimize memory
            manager.optimize_memory()
            
            # Get performance stats
            stats = manager.get_performance_stats()
            
            assert 'enabled' in stats
            assert 'current_scale' in stats
            assert 'memory_usage' in stats
            assert 'performance_metrics' in stats
            
            if torch.cuda.is_available():
                assert len(stats['memory_usage']) > 0
                assert 'allocated_gb' in stats['memory_usage'][0]
            
            logger.info("✅ Mixed precision with memory tracking test successful")
            return True
            
        except Exception as e:
            logger.error(f"❌ Mixed precision with memory tracking test failed: {e}")
            return False
    
    def test_adaptive_scaling_decision(self) -> Any:
        """Test adaptive scaling decision making."""
        logger.info("=== Testing Adaptive Scaling Decision ===")
        
        if not MIXED_PRECISION_AVAILABLE:
            logger.warning("Mixed precision system not available")
            return False
        
        try:
            # Create simple model
            model = nn.Sequential(
                nn.Linear(784, 512),
                nn.ReLU(),
                nn.Linear(512, 10)
            ).to(self.device)
            
            # Create adaptive configuration
            config = MixedPrecisionConfig(
                enabled=True,
                dtype=torch.float16,
                autocast_enabled=True,
                grad_scaler_enabled=True,
                dynamic_scaling=True,
                memory_efficient=True
            )
            
            manager = AdaptiveMixedPrecisionManager(config)
            
            # Test adaptive decision
            should_scale = manager._adaptive_scaling_decision(0.9)  # High performance
            assert should_scale == True
            
            should_scale = manager._adaptive_scaling_decision(0.5)  # Low performance
            assert should_scale == False
            
            logger.info("✅ Adaptive scaling decision test successful")
            return True
            
        except Exception as e:
            logger.error(f"❌ Adaptive scaling decision test failed: {e}")
            return False
    
    def test_mixed_precision_trainer(self) -> Any:
        """Test MixedPrecisionTrainer."""
        logger.info("=== Testing MixedPrecisionTrainer ===")
        
        if not MIXED_PRECISION_AVAILABLE:
            logger.warning("Mixed precision system not available")
            return False
        
        try:
            # Create simple model
            model = nn.Sequential(
                nn.Linear(784, 512),
                nn.ReLU(),
                nn.Linear(512, 10)
            ).to(self.device)
            
            # Create configuration
            config = MixedPrecisionConfig(
                enabled=True,
                dtype=torch.float16,
                autocast_enabled=True,
                grad_scaler_enabled=True,
                memory_efficient=True,
                track_performance=True,
                log_scaling=True
            )
            
            # Create trainer
            trainer = MixedPrecisionTrainer(model, config)
            
            assert trainer.model == model
            assert trainer.config == config
            assert trainer.mixed_precision_manager.config == config
            
            # Create dummy dataset
            data = torch.randn(100, 784).to(self.device)
            targets = torch.randint(0, 10, (100,)).to(self.device)
            dataset = torch.utils.data.TensorDataset(data, targets)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
            
            # Test training step
            step_result = trainer.train_step(data[:8], targets[:8], step=0)
            
            assert 'loss' in step_result
            assert 'scaled_loss' in step_result
            assert 'accuracy' in step_result
            assert 'scale' in step_result
            assert 'memory_allocated_gb' in step_result
            assert 'training_time' in step_result
            
            logger.info("✅ MixedPrecisionTrainer test successful")
            return True
            
        except Exception as e:
            logger.error(f"❌ MixedPrecisionTrainer test failed: {e}")
            return False
    
    def test_mixed_precision_training_epoch(self) -> Any:
        """Test complete training epoch with mixed precision."""
        logger.info("=== Testing Mixed Precision Training Epoch ===")
        
        if not MIXED_PRECISION_AVAILABLE:
            logger.warning("Mixed precision system not available")
            return False
        
        try:
            # Create simple model
            model = nn.Sequential(
                nn.Linear(784, 512),
                nn.ReLU(),
                nn.Linear(512, 10)
            ).to(self.device)
            
            # Create configuration
            config = MixedPrecisionConfig(
                enabled=True,
                dtype=torch.float16,
                autocast_enabled=True,
                grad_scaler_enabled=True,
                memory_efficient=True,
                track_performance=True,
                log_scaling=True
            )
            
            # Create trainer
            trainer = MixedPrecisionTrainer(model, config)
            
            # Create dummy dataset
            data = torch.randn(100, 784).to(self.device)
            targets = torch.randint(0, 10, (100,)).to(self.device)
            dataset = torch.utils.data.TensorDataset(data, targets)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
            
            # Test training epoch
            epoch_results = trainer.train_epoch(dataloader, epoch=1)
            
            assert 'avg_loss' in epoch_results
            assert 'avg_accuracy' in epoch_results
            assert 'final_scale' in epoch_results
            assert 'performance_stats' in epoch_results
            
            # Check performance stats
            stats = epoch_results['performance_stats']
            assert 'enabled' in stats
            assert 'current_scale' in stats
            assert 'scaling_history' in stats
            assert 'memory_usage' in stats
            
            logger.info("✅ Mixed precision training epoch test successful")
            return True
            
        except Exception as e:
            logger.error(f"❌ Mixed precision training epoch test failed: {e}")
            return False
    
    def test_mixed_precision_validation(self) -> Any:
        """Test validation with mixed precision."""
        logger.info("=== Testing Mixed Precision Validation ===")
        
        if not MIXED_PRECISION_AVAILABLE:
            logger.warning("Mixed precision system not available")
            return False
        
        try:
            # Create simple model
            model = nn.Sequential(
                nn.Linear(784, 512),
                nn.ReLU(),
                nn.Linear(512, 10)
            ).to(self.device)
            
            # Create configuration
            config = MixedPrecisionConfig(
                enabled=True,
                dtype=torch.float16,
                autocast_enabled=True,
                grad_scaler_enabled=True,
                memory_efficient=True
            )
            
            # Create trainer
            trainer = MixedPrecisionTrainer(model, config)
            
            # Create dummy dataset
            data = torch.randn(100, 784).to(self.device)
            targets = torch.randint(0, 10, (100,)).to(self.device)
            dataset = torch.utils.data.TensorDataset(data, targets)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
            
            # Test validation
            val_results = trainer.validate(dataloader)
            
            assert 'loss' in val_results
            assert 'accuracy' in val_results
            
            logger.info("✅ Mixed precision validation test successful")
            return True
            
        except Exception as e:
            logger.error(f"❌ Mixed precision validation test failed: {e}")
            return False
    
    def test_integration_with_optimization_demo(self) -> Any:
        """Test integration with optimization demo."""
        logger.info("=== Testing Integration with Optimization Demo ===")
        
        if not MIXED_PRECISION_AVAILABLE or not OPTIMIZATION_AVAILABLE:
            logger.warning("Required systems not available")
            return False
        
        try:
            # Create model and config
            model_config = ModelConfig()
            model = OptimizedNeuralNetwork(model_config)
            
            # Create mixed precision config
            mixed_precision_config = MixedPrecisionConfig(
                enabled=True,
                dtype=torch.float16,
                autocast_enabled=True,
                grad_scaler_enabled=True,
                dynamic_scaling=True,
                memory_efficient=True,
                track_performance=True
            )
            
            # Create adaptive mixed precision manager
            mixed_precision_manager = AdaptiveMixedPrecisionManager(mixed_precision_config)
            
            # Test integration
            
            trainer = OptimizedTrainer(
                model, model_config, 
                mixed_precision_manager=mixed_precision_manager
            )
            
            # Create dataset
            data = torch.randn(100, model_config.input_size)
            targets = torch.randint(0, model_config.output_size, (100,))
            dataset = torch.utils.data.TensorDataset(data, targets)
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
    
    def test_performance_comparison(self) -> Any:
        """Test performance comparison between mixed precision and full precision."""
        logger.info("=== Testing Performance Comparison ===")
        
        if not MIXED_PRECISION_AVAILABLE:
            logger.warning("Mixed precision system not available")
            return False
        
        try:
            # Create simple model
            model = nn.Sequential(
                nn.Linear(784, 512),
                nn.ReLU(),
                nn.Linear(512, 10)
            ).to(self.device)
            
            # Create mixed precision configuration
            mixed_precision_config = MixedPrecisionConfig(
                enabled=True,
                dtype=torch.float16,
                autocast_enabled=True,
                grad_scaler_enabled=True,
                memory_efficient=True,
                track_performance=True
            )
            
            # Create full precision configuration
            full_precision_config = MixedPrecisionConfig(
                enabled=False,
                dtype=torch.float32,
                autocast_enabled=False,
                grad_scaler_enabled=False,
                memory_efficient=True,
                track_performance=True
            )
            
            # Create trainers
            mixed_precision_trainer = MixedPrecisionTrainer(model, mixed_precision_config)
            full_precision_trainer = MixedPrecisionTrainer(model, full_precision_config)
            
            # Create dummy dataset
            data = torch.randn(100, 784).to(self.device)
            targets = torch.randint(0, 10, (100,)).to(self.device)
            dataset = torch.utils.data.TensorDataset(data, targets)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
            
            # Test mixed precision performance
            start_time = time.time()
            mixed_precision_results = mixed_precision_trainer.train_epoch(dataloader, epoch=1)
            mixed_precision_time = time.time() - start_time
            
            # Test full precision performance
            start_time = time.time()
            full_precision_results = full_precision_trainer.train_epoch(dataloader, epoch=1)
            full_precision_time = time.time() - start_time
            
            # Compare performance
            logger.info(f"Mixed Precision Time: {mixed_precision_time:.4f}s")
            logger.info(f"Full Precision Time: {full_precision_time:.4f}s")
            logger.info(f"Speedup: {full_precision_time / mixed_precision_time:.2f}x")
            
            # Check that mixed precision is faster (usually)
            if torch.cuda.is_available():
                assert mixed_precision_time <= full_precision_time * 1.5  # Allow some variance
            
            logger.info("✅ Performance comparison test successful")
            return True
            
        except Exception as e:
            logger.error(f"❌ Performance comparison test failed: {e}")
            return False
    
    def test_memory_efficiency(self) -> Any:
        """Test memory efficiency of mixed precision training."""
        logger.info("=== Testing Memory Efficiency ===")
        
        if not MIXED_PRECISION_AVAILABLE:
            logger.warning("Mixed precision system not available")
            return False
        
        try:
            # Create simple model
            model = nn.Sequential(
                nn.Linear(784, 512),
                nn.ReLU(),
                nn.Linear(512, 10)
            ).to(self.device)
            
            # Create mixed precision configuration
            mixed_precision_config = MixedPrecisionConfig(
                enabled=True,
                dtype=torch.float16,
                autocast_enabled=True,
                grad_scaler_enabled=True,
                memory_efficient=True,
                clear_cache=True,
                optimize_memory=True,
                profile_memory=True
            )
            
            # Create full precision configuration
            full_precision_config = MixedPrecisionConfig(
                enabled=False,
                dtype=torch.float32,
                autocast_enabled=False,
                grad_scaler_enabled=False,
                memory_efficient=True,
                profile_memory=True
            )
            
            # Create trainers
            mixed_precision_trainer = MixedPrecisionTrainer(model, mixed_precision_config)
            full_precision_trainer = MixedPrecisionTrainer(model, full_precision_config)
            
            # Create dummy dataset
            data = torch.randn(100, 784).to(self.device)
            targets = torch.randint(0, 10, (100,)).to(self.device)
            dataset = torch.utils.data.TensorDataset(data, targets)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
            
            # Test mixed precision memory usage
            mixed_precision_results = mixed_precision_trainer.train_epoch(dataloader, epoch=1)
            mixed_precision_stats = mixed_precision_results['performance_stats']
            
            # Test full precision memory usage
            full_precision_results = full_precision_trainer.train_epoch(dataloader, epoch=1)
            full_precision_stats = full_precision_results['performance_stats']
            
            # Compare memory usage
            if torch.cuda.is_available():
                mixed_precision_memory = mixed_precision_stats['memory_usage'][-1]['allocated_gb'] if mixed_precision_stats['memory_usage'] else 0
                full_precision_memory = full_precision_stats['memory_usage'][-1]['allocated_gb'] if full_precision_stats['memory_usage'] else 0
                
                logger.info(f"Mixed Precision Memory: {mixed_precision_memory:.2f} GB")
                logger.info(f"Full Precision Memory: {full_precision_memory:.2f} GB")
                logger.info(f"Memory Reduction: {(1 - mixed_precision_memory / full_precision_memory) * 100:.1f}%")
                
                # Check that mixed precision uses less memory (usually)
                assert mixed_precision_memory <= full_precision_memory * 1.2  # Allow some variance
            
            logger.info("✅ Memory efficiency test successful")
            return True
            
        except Exception as e:
            logger.error(f"❌ Memory efficiency test failed: {e}")
            return False
    
    def run_all_tests(self) -> Any:
        """Run all mixed precision tests."""
        logger.info("Starting comprehensive mixed precision tests")
        
        tests = [
            ("MixedPrecisionConfig", self.test_mixed_precision_config),
            ("MixedPrecisionManager Creation", self.test_mixed_precision_manager_creation),
            ("AdaptiveMixedPrecisionManager", self.test_adaptive_mixed_precision_manager),
            ("Basic Mixed Precision Training", self.test_basic_mixed_precision_training),
            ("Memory Tracking", self.test_mixed_precision_with_memory_tracking),
            ("Adaptive Scaling Decision", self.test_adaptive_scaling_decision),
            ("MixedPrecisionTrainer", self.test_mixed_precision_trainer),
            ("Training Epoch", self.test_mixed_precision_training_epoch),
            ("Validation", self.test_mixed_precision_validation),
            ("Integration Test", self.test_integration_with_optimization_demo),
            ("Performance Comparison", self.test_performance_comparison),
            ("Memory Efficiency", self.test_memory_efficiency)
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
            logger.info("🎉 All mixed precision tests passed!")
        else:
            logger.warning(f"⚠️ {total - passed} tests failed")
        
        return results

def main():
    """Main test function."""
    logger.info("=== Mixed Precision Training System Test Suite ===")
    
    if not MIXED_PRECISION_AVAILABLE:
        logger.error("Mixed precision system not available. Please install required dependencies.")
        return
    
    # Create test suite
    test_suite = TestMixedPrecision()
    
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
        logger.info("🎉 All mixed precision tests completed successfully!")
    else:
        logger.warning(f"⚠️ {total - passed} tests failed. Check logs for details.")
    
    logger.info("=== Test Suite Completed ===")

match __name__:
    case "__main__":
    main() 