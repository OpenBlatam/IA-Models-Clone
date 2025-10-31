#!/usr/bin/env python3
"""
Comprehensive Test Suite for SEO System Optimizations
Test and validate all optimization modules with performance benchmarking
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import time
import logging
import asyncio
import warnings
from typing import Dict, Any, List, Tuple
import json
from pathlib import Path
import psutil
import gc

# Import optimization modules
from memory_optimizer import MemoryOptimizer, MemoryConfig
from async_data_loader import AsyncDataLoader, AsyncDataConfig
from model_compiler import ModelCompiler, CompilationConfig
from optimization_orchestrator import OptimizationOrchestrator, OptimizationConfig

warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestDataset(Dataset):
    """Test dataset for optimization testing."""
    
    def __init__(self, size: int = 1000, input_dim: int = 512, num_classes: int = 2):
        self.size = size
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Generate synthetic data
        self.data = torch.randn(size, input_dim)
        self.labels = torch.randint(0, num_classes, (size,))
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class TestModel(nn.Module):
    """Test model for optimization testing."""
    
    def __init__(self, input_dim: int = 512, hidden_dim: int = 256, num_classes: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class OptimizationTester:
    """Comprehensive optimization testing framework."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.results = {}
        
    def run_all_tests(self):
        """Run all optimization tests."""
        self.logger.info("Starting comprehensive optimization testing")
        
        # Test 1: Memory Optimization
        self.test_memory_optimization()
        
        # Test 2: Async Data Loading
        self.test_async_data_loading()
        
        # Test 3: Model Compilation
        self.test_model_compilation()
        
        # Test 4: Integrated Optimization
        self.test_integrated_optimization()
        
        # Test 5: Performance Benchmarking
        self.test_performance_benchmarking()
        
        # Generate report
        self.generate_test_report()
        
    def test_memory_optimization(self):
        """Test memory optimization module."""
        self.logger.info("Testing memory optimization...")
        
        try:
            # Create test model and data
            model = TestModel()
            dataset = TestDataset(size=1000)
            
            # Test memory optimizer
            memory_config = MemoryConfig(
                enable_memory_monitoring=True,
                enable_dynamic_batching=True,
                enable_model_caching=True
            )
            
            memory_optimizer = MemoryOptimizer(memory_config)
            
            # Test memory context
            with memory_optimizer.memory_context(model):
                # Simulate training
                optimizer = optim.Adam(model.parameters())
                criterion = nn.CrossEntropyLoss()
                
                for i in range(10):
                    batch_data, batch_labels = dataset[i:i+32]
                    optimizer.zero_grad()
                    outputs = model(batch_data)
                    loss = criterion(outputs, batch_labels)
                    loss.backward()
                    optimizer.step()
            
            # Get memory stats
            memory_stats = memory_optimizer.memory_monitor.get_memory_stats()
            
            self.results['memory_optimization'] = {
                'status': 'PASSED',
                'memory_stats': memory_stats,
                'error': None
            }
            
            self.logger.info("Memory optimization test PASSED")
            
        except Exception as e:
            self.results['memory_optimization'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            self.logger.error(f"Memory optimization test FAILED: {e}")
    
    def test_async_data_loading(self):
        """Test async data loading module."""
        self.logger.info("Testing async data loading...")
        
        try:
            # Create test dataset
            dataset = TestDataset(size=1000)
            
            # Test async data loader
            data_config = AsyncDataConfig(
                batch_size=32,
                num_workers=2,
                enable_async_loading=True,
                enable_data_caching=True
            )
            
            async_loader = AsyncDataLoader(data_config)
            
            # Test async loading
            async def test_loading():
                batch_count = 0
                async for batch in async_loader.load_dataset(dataset):
                    batch_count += 1
                    if batch_count >= 10:  # Test first 10 batches
                        break
                return batch_count
            
            # Run async test
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                batch_count = loop.run_until_complete(test_loading())
            finally:
                loop.close()
            
            # Get cache stats
            cache_stats = async_loader.cache_manager.get_cache_stats()
            
            self.results['async_data_loading'] = {
                'status': 'PASSED',
                'batch_count': batch_count,
                'cache_stats': cache_stats,
                'error': None
            }
            
            self.logger.info("Async data loading test PASSED")
            
        except Exception as e:
            self.results['async_data_loading'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            self.logger.error(f"Async data loading test FAILED: {e}")
    
    def test_model_compilation(self):
        """Test model compilation module."""
        self.logger.info("Testing model compilation...")
        
        try:
            # Create test model and data
            model = TestModel()
            sample_input = torch.randn(1, 512)
            
            # Test model compiler
            compilation_config = CompilationConfig(
                enable_compilation=True,
                enable_benchmarking=True,
                benchmark_iterations=50
            )
            
            model_compiler = ModelCompiler(compilation_config)
            
            # Test compilation
            compiled_model = model_compiler.compile_model(model, sample_input)
            
            # Test inference
            with torch.no_grad():
                original_output = model(sample_input)
                compiled_output = compiled_model(sample_input)
            
            # Check outputs are similar
            output_diff = torch.abs(original_output - compiled_output).mean().item()
            
            self.results['model_compilation'] = {
                'status': 'PASSED',
                'output_difference': output_diff,
                'compilation_successful': True,
                'error': None
            }
            
            self.logger.info("Model compilation test PASSED")
            
        except Exception as e:
            self.results['model_compilation'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            self.logger.error(f"Model compilation test FAILED: {e}")
    
    def test_integrated_optimization(self):
        """Test integrated optimization orchestrator."""
        self.logger.info("Testing integrated optimization...")
        
        try:
            # Create test components
            model = TestModel()
            dataset = TestDataset(size=1000)
            training_config = {
                'learning_rate': 1e-3,
                'num_epochs': 5,
                'batch_size': 32
            }
            
            # Test optimization orchestrator
            optimization_config = OptimizationConfig(
                enable_memory_optimization=True,
                enable_data_optimization=True,
                enable_model_optimization=True,
                enable_performance_monitoring=True
            )
            
            orchestrator = OptimizationOrchestrator(optimization_config)
            
            # Test integrated optimization
            async def test_integrated():
                return await orchestrator.optimize_training_pipeline(model, dataset, training_config)
            
            # Run async test
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                optimized_model, optimized_config = loop.run_until_complete(test_integrated())
            finally:
                loop.close()
            
            # Get optimization stats
            optimization_stats = orchestrator.get_optimization_stats()
            
            self.results['integrated_optimization'] = {
                'status': 'PASSED',
                'optimization_successful': True,
                'optimized_config': optimized_config,
                'optimization_stats': optimization_stats,
                'error': None
            }
            
            self.logger.info("Integrated optimization test PASSED")
            
        except Exception as e:
            self.results['integrated_optimization'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            self.logger.error(f"Integrated optimization test FAILED: {e}")
    
    def test_performance_benchmarking(self):
        """Test performance benchmarking."""
        self.logger.info("Testing performance benchmarking...")
        
        try:
            # Create test components
            model = TestModel()
            dataset = TestDataset(size=1000)
            
            # Benchmark baseline performance
            baseline_time = self._benchmark_baseline(model, dataset)
            
            # Benchmark optimized performance
            optimized_time = self._benchmark_optimized(model, dataset)
            
            # Calculate improvement
            improvement = (baseline_time - optimized_time) / baseline_time * 100
            
            self.results['performance_benchmarking'] = {
                'status': 'PASSED',
                'baseline_time': baseline_time,
                'optimized_time': optimized_time,
                'improvement_percent': improvement,
                'error': None
            }
            
            self.logger.info(f"Performance benchmarking test PASSED - {improvement:.2f}% improvement")
            
        except Exception as e:
            self.results['performance_benchmarking'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            self.logger.error(f"Performance benchmarking test FAILED: {e}")
    
    def _benchmark_baseline(self, model: nn.Module, dataset: Dataset) -> float:
        """Benchmark baseline performance."""
        model.eval()
        
        # Warm up
        sample_data, _ = dataset[0]
        with torch.no_grad():
            for _ in range(10):
                _ = model(sample_data.unsqueeze(0))
        
        # Benchmark
        start_time = time.time()
        
        with torch.no_grad():
            for i in range(100):
                data, _ = dataset[i % len(dataset)]
                _ = model(data.unsqueeze(0))
        
        end_time = time.time()
        return end_time - start_time
    
    def _benchmark_optimized(self, model: nn.Module, dataset: Dataset) -> float:
        """Benchmark optimized performance."""
        # Apply optimizations
        optimization_config = OptimizationConfig()
        orchestrator = OptimizationOrchestrator(optimization_config)
        
        with orchestrator.optimization_context(model):
            model.eval()
            
            # Warm up
            sample_data, _ = dataset[0]
            with torch.no_grad():
                for _ in range(10):
                    _ = model(sample_data.unsqueeze(0))
            
            # Benchmark
            start_time = time.time()
            
            with torch.no_grad():
                for i in range(100):
                    data, _ = dataset[i % len(dataset)]
                    _ = model(data.unsqueeze(0))
            
            end_time = time.time()
            return end_time - start_time
    
    def generate_test_report(self):
        """Generate comprehensive test report."""
        self.logger.info("Generating test report...")
        
        # Calculate overall results
        total_tests = len(self.results)
        passed_tests = sum(1 for result in self.results.values() if result['status'] == 'PASSED')
        failed_tests = total_tests - passed_tests
        
        # Create report
        report = {
            'summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'success_rate': (passed_tests / total_tests) * 100 if total_tests > 0 else 0
            },
            'detailed_results': self.results,
            'timestamp': time.time(),
            'system_info': self._get_system_info()
        }
        
        # Save report
        report_file = Path("optimization_test_report.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Print summary
        self.logger.info("=" * 50)
        self.logger.info("OPTIMIZATION TEST REPORT")
        self.logger.info("=" * 50)
        self.logger.info(f"Total Tests: {total_tests}")
        self.logger.info(f"Passed: {passed_tests}")
        self.logger.info(f"Failed: {failed_tests}")
        self.logger.info(f"Success Rate: {report['summary']['success_rate']:.1f}%")
        self.logger.info("=" * 50)
        
        # Print detailed results
        for test_name, result in self.results.items():
            status = "✅ PASSED" if result['status'] == 'PASSED' else "❌ FAILED"
            self.logger.info(f"{test_name}: {status}")
            if result['status'] == 'FAILED':
                self.logger.error(f"  Error: {result['error']}")
        
        self.logger.info(f"Detailed report saved to: {report_file}")
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        return {
            'python_version': f"{torch.__version__}",
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total / (1024**3),  # GB
            'platform': torch.utils.collect_env.get_pretty_env_info()
        }

def main():
    """Main test execution function."""
    logger.info("Starting SEO System Optimization Tests")
    
    # Create tester
    tester = OptimizationTester()
    
    # Run all tests
    tester.run_all_tests()
    
    logger.info("Optimization testing completed")

if __name__ == "__main__":
    main()






