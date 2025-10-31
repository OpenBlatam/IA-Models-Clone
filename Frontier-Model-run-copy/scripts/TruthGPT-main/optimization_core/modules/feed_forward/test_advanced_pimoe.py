"""
Comprehensive test suite for advanced PiMoE features
Tests all advanced routing, optimization, and performance features
"""

import torch
import torch.nn as nn
import pytest
import numpy as np
from typing import Dict, List, Tuple, Any
import time
import unittest
from unittest.mock import Mock, patch

from .advanced_pimoe_routing import (
    AdvancedPiMoESystem,
    RoutingStrategy,
    AdvancedRoutingConfig,
    AttentionBasedRouter,
    HierarchicalRouter,
    DynamicExpertScaler,
    CrossExpertCommunicator,
    NeuralArchitectureSearchRouter,
    create_advanced_pimoe_system
)
from .pimoe_performance_optimizer import (
    PiMoEPerformanceOptimizer,
    PerformanceConfig,
    OptimizationLevel,
    MemoryOptimizer,
    ComputationalOptimizer,
    ParallelProcessor,
    CacheManager,
    HardwareOptimizer,
    create_performance_optimizer
)
from .ultimate_pimoe_system import (
    UltimatePiMoESystem,
    UltimatePiMoEConfig,
    create_ultimate_pimoe_system,
    run_ultimate_pimoe_demo
)
from .pimoe_router import ExpertType

class TestAdvancedRouting(unittest.TestCase):
    """Test cases for advanced routing algorithms."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.hidden_size = 256
        self.num_experts = 4
        self.batch_size = 2
        self.seq_len = 32
        self.expert_types = [
            ExpertType.REASONING,
            ExpertType.COMPUTATION,
            ExpertType.MATHEMATICAL,
            ExpertType.LOGICAL
        ]
        
        self.test_input = torch.randn(self.batch_size, self.seq_len, self.hidden_size)
        self.attention_mask = torch.ones(self.batch_size, self.seq_len)
    
    def test_attention_based_router(self):
        """Test attention-based router."""
        router = AttentionBasedRouter(
            hidden_size=self.hidden_size,
            num_experts=self.num_experts,
            expert_types=self.expert_types,
            attention_heads=4
        )
        
        # Test basic forward pass
        output = router(self.test_input)
        self.assertEqual(output.shape, self.test_input.shape)
        
        # Test with attention weights
        output, attention_info = router(self.test_input, return_attention_weights=True)
        self.assertEqual(output.shape, self.test_input.shape)
        self.assertIn('routing_decisions', attention_info)
        self.assertIn('attention_weights', attention_info)
    
    def test_hierarchical_router(self):
        """Test hierarchical router."""
        router = HierarchicalRouter(
            hidden_size=self.hidden_size,
            num_experts=self.num_experts,
            expert_types=self.expert_types,
            hierarchical_levels=3
        )
        
        # Test basic forward pass
        output = router(self.test_input)
        self.assertEqual(output.shape, self.test_input.shape)
        
        # Test with hierarchical info
        output, hierarchical_info = router(self.test_input, return_hierarchical_info=True)
        self.assertEqual(output.shape, self.test_input.shape)
        self.assertIn('level_outputs', hierarchical_info)
        self.assertIn('level_decisions', hierarchical_info)
    
    def test_dynamic_expert_scaler(self):
        """Test dynamic expert scaler."""
        scaler = DynamicExpertScaler(
            base_num_experts=self.num_experts,
            max_num_experts=8,
            scaling_threshold=0.8
        )
        
        # Test scaling decision
        expert_loads = torch.rand(self.num_experts)
        expert_performance = torch.rand(self.num_experts)
        
        scaling_decision = scaler(expert_loads, expert_performance)
        self.assertIn('scaling_decision', scaling_decision)
        self.assertIn('action', scaling_decision)
        self.assertIn('current_experts', scaling_decision)
    
    def test_cross_expert_communicator(self):
        """Test cross-expert communicator."""
        communicator = CrossExpertCommunicator(
            hidden_size=self.hidden_size,
            num_experts=self.num_experts
        )
        
        # Test communication
        expert_outputs = [torch.randn(self.batch_size, self.hidden_size) for _ in range(self.num_experts)]
        expert_ids = list(range(self.num_experts))
        
        communicated_outputs = communicator(expert_outputs, expert_ids)
        self.assertEqual(len(communicated_outputs), len(expert_outputs))
        
        # Test with communication info
        outputs, comm_info = communicator(expert_outputs, expert_ids, return_communication_info=True)
        self.assertIn('attention_weights', comm_info)
        self.assertIn('communication_channels', comm_info)
    
    def test_neural_architecture_search(self):
        """Test neural architecture search."""
        nas_router = NeuralArchitectureSearchRouter(
            hidden_size=self.hidden_size,
            search_space_size=50
        )
        
        # Test architecture evaluation
        architecture = {
            'num_layers': 2,
            'hidden_sizes': self.hidden_size,
            'activations': 'relu',
            'dropout_rates': 0.1,
            'normalization': 'layer_norm'
        }
        
        performance_metrics = {
            'latency_ms': 10.0,
            'throughput_tokens_per_sec': 1000.0,
            'memory_usage_mb': 100.0
        }
        
        fitness = nas_router.evaluate_architecture(architecture, performance_metrics)
        self.assertIsInstance(fitness, float)
        
        # Test population evolution
        performance_data = {0: performance_metrics}
        new_population = nas_router.evolve_population(performance_data)
        self.assertEqual(len(new_population), nas_router.population_size)
    
    def test_advanced_pimoe_system(self):
        """Test advanced PiMoE system."""
        routing_config = AdvancedRoutingConfig(
            strategy=RoutingStrategy.ATTENTION_BASED,
            attention_heads=4
        )
        
        system = AdvancedPiMoESystem(
            hidden_size=self.hidden_size,
            num_experts=self.num_experts,
            expert_types=self.expert_types,
            routing_config=routing_config
        )
        
        # Test forward pass
        output = system(self.test_input)
        self.assertEqual(output.shape, self.test_input.shape)
        
        # Test with advanced info
        output, advanced_info = system(self.test_input, return_advanced_info=True)
        self.assertEqual(output.shape, self.test_input.shape)
        self.assertIn('routing_info', advanced_info)
        self.assertIn('expert_outputs', advanced_info)
    
    def test_create_advanced_pimoe_system(self):
        """Test factory function for advanced PiMoE system."""
        system = create_advanced_pimoe_system(
            hidden_size=self.hidden_size,
            num_experts=self.num_experts,
            routing_strategy=RoutingStrategy.ATTENTION_BASED
        )
        
        self.assertIsInstance(system, AdvancedPiMoESystem)
        self.assertEqual(system.hidden_size, self.hidden_size)
        self.assertEqual(system.num_experts, self.num_experts)

class TestPerformanceOptimization(unittest.TestCase):
    """Test cases for performance optimization."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = PerformanceConfig(
            optimization_level=OptimizationLevel.ADVANCED,
            enable_memory_optimization=True,
            enable_computational_optimization=True,
            enable_parallel_processing=True,
            enable_caching=True,
            enable_hardware_optimization=True
        )
        
        self.hidden_size = 256
        self.num_experts = 4
        self.test_input = torch.randn(2, 32, self.hidden_size)
    
    def test_memory_optimizer(self):
        """Test memory optimizer."""
        optimizer = MemoryOptimizer(self.config)
        
        # Test memory optimization
        model = nn.Linear(self.hidden_size, self.hidden_size)
        optimizations = optimizer.optimize_memory_usage(model)
        
        self.assertIn('gradient_checkpointing', optimizations)
        self.assertIn('memory_efficient_attention', optimizations)
        self.assertIn('mixed_precision', optimizations)
        self.assertIn('memory_cleanup', optimizations)
        
        # Test memory monitoring
        memory_stats = optimizer.monitor_memory_usage()
        self.assertIn('current_memory', memory_stats)
        self.assertIn('max_memory', memory_stats)
        self.assertIn('memory_usage_ratio', memory_stats)
    
    def test_computational_optimizer(self):
        """Test computational optimizer."""
        optimizer = ComputationalOptimizer(self.config)
        
        # Test computational optimization
        model = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        
        optimizations = optimizer.optimize_computations(model)
        self.assertIn('kernel_fusion', optimizations)
        self.assertIn('operator_optimization', optimizations)
        self.assertIn('batch_optimization', optimizations)
        self.assertIn('expert_parallelism', optimizations)
    
    def test_parallel_processor(self):
        """Test parallel processor."""
        processor = ParallelProcessor(self.config)
        
        # Test parallel processing
        expert_inputs = [torch.randn(2, self.hidden_size) for _ in range(self.num_experts)]
        expert_networks = [nn.Linear(self.hidden_size, self.hidden_size) for _ in range(self.num_experts)]
        
        results = processor.process_experts_parallel(expert_inputs, expert_networks)
        self.assertEqual(len(results), len(expert_inputs))
        
        # Test with results info
        results_info = processor.process_experts_parallel(
            expert_inputs, expert_networks, return_results=False
        )
        self.assertIn('results', results_info)
        self.assertIn('processing_time', results_info)
        self.assertIn('speedup', results_info)
    
    def test_cache_manager(self):
        """Test cache manager."""
        manager = CacheManager(self.config)
        
        # Test caching
        def computation_func(x):
            return x * 2
        
        # First call (cache miss)
        result1 = manager.get_cached_result("test_key", computation_func, 5)
        self.assertEqual(result1, 10)
        
        # Second call (cache hit)
        result2 = manager.get_cached_result("test_key", computation_func, 5)
        self.assertEqual(result2, 10)
        
        # Test cache stats
        stats = manager.get_cache_stats()
        self.assertIn('hits', stats)
        self.assertIn('misses', stats)
        self.assertIn('hit_rate', stats)
    
    def test_hardware_optimizer(self):
        """Test hardware optimizer."""
        optimizer = HardwareOptimizer(self.config)
        
        # Test hardware detection
        hardware_info = optimizer.hardware_info
        self.assertIn('cuda_available', hardware_info)
        self.assertIn('cpu_count', hardware_info)
        
        # Test hardware optimization
        model = nn.Linear(self.hidden_size, self.hidden_size)
        optimizations = optimizer.optimize_for_hardware(model)
        
        self.assertIn('cuda', optimizations)
        self.assertIn('cpu', optimizations)
        self.assertIn('memory', optimizations)
    
    def test_performance_optimizer(self):
        """Test comprehensive performance optimizer."""
        optimizer = PiMoEPerformanceOptimizer(self.config)
        
        # Test system optimization
        model = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        
        optimization_results = optimizer.optimize_system(model)
        self.assertIn('memory', optimization_results)
        self.assertIn('computational', optimization_results)
        self.assertIn('hardware', optimization_results)
        
        # Test inference optimization
        inference_optimizations = optimizer.optimize_inference(model)
        self.assertIn('mixed_precision', inference_optimizations)
        
        # Test performance metrics
        metrics = optimizer.get_performance_metrics()
        self.assertIn('optimization_stats', metrics)
        self.assertIn('memory_stats', metrics)
        self.assertIn('parallel_stats', metrics)
        self.assertIn('cache_stats', metrics)
        self.assertIn('hardware_info', metrics)
    
    def test_benchmark_performance(self):
        """Test performance benchmarking."""
        optimizer = PiMoEPerformanceOptimizer(self.config)
        
        model = nn.Linear(self.hidden_size, self.hidden_size)
        benchmark_results = optimizer.benchmark_performance(model, self.test_input, num_iterations=10)
        
        self.assertIn('total_time', benchmark_results)
        self.assertIn('average_time', benchmark_results)
        self.assertIn('throughput', benchmark_results)
        self.assertIn('iterations', benchmark_results)

class TestUltimatePiMoESystem(unittest.TestCase):
    """Test cases for the ultimate PiMoE system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.hidden_size = 256
        self.num_experts = 4
        self.batch_size = 2
        self.seq_len = 32
        self.test_input = torch.randn(self.batch_size, self.seq_len, self.hidden_size)
    
    def test_ultimate_pimoe_config(self):
        """Test ultimate PiMoE configuration."""
        config = UltimatePiMoEConfig(
            hidden_size=self.hidden_size,
            num_experts=self.num_experts,
            routing_strategy=RoutingStrategy.ATTENTION_BASED,
            optimization_level=OptimizationLevel.ADVANCED,
            enable_all_features=True
        )
        
        self.assertEqual(config.hidden_size, self.hidden_size)
        self.assertEqual(config.num_experts, self.num_experts)
        self.assertEqual(config.routing_strategy, RoutingStrategy.ATTENTION_BASED)
        self.assertEqual(config.optimization_level, OptimizationLevel.ADVANCED)
        self.assertTrue(config.enable_dynamic_scaling)
        self.assertTrue(config.enable_cross_expert_communication)
        self.assertTrue(config.enable_performance_optimization)
    
    def test_ultimate_pimoe_system(self):
        """Test ultimate PiMoE system."""
        config = UltimatePiMoEConfig(
            hidden_size=self.hidden_size,
            num_experts=self.num_experts,
            enable_all_features=True
        )
        
        system = UltimatePiMoESystem(config)
        
        # Test basic forward pass
        output = system(self.test_input)
        self.assertEqual(output.shape, self.test_input.shape)
        
        # Test with comprehensive info
        output, comprehensive_info = system(self.test_input, return_comprehensive_info=True)
        self.assertEqual(output.shape, self.test_input.shape)
        self.assertIn('routing_info', comprehensive_info)
        self.assertIn('performance_metrics', comprehensive_info)
        self.assertIn('system_stats', comprehensive_info)
    
    def test_system_optimization(self):
        """Test system optimization."""
        system = create_ultimate_pimoe_system(
            hidden_size=self.hidden_size,
            num_experts=self.num_experts,
            enable_all_features=True
        )
        
        # Test optimization for inference
        system.optimize_for_inference()
        
        # Test benchmarking
        benchmark_results = system.benchmark_system(self.test_input, num_iterations=10)
        self.assertIn('total_time', benchmark_results)
        self.assertIn('average_time', benchmark_results)
        self.assertIn('throughput', benchmark_results)
    
    def test_system_statistics(self):
        """Test system statistics."""
        system = create_ultimate_pimoe_system(
            hidden_size=self.hidden_size,
            num_experts=self.num_experts
        )
        
        stats = system.get_system_stats()
        self.assertIn('system_type', stats)
        self.assertIn('hidden_size', stats)
        self.assertIn('num_experts', stats)
        self.assertIn('expert_types', stats)
        self.assertIn('routing_strategy', stats)
        self.assertIn('optimization_level', stats)
        self.assertIn('features_enabled', stats)
    
    def test_create_ultimate_pimoe_system(self):
        """Test factory function for ultimate PiMoE system."""
        system = create_ultimate_pimoe_system(
            hidden_size=self.hidden_size,
            num_experts=self.num_experts,
            routing_strategy=RoutingStrategy.ATTENTION_BASED,
            optimization_level=OptimizationLevel.ADVANCED,
            enable_all_features=True
        )
        
        self.assertIsInstance(system, UltimatePiMoESystem)
        self.assertEqual(system.hidden_size, self.hidden_size)
        self.assertEqual(system.num_experts, self.num_experts)
    
    def test_performance_tracker(self):
        """Test performance tracker."""
        from .ultimate_pimoe_system import PerformanceTracker
        
        tracker = PerformanceTracker()
        
        # Test adding metrics
        metrics = {
            'latency_ms': 10.0,
            'throughput_tokens_per_sec': 1000.0,
            'memory_usage_mb': 100.0
        }
        
        tracker.add_metrics(metrics)
        
        # Test getting metrics
        tracker_metrics = tracker.get_metrics()
        self.assertIn('current_metrics', tracker_metrics)
        self.assertIn('aggregated_metrics', tracker_metrics)
        self.assertIn('history_length', tracker_metrics)
    
    def test_adaptation_tracker(self):
        """Test adaptation tracker."""
        from .ultimate_pimoe_system import AdaptationTracker
        
        tracker = AdaptationTracker()
        
        # Test adding adaptation
        adaptation_info = {
            'success': True,
            'adaptation_type': 'routing_adjustment',
            'performance_improvement': 0.1
        }
        
        tracker.add_adaptation(adaptation_info)
        
        # Test getting adaptation info
        adaptation_info_result = tracker.get_adaptation_info()
        self.assertIn('adaptation_history', adaptation_info_result)
        self.assertIn('adaptation_stats', adaptation_info_result)
        self.assertIn('learning_progress', adaptation_info_result)

class TestIntegration(unittest.TestCase):
    """Test integration between components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.hidden_size = 256
        self.num_experts = 4
        self.test_input = torch.randn(2, 32, self.hidden_size)
    
    def test_full_integration(self):
        """Test full integration of all components."""
        # Create ultimate system
        system = create_ultimate_pimoe_system(
            hidden_size=self.hidden_size,
            num_experts=self.num_experts,
            routing_strategy=RoutingStrategy.ATTENTION_BASED,
            optimization_level=OptimizationLevel.ADVANCED,
            enable_all_features=True
        )
        
        # Test forward pass
        output = system(self.test_input)
        self.assertEqual(output.shape, self.test_input.shape)
        
        # Test with comprehensive info
        output, info = system(self.test_input, return_comprehensive_info=True)
        self.assertEqual(output.shape, self.test_input.shape)
        self.assertIn('routing_info', info)
        self.assertIn('performance_metrics', info)
        self.assertIn('system_stats', info)
        
        # Test optimization
        system.optimize_for_inference()
        
        # Test benchmarking
        benchmark_results = system.benchmark_system(self.test_input, num_iterations=5)
        self.assertIn('total_time', benchmark_results)
        self.assertIn('throughput', benchmark_results)
    
    def test_performance_comparison(self):
        """Test performance comparison between different configurations."""
        configs = [
            {
                'routing_strategy': RoutingStrategy.ATTENTION_BASED,
                'optimization_level': OptimizationLevel.BASIC,
                'enable_all_features': False
            },
            {
                'routing_strategy': RoutingStrategy.ATTENTION_BASED,
                'optimization_level': OptimizationLevel.ADVANCED,
                'enable_all_features': True
            }
        ]
        
        results = {}
        
        for i, config in enumerate(configs):
            system = create_ultimate_pimoe_system(
                hidden_size=self.hidden_size,
                num_experts=self.num_experts,
                **config
            )
            
            # Benchmark
            benchmark_results = system.benchmark_system(self.test_input, num_iterations=5)
            results[f'config_{i}'] = benchmark_results
        
        # Compare results
        self.assertIn('config_0', results)
        self.assertIn('config_1', results)
        
        # Advanced config should generally perform better
        # (though this may vary based on hardware and specific conditions)
        self.assertIsInstance(results['config_0']['total_time'], float)
        self.assertIsInstance(results['config_1']['total_time'], float)

def run_advanced_tests():
    """Run all advanced tests."""
    test_suites = [
        TestAdvancedRouting,
        TestPerformanceOptimization,
        TestUltimatePiMoESystem,
        TestIntegration
    ]
    
    all_tests = unittest.TestSuite()
    
    for test_suite in test_suites:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_suite)
        all_tests.addTests(tests)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(all_tests)
    
    return result

if __name__ == "__main__":
    # Run all advanced tests
    result = run_advanced_tests()
    
    # Print summary
    if result.wasSuccessful():
        print("\n✅ All advanced tests passed successfully!")
    else:
        print(f"\n❌ {len(result.failures)} failures, {len(result.errors)} errors")
        for failure in result.failures:
            print(f"FAIL: {failure[0]}")
            print(f"  {failure[1]}")
        for error in result.errors:
            print(f"ERROR: {error[0]}")
            print(f"  {error[1]}")


