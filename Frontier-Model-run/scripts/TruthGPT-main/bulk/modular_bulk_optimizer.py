#!/usr/bin/env python3
"""
Modular Bulk Optimizer - The most advanced modular optimization system
Provides clean, modular architecture with separation of concerns
"""

import torch
import torch.nn as nn
import asyncio
import time
import logging
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import numpy as np
from pathlib import Path
import json
import uuid
from datetime import datetime, timezone
import argparse
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.base_optimizer import BaseOptimizer, OptimizationResult, ModelProfile
from core.model_analyzer import ModelAnalyzer
from core.performance_metrics import PerformanceMetrics
from core.config_manager import ConfigManager
from orchestrator.optimization_orchestrator import OptimizationOrchestrator, OrchestrationConfig
from strategies.transformer_strategy import TransformerOptimizationStrategy

@dataclass
class ModularOptimizerConfig:
    """Configuration for modular bulk optimizer."""
    # Core settings
    max_concurrent_optimizations: int = 4
    optimization_timeout: int = 300
    enable_parallel_processing: bool = True
    enable_adaptive_selection: bool = True
    enable_performance_monitoring: bool = True
    enable_resource_management: bool = True
    
    # Strategy settings
    strategy_weights: Dict[str, float] = field(default_factory=lambda: {
        'transformer': 0.3,
        'llm': 0.2,
        'diffusion': 0.2,
        'quantum': 0.1,
        'performance': 0.2
    })
    
    # Performance settings
    target_improvement: float = 0.5
    max_optimization_time: int = 600
    enable_caching: bool = True
    enable_logging: bool = True
    
    # Advanced settings
    enable_ai_selection: bool = True
    enable_quantum_optimization: bool = True
    enable_neural_architecture_search: bool = True
    enable_ultra_performance_optimization: bool = True

class ModularBulkOptimizer:
    """Modular bulk optimizer with clean architecture."""
    
    def __init__(self, config: Optional[ModularOptimizerConfig] = None):
        self.config = config or ModularOptimizerConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize core components
        self.model_analyzer = ModelAnalyzer()
        self.performance_metrics = PerformanceMetrics()
        self.config_manager = ConfigManager()
        
        # Initialize orchestration
        orchestration_config = OrchestrationConfig(
            max_concurrent_optimizations=self.config.max_concurrent_optimizations,
            optimization_timeout=self.config.optimization_timeout,
            enable_parallel_processing=self.config.enable_parallel_processing,
            enable_adaptive_selection=self.config.enable_adaptive_selection,
            enable_performance_monitoring=self.config.enable_performance_monitoring,
            enable_resource_management=self.config.enable_resource_management,
            strategy_weights=self.config.strategy_weights
        )
        
        self.orchestrator = OptimizationOrchestrator(orchestration_config)
        
        # Performance tracking
        self.optimization_history = []
        self.performance_statistics = {}
        
        # Setup logging
        if self.config.enable_logging:
            self._setup_logging()
        
        self.logger.info("Modular bulk optimizer initialized")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        try:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler('modular_optimizer.log'),
                    logging.StreamHandler()
                ]
            )
            
            self.logger.info("Logging configured")
            
        except Exception as e:
            self.logger.error(f"Logging setup failed: {e}")
    
    async def optimize_models(self, models: List[Tuple[str, nn.Module]], 
                            target_improvement: Optional[float] = None,
                            preferred_strategies: Optional[List[str]] = None,
                            constraints: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Optimize multiple models using modular approach."""
        try:
            self.logger.info(f"Starting optimization of {len(models)} models")
            
            # Use config target if not specified
            target = target_improvement or self.config.target_improvement
            
            # Start performance monitoring
            if self.config.enable_performance_monitoring:
                self.performance_metrics.start_monitoring()
            
            # Execute orchestrated optimization
            results = await self.orchestrator.optimize_models_batch(
                models, target, preferred_strategies
            )
            
            # Process results
            processed_results = []
            for result in results:
                processed_result = {
                    'model_name': result.task_id,
                    'success': result.success,
                    'applied_strategies': result.applied_strategies,
                    'total_improvement': result.total_improvement,
                    'execution_time': result.execution_time,
                    'resource_usage': result.resource_usage,
                    'optimization_results': [
                        {
                            'method': opt_result.optimization_method,
                            'improvement': opt_result.performance_improvements,
                            'time': opt_result.optimization_time
                        }
                        for opt_result in result.optimization_results
                    ],
                    'metadata': result.metadata,
                    'error': result.error
                }
                processed_results.append(processed_result)
            
            # Update performance statistics
            self._update_performance_statistics(processed_results)
            
            # Stop performance monitoring
            if self.config.enable_performance_monitoring:
                self.performance_metrics.stop_monitoring()
            
            self.logger.info(f"Optimization completed: {len(processed_results)} models processed")
            return processed_results
            
        except Exception as e:
            self.logger.error(f"Model optimization failed: {e}")
            return []
    
    async def optimize_single_model(self, model: nn.Module, model_name: str,
                                  target_improvement: Optional[float] = None,
                                  preferred_strategies: Optional[List[str]] = None,
                                  constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Optimize a single model."""
        try:
            self.logger.info(f"Starting optimization of model: {model_name}")
            
            # Use config target if not specified
            target = target_improvement or self.config.target_improvement
            
            # Start performance monitoring
            if self.config.enable_performance_monitoring:
                self.performance_metrics.start_monitoring()
            
            # Execute orchestrated optimization
            result = await self.orchestrator.optimize_model(
                model, model_name, target, preferred_strategies, constraints
            )
            
            # Process result
            processed_result = {
                'model_name': model_name,
                'success': result.success,
                'applied_strategies': result.applied_strategies,
                'total_improvement': result.total_improvement,
                'execution_time': result.execution_time,
                'resource_usage': result.resource_usage,
                'optimization_results': [
                    {
                        'method': opt_result.optimization_method,
                        'improvement': opt_result.performance_improvements,
                        'time': opt_result.optimization_time
                    }
                    for opt_result in result.optimization_results
                ],
                'metadata': result.metadata,
                'error': result.error
            }
            
            # Update performance statistics
            self._update_performance_statistics([processed_result])
            
            # Stop performance monitoring
            if self.config.enable_performance_monitoring:
                self.performance_metrics.stop_monitoring()
            
            self.logger.info(f"Single model optimization completed: {model_name}")
            return processed_result
            
        except Exception as e:
            self.logger.error(f"Single model optimization failed: {e}")
            return {
                'model_name': model_name,
                'success': False,
                'error': str(e)
            }
    
    def _update_performance_statistics(self, results: List[Dict[str, Any]]):
        """Update performance statistics."""
        try:
            successful_results = [r for r in results if r['success']]
            
            if not successful_results:
                return
            
            # Calculate statistics
            avg_improvement = np.mean([r['total_improvement'] for r in successful_results])
            avg_execution_time = np.mean([r['execution_time'] for r in successful_results])
            
            # Strategy usage
            strategy_usage = {}
            for result in successful_results:
                for strategy in result['applied_strategies']:
                    strategy_usage[strategy] = strategy_usage.get(strategy, 0) + 1
            
            # Update statistics
            self.performance_statistics = {
                'total_optimizations': len(results),
                'successful_optimizations': len(successful_results),
                'success_rate': len(successful_results) / len(results),
                'avg_improvement': avg_improvement,
                'avg_execution_time': avg_execution_time,
                'strategy_usage': strategy_usage,
                'last_updated': datetime.now(timezone.utc).isoformat()
            }
            
            # Store in history
            self.optimization_history.extend(results)
            
        except Exception as e:
            self.logger.error(f"Performance statistics update failed: {e}")
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        return self.performance_statistics.copy()
    
    def get_orchestration_statistics(self) -> Dict[str, Any]:
        """Get orchestration statistics."""
        return self.orchestrator.get_orchestration_statistics()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        try:
            if self.config.enable_performance_monitoring:
                return self.performance_metrics.get_performance_statistics().__dict__
            else:
                return {}
        except Exception as e:
            self.logger.error(f"Performance metrics retrieval failed: {e}")
            return {}
    
    def export_results(self, filepath: str) -> bool:
        """Export optimization results."""
        try:
            export_data = {
                'optimization_history': self.optimization_history,
                'performance_statistics': self.performance_statistics,
                'orchestration_statistics': self.get_orchestration_statistics(),
                'performance_metrics': self.get_performance_metrics(),
                'config': self.config.__dict__,
                'export_timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            self.logger.info(f"Results exported to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Results export failed: {e}")
            return False
    
    def cleanup(self):
        """Cleanup resources."""
        try:
            if hasattr(self, 'orchestrator'):
                self.orchestrator.cleanup()
            
            if hasattr(self, 'performance_metrics'):
                self.performance_metrics.cleanup()
            
            self.logger.info("Modular bulk optimizer cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
    
    def __del__(self):
        """Destructor."""
        self.cleanup()

def create_modular_optimizer(config: Optional[ModularOptimizerConfig] = None) -> ModularBulkOptimizer:
    """Create modular bulk optimizer."""
    return ModularBulkOptimizer(config)

async def main():
    """Main function for testing."""
    # Create test models
    class TestModel(nn.Module):
        def __init__(self, size=100):
            super().__init__()
            self.linear1 = nn.Linear(size, size // 2)
            self.linear2 = nn.Linear(size // 2, 10)
            self.relu = nn.ReLU()
        
        def forward(self, x):
            x = self.relu(self.linear1(x))
            x = self.linear2(x)
            return x
    
    # Create models
    models = [
        ("test_model_1", TestModel(100)),
        ("test_model_2", TestModel(200)),
        ("test_model_3", TestModel(300))
    ]
    
    # Create optimizer
    config = ModularOptimizerConfig(
        max_concurrent_optimizations=2,
        target_improvement=0.3,
        enable_performance_monitoring=True
    )
    
    optimizer = create_modular_optimizer(config)
    
    print("🚀 Modular Bulk Optimization Demo")
    print("=" * 60)
    
    # Run optimization
    results = await optimizer.optimize_models(models)
    
    print(f"\n📊 Optimization Results:")
    print(f"   - Total models: {len(results)}")
    
    successful = [r for r in results if r['success']]
    print(f"   - Successful: {len(successful)}")
    print(f"   - Success rate: {len(successful)/len(results)*100:.1f}%")
    
    if successful:
        avg_improvement = np.mean([r['total_improvement'] for r in successful])
        avg_time = np.mean([r['execution_time'] for r in successful])
        print(f"   - Average improvement: {avg_improvement:.2%}")
        print(f"   - Average time: {avg_time:.2f}s")
    
    print(f"\n🔍 Detailed Results:")
    for result in results:
        if result['success']:
            print(f"   ✅ {result['model_name']}: {result['total_improvement']:.2%} improvement using {result['applied_strategies']}")
        else:
            print(f"   ❌ {result['model_name']}: {result.get('error', 'Unknown error')}")
    
    # Get statistics
    stats = optimizer.get_optimization_statistics()
    print(f"\n📈 Statistics:")
    print(f"   - Success rate: {stats.get('success_rate', 0):.1%}")
    print(f"   - Average improvement: {stats.get('avg_improvement', 0):.2%}")
    print(f"   - Strategy usage: {stats.get('strategy_usage', {})}")
    
    # Cleanup
    optimizer.cleanup()
    
    print("\n🎉 Modular bulk optimization demo completed!")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Modular Bulk Optimizer")
    parser.add_argument("--mode", choices=["demo", "optimize", "status"], default="demo",
                       help="Operation mode")
    parser.add_argument("--models", nargs="+", help="Model names to optimize")
    parser.add_argument("--target", type=float, default=0.5, help="Target improvement")
    parser.add_argument("--strategies", nargs="+", help="Preferred strategies")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--output", help="Output file path")
    
    args = parser.parse_args()
    
    if args.mode == "demo":
        asyncio.run(main())
    elif args.mode == "optimize":
        print("Optimization mode not implemented yet")
    elif args.mode == "status":
        print("Status mode not implemented yet")
    else:
        print(f"Unknown mode: {args.mode}")
