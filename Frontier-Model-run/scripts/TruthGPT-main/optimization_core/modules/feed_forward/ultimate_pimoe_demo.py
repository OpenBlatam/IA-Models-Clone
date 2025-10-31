"""
Ultimate PiMoE System Demonstration
Comprehensive demonstration of all advanced PiMoE features and improvements
"""

import torch
import torch.nn as nn
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
import json
from dataclasses import dataclass

from .ultimate_pimoe_system import (
    create_ultimate_pimoe_system,
    run_ultimate_pimoe_demo,
    UltimatePiMoEConfig
)
from .advanced_pimoe_routing import RoutingStrategy
from .pimoe_performance_optimizer import OptimizationLevel
from .pimoe_router import ExpertType

@dataclass
class DemoResults:
    """Results from the ultimate PiMoE demonstration."""
    system_stats: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    benchmark_results: Dict[str, Any]
    routing_analysis: Dict[str, Any]
    optimization_effects: Dict[str, Any]
    comparison_results: Dict[str, Any]

class UltimatePiMoEDemonstration:
    """
    Comprehensive demonstration of the Ultimate PiMoE system.
    """
    
    def __init__(self, config: UltimatePiMoEConfig):
        self.config = config
        self.results = {}
        
        # Initialize system
        self.system = create_ultimate_pimoe_system(
            hidden_size=config.hidden_size,
            num_experts=config.num_experts,
            expert_types=config.expert_types,
            routing_strategy=config.routing_strategy,
            optimization_level=config.optimization_level,
            enable_all_features=config.enable_all_features
        )
        
        # Generate test data
        self.test_data = self._generate_test_data()
        
    def _generate_test_data(self) -> Dict[str, torch.Tensor]:
        """Generate diverse test data for demonstration."""
        batch_size = self.config.batch_size
        seq_len = self.config.sequence_length
        hidden_size = self.config.hidden_size
        
        return {
            'mathematical': torch.randn(batch_size, seq_len, hidden_size) * 0.5 + 1.0,
            'language': torch.randn(batch_size, seq_len, hidden_size) * 0.3,
            'reasoning': torch.randn(batch_size, seq_len, hidden_size) * 0.4 + 0.2,
            'creative': torch.randn(batch_size, seq_len, hidden_size) * 0.8,
            'analytical': torch.randn(batch_size, seq_len, hidden_size) * 0.6 + 0.1,
            'logical': torch.randn(batch_size, seq_len, hidden_size) * 0.4 - 0.1
        }
    
    def run_comprehensive_demo(self) -> DemoResults:
        """Run comprehensive demonstration."""
        print("🚀 Ultimate PiMoE System Demonstration")
        print("=" * 60)
        
        # 1. System Configuration Analysis
        print("\n📋 System Configuration Analysis")
        config_analysis = self._analyze_system_configuration()
        
        # 2. Performance Benchmarking
        print("\n⚡ Performance Benchmarking")
        performance_results = self._run_performance_benchmarks()
        
        # 3. Routing Analysis
        print("\n🎯 Routing Analysis")
        routing_results = self._analyze_routing_behavior()
        
        # 4. Optimization Effects
        print("\n🔧 Optimization Effects Analysis")
        optimization_results = self._analyze_optimization_effects()
        
        # 5. Feature Comparison
        print("\n🔄 Feature Comparison")
        comparison_results = self._compare_feature_combinations()
        
        # 6. Advanced Features Demo
        print("\n🧠 Advanced Features Demonstration")
        advanced_results = self._demonstrate_advanced_features()
        
        # Compile results
        self.results = DemoResults(
            system_stats=config_analysis,
            performance_metrics=performance_results,
            benchmark_results=performance_results,
            routing_analysis=routing_results,
            optimization_effects=optimization_results,
            comparison_results=comparison_results
        )
        
        # Generate visualizations
        self._generate_visualizations()
        
        # Print summary
        self._print_demo_summary()
        
        return self.results
    
    def _analyze_system_configuration(self) -> Dict[str, Any]:
        """Analyze system configuration and capabilities."""
        stats = self.system.get_system_stats()
        
        print(f"  System Type: {stats['system_type']}")
        print(f"  Hidden Size: {stats['hidden_size']}")
        print(f"  Number of Experts: {stats['num_experts']}")
        print(f"  Expert Types: {stats['expert_types']}")
        print(f"  Routing Strategy: {stats['routing_strategy']}")
        print(f"  Optimization Level: {stats['optimization_level']}")
        
        features_enabled = stats['features_enabled']
        enabled_count = sum(features_enabled.values())
        total_count = len(features_enabled)
        
        print(f"  Features Enabled: {enabled_count}/{total_count}")
        for feature, enabled in features_enabled.items():
            status = "✅" if enabled else "❌"
            print(f"    {status} {feature.replace('_', ' ').title()}")
        
        return stats
    
    def _run_performance_benchmarks(self) -> Dict[str, Any]:
        """Run comprehensive performance benchmarks."""
        results = {}
        
        # Test different data types
        for data_type, data in self.test_data.items():
            print(f"  Testing {data_type} sequences...")
            
            # Benchmark performance
            benchmark_results = self.system.benchmark_system(data, num_iterations=20)
            
            # Process with comprehensive info
            output, comprehensive_info = self.system(data, return_comprehensive_info=True)
            
            results[data_type] = {
                'benchmark': benchmark_results,
                'performance_metrics': comprehensive_info['performance_metrics'],
                'routing_info': comprehensive_info['routing_info'],
                'output_shape': output.shape
            }
            
            print(f"    Latency: {benchmark_results['average_time']:.4f} s")
            print(f"    Throughput: {benchmark_results['throughput']:.2f} tokens/sec")
        
        return results
    
    def _analyze_routing_behavior(self) -> Dict[str, Any]:
        """Analyze routing behavior across different data types."""
        routing_analysis = {}
        
        for data_type, data in self.test_data.items():
            print(f"  Analyzing {data_type} routing...")
            
            # Get routing information
            output, comprehensive_info = self.system(data, return_comprehensive_info=True)
            routing_info = comprehensive_info['routing_info']
            
            if routing_info and 'routing_decisions' in routing_info:
                decisions = routing_info['routing_decisions']
                
                # Analyze expert usage
                expert_usage = {}
                expert_type_usage = {}
                confidence_scores = []
                
                for decision in decisions:
                    expert_id = decision.expert_id
                    expert_type = decision.expert_type
                    confidence = decision.confidence
                    
                    expert_usage[expert_id] = expert_usage.get(expert_id, 0) + 1
                    expert_type_usage[expert_type.value] = expert_type_usage.get(expert_type.value, 0) + 1
                    confidence_scores.append(confidence)
                
                # Calculate statistics
                unique_experts = len(expert_usage)
                avg_confidence = np.mean(confidence_scores)
                confidence_std = np.std(confidence_scores)
                
                routing_analysis[data_type] = {
                    'unique_experts_used': unique_experts,
                    'expert_usage_distribution': expert_usage,
                    'expert_type_distribution': expert_type_usage,
                    'average_confidence': avg_confidence,
                    'confidence_std': confidence_std,
                    'routing_entropy': self._calculate_routing_entropy(expert_usage)
                }
                
                print(f"    Unique experts used: {unique_experts}")
                print(f"    Average confidence: {avg_confidence:.3f}")
                print(f"    Routing entropy: {routing_analysis[data_type]['routing_entropy']:.3f}")
        
        return routing_analysis
    
    def _calculate_routing_entropy(self, expert_usage: Dict[int, int]) -> float:
        """Calculate entropy of expert usage distribution."""
        total_usage = sum(expert_usage.values())
        if total_usage == 0:
            return 0.0
        
        entropy = 0.0
        for count in expert_usage.values():
            p = count / total_usage
            if p > 0:
                entropy -= p * np.log2(p)
        
        return entropy
    
    def _analyze_optimization_effects(self) -> Dict[str, Any]:
        """Analyze the effects of different optimizations."""
        optimization_effects = {}
        
        # Test with different optimization levels
        optimization_levels = [
            OptimizationLevel.BASIC,
            OptimizationLevel.INTERMEDIATE,
            OptimizationLevel.ADVANCED,
            OptimizationLevel.EXTREME
        ]
        
        for level in optimization_levels:
            print(f"  Testing {level.value} optimization...")
            
            # Create system with specific optimization level
            test_system = create_ultimate_pimoe_system(
                hidden_size=self.config.hidden_size,
                num_experts=self.config.num_experts,
                optimization_level=level,
                enable_all_features=True
            )
            
            # Benchmark performance
            test_data = self.test_data['mathematical']
            benchmark_results = test_system.benchmark_system(test_data, num_iterations=10)
            
            optimization_effects[level.value] = {
                'latency': benchmark_results['average_time'],
                'throughput': benchmark_results['throughput'],
                'total_time': benchmark_results['total_time']
            }
            
            print(f"    Latency: {benchmark_results['average_time']:.4f} s")
            print(f"    Throughput: {benchmark_results['throughput']:.2f} tokens/sec")
        
        return optimization_effects
    
    def _compare_feature_combinations(self) -> Dict[str, Any]:
        """Compare different feature combinations."""
        feature_combinations = [
            {'name': 'Basic', 'enable_all_features': False},
            {'name': 'Advanced Routing', 'routing_strategy': RoutingStrategy.ATTENTION_BASED, 'enable_all_features': False},
            {'name': 'Performance Optimization', 'optimization_level': OptimizationLevel.ADVANCED, 'enable_all_features': False},
            {'name': 'All Features', 'enable_all_features': True}
        ]
        
        comparison_results = {}
        
        for combo in feature_combinations:
            print(f"  Testing {combo['name']} configuration...")
            
            # Create system with specific configuration
            test_system = create_ultimate_pimoe_system(
                hidden_size=self.config.hidden_size,
                num_experts=self.config.num_experts,
                **{k: v for k, v in combo.items() if k != 'name'}
            )
            
            # Benchmark performance
            test_data = self.test_data['mathematical']
            benchmark_results = test_system.benchmark_system(test_data, num_iterations=10)
            
            comparison_results[combo['name']] = {
                'latency': benchmark_results['average_time'],
                'throughput': benchmark_results['throughput'],
                'total_time': benchmark_results['total_time']
            }
            
            print(f"    Latency: {benchmark_results['average_time']:.4f} s")
            print(f"    Throughput: {benchmark_results['throughput']:.2f} tokens/sec")
        
        return comparison_results
    
    def _demonstrate_advanced_features(self) -> Dict[str, Any]:
        """Demonstrate advanced features."""
        advanced_results = {}
        
        # Test dynamic scaling
        print("  Testing dynamic expert scaling...")
        if hasattr(self.system, 'expert_scaler') and self.system.expert_scaler:
            expert_loads = torch.rand(self.config.num_experts)
            expert_performance = torch.rand(self.config.num_experts)
            scaling_decision = self.system.expert_scaler(expert_loads, expert_performance)
            advanced_results['dynamic_scaling'] = scaling_decision
            print(f"    Scaling decision: {scaling_decision['action']}")
            print(f"    Performance score: {scaling_decision['scaling_decision']:.3f}")
        
        # Test cross-expert communication
        print("  Testing cross-expert communication...")
        if hasattr(self.system, 'communicator') and self.system.communicator:
            # Simulate expert outputs
            expert_outputs = [torch.randn(2, self.config.hidden_size) for _ in range(self.config.num_experts)]
            expert_ids = list(range(self.config.num_experts))
            
            communicated_outputs = self.system.communicator(expert_outputs, expert_ids)
            advanced_results['cross_expert_communication'] = {
                'input_experts': len(expert_outputs),
                'output_experts': len(communicated_outputs),
                'communication_successful': True
            }
            print(f"    Communication successful: {len(communicated_outputs)} experts processed")
        
        # Test neural architecture search
        print("  Testing neural architecture search...")
        if hasattr(self.system, 'nas_router') and self.system.nas_router:
            # Test architecture evaluation
            architecture = {
                'num_layers': 2,
                'hidden_sizes': self.config.hidden_size,
                'activations': 'relu',
                'dropout_rates': 0.1,
                'normalization': 'layer_norm'
            }
            
            performance_metrics = {
                'latency_ms': 10.0,
                'throughput_tokens_per_sec': 1000.0,
                'memory_usage_mb': 100.0
            }
            
            fitness = self.system.nas_router.evaluate_architecture(architecture, performance_metrics)
            advanced_results['neural_architecture_search'] = {
                'architecture_fitness': fitness,
                'search_space_size': self.system.nas_router.search_space_size,
                'population_size': self.system.nas_router.population_size
            }
            print(f"    Architecture fitness: {fitness:.3f}")
            print(f"    Search space size: {self.system.nas_router.search_space_size}")
        
        return advanced_results
    
    def _generate_visualizations(self):
        """Generate visualization plots."""
        try:
            # Performance comparison plot
            self._plot_performance_comparison()
            
            # Routing analysis plot
            self._plot_routing_analysis()
            
            # Optimization effects plot
            self._plot_optimization_effects()
            
            print("\n📊 Visualizations generated successfully!")
        except Exception as e:
            print(f"\n⚠️  Visualization generation failed: {e}")
    
    def _plot_performance_comparison(self):
        """Plot performance comparison across data types."""
        if not self.results.performance_metrics:
            return
        
        data_types = list(self.results.performance_metrics.keys())
        latencies = [self.results.performance_metrics[dt]['benchmark']['average_time'] for dt in data_types]
        throughputs = [self.results.performance_metrics[dt]['benchmark']['throughput'] for dt in data_types]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Latency comparison
        ax1.bar(data_types, latencies, color=['blue', 'green', 'red', 'orange', 'purple', 'brown'])
        ax1.set_title('Latency Comparison by Data Type')
        ax1.set_ylabel('Latency (seconds)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Throughput comparison
        ax2.bar(data_types, throughputs, color=['blue', 'green', 'red', 'orange', 'purple', 'brown'])
        ax2.set_title('Throughput Comparison by Data Type')
        ax2.set_ylabel('Throughput (tokens/sec)')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('ultimate_pimoe_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_routing_analysis(self):
        """Plot routing analysis results."""
        if not self.results.routing_analysis:
            return
        
        data_types = list(self.results.routing_analysis.keys())
        unique_experts = [self.results.routing_analysis[dt]['unique_experts_used'] for dt in data_types]
        confidences = [self.results.routing_analysis[dt]['average_confidence'] for dt in data_types]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Unique experts used
        ax1.bar(data_types, unique_experts, color=['blue', 'green', 'red', 'orange', 'purple', 'brown'])
        ax1.set_title('Unique Experts Used by Data Type')
        ax1.set_ylabel('Number of Experts')
        ax1.tick_params(axis='x', rotation=45)
        
        # Average confidence
        ax2.bar(data_types, confidences, color=['blue', 'green', 'red', 'orange', 'purple', 'brown'])
        ax2.set_title('Average Routing Confidence by Data Type')
        ax2.set_ylabel('Confidence Score')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('ultimate_pimoe_routing_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_optimization_effects(self):
        """Plot optimization effects."""
        if not self.results.optimization_effects:
            return
        
        levels = list(self.results.optimization_effects.keys())
        latencies = [self.results.optimization_effects[level]['latency'] for level in levels]
        throughputs = [self.results.optimization_effects[level]['throughput'] for level in levels]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Latency by optimization level
        ax1.plot(levels, latencies, 'b-o', linewidth=2, markersize=8)
        ax1.set_title('Latency by Optimization Level')
        ax1.set_ylabel('Latency (seconds)')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Throughput by optimization level
        ax2.plot(levels, throughputs, 'g-o', linewidth=2, markersize=8)
        ax2.set_title('Throughput by Optimization Level')
        ax2.set_ylabel('Throughput (tokens/sec)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('ultimate_pimoe_optimization_effects.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _print_demo_summary(self):
        """Print comprehensive demo summary."""
        print("\n" + "=" * 60)
        print("📋 ULTIMATE PIMOE DEMO SUMMARY")
        print("=" * 60)
        
        # System configuration
        print(f"\n🔧 System Configuration:")
        print(f"  Hidden Size: {self.config.hidden_size}")
        print(f"  Number of Experts: {self.config.num_experts}")
        print(f"  Routing Strategy: {self.config.routing_strategy.value}")
        print(f"  Optimization Level: {self.config.optimization_level.value}")
        print(f"  All Features Enabled: {self.config.enable_all_features}")
        
        # Performance summary
        if self.results.performance_metrics:
            print(f"\n⚡ Performance Summary:")
            for data_type, metrics in self.results.performance_metrics.items():
                benchmark = metrics['benchmark']
                print(f"  {data_type.title()}:")
                print(f"    Latency: {benchmark['average_time']:.4f} s")
                print(f"    Throughput: {benchmark['throughput']:.2f} tokens/sec")
        
        # Routing analysis summary
        if self.results.routing_analysis:
            print(f"\n🎯 Routing Analysis Summary:")
            for data_type, analysis in self.results.routing_analysis.items():
                print(f"  {data_type.title()}:")
                print(f"    Unique Experts: {analysis['unique_experts_used']}")
                print(f"    Average Confidence: {analysis['average_confidence']:.3f}")
                print(f"    Routing Entropy: {analysis['routing_entropy']:.3f}")
        
        # Optimization effects summary
        if self.results.optimization_effects:
            print(f"\n🔧 Optimization Effects Summary:")
            for level, effects in self.results.optimization_effects.items():
                print(f"  {level.title()}:")
                print(f"    Latency: {effects['latency']:.4f} s")
                print(f"    Throughput: {effects['throughput']:.2f} tokens/sec")
        
        # Advanced features summary
        if self.results.comparison_results:
            print(f"\n🧠 Advanced Features Summary:")
            for feature, results in self.results.comparison_results.items():
                print(f"  {feature}:")
                print(f"    Latency: {results['latency']:.4f} s")
                print(f"    Throughput: {results['throughput']:.2f} tokens/sec")
        
        print(f"\n✅ Ultimate PiMoE demonstration completed successfully!")
        print(f"📊 Results saved and visualizations generated!")

def run_ultimate_pimoe_demonstration():
    """Run the ultimate PiMoE demonstration."""
    
    # Configuration
    config = UltimatePiMoEConfig(
        hidden_size=512,
        num_experts=8,
        expert_types=[
            ExpertType.REASONING,
            ExpertType.COMPUTATION,
            ExpertType.MATHEMATICAL,
            ExpertType.LOGICAL,
            ExpertType.LANGUAGE,
            ExpertType.CREATIVE,
            ExpertType.ANALYTICAL
        ],
        routing_strategy=RoutingStrategy.ATTENTION_BASED,
        optimization_level=OptimizationLevel.ADVANCED,
        enable_all_features=True,
        batch_size=2,
        sequence_length=128
    )
    
    # Create demonstration
    demo = UltimatePiMoEDemonstration(config)
    
    # Run comprehensive demonstration
    results = demo.run_comprehensive_demo()
    
    # Save results
    with open('ultimate_pimoe_demo_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to ultimate_pimoe_demo_results.json")
    
    return results

if __name__ == "__main__":
    # Run the ultimate demonstration
    results = run_ultimate_pimoe_demonstration()




