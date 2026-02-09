#!/usr/bin/env python3
"""
Model Analyzer - Comprehensive model analysis and profiling
Provides detailed analysis of model characteristics for optimization decisions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import numpy as np
from pathlib import Path
import json
import uuid
from datetime import datetime, timezone
from collections import defaultdict, Counter
import networkx as nx

@dataclass
class LayerInfo:
    """Information about a model layer."""
    name: str
    type: str
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    parameters: int
    memory_usage: float
    computational_cost: float
    is_trainable: bool

@dataclass
class ArchitectureInfo:
    """Architecture analysis information."""
    total_layers: int
    layer_types: Dict[str, int]
    parameter_distribution: Dict[str, int]
    memory_distribution: Dict[str, float]
    computational_distribution: Dict[str, float]
    connectivity_graph: nx.DiGraph
    critical_path: List[str]
    bottlenecks: List[str]

@dataclass
class PerformanceProfile:
    """Model performance profile."""
    inference_time: float
    memory_peak: float
    memory_average: float
    throughput: float
    latency: float
    efficiency_score: float
    scalability_score: float

class ModelAnalyzer:
    """Comprehensive model analyzer."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.analysis_cache = {}
        self.performance_cache = {}
    
    def analyze_model(self, model: nn.Module, model_name: str = "model") -> Dict[str, Any]:
        """Comprehensive model analysis."""
        try:
            # Check cache
            cache_key = f"{model_name}_{id(model)}"
            if cache_key in self.analysis_cache:
                return self.analysis_cache[cache_key]
            
            # Basic model info
            basic_info = self._analyze_basic_info(model, model_name)
            
            # Architecture analysis
            architecture_info = self._analyze_architecture(model)
            
            # Layer analysis
            layer_info = self._analyze_layers(model)
            
            # Performance analysis
            performance_profile = self._analyze_performance(model)
            
            # Optimization recommendations
            recommendations = self._generate_recommendations(basic_info, architecture_info, performance_profile)
            
            # Compile results
            analysis_result = {
                'model_name': model_name,
                'analysis_timestamp': datetime.now(timezone.utc).isoformat(),
                'basic_info': basic_info,
                'architecture_info': architecture_info,
                'layer_info': layer_info,
                'performance_profile': performance_profile,
                'recommendations': recommendations,
                'analysis_metadata': {
                    'total_analysis_time': time.time(),
                    'cache_hit': False
                }
            }
            
            # Cache result
            self.analysis_cache[cache_key] = analysis_result
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Model analysis failed: {e}")
            return {
                'model_name': model_name,
                'error': str(e),
                'analysis_timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    def _analyze_basic_info(self, model: nn.Module, model_name: str) -> Dict[str, Any]:
        """Analyze basic model information."""
        try:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            frozen_params = total_params - trainable_params
            
            # Memory usage estimation
            memory_usage = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)  # MB
            
            # Model depth
            max_depth = self._calculate_model_depth(model)
            
            # Complexity metrics
            complexity_score = self._calculate_complexity_score(model)
            
            return {
                'model_name': model_name,
                'model_type': type(model).__name__,
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'frozen_parameters': frozen_params,
                'memory_usage_mb': memory_usage,
                'model_depth': max_depth,
                'complexity_score': complexity_score,
                'parameter_efficiency': trainable_params / max(total_params, 1)
            }
            
        except Exception as e:
            self.logger.error(f"Basic info analysis failed: {e}")
            return {}
    
    def _analyze_architecture(self, model: nn.Module) -> ArchitectureInfo:
        """Analyze model architecture."""
        try:
            layers = list(model.modules())
            layer_types = Counter(type(layer).__name__ for layer in layers if len(list(layer.parameters())) > 0)
            
            # Parameter distribution
            parameter_distribution = defaultdict(int)
            memory_distribution = defaultdict(float)
            computational_distribution = defaultdict(float)
            
            for name, module in model.named_modules():
                if len(list(module.parameters())) > 0:
                    layer_type = type(module).__name__
                    params = sum(p.numel() for p in module.parameters())
                    memory = sum(p.numel() * p.element_size() for p in module.parameters()) / (1024**2)
                    
                    parameter_distribution[layer_type] += params
                    memory_distribution[layer_type] += memory
                    computational_distribution[layer_type] += self._estimate_computational_cost(module)
            
            # Build connectivity graph
            connectivity_graph = self._build_connectivity_graph(model)
            
            # Find critical path
            critical_path = self._find_critical_path(connectivity_graph)
            
            # Identify bottlenecks
            bottlenecks = self._identify_bottlenecks(connectivity_graph, parameter_distribution)
            
            return ArchitectureInfo(
                total_layers=len(layers),
                layer_types=dict(layer_types),
                parameter_distribution=dict(parameter_distribution),
                memory_distribution=dict(memory_distribution),
                computational_distribution=dict(computational_distribution),
                connectivity_graph=connectivity_graph,
                critical_path=critical_path,
                bottlenecks=bottlenecks
            )
            
        except Exception as e:
            self.logger.error(f"Architecture analysis failed: {e}")
            return ArchitectureInfo(
                total_layers=0,
                layer_types={},
                parameter_distribution={},
                memory_distribution={},
                computational_distribution={},
                connectivity_graph=nx.DiGraph(),
                critical_path=[],
                bottlenecks=[]
            )
    
    def _analyze_layers(self, model: nn.Module) -> List[LayerInfo]:
        """Analyze individual layers."""
        layer_info_list = []
        
        try:
            for name, module in model.named_modules():
                if len(list(module.parameters())) > 0:
                    # Get layer information
                    params = sum(p.numel() for p in module.parameters())
                    memory = sum(p.numel() * p.element_size() for p in module.parameters()) / (1024**2)
                    computational_cost = self._estimate_computational_cost(module)
                    is_trainable = any(p.requires_grad for p in module.parameters())
                    
                    # Estimate shapes (simplified)
                    input_shape = self._estimate_input_shape(module)
                    output_shape = self._estimate_output_shape(module, input_shape)
                    
                    layer_info = LayerInfo(
                        name=name,
                        type=type(module).__name__,
                        input_shape=input_shape,
                        output_shape=output_shape,
                        parameters=params,
                        memory_usage=memory,
                        computational_cost=computational_cost,
                        is_trainable=is_trainable
                    )
                    
                    layer_info_list.append(layer_info)
            
            return layer_info_list
            
        except Exception as e:
            self.logger.error(f"Layer analysis failed: {e}")
            return []
    
    def _analyze_performance(self, model: nn.Module) -> PerformanceProfile:
        """Analyze model performance."""
        try:
            # Check cache
            cache_key = f"performance_{id(model)}"
            if cache_key in self.performance_cache:
                return self.performance_cache[cache_key]
            
            # Benchmark model
            inference_time, memory_peak, memory_average = self._benchmark_model(model)
            
            # Calculate metrics
            throughput = 1.0 / max(inference_time, 0.001)
            latency = inference_time * 1000  # Convert to ms
            efficiency_score = self._calculate_efficiency_score(model, inference_time, memory_peak)
            scalability_score = self._calculate_scalability_score(model)
            
            profile = PerformanceProfile(
                inference_time=inference_time,
                memory_peak=memory_peak,
                memory_average=memory_average,
                throughput=throughput,
                latency=latency,
                efficiency_score=efficiency_score,
                scalability_score=scalability_score
            )
            
            # Cache result
            self.performance_cache[cache_key] = profile
            
            return profile
            
        except Exception as e:
            self.logger.error(f"Performance analysis failed: {e}")
            return PerformanceProfile(
                inference_time=0.0,
                memory_peak=0.0,
                memory_average=0.0,
                throughput=0.0,
                latency=0.0,
                efficiency_score=0.0,
                scalability_score=0.0
            )
    
    def _calculate_model_depth(self, model: nn.Module) -> int:
        """Calculate model depth."""
        try:
            max_depth = 0
            
            def traverse(module, depth):
                nonlocal max_depth
                max_depth = max(max_depth, depth)
                for child in module.children():
                    traverse(child, depth + 1)
            
            traverse(model, 0)
            return max_depth
            
        except Exception as e:
            self.logger.warning(f"Depth calculation failed: {e}")
            return 0
    
    def _calculate_complexity_score(self, model: nn.Module) -> float:
        """Calculate model complexity score."""
        try:
            total_params = sum(p.numel() for p in model.parameters())
            num_layers = len(list(model.modules()))
            
            # Normalize complexity
            param_complexity = min(1.0, total_params / 10000000)  # 10M params = 1.0
            layer_complexity = min(1.0, num_layers / 100)  # 100 layers = 1.0
            
            return (param_complexity + layer_complexity) / 2.0
            
        except Exception as e:
            self.logger.warning(f"Complexity calculation failed: {e}")
            return 0.0
    
    def _estimate_computational_cost(self, module: nn.Module) -> float:
        """Estimate computational cost of a module."""
        try:
            if isinstance(module, nn.Linear):
                return module.in_features * module.out_features
            elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                # Simplified convolution cost estimation
                kernel_size = module.kernel_size if isinstance(module.kernel_size, int) else module.kernel_size[0]
                return module.in_channels * module.out_channels * kernel_size
            else:
                # Default estimation
                return sum(p.numel() for p in module.parameters())
                
        except Exception as e:
            self.logger.warning(f"Computational cost estimation failed: {e}")
            return 0.0
    
    def _build_connectivity_graph(self, model: nn.Module) -> nx.DiGraph:
        """Build connectivity graph of the model."""
        graph = nx.DiGraph()
        
        try:
            for name, module in model.named_modules():
                if len(list(module.parameters())) > 0:
                    graph.add_node(name, module_type=type(module).__name__)
                    
                    # Add edges to children
                    for child_name, child_module in module.named_children():
                        if len(list(child_module.parameters())) > 0:
                            graph.add_edge(name, f"{name}.{child_name}")
            
            return graph
            
        except Exception as e:
            self.logger.warning(f"Connectivity graph construction failed: {e}")
            return nx.DiGraph()
    
    def _find_critical_path(self, graph: nx.DiGraph) -> List[str]:
        """Find critical path in the model."""
        try:
            if not graph.nodes():
                return []
            
            # Find longest path (simplified)
            try:
                longest_path = nx.dag_longest_path(graph)
                return longest_path
            except:
                # Fallback to topological sort
                return list(nx.topological_sort(graph))
                
        except Exception as e:
            self.logger.warning(f"Critical path finding failed: {e}")
            return []
    
    def _identify_bottlenecks(self, graph: nx.DiGraph, parameter_distribution: Dict[str, int]) -> List[str]:
        """Identify potential bottlenecks."""
        try:
            bottlenecks = []
            
            # Find nodes with high parameter count
            for node in graph.nodes():
                if graph.nodes[node].get('module_type') in parameter_distribution:
                    if parameter_distribution[graph.nodes[node]['module_type']] > 1000000:  # 1M params
                        bottlenecks.append(node)
            
            return bottlenecks
            
        except Exception as e:
            self.logger.warning(f"Bottleneck identification failed: {e}")
            return []
    
    def _benchmark_model(self, model: nn.Module) -> Tuple[float, float, float]:
        """Benchmark model performance."""
        try:
            model.eval()
            
            # Create dummy input
            dummy_input = torch.randn(1, 3, 224, 224)  # Standard input size
            
            # Warmup
            with torch.no_grad():
                for _ in range(10):
                    _ = model(dummy_input)
            
            # Benchmark
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            
            for _ in range(100):
                with torch.no_grad():
                    _ = model(dummy_input)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()
            
            inference_time = (end_time - start_time) / 100
            
            # Memory usage (simplified)
            memory_peak = 0.0
            memory_average = 0.0
            
            if torch.cuda.is_available():
                memory_peak = torch.cuda.max_memory_allocated() / (1024**3)  # GB
                memory_average = torch.cuda.memory_allocated() / (1024**3)  # GB
            
            return inference_time, memory_peak, memory_average
            
        except Exception as e:
            self.logger.warning(f"Model benchmarking failed: {e}")
            return 0.0, 0.0, 0.0
    
    def _calculate_efficiency_score(self, model: nn.Module, inference_time: float, memory_peak: float) -> float:
        """Calculate model efficiency score."""
        try:
            total_params = sum(p.numel() for p in model.parameters())
            
            # Efficiency based on parameters vs performance
            param_efficiency = 1000000 / max(total_params, 1)  # 1M params = 1.0
            time_efficiency = 1.0 / max(inference_time, 0.001)
            memory_efficiency = 1.0 / max(memory_peak, 0.001)
            
            # Combine scores
            efficiency_score = (param_efficiency + time_efficiency + memory_efficiency) / 3.0
            return min(1.0, efficiency_score)
            
        except Exception as e:
            self.logger.warning(f"Efficiency score calculation failed: {e}")
            return 0.0
    
    def _calculate_scalability_score(self, model: nn.Module) -> float:
        """Calculate model scalability score."""
        try:
            total_params = sum(p.numel() for p in model.parameters())
            num_layers = len(list(model.modules()))
            
            # Scalability based on model size and complexity
            param_scalability = 1.0 / (1.0 + total_params / 10000000)  # Penalty for large models
            layer_scalability = 1.0 / (1.0 + num_layers / 100)  # Penalty for deep models
            
            return (param_scalability + layer_scalability) / 2.0
            
        except Exception as e:
            self.logger.warning(f"Scalability score calculation failed: {e}")
            return 0.0
    
    def _estimate_input_shape(self, module: nn.Module) -> Tuple[int, ...]:
        """Estimate input shape for a module."""
        # Simplified shape estimation
        if isinstance(module, nn.Linear):
            return (module.in_features,)
        elif isinstance(module, nn.Conv2d):
            return (module.in_channels, 224, 224)  # Default image size
        else:
            return (1, 128)  # Default shape
    
    def _estimate_output_shape(self, module: nn.Module, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Estimate output shape for a module."""
        # Simplified shape estimation
        if isinstance(module, nn.Linear):
            return (module.out_features,)
        elif isinstance(module, nn.Conv2d):
            # Simplified convolution output calculation
            return (module.out_channels, 224, 224)
        else:
            return input_shape  # Default to input shape
    
    def _generate_recommendations(self, basic_info: Dict[str, Any], 
                                architecture_info: ArchitectureInfo, 
                                performance_profile: PerformanceProfile) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        try:
            # Memory recommendations
            if basic_info.get('memory_usage_mb', 0) > 1000:
                recommendations.append("Consider quantization to reduce memory usage")
            
            # Parameter recommendations
            if basic_info.get('total_parameters', 0) > 10000000:
                recommendations.append("Consider pruning to reduce parameter count")
            
            # Performance recommendations
            if performance_profile.efficiency_score < 0.5:
                recommendations.append("Consider architecture optimization for better efficiency")
            
            # Bottleneck recommendations
            if architecture_info.bottlenecks:
                recommendations.append(f"Optimize bottlenecks: {', '.join(architecture_info.bottlenecks[:3])}")
            
            # Layer type recommendations
            if architecture_info.layer_types.get('Linear', 0) > 10:
                recommendations.append("Consider replacing some Linear layers with more efficient alternatives")
            
            return recommendations
            
        except Exception as e:
            self.logger.warning(f"Recommendation generation failed: {e}")
            return ["Unable to generate recommendations"]
    
    def clear_cache(self):
        """Clear analysis cache."""
        self.analysis_cache.clear()
        self.performance_cache.clear()
        self.logger.info("Analysis cache cleared")
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'analysis_cache_size': len(self.analysis_cache),
            'performance_cache_size': len(self.performance_cache),
            'total_cache_size': len(self.analysis_cache) + len(self.performance_cache)
        }
