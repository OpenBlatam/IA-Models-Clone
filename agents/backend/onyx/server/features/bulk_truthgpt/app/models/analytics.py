"""
Analytics models for Ultimate Enhanced Supreme Production system
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from datetime import datetime

@dataclass
class UsageMetrics:
    """Usage metrics model."""
    total_queries: int = 0
    total_documents_generated: int = 0
    total_processing_time: float = 0.0
    average_processing_time: float = 0.0
    average_documents_per_query: float = 0.0
    peak_concurrent_generations: int = 0
    optimization_usage: Dict[str, int] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.optimization_usage is None:
            self.optimization_usage = {}
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

@dataclass
class PerformanceAnalytics:
    """Performance analytics model."""
    supreme_speed_improvement: float = 0.0
    supreme_memory_reduction: float = 0.0
    supreme_accuracy_preservation: float = 0.0
    supreme_energy_efficiency: float = 0.0
    
    ultra_fast_speed_improvement: float = 0.0
    ultra_fast_memory_reduction: float = 0.0
    ultra_fast_accuracy_preservation: float = 0.0
    ultra_fast_energy_efficiency: float = 0.0
    
    refactored_ultimate_hybrid_speed_improvement: float = 0.0
    refactored_ultimate_hybrid_memory_reduction: float = 0.0
    refactored_ultimate_hybrid_accuracy_preservation: float = 0.0
    refactored_ultimate_hybrid_energy_efficiency: float = 0.0
    
    cuda_kernel_speed_improvement: float = 0.0
    cuda_kernel_memory_reduction: float = 0.0
    cuda_kernel_accuracy_preservation: float = 0.0
    cuda_kernel_energy_efficiency: float = 0.0
    
    gpu_utilization_speed_improvement: float = 0.0
    gpu_utilization_memory_reduction: float = 0.0
    gpu_utilization_accuracy_preservation: float = 0.0
    gpu_utilization_energy_efficiency: float = 0.0
    
    memory_optimization_speed_improvement: float = 0.0
    memory_optimization_memory_reduction: float = 0.0
    memory_optimization_accuracy_preservation: float = 0.0
    memory_optimization_energy_efficiency: float = 0.0
    
    reward_function_speed_improvement: float = 0.0
    reward_function_memory_reduction: float = 0.0
    reward_function_accuracy_preservation: float = 0.0
    reward_function_energy_efficiency: float = 0.0
    
    truthgpt_adapter_speed_improvement: float = 0.0
    truthgpt_adapter_memory_reduction: float = 0.0
    truthgpt_adapter_accuracy_preservation: float = 0.0
    truthgpt_adapter_energy_efficiency: float = 0.0
    
    microservices_speed_improvement: float = 0.0
    microservices_memory_reduction: float = 0.0
    microservices_accuracy_preservation: float = 0.0
    microservices_energy_efficiency: float = 0.0
    
    combined_ultimate_enhanced_speed_improvement: float = 0.0
    combined_ultimate_enhanced_memory_reduction: float = 0.0
    combined_ultimate_enhanced_accuracy_preservation: float = 0.0
    combined_ultimate_enhanced_energy_efficiency: float = 0.0
    
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

@dataclass
class AnalyticsData:
    """Analytics data model."""
    usage_metrics: UsageMetrics
    performance_analytics: PerformanceAnalytics
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'usage_metrics': {
                'total_queries': self.usage_metrics.total_queries,
                'total_documents_generated': self.usage_metrics.total_documents_generated,
                'total_processing_time': self.usage_metrics.total_processing_time,
                'average_processing_time': self.usage_metrics.average_processing_time,
                'average_documents_per_query': self.usage_metrics.average_documents_per_query,
                'peak_concurrent_generations': self.usage_metrics.peak_concurrent_generations,
                'optimization_usage': self.usage_metrics.optimization_usage,
                'timestamp': self.usage_metrics.timestamp.isoformat() if self.usage_metrics.timestamp else None
            },
            'performance_analytics': {
                'supreme_metrics': {
                    'speed_improvement': self.performance_analytics.supreme_speed_improvement,
                    'memory_reduction': self.performance_analytics.supreme_memory_reduction,
                    'accuracy_preservation': self.performance_analytics.supreme_accuracy_preservation,
                    'energy_efficiency': self.performance_analytics.supreme_energy_efficiency
                },
                'ultra_fast_metrics': {
                    'speed_improvement': self.performance_analytics.ultra_fast_speed_improvement,
                    'memory_reduction': self.performance_analytics.ultra_fast_memory_reduction,
                    'accuracy_preservation': self.performance_analytics.ultra_fast_accuracy_preservation,
                    'energy_efficiency': self.performance_analytics.ultra_fast_energy_efficiency
                },
                'refactored_ultimate_hybrid_metrics': {
                    'speed_improvement': self.performance_analytics.refactored_ultimate_hybrid_speed_improvement,
                    'memory_reduction': self.performance_analytics.refactored_ultimate_hybrid_memory_reduction,
                    'accuracy_preservation': self.performance_analytics.refactored_ultimate_hybrid_accuracy_preservation,
                    'energy_efficiency': self.performance_analytics.refactored_ultimate_hybrid_energy_efficiency
                },
                'cuda_kernel_metrics': {
                    'speed_improvement': self.performance_analytics.cuda_kernel_speed_improvement,
                    'memory_reduction': self.performance_analytics.cuda_kernel_memory_reduction,
                    'accuracy_preservation': self.performance_analytics.cuda_kernel_accuracy_preservation,
                    'energy_efficiency': self.performance_analytics.cuda_kernel_energy_efficiency
                },
                'gpu_utilization_metrics': {
                    'speed_improvement': self.performance_analytics.gpu_utilization_speed_improvement,
                    'memory_reduction': self.performance_analytics.gpu_utilization_memory_reduction,
                    'accuracy_preservation': self.performance_analytics.gpu_utilization_accuracy_preservation,
                    'energy_efficiency': self.performance_analytics.gpu_utilization_energy_efficiency
                },
                'memory_optimization_metrics': {
                    'speed_improvement': self.performance_analytics.memory_optimization_speed_improvement,
                    'memory_reduction': self.performance_analytics.memory_optimization_memory_reduction,
                    'accuracy_preservation': self.performance_analytics.memory_optimization_accuracy_preservation,
                    'energy_efficiency': self.performance_analytics.memory_optimization_energy_efficiency
                },
                'reward_function_metrics': {
                    'speed_improvement': self.performance_analytics.reward_function_speed_improvement,
                    'memory_reduction': self.performance_analytics.reward_function_memory_reduction,
                    'accuracy_preservation': self.performance_analytics.reward_function_accuracy_preservation,
                    'energy_efficiency': self.performance_analytics.reward_function_energy_efficiency
                },
                'truthgpt_adapter_metrics': {
                    'speed_improvement': self.performance_analytics.truthgpt_adapter_speed_improvement,
                    'memory_reduction': self.performance_analytics.truthgpt_adapter_memory_reduction,
                    'accuracy_preservation': self.performance_analytics.truthgpt_adapter_accuracy_preservation,
                    'energy_efficiency': self.performance_analytics.truthgpt_adapter_energy_efficiency
                },
                'microservices_metrics': {
                    'speed_improvement': self.performance_analytics.microservices_speed_improvement,
                    'memory_reduction': self.performance_analytics.microservices_memory_reduction,
                    'accuracy_preservation': self.performance_analytics.microservices_accuracy_preservation,
                    'energy_efficiency': self.performance_analytics.microservices_energy_efficiency
                },
                'combined_ultimate_enhanced_metrics': {
                    'speed_improvement': self.performance_analytics.combined_ultimate_enhanced_speed_improvement,
                    'memory_reduction': self.performance_analytics.combined_ultimate_enhanced_memory_reduction,
                    'accuracy_preservation': self.performance_analytics.combined_ultimate_enhanced_accuracy_preservation,
                    'energy_efficiency': self.performance_analytics.combined_ultimate_enhanced_energy_efficiency
                },
                'timestamp': self.performance_analytics.timestamp.isoformat() if self.performance_analytics.timestamp else None
            },
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }









