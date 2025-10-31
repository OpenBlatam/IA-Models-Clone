"""
Optimization models for Ultimate Enhanced Supreme Production system
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from datetime import datetime

@dataclass
class OptimizationResult:
    """Optimization result model."""
    speed_improvement: float
    memory_reduction: float
    accuracy_preservation: float
    energy_efficiency: float
    optimization_time: float
    level: str
    techniques_applied: List[str]
    performance_metrics: Dict[str, Any]
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

@dataclass
class OptimizationMetrics:
    """Optimization metrics model."""
    supreme_speed_improvement: float = 0.0
    supreme_memory_reduction: float = 0.0
    supreme_accuracy_preservation: float = 0.0
    supreme_energy_efficiency: float = 0.0
    supreme_optimization_time: float = 0.0
    
    ultra_fast_speed_improvement: float = 0.0
    ultra_fast_memory_reduction: float = 0.0
    ultra_fast_accuracy_preservation: float = 0.0
    ultra_fast_energy_efficiency: float = 0.0
    ultra_fast_optimization_time: float = 0.0
    
    refactored_ultimate_hybrid_speed_improvement: float = 0.0
    refactored_ultimate_hybrid_memory_reduction: float = 0.0
    refactored_ultimate_hybrid_accuracy_preservation: float = 0.0
    refactored_ultimate_hybrid_energy_efficiency: float = 0.0
    refactored_ultimate_hybrid_optimization_time: float = 0.0
    
    cuda_kernel_speed_improvement: float = 0.0
    cuda_kernel_memory_reduction: float = 0.0
    cuda_kernel_accuracy_preservation: float = 0.0
    cuda_kernel_energy_efficiency: float = 0.0
    cuda_kernel_optimization_time: float = 0.0
    
    gpu_utilization_speed_improvement: float = 0.0
    gpu_utilization_memory_reduction: float = 0.0
    gpu_utilization_accuracy_preservation: float = 0.0
    gpu_utilization_energy_efficiency: float = 0.0
    gpu_utilization_optimization_time: float = 0.0
    
    memory_optimization_speed_improvement: float = 0.0
    memory_optimization_memory_reduction: float = 0.0
    memory_optimization_accuracy_preservation: float = 0.0
    memory_optimization_energy_efficiency: float = 0.0
    memory_optimization_optimization_time: float = 0.0
    
    reward_function_speed_improvement: float = 0.0
    reward_function_memory_reduction: float = 0.0
    reward_function_accuracy_preservation: float = 0.0
    reward_function_energy_efficiency: float = 0.0
    reward_function_optimization_time: float = 0.0
    
    truthgpt_adapter_speed_improvement: float = 0.0
    truthgpt_adapter_memory_reduction: float = 0.0
    truthgpt_adapter_accuracy_preservation: float = 0.0
    truthgpt_adapter_energy_efficiency: float = 0.0
    truthgpt_adapter_optimization_time: float = 0.0
    
    microservices_speed_improvement: float = 0.0
    microservices_memory_reduction: float = 0.0
    microservices_accuracy_preservation: float = 0.0
    microservices_energy_efficiency: float = 0.0
    microservices_optimization_time: float = 0.0
    
    combined_ultimate_enhanced_speed_improvement: float = 0.0
    combined_ultimate_enhanced_memory_reduction: float = 0.0
    combined_ultimate_enhanced_accuracy_preservation: float = 0.0
    combined_ultimate_enhanced_energy_efficiency: float = 0.0
    combined_ultimate_enhanced_optimization_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'supreme_metrics': {
                'speed_improvement': self.supreme_speed_improvement,
                'memory_reduction': self.supreme_memory_reduction,
                'accuracy_preservation': self.supreme_accuracy_preservation,
                'energy_efficiency': self.supreme_energy_efficiency,
                'optimization_time': self.supreme_optimization_time
            },
            'ultra_fast_metrics': {
                'speed_improvement': self.ultra_fast_speed_improvement,
                'memory_reduction': self.ultra_fast_memory_reduction,
                'accuracy_preservation': self.ultra_fast_accuracy_preservation,
                'energy_efficiency': self.ultra_fast_energy_efficiency,
                'optimization_time': self.ultra_fast_optimization_time
            },
            'refactored_ultimate_hybrid_metrics': {
                'speed_improvement': self.refactored_ultimate_hybrid_speed_improvement,
                'memory_reduction': self.refactored_ultimate_hybrid_memory_reduction,
                'accuracy_preservation': self.refactored_ultimate_hybrid_accuracy_preservation,
                'energy_efficiency': self.refactored_ultimate_hybrid_energy_efficiency,
                'optimization_time': self.refactored_ultimate_hybrid_optimization_time
            },
            'cuda_kernel_metrics': {
                'speed_improvement': self.cuda_kernel_speed_improvement,
                'memory_reduction': self.cuda_kernel_memory_reduction,
                'accuracy_preservation': self.cuda_kernel_accuracy_preservation,
                'energy_efficiency': self.cuda_kernel_energy_efficiency,
                'optimization_time': self.cuda_kernel_optimization_time
            },
            'gpu_utilization_metrics': {
                'speed_improvement': self.gpu_utilization_speed_improvement,
                'memory_reduction': self.gpu_utilization_memory_reduction,
                'accuracy_preservation': self.gpu_utilization_accuracy_preservation,
                'energy_efficiency': self.gpu_utilization_energy_efficiency,
                'optimization_time': self.gpu_utilization_optimization_time
            },
            'memory_optimization_metrics': {
                'speed_improvement': self.memory_optimization_speed_improvement,
                'memory_reduction': self.memory_optimization_memory_reduction,
                'accuracy_preservation': self.memory_optimization_accuracy_preservation,
                'energy_efficiency': self.memory_optimization_energy_efficiency,
                'optimization_time': self.memory_optimization_optimization_time
            },
            'reward_function_metrics': {
                'speed_improvement': self.reward_function_speed_improvement,
                'memory_reduction': self.reward_function_memory_reduction,
                'accuracy_preservation': self.reward_function_accuracy_preservation,
                'energy_efficiency': self.reward_function_energy_efficiency,
                'optimization_time': self.reward_function_optimization_time
            },
            'truthgpt_adapter_metrics': {
                'speed_improvement': self.truthgpt_adapter_speed_improvement,
                'memory_reduction': self.truthgpt_adapter_memory_reduction,
                'accuracy_preservation': self.truthgpt_adapter_accuracy_preservation,
                'energy_efficiency': self.truthgpt_adapter_energy_efficiency,
                'optimization_time': self.truthgpt_adapter_optimization_time
            },
            'microservices_metrics': {
                'speed_improvement': self.microservices_speed_improvement,
                'memory_reduction': self.microservices_memory_reduction,
                'accuracy_preservation': self.microservices_accuracy_preservation,
                'energy_efficiency': self.microservices_energy_efficiency,
                'optimization_time': self.microservices_optimization_time
            },
            'combined_ultimate_enhanced_metrics': {
                'speed_improvement': self.combined_ultimate_enhanced_speed_improvement,
                'memory_reduction': self.combined_ultimate_enhanced_memory_reduction,
                'accuracy_preservation': self.combined_ultimate_enhanced_accuracy_preservation,
                'energy_efficiency': self.combined_ultimate_enhanced_energy_efficiency,
                'optimization_time': self.combined_ultimate_enhanced_optimization_time
            }
        }









