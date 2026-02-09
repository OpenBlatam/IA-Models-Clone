"""
Generation models for Ultimate Enhanced Supreme Production system
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from datetime import datetime

@dataclass
class GenerationRequest:
    """Generation request model."""
    query: str
    max_documents: Optional[int] = None
    optimization_level: Optional[str] = None
    supreme_optimization_enabled: bool = True
    ultra_fast_optimization_enabled: bool = True
    refactored_ultimate_hybrid_optimization_enabled: bool = True
    cuda_kernel_optimization_enabled: bool = True
    gpu_utils_optimization_enabled: bool = True
    memory_utils_optimization_enabled: bool = True
    reward_function_optimization_enabled: bool = True
    truthgpt_adapter_optimization_enabled: bool = True
    microservices_optimization_enabled: bool = True
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

@dataclass
class Document:
    """Document model."""
    id: str
    content: str
    supreme_optimization: Dict[str, Any]
    ultra_fast_optimization: Dict[str, Any]
    refactored_ultimate_hybrid_optimization: Dict[str, Any]
    cuda_kernel_optimization: Dict[str, Any]
    gpu_utils_optimization: Dict[str, Any]
    memory_utils_optimization: Dict[str, Any]
    reward_function_optimization: Dict[str, Any]
    truthgpt_adapter_optimization: Dict[str, Any]
    microservices_optimization: Dict[str, Any]
    combined_ultimate_enhanced_speedup: float
    generation_time: float
    quality_score: float
    diversity_score: float
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

@dataclass
class GenerationResponse:
    """Generation response model."""
    query: str
    documents_generated: int
    processing_time: float
    supreme_optimization: Dict[str, Any]
    ultra_fast_optimization: Dict[str, Any]
    refactored_ultimate_hybrid_optimization: Dict[str, Any]
    cuda_kernel_optimization: Dict[str, Any]
    gpu_utils_optimization: Dict[str, Any]
    memory_utils_optimization: Dict[str, Any]
    reward_function_optimization: Dict[str, Any]
    truthgpt_adapter_optimization: Dict[str, Any]
    microservices_optimization: Dict[str, Any]
    combined_ultimate_enhanced_metrics: Dict[str, Any]
    documents: List[Document]
    total_documents: int
    ultimate_enhanced_supreme_ready: bool
    ultra_fast_ready: bool
    refactored_ultimate_hybrid_ready: bool
    cuda_kernel_ready: bool
    gpu_utils_ready: bool
    memory_utils_ready: bool
    reward_function_ready: bool
    truthgpt_adapter_ready: bool
    microservices_ready: bool
    ultimate_ready: bool
    ultra_advanced_ready: bool
    advanced_ready: bool
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'query': self.query,
            'documents_generated': self.documents_generated,
            'processing_time': self.processing_time,
            'supreme_optimization': self.supreme_optimization,
            'ultra_fast_optimization': self.ultra_fast_optimization,
            'refactored_ultimate_hybrid_optimization': self.refactored_ultimate_hybrid_optimization,
            'cuda_kernel_optimization': self.cuda_kernel_optimization,
            'gpu_utils_optimization': self.gpu_utils_optimization,
            'memory_utils_optimization': self.memory_utils_optimization,
            'reward_function_optimization': self.reward_function_optimization,
            'truthgpt_adapter_optimization': self.truthgpt_adapter_optimization,
            'microservices_optimization': self.microservices_optimization,
            'combined_ultimate_enhanced_metrics': self.combined_ultimate_enhanced_metrics,
            'documents': [doc.__dict__ for doc in self.documents],
            'total_documents': self.total_documents,
            'ultimate_enhanced_supreme_ready': self.ultimate_enhanced_supreme_ready,
            'ultra_fast_ready': self.ultra_fast_ready,
            'refactored_ultimate_hybrid_ready': self.refactored_ultimate_hybrid_ready,
            'cuda_kernel_ready': self.cuda_kernel_ready,
            'gpu_utils_ready': self.gpu_utils_ready,
            'memory_utils_ready': self.memory_utils_ready,
            'reward_function_ready': self.reward_function_ready,
            'truthgpt_adapter_ready': self.truthgpt_adapter_ready,
            'microservices_ready': self.microservices_ready,
            'ultimate_ready': self.ultimate_ready,
            'ultra_advanced_ready': self.ultra_advanced_ready,
            'advanced_ready': self.advanced_ready,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }









