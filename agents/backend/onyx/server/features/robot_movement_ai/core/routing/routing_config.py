"""
Routing Configuration
====================

Configuración centralizada para el sistema de routing.
Sigue las mejores prácticas de configuración para proyectos de ML.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuración para modelos de deep learning."""
    
    # GNN Configuration
    gnn_input_dim: int = 10
    gnn_hidden_dim: int = 128
    gnn_output_dim: int = 64
    gnn_num_layers: int = 3
    gnn_num_heads: int = 8
    gnn_dropout: float = 0.1
    
    # Transformer Configuration
    transformer_d_model: int = 128
    transformer_nhead: int = 8
    transformer_num_layers: int = 6
    transformer_dim_feedforward: int = 512
    transformer_dropout: float = 0.1
    
    # Deep Optimizer Configuration
    deep_optimizer_input_dim: int = 10
    deep_optimizer_hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    deep_optimizer_dropout: float = 0.2
    
    # Training Configuration
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 10
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0
    
    # Mixed Precision
    use_mixed_precision: bool = True
    use_gpu: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir configuración a diccionario."""
        return {
            'gnn': {
                'input_dim': self.gnn_input_dim,
                'hidden_dim': self.gnn_hidden_dim,
                'output_dim': self.gnn_output_dim,
                'num_layers': self.gnn_num_layers,
                'num_heads': self.gnn_num_heads,
                'dropout': self.gnn_dropout
            },
            'transformer': {
                'd_model': self.transformer_d_model,
                'nhead': self.transformer_nhead,
                'num_layers': self.transformer_num_layers,
                'dim_feedforward': self.transformer_dim_feedforward,
                'dropout': self.transformer_dropout
            },
            'deep_optimizer': {
                'input_dim': self.deep_optimizer_input_dim,
                'hidden_dims': self.deep_optimizer_hidden_dims,
                'dropout': self.deep_optimizer_dropout
            },
            'training': {
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'num_epochs': self.num_epochs,
                'weight_decay': self.weight_decay,
                'gradient_clip_norm': self.gradient_clip_norm
            },
            'hardware': {
                'use_mixed_precision': self.use_mixed_precision,
                'use_gpu': self.use_gpu
            }
        }


@dataclass
class RoutingConfig:
    """Configuración general del sistema de routing."""
    
    # Cache Configuration
    cache_max_size: int = 10000
    cache_ttl: float = 3600.0  # seconds
    
    # Precomputation Configuration
    precomputation_max_nodes: int = 100
    enable_precomputation: bool = False
    
    # Batch Processing Configuration
    batch_processing_enabled: bool = True
    batch_size: int = 32
    max_workers: Optional[int] = None
    
    # Performance Monitoring
    enable_performance_monitoring: bool = True
    performance_window_size: int = 1000
    
    # Model Configuration
    model_config: ModelConfig = field(default_factory=ModelConfig)
    
    # Feature Extraction
    node_feature_dim: int = 10
    edge_feature_dim: int = 5
    
    # Routing Strategy Defaults
    default_strategy: str = "adaptive"
    enable_deep_learning: bool = True
    enable_transformer: bool = True
    enable_llm: bool = True
    
    # Extreme Performance Options
    use_onnx: bool = True
    use_quantization: bool = False
    use_tensorrt: bool = False
    use_kernel_fusion: bool = True
    use_graph_optimization: bool = True
    use_memory_optimization: bool = True
    
    # System-Level Optimizations
    enable_cpu_affinity: bool = False  # Solo habilitar si es necesario
    enable_system_monitoring: bool = True
    
    # Cache Optimizations
    use_distributed_cache: bool = False
    redis_host: str = "localhost"
    
    # Security Optimizations
    enable_rate_limiting: bool = True
    
    # Monitoring Optimizations
    enable_prometheus: bool = True
    
    # Logging Optimizations
    enable_structured_logging: bool = True
    log_file: Optional[str] = None
    
    # Backup Optimizations
    enable_auto_backup: bool = False
    snapshot_dir: str = "snapshots"
    auto_backup_interval: float = 3600.0  # seconds
    
    # API Optimizations
    enable_response_cache: bool = True
    
    # Serialization Optimizations
    serialization_format: str = "auto"  # auto, json, msgpack, pickle
    use_compression: bool = False
    
    # Configuration Optimizations
    config_file: Optional[str] = None
    
    # Scalability Optimizations
    num_shards: int = 4
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir configuración completa a diccionario."""
        return {
            'cache': {
                'max_size': self.cache_max_size,
                'ttl': self.cache_ttl
            },
            'precomputation': {
                'max_nodes': self.precomputation_max_nodes,
                'enabled': self.enable_precomputation
            },
            'batch_processing': {
                'enabled': self.batch_processing_enabled,
                'batch_size': self.batch_size,
                'max_workers': self.max_workers
            },
            'performance': {
                'monitoring_enabled': self.enable_performance_monitoring,
                'window_size': self.performance_window_size
            },
            'model': self.model_config.to_dict(),
            'features': {
                'node_dim': self.node_feature_dim,
                'edge_dim': self.edge_feature_dim
            },
            'routing': {
                'default_strategy': self.default_strategy,
                'enable_deep_learning': self.enable_deep_learning,
                'enable_transformer': self.enable_transformer,
                'enable_llm': self.enable_llm
            }
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RoutingConfig':
        """Crear configuración desde diccionario."""
        model_config = ModelConfig()
        if 'model' in config_dict:
            model_dict = config_dict['model']
            if 'gnn' in model_dict:
                gnn = model_dict['gnn']
                model_config.gnn_input_dim = gnn.get('input_dim', 10)
                model_config.gnn_hidden_dim = gnn.get('hidden_dim', 128)
                model_config.gnn_output_dim = gnn.get('output_dim', 64)
                model_config.gnn_num_layers = gnn.get('num_layers', 3)
                model_config.gnn_num_heads = gnn.get('num_heads', 8)
                model_config.gnn_dropout = gnn.get('dropout', 0.1)
        
        return cls(
            cache_max_size=config_dict.get('cache', {}).get('max_size', 10000),
            cache_ttl=config_dict.get('cache', {}).get('ttl', 3600.0),
            precomputation_max_nodes=config_dict.get('precomputation', {}).get('max_nodes', 100),
            enable_precomputation=config_dict.get('precomputation', {}).get('enabled', False),
            batch_processing_enabled=config_dict.get('batch_processing', {}).get('enabled', True),
            batch_size=config_dict.get('batch_processing', {}).get('batch_size', 32),
            enable_performance_monitoring=config_dict.get('performance', {}).get('monitoring_enabled', True),
            model_config=model_config,
            default_strategy=config_dict.get('routing', {}).get('default_strategy', 'adaptive'),
            enable_deep_learning=config_dict.get('routing', {}).get('enable_deep_learning', True),
            enable_transformer=config_dict.get('routing', {}).get('enable_transformer', True),
            enable_llm=config_dict.get('routing', {}).get('enable_llm', True)
        )


# Configuración por defecto
DEFAULT_ROUTING_CONFIG = RoutingConfig()

