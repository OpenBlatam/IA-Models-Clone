"""
Routing Data Validation
========================

Sistema de validación robusto para datos de routing.
Incluye validación de inputs, detección de NaN/Inf, y sanitización.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class DataValidator:
    """Validador profesional de datos."""
    
    @staticmethod
    def validate_tensor(
        tensor: Any,
        name: str = "tensor",
        check_nan: bool = True,
        check_inf: bool = True,
        check_finite: bool = True,
        allow_empty: bool = False
    ) -> Tuple[bool, Optional[str]]:
        """
        Validar tensor.
        
        Args:
            tensor: Tensor a validar
            name: Nombre del tensor para mensajes de error
            check_nan: Verificar NaN
            check_inf: Verificar Inf
            check_finite: Verificar valores finitos
            allow_empty: Permitir tensores vacíos
        
        Returns:
            (is_valid, error_message)
        """
        if not TORCH_AVAILABLE:
            return True, None
        
        if not isinstance(tensor, torch.Tensor):
            return False, f"{name} is not a torch.Tensor"
        
        if not allow_empty and tensor.numel() == 0:
            return False, f"{name} is empty"
        
        if check_nan and torch.isnan(tensor).any():
            return False, f"{name} contains NaN values"
        
        if check_inf and torch.isinf(tensor).any():
            return False, f"{name} contains Inf values"
        
        if check_finite and not torch.isfinite(tensor).all():
            return False, f"{name} contains non-finite values"
        
        return True, None
    
    @staticmethod
    def sanitize_tensor(
        tensor: torch.Tensor,
        replace_nan: float = 0.0,
        replace_inf: float = 0.0,
        clip_values: Optional[Tuple[float, float]] = None
    ) -> torch.Tensor:
        """
        Sanitizar tensor reemplazando NaN/Inf.
        
        Args:
            tensor: Tensor a sanitizar
            replace_nan: Valor para reemplazar NaN
            replace_inf: Valor para reemplazar Inf
            clip_values: Tupla (min, max) para clip de valores
        
        Returns:
            Tensor sanitizado
        """
        if not TORCH_AVAILABLE:
            return tensor
        
        # Reemplazar NaN
        tensor = torch.where(torch.isnan(tensor), torch.tensor(replace_nan, device=tensor.device), tensor)
        
        # Reemplazar Inf
        tensor = torch.where(torch.isinf(tensor), torch.tensor(replace_inf, device=tensor.device), tensor)
        
        # Clip valores si se especifica
        if clip_values:
            tensor = torch.clamp(tensor, clip_values[0], clip_values[1])
        
        return tensor
    
    @staticmethod
    def validate_route_path(
        path: List[str],
        nodes: Dict[str, Any],
        min_length: int = 2
    ) -> Tuple[bool, Optional[str]]:
        """
        Validar ruta.
        
        Args:
            path: Lista de IDs de nodos
            nodes: Diccionario de nodos disponibles
            min_length: Longitud mínima de la ruta
        
        Returns:
            (is_valid, error_message)
        """
        if not isinstance(path, list):
            return False, "Path must be a list"
        
        if len(path) < min_length:
            return False, f"Path must have at least {min_length} nodes"
        
        for node_id in path:
            if node_id not in nodes:
                return False, f"Node {node_id} not found in graph"
        
        return True, None
    
    @staticmethod
    def validate_graph(
        nodes: Dict[str, Any],
        edges: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """
        Validar grafo completo.
        
        Args:
            nodes: Diccionario de nodos
            edges: Diccionario de aristas
        
        Returns:
            (is_valid, error_message)
        """
        if not nodes:
            return False, "Graph must have at least one node"
        
        for edge_id, edge in edges.items():
            if hasattr(edge, 'from_node'):
                from_node = edge.from_node
                to_node = edge.to_node
            else:
                from_node = edge.get('from_node')
                to_node = edge.get('to_node')
            
            if from_node not in nodes:
                return False, f"Edge {edge_id} references unknown from_node: {from_node}"
            
            if to_node not in nodes:
                return False, f"Edge {edge_id} references unknown to_node: {to_node}"
        
        return True, None


class GradientMonitor:
    """Monitor de gradientes para debugging."""
    
    def __init__(self, model: Any, log_interval: int = 100):
        """
        Inicializar monitor de gradientes.
        
        Args:
            model: Modelo PyTorch
            log_interval: Intervalo de logging
        """
        self.model = model
        self.log_interval = log_interval
        self.gradient_history: List[Dict[str, float]] = []
    
    def check_gradients(self, step: int = 0) -> Dict[str, Any]:
        """
        Verificar gradientes del modelo.
        
        Args:
            step: Paso actual
        
        Returns:
            Diccionario con estadísticas de gradientes
        """
        if not TORCH_AVAILABLE:
            return {}
        
        stats = {
            'step': step,
            'grad_norms': {},
            'grad_means': {},
            'grad_stds': {},
            'has_nan': False,
            'has_inf': False,
            'zero_grads': 0,
            'total_params': 0
        }
        
        total_norm = 0.0
        param_count = 0
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad = param.grad.data
                
                # Verificar NaN/Inf
                if torch.isnan(grad).any():
                    stats['has_nan'] = True
                    logger.warning(f"NaN gradient detected in {name}")
                
                if torch.isinf(grad).any():
                    stats['has_inf'] = True
                    logger.warning(f"Inf gradient detected in {name}")
                
                # Estadísticas
                param_norm = grad.norm(2).item()
                stats['grad_norms'][name] = param_norm
                stats['grad_means'][name] = grad.mean().item()
                stats['grad_stds'][name] = grad.std().item()
                
                total_norm += param_norm ** 2
                param_count += 1
            else:
                stats['zero_grads'] += 1
            
            stats['total_params'] += 1
        
        stats['total_grad_norm'] = total_norm ** 0.5
        stats['params_with_grad'] = param_count
        
        if step % self.log_interval == 0:
            self.gradient_history.append(stats)
        
        return stats
    
    def get_gradient_summary(self) -> Dict[str, Any]:
        """Obtener resumen de historial de gradientes."""
        if not self.gradient_history:
            return {}
        
        total_norms = [h['total_grad_norm'] for h in self.gradient_history]
        
        return {
            'mean_grad_norm': np.mean(total_norms),
            'std_grad_norm': np.std(total_norms),
            'max_grad_norm': np.max(total_norms),
            'min_grad_norm': np.min(total_norms),
            'nan_detections': sum(1 for h in self.gradient_history if h['has_nan']),
            'inf_detections': sum(1 for h in self.gradient_history if h['has_inf'])
        }

