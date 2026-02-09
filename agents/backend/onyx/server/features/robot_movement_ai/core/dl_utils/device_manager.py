"""
Device Manager - Modular Device Management
==========================================

Gestión modular de dispositivos (CPU/GPU) para deep learning.
"""

import logging
from typing import Optional, List, Dict, Any
import torch

logger = logging.getLogger(__name__)


class DeviceManager:
    """
    Gestor de dispositivos para deep learning.
    
    Maneja la selección y configuración de dispositivos
    (CPU, GPU, multi-GPU) de manera modular.
    """
    
    def __init__(
        self,
        device: Optional[str] = None,
        use_mixed_precision: bool = True,
        allow_fallback: bool = True
    ):
        """
        Inicializar gestor de dispositivos.
        
        Args:
            device: Dispositivo específico ('cuda', 'cpu', 'cuda:0', etc.)
            use_mixed_precision: Usar mixed precision si está disponible
            allow_fallback: Permitir fallback a CPU si GPU no está disponible
        """
        self.use_mixed_precision = use_mixed_precision
        self.allow_fallback = allow_fallback
        self.device = self._get_device(device)
        self.is_cuda_available = torch.cuda.is_available()
        self.num_gpus = torch.cuda.device_count() if self.is_cuda_available else 0
        
        logger.info(f"Device Manager initialized: {self.device}")
        if self.is_cuda_available:
            logger.info(f"CUDA available: {self.num_gpus} GPU(s)")
            for i in range(self.num_gpus):
                logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    def _get_device(self, device: Optional[str]) -> torch.device:
        """
        Obtener dispositivo.
        
        Args:
            device: Dispositivo especificado
            
        Returns:
            torch.device
        """
        if device is None:
            if torch.cuda.is_available():
                return torch.device('cuda')
            else:
                if not self.allow_fallback:
                    raise RuntimeError("CUDA not available and fallback disabled")
                return torch.device('cpu')
        
        if device.startswith('cuda'):
            if not torch.cuda.is_available():
                if not self.allow_fallback:
                    raise RuntimeError("CUDA not available and fallback disabled")
                logger.warning("CUDA requested but not available, falling back to CPU")
                return torch.device('cpu')
            return torch.device(device)
        
        return torch.device(device)
    
    def move_to_device(self, obj: Any) -> Any:
        """
        Mover objeto a dispositivo.
        
        Args:
            obj: Objeto a mover (tensor, modelo, etc.)
            
        Returns:
            Objeto movido al dispositivo
        """
        if isinstance(obj, torch.Tensor):
            return obj.to(self.device)
        elif isinstance(obj, torch.nn.Module):
            return obj.to(self.device)
        elif isinstance(obj, dict):
            return {k: self.move_to_device(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return type(obj)(self.move_to_device(item) for item in obj)
        else:
            return obj
    
    def get_device_info(self) -> Dict[str, Any]:
        """
        Obtener información del dispositivo.
        
        Returns:
            Diccionario con información del dispositivo
        """
        info = {
            'device': str(self.device),
            'cuda_available': self.is_cuda_available,
            'num_gpus': self.num_gpus,
            'mixed_precision': self.use_mixed_precision and self.is_cuda_available
        }
        
        if self.is_cuda_available:
            info['gpu_names'] = [
                torch.cuda.get_device_name(i) for i in range(self.num_gpus)
            ]
            info['cuda_version'] = torch.version.cuda
            info['cudnn_version'] = torch.backends.cudnn.version()
            
            # Memoria GPU
            if self.device.type == 'cuda':
                info['gpu_memory'] = {
                    'allocated': torch.cuda.memory_allocated(self.device) / 1024**3,  # GB
                    'reserved': torch.cuda.memory_reserved(self.device) / 1024**3,  # GB
                    'max_allocated': torch.cuda.max_memory_allocated(self.device) / 1024**3  # GB
                }
        
        return info
    
    def clear_cache(self):
        """Limpiar caché de GPU."""
        if self.is_cuda_available:
            torch.cuda.empty_cache()
            logger.debug("GPU cache cleared")
    
    def reset_peak_memory_stats(self):
        """Resetear estadísticas de memoria."""
        if self.is_cuda_available and self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
    
    def enable_mixed_precision(self) -> bool:
        """
        Habilitar mixed precision si está disponible.
        
        Returns:
            True si se habilitó, False en caso contrario
        """
        if self.is_cuda_available and self.use_mixed_precision:
            logger.info("Mixed precision enabled")
            return True
        else:
            logger.info("Mixed precision not available or disabled")
            return False
    
    def setup_multi_gpu(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Configurar multi-GPU si está disponible.
        
        Args:
            model: Modelo a configurar
            
        Returns:
            Modelo configurado para multi-GPU
        """
        if self.num_gpus > 1:
            logger.info(f"Setting up multi-GPU with {self.num_gpus} GPUs")
            return torch.nn.DataParallel(model)
        return model


# Instancia global
_device_manager: Optional[DeviceManager] = None


def get_device_manager(
    device: Optional[str] = None,
    use_mixed_precision: bool = True
) -> DeviceManager:
    """
    Obtener instancia global del gestor de dispositivos.
    
    Args:
        device: Dispositivo específico
        use_mixed_precision: Usar mixed precision
        
    Returns:
        DeviceManager
    """
    global _device_manager
    if _device_manager is None:
        _device_manager = DeviceManager(device=device, use_mixed_precision=use_mixed_precision)
    return _device_manager













