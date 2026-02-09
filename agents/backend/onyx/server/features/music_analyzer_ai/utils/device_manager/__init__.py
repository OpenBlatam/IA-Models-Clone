"""
Device Manager Submodule
Aggregates device management components.
"""

from typing import Optional
from .manager import DeviceManager as BaseDeviceManager
from .compilation import ModelCompilationMixin


class DeviceManager(BaseDeviceManager):
    """
    Complete device manager combining all functionality.
    """
    
    def __init__(self, device: Optional[str] = None):
        super().__init__(device=device)
        self._compilation = ModelCompilationMixin()
    
    def compile_model(self, model, mode: str = "reduce-overhead"):
        """Compile model for faster execution."""
        return self._compilation.compile_model(model, mode)


# Global device manager
_device_manager: Optional[DeviceManager] = None


def get_device_manager(device: Optional[str] = None) -> DeviceManager:
    """Get global device manager"""
    global _device_manager
    if _device_manager is None:
        _device_manager = DeviceManager(device)
    return _device_manager


__all__ = [
    "DeviceManager",
    "ModelCompilationMixin",
    "get_device_manager",
]

