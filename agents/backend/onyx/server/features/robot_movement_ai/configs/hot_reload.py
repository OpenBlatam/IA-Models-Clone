"""Hot Reload Manager - Recarga dinámica de configuración"""
import asyncio
from pathlib import Path
from typing import Callable, List


class HotReloadManager:
    """Gestor de recarga dinámica de configuración"""
    
    def __init__(self):
        self.watchers: List[Callable] = []
        self._running = False
    
    def watch(self, callback: Callable):
        """Registra un callback para cambios de configuración"""
        self.watchers.append(callback)
    
    async def start(self, config_path: Path):
        """Inicia el monitoreo de cambios"""
        self._running = True
        last_modified = config_path.stat().st_mtime if config_path.exists() else 0
        
        while self._running:
            if config_path.exists():
                current_modified = config_path.stat().st_mtime
                if current_modified > last_modified:
                    last_modified = current_modified
                    for callback in self.watchers:
                        await callback()
            await asyncio.sleep(1)
    
    def stop(self):
        """Detiene el monitoreo"""
        self._running = False

