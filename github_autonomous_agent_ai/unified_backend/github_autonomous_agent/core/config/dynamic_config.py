"""
Sistema de Configuración Dinámica con Hot-Reload.
"""

import asyncio
import json
from typing import Dict, Any, Optional, Callable, List
from pathlib import Path
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)


class ConfigChangeHandler(FileSystemEventHandler):
    """Handler para cambios en archivos de configuración."""
    
    def __init__(self, config_manager: 'DynamicConfigManager'):
        """
        Inicializar handler.
        
        Args:
            config_manager: Manager de configuración
        """
        self.config_manager = config_manager
    
    def on_modified(self, event):
        """Manejar modificación de archivo."""
        if not event.is_directory and event.src_path.endswith(('.json', '.yaml', '.yml')):
            logger.info(f"Archivo de configuración modificado: {event.src_path}")
            asyncio.create_task(self.config_manager.reload_config(event.src_path))


class DynamicConfigManager:
    """Manager de configuración dinámica con hot-reload."""
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Inicializar manager de configuración dinámica.
        
        Args:
            config_dir: Directorio de configuración
        """
        if config_dir is None:
            config_dir = Path(settings.STORAGE_PATH) / "config"
        self.config_dir = config_dir
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.configs: Dict[str, Dict[str, Any]] = {}
        self.watchers: Dict[str, Observer] = {}
        self.change_handlers: List[Callable[[str, Dict[str, Any]], None]] = []
        self.running = False
    
    def load_config(self, name: str, file_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Cargar configuración desde archivo.
        
        Args:
            name: Nombre de la configuración
            file_path: Ruta al archivo (opcional)
            
        Returns:
            Configuración cargada
        """
        if file_path is None:
            file_path = self.config_dir / f"{name}.json"
        
        try:
            if file_path.suffix == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            elif file_path.suffix in ('.yaml', '.yml'):
                import yaml
                with open(file_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
            else:
                raise ValueError(f"Formato no soportado: {file_path.suffix}")
            
            config["_loaded_at"] = datetime.now().isoformat()
            config["_file_path"] = str(file_path)
            self.configs[name] = config
            
            logger.info(f"Configuración cargada: {name}")
            return config
            
        except Exception as e:
            logger.error(f"Error cargando configuración {name}: {e}", exc_info=True)
            return {}
    
    def get_config(self, name: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Obtener configuración.
        
        Args:
            name: Nombre de la configuración
            default: Valor por defecto
            
        Returns:
            Configuración
        """
        return self.configs.get(name, default or {})
    
    def set_config(self, name: str, config: Dict[str, Any], save: bool = True) -> None:
        """
        Establecer configuración.
        
        Args:
            name: Nombre de la configuración
            config: Configuración
            save: Si guardar en archivo
        """
        config["_updated_at"] = datetime.now().isoformat()
        self.configs[name] = config
        
        if save:
            file_path = Path(config.get("_file_path", self.config_dir / f"{name}.json"))
            try:
                if file_path.suffix == '.json':
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(config, f, indent=2, ensure_ascii=False)
                elif file_path.suffix in ('.yaml', '.yml'):
                    import yaml
                    with open(file_path, 'w', encoding='utf-8') as f:
                        yaml.dump(config, f, default_flow_style=False)
                
                logger.info(f"Configuración guardada: {name}")
            except Exception as e:
                logger.error(f"Error guardando configuración {name}: {e}", exc_info=True)
        
        # Notificar cambios
        self._notify_change(name, config)
    
    async def reload_config(self, file_path: str) -> None:
        """
        Recargar configuración desde archivo.
        
        Args:
            file_path: Ruta al archivo
        """
        file_path_obj = Path(file_path)
        name = file_path_obj.stem
        
        logger.info(f"Recargando configuración: {name}")
        config = self.load_config(name, file_path_obj)
        self._notify_change(name, config)
    
    def watch_config(self, name: str, file_path: Optional[Path] = None) -> None:
        """
        Observar archivo de configuración para cambios.
        
        Args:
            name: Nombre de la configuración
            file_path: Ruta al archivo (opcional)
        """
        if file_path is None:
            file_path = self.config_dir / f"{name}.json"
        
        if not file_path.exists():
            logger.warning(f"Archivo de configuración no existe: {file_path}")
            return
        
        observer = Observer()
        handler = ConfigChangeHandler(self)
        observer.schedule(handler, str(file_path.parent), recursive=False)
        observer.start()
        
        self.watchers[name] = observer
        logger.info(f"Observando configuración: {name} ({file_path})")
    
    def register_change_handler(self, handler: Callable[[str, Dict[str, Any]], None]) -> None:
        """
        Registrar handler para cambios de configuración.
        
        Args:
            handler: Función que maneja cambios
        """
        self.change_handlers.append(handler)
    
    def _notify_change(self, name: str, config: Dict[str, Any]) -> None:
        """Notificar cambios a handlers."""
        for handler in self.change_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    asyncio.create_task(handler(name, config))
                else:
                    handler(name, config)
            except Exception as e:
                logger.error(f"Error en change handler: {e}", exc_info=True)
    
    def stop_watching(self, name: Optional[str] = None) -> None:
        """
        Detener observación de configuración.
        
        Args:
            name: Nombre de la configuración (opcional, todas si None)
        """
        if name:
            if name in self.watchers:
                self.watchers[name].stop()
                del self.watchers[name]
        else:
            for observer in self.watchers.values():
                observer.stop()
            self.watchers.clear()
    
    def get_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """Obtener todas las configuraciones."""
        return self.configs.copy()


# Instancia global
_config_manager: Optional[DynamicConfigManager] = None


def get_config_manager() -> DynamicConfigManager:
    """Obtener manager de configuración dinámica."""
    global _config_manager
    if _config_manager is None:
        _config_manager = DynamicConfigManager()
    return _config_manager



