"""
JSON Storage - Almacenamiento en JSON
=====================================

Sistema de almacenamiento usando archivos JSON.
"""

import json
import os
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
from .storage_interface import StorageInterface

logger = logging.getLogger(__name__)


class JSONStorage(StorageInterface):
    """Almacenamiento usando archivos JSON"""
    
    def __init__(self, base_path: str = "data/storage"):
        """
        Inicializar almacenamiento JSON
        
        Args:
            base_path: Ruta base para almacenamiento
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"JSON Storage inicializado en {base_path}")
    
    def _get_file_path(self, key: str) -> Path:
        """Obtener ruta del archivo para una clave"""
        # Sanitizar clave para usar como nombre de archivo
        safe_key = key.replace("/", "_").replace("\\", "_")
        return self.base_path / f"{safe_key}.json"
    
    def save(self, key: str, data: Dict[str, Any]) -> bool:
        """
        Guardar datos
        
        Args:
            key: Clave única
            data: Datos a guardar
            
        Returns:
            True si se guardó exitosamente
        """
        try:
            file_path = self._get_file_path(key)
            
            # Agregar metadata
            data_with_meta = {
                "key": key,
                "data": data,
                "updated_at": datetime.now().isoformat()
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data_with_meta, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Datos guardados: {key}")
            return True
            
        except Exception as e:
            logger.error(f"Error guardando {key}: {e}")
            return False
    
    def load(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Cargar datos
        
        Args:
            key: Clave única
            
        Returns:
            Datos cargados o None
        """
        try:
            file_path = self._get_file_path(key)
            
            if not file_path.exists():
                return None
            
            with open(file_path, 'r', encoding='utf-8') as f:
                stored = json.load(f)
                return stored.get("data")
            
        except Exception as e:
            logger.error(f"Error cargando {key}: {e}")
            return None
    
    def delete(self, key: str) -> bool:
        """
        Eliminar datos
        
        Args:
            key: Clave única
            
        Returns:
            True si se eliminó exitosamente
        """
        try:
            file_path = self._get_file_path(key)
            
            if file_path.exists():
                file_path.unlink()
                logger.debug(f"Datos eliminados: {key}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error eliminando {key}: {e}")
            return False
    
    def list_keys(self, prefix: Optional[str] = None) -> List[str]:
        """
        Listar claves
        
        Args:
            prefix: Prefijo para filtrar (opcional)
            
        Returns:
            Lista de claves
        """
        try:
            keys = []
            
            for file_path in self.base_path.glob("*.json"):
                # Extraer clave del nombre de archivo
                key = file_path.stem.replace("_", "/")
                
                if prefix is None or key.startswith(prefix):
                    keys.append(key)
            
            return sorted(keys)
            
        except Exception as e:
            logger.error(f"Error listando claves: {e}")
            return []
    
    def exists(self, key: str) -> bool:
        """
        Verificar si existe una clave
        
        Args:
            key: Clave única
            
        Returns:
            True si existe
        """
        file_path = self._get_file_path(key)
        return file_path.exists()
    
    def save_batch(self, items: Dict[str, Dict[str, Any]]) -> int:
        """
        Guardar múltiples items
        
        Args:
            items: Dict con clave -> datos
            
        Returns:
            Número de items guardados exitosamente
        """
        saved = 0
        for key, data in items.items():
            if self.save(key, data):
                saved += 1
        return saved
    
    def load_all(self, prefix: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Cargar todos los items con un prefijo
        
        Args:
            prefix: Prefijo para filtrar
            
        Returns:
            Dict con clave -> datos
        """
        keys = self.list_keys(prefix)
        return {key: self.load(key) for key in keys if self.load(key) is not None}



