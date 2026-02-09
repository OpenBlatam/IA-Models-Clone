"""
Import Utilities
================

Utilidades para importar datos y configuraciones.
"""

import json
import yaml
import csv
import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class ImportManager:
    """
    Gestor de importaciones.
    
    Gestiona importaciones de datos en múltiples formatos.
    """
    
    def __init__(self):
        """Inicializar gestor de importaciones."""
        self.import_history: List[Dict[str, Any]] = []
    
    def import_json(self, filepath: str) -> Any:
        """
        Importar datos desde JSON.
        
        Args:
            filepath: Ruta del archivo
            
        Returns:
            Datos importados
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self._record_import("json", filepath, data)
        logger.info(f"Data imported from JSON: {filepath}")
        
        return data
    
    def import_yaml(self, filepath: str) -> Any:
        """
        Importar datos desde YAML.
        
        Args:
            filepath: Ruta del archivo
            
        Returns:
            Datos importados
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        self._record_import("yaml", filepath, data)
        logger.info(f"Data imported from YAML: {filepath}")
        
        return data
    
    def import_csv(
        self,
        filepath: str,
        delimiter: str = ','
    ) -> List[Dict[str, Any]]:
        """
        Importar datos desde CSV.
        
        Args:
            filepath: Ruta del archivo
            delimiter: Delimitador
            
        Returns:
            Lista de diccionarios
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            data = list(reader)
        
        self._record_import("csv", filepath, data)
        logger.info(f"Data imported from CSV: {filepath}")
        
        return data
    
    def import_file(self, filepath: str) -> Any:
        """
        Importar datos detectando formato automáticamente.
        
        Args:
            filepath: Ruta del archivo
            
        Returns:
            Datos importados
        """
        path = Path(filepath)
        suffix = path.suffix.lower()
        
        if suffix == '.json':
            return self.import_json(filepath)
        elif suffix in ['.yaml', '.yml']:
            return self.import_yaml(filepath)
        elif suffix == '.csv':
            return self.import_csv(filepath)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    
    def _record_import(self, format: str, filepath: str, data: Any) -> None:
        """Registrar importación."""
        self.import_history.append({
            "format": format,
            "filepath": filepath,
            "timestamp": __import__('datetime').datetime.now().isoformat(),
            "data_size": len(str(data))
        })
    
    def get_import_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Obtener historial de importaciones."""
        return self.import_history[-limit:]


# Instancia global
_import_manager: Optional[ImportManager] = None


def get_import_manager() -> ImportManager:
    """Obtener instancia global del gestor de importaciones."""
    global _import_manager
    if _import_manager is None:
        _import_manager = ImportManager()
    return _import_manager






