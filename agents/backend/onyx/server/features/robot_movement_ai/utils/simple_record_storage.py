"""
Simple Record Storage - Versión Simplificada y Refactorizada
============================================================

Versión simplificada del almacenamiento de registros que demuestra
las mejores prácticas de manera clara y concisa.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class SimpleRecordStorage:
    """Almacenamiento simple de registros con mejores prácticas."""
    
    def __init__(self, file_path: str):
        if not file_path or not isinstance(file_path, str):
            raise ValueError("file_path debe ser una cadena no vacía")
        
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not self.file_path.exists():
            self._init_file()
    
    def _init_file(self) -> None:
        """Inicializar archivo vacío."""
        try:
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump({"records": []}, f, indent=2)
        except (IOError, OSError) as e:
            raise RuntimeError(f"Error al inicializar archivo: {e}") from e
    
    def read(self) -> List[Dict[str, Any]]:
        """Leer todos los registros."""
        if not self.file_path.exists():
            return []
        
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return data.get('records', []) if isinstance(data, dict) else []
        except json.JSONDecodeError as e:
            raise RuntimeError(f"JSON inválido: {e}") from e
        except (IOError, OSError) as e:
            raise RuntimeError(f"Error al leer: {e}") from e
    
    def write(self, records: List[Dict[str, Any]]) -> bool:
        """Escribir registros al archivo."""
        if not isinstance(records, list):
            raise ValueError("records debe ser una lista")
        
        try:
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump({"records": records}, f, indent=2)
            return True
        except (IOError, OSError) as e:
            raise RuntimeError(f"Error al escribir: {e}") from e
    
    def update(self, record_id: str, updates: Dict[str, Any]) -> bool:
        """Actualizar un registro por ID."""
        if not record_id or not isinstance(record_id, str):
            raise ValueError("record_id debe ser una cadena no vacía")
        
        if not isinstance(updates, dict):
            raise ValueError("updates debe ser un diccionario")
        
        try:
            records = self.read()
            
            for i, record in enumerate(records):
                if isinstance(record, dict) and record.get('id') == record_id:
                    records[i].update(updates)
                    if 'id' not in records[i]:
                        records[i]['id'] = record_id
                    
                    self.write(records)
                    return True
            
            return False
        except (RuntimeError, ValueError) as e:
            raise
        except Exception as e:
            raise RuntimeError(f"Error inesperado: {e}") from e


