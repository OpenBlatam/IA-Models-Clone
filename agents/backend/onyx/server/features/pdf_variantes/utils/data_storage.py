"""
Data Storage - Almacenamiento de datos en archivos JSON
========================================================

Clase refactorizada para manejo seguro de archivos con context managers,
manejo de errores apropiado y estructura de código mejorada.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class DataStorage:
    """
    Clase para almacenamiento de datos en archivos JSON.
    
    Proporciona métodos write, read y update con:
    - Uso de context managers para operaciones de archivo
    - Manejo apropiado de errores
    - Validación de entradas del usuario
    """
    
    def __init__(self, file_path: str):
        """
        Inicializar almacenamiento de datos.
        
        Args:
            file_path: Ruta al archivo JSON donde se almacenarán los datos
            
        Raises:
            ValueError: Si file_path es None o vacío
            OSError: Si no se puede crear el directorio padre
        """
        if not file_path or not isinstance(file_path, str):
            raise ValueError("file_path debe ser una cadena no vacía")
        
        self.file_path = Path(file_path)
        self._ensure_directory_exists()
    
    def _ensure_directory_exists(self) -> None:
        """Asegurar que el directorio padre existe."""
        try:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"Error creando directorio {self.file_path.parent}: {e}")
            raise
    
    def write(self, data: Dict[str, Any]) -> bool:
        """
        Escribir datos al archivo.
        
        Args:
            data: Diccionario con los datos a escribir
            
        Returns:
            True si la escritura fue exitosa, False en caso contrario
            
        Raises:
            ValueError: Si data no es un diccionario válido
            TypeError: Si data no es un diccionario
        """
        if not isinstance(data, dict):
            raise TypeError("data debe ser un diccionario")
        
        if not data:
            raise ValueError("data no puede estar vacío")
        
        try:
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Datos escritos exitosamente en {self.file_path}")
            return True
            
        except json.JSONEncodeError as e:
            logger.error(f"Error serializando datos a JSON: {e}")
            return False
        except OSError as e:
            logger.error(f"Error de E/S al escribir archivo {self.file_path}: {e}")
            return False
        except Exception as e:
            logger.error(f"Error inesperado al escribir archivo: {e}")
            return False
    
    def read(self) -> Optional[Dict[str, Any]]:
        """
        Leer datos del archivo.
        
        Returns:
            Diccionario con los datos leídos, o None si el archivo no existe
            o hay un error al leerlo
            
        Raises:
            json.JSONDecodeError: Si el archivo contiene JSON inválido
        """
        if not self.file_path.exists():
            logger.warning(f"Archivo no existe: {self.file_path}")
            return None
        
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, dict):
                logger.error(f"Datos en archivo no son un diccionario válido")
                return None
            
            logger.debug(f"Datos leídos exitosamente de {self.file_path}")
            return data
            
        except json.JSONDecodeError as e:
            logger.error(f"Error decodificando JSON en {self.file_path}: {e}")
            return None
        except OSError as e:
            logger.error(f"Error de E/S al leer archivo {self.file_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error inesperado al leer archivo: {e}")
            return None
    
    def update(self, record_id: str, updates: Dict[str, Any]) -> bool:
        """
        Actualizar un registro específico en el archivo.
        
        Args:
            record_id: ID del registro a actualizar
            updates: Diccionario con los campos a actualizar
            
        Returns:
            True si la actualización fue exitosa, False en caso contrario
            
        Raises:
            ValueError: Si record_id o updates son inválidos
            TypeError: Si los argumentos no son del tipo correcto
        """
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError("record_id debe ser una cadena no vacía")
        
        if not isinstance(updates, dict):
            raise TypeError("updates debe ser un diccionario")
        
        if not updates:
            raise ValueError("updates no puede estar vacío")
        
        try:
            data = self.read()
            
            if data is None:
                logger.warning(f"No se pudo leer el archivo para actualizar")
                return False
            
            if 'records' not in data:
                data['records'] = {}
            
            if record_id not in data['records']:
                logger.warning(f"Registro con ID '{record_id}' no encontrado")
                return False
            
            data['records'][record_id].update(updates)
            
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Registro '{record_id}' actualizado exitosamente")
            return True
            
        except json.JSONEncodeError as e:
            logger.error(f"Error serializando datos actualizados a JSON: {e}")
            return False
        except OSError as e:
            logger.error(f"Error de E/S al actualizar archivo {self.file_path}: {e}")
            return False
        except Exception as e:
            logger.error(f"Error inesperado al actualizar registro: {e}")
            return False
    
    def add_record(self, record_id: str, record_data: Dict[str, Any]) -> bool:
        """
        Agregar un nuevo registro al archivo.
        
        Args:
            record_id: ID único para el nuevo registro
            record_data: Datos del registro
            
        Returns:
            True si el registro fue agregado exitosamente, False en caso contrario
        """
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError("record_id debe ser una cadena no vacía")
        
        if not isinstance(record_data, dict):
            raise TypeError("record_data debe ser un diccionario")
        
        try:
            data = self.read()
            
            if data is None:
                data = {'records': {}}
            
            if 'records' not in data:
                data['records'] = {}
            
            if record_id in data['records']:
                logger.warning(f"Registro con ID '{record_id}' ya existe")
                return False
            
            data['records'][record_id] = record_data
            
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Registro '{record_id}' agregado exitosamente")
            return True
            
        except Exception as e:
            logger.error(f"Error agregando registro: {e}")
            return False
    
    def get_record(self, record_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtener un registro específico por su ID.
        
        Args:
            record_id: ID del registro a obtener
            
        Returns:
            Diccionario con los datos del registro, o None si no se encuentra
        """
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError("record_id debe ser una cadena no vacía")
        
        data = self.read()
        
        if data is None or 'records' not in data:
            return None
        
        return data['records'].get(record_id)
    
    def delete_record(self, record_id: str) -> bool:
        """
        Eliminar un registro del archivo.
        
        Args:
            record_id: ID del registro a eliminar
            
        Returns:
            True si el registro fue eliminado exitosamente, False en caso contrario
        """
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError("record_id debe ser una cadena no vacía")
        
        try:
            data = self.read()
            
            if data is None or 'records' not in data:
                return False
            
            if record_id not in data['records']:
                logger.warning(f"Registro con ID '{record_id}' no encontrado")
                return False
            
            del data['records'][record_id]
            
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Registro '{record_id}' eliminado exitosamente")
            return True
            
        except Exception as e:
            logger.error(f"Error eliminando registro: {e}")
            return False
    
    def list_records(self) -> List[str]:
        """
        Listar todos los IDs de registros.
        
        Returns:
            Lista de IDs de registros
        """
        data = self.read()
        
        if data is None or 'records' not in data:
            return []
        
        return list(data['records'].keys())


