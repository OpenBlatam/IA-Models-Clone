"""
Record Storage - Almacenamiento de registros

Sistema de almacenamiento de registros en archivo JSON con manejo adecuado de errores.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class RecordStorage:
    """
    Almacenamiento de registros en archivo JSON.
    
    Proporciona métodos para leer, escribir y actualizar registros
    con manejo adecuado de archivos y errores.
    """
    
    def __init__(self, file_path: str):
        """
        Inicializar almacenamiento de registros.
        
        Args:
            file_path: Ruta al archivo JSON para almacenar registros
            
        Raises:
            ValueError: Si file_path está vacío o es inválido
        """
        if not file_path or not isinstance(file_path, str):
            raise ValueError("file_path debe ser una cadena no vacía")
        
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not self.file_path.exists():
            self._initialize_file()
    
    def _initialize_file(self) -> None:
        """Inicializar el archivo de almacenamiento con una lista vacía de registros."""
        try:
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump({"records": []}, f, indent=2, ensure_ascii=False)
            logger.info(f"Archivo de almacenamiento inicializado: {self.file_path}")
        except (IOError, OSError) as e:
            logger.error(f"Error al inicializar archivo de almacenamiento: {e}")
            raise RuntimeError(f"No se puede inicializar el archivo de almacenamiento: {e}") from e
    
    def read(self) -> List[Dict[str, Any]]:
        """
        Leer todos los registros del archivo.
        
        Returns:
            Lista de registros. Retorna lista vacía si el archivo no existe o es inválido.
            
        Raises:
            RuntimeError: Si el archivo no se puede leer o contiene JSON inválido
        """
        if not self.file_path.exists():
            logger.warning(f"El archivo de almacenamiento no existe: {self.file_path}")
            return []
        
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                
                if not content:
                    logger.debug(f"Archivo vacío: {self.file_path}")
                    return []
                
                data = json.loads(content)
            
            if not isinstance(data, dict):
                logger.error(f"Formato de archivo inválido: el contenido no es un diccionario en {self.file_path}")
                return []
            
            if 'records' not in data:
                logger.error("Formato de archivo inválido: falta la clave 'records'")
                return []
            
            records = data.get('records', [])
            if not isinstance(records, list):
                logger.error("Formato de archivo inválido: 'records' no es una lista")
                return []
            
            logger.debug(f"Leídos {len(records)} registros de {self.file_path}")
            return records
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON inválido en archivo de almacenamiento {self.file_path}: {e}")
            raise RuntimeError(f"No se puede analizar el archivo de almacenamiento: {e}") from e
        except (OSError, IOError) as e:
            logger.error(f"Error de E/S al leer archivo de almacenamiento {self.file_path}: {e}")
            raise RuntimeError(f"No se puede leer el archivo de almacenamiento: {e}") from e
        except Exception as e:
            logger.error(f"Error inesperado al leer el archivo: {e}")
            raise RuntimeError(f"Error inesperado al leer el archivo: {e}") from e
    
    def write(self, records: List[Dict[str, Any]]) -> bool:
        """
        Escribir registros al archivo, reemplazando todos los registros existentes.
        
        Args:
            records: Lista de diccionarios de registros a escribir
            
        Returns:
            True si la escritura fue exitosa, False en caso contrario
            
        Raises:
            TypeError: Si records no es una lista
            ValueError: Si records contiene elementos inválidos
            RuntimeError: Si el archivo no se puede escribir
        """
        if not isinstance(records, list):
            raise TypeError("records debe ser una lista")
        
        for i, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(f"El elemento en índice {i} no es un diccionario válido")
        
        try:
            data = {"records": records}
            
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Escritos {len(records)} registros en {self.file_path}")
            return True
            
        except (OSError, IOError) as e:
            logger.error(f"Error de E/S al escribir archivo de almacenamiento {self.file_path}: {e}")
            raise RuntimeError(f"No se puede escribir el archivo de almacenamiento: {e}") from e
        except (TypeError, ValueError) as e:
            logger.error(f"Error al serializar registros: {e}")
            raise RuntimeError(f"No se pueden serializar los registros: {e}") from e
        except json.JSONEncodeError as e:
            logger.error(f"Error al codificar JSON: {e}")
            raise RuntimeError(f"Error al codificar los registros a JSON: {e}") from e
        except Exception as e:
            logger.error(f"Error inesperado al escribir: {e}")
            raise RuntimeError(f"Error inesperado al escribir: {e}") from e
    
    def update(self, record_id: str, updates: Dict[str, Any]) -> bool:
        """
        Actualizar un registro específico por ID.
        
        Args:
            record_id: El ID del registro a actualizar
            updates: Diccionario de campos a actualizar
            
        Returns:
            True si el registro fue encontrado y actualizado, False en caso contrario
            
        Raises:
            ValueError: Si record_id está vacío o updates no es un diccionario
            TypeError: Si los tipos de los argumentos son incorrectos
            RuntimeError: Si las operaciones de archivo fallan
        """
        if not isinstance(record_id, str):
            raise TypeError("record_id debe ser una cadena")
        
        if not record_id or not record_id.strip():
            raise ValueError("record_id debe ser una cadena no vacía")
        
        if not isinstance(updates, dict):
            raise TypeError("updates debe ser un diccionario")
        
        if not updates:
            logger.warning("No se proporcionaron actualizaciones")
            return False
        
        try:
            records = self.read()
            
            if records is None:
                logger.error("No se pudo leer los registros para actualizar")
                return False
            
            if not isinstance(records, list):
                logger.error("Los registros leídos no son una lista válida")
                return False
            
            record_found = False
            for i, record in enumerate(records):
                if not isinstance(record, dict):
                    logger.warning(f"Registro inválido en índice {i}, omitiendo")
                    continue
                
                if record.get('id') == record_id:
                    original_id = record.get('id')
                    records[i].update(updates)
                    if 'id' not in records[i] or records[i].get('id') != original_id:
                        records[i]['id'] = original_id
                    record_found = True
                    logger.debug(f"Registro actualizado con id: {record_id}")
                    break
            
            if not record_found:
                logger.warning(f"Registro con id '{record_id}' no encontrado")
                return False
            
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump({"records": records}, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Registro actualizado exitosamente: {record_id}")
            return True
            
        except (OSError, IOError) as e:
            logger.error(f"Error de E/S al actualizar el archivo {self.file_path}: {e}")
            raise RuntimeError(f"Error al escribir el archivo durante la actualización: {e}") from e
        except (TypeError, ValueError) as e:
            logger.error(f"Error de validación al actualizar registro: {e}")
            raise
        except json.JSONEncodeError as e:
            logger.error(f"Error al codificar JSON durante la actualización: {e}")
            raise RuntimeError(f"Error al serializar los registros actualizados: {e}") from e
        except RuntimeError as e:
            logger.error(f"Error al actualizar registro: {e}")
            raise
        except Exception as e:
            logger.error(f"Error inesperado durante la actualización: {e}")
            raise RuntimeError(f"Error inesperado durante la actualización: {e}") from e
    
    def add(self, record: Dict[str, Any]) -> bool:
        """
        Agregar un nuevo registro al almacenamiento.
        
        Args:
            record: Diccionario que representa el registro a agregar
            
        Returns:
            True si el registro fue agregado exitosamente, False en caso contrario
            
        Raises:
            ValueError: Si record no es un diccionario o falta el campo 'id' requerido
        """
        if not isinstance(record, dict):
            raise ValueError("record debe ser un diccionario")
        
        if 'id' not in record:
            raise ValueError("record debe contener un campo 'id'")
        
        try:
            records = self.read()
            
            for existing_record in records:
                if isinstance(existing_record, dict) and existing_record.get('id') == record['id']:
                    logger.warning(f"Registro con id '{record['id']}' ya existe")
                    return False
            
            records.append(record)
            self.write(records)
            logger.info(f"Nuevo registro agregado con id: {record['id']}")
            return True
            
        except (RuntimeError, ValueError) as e:
            logger.error(f"Error al agregar registro: {e}")
            raise
    
    def delete(self, record_id: str) -> bool:
        """
        Eliminar un registro por ID.
        
        Args:
            record_id: El ID del registro a eliminar
            
        Returns:
            True si el registro fue encontrado y eliminado, False en caso contrario
            
        Raises:
            ValueError: Si record_id está vacío
        """
        if not record_id or not isinstance(record_id, str):
            raise ValueError("record_id debe ser una cadena no vacía")
        
        try:
            records = self.read()
            original_count = len(records)
            
            records = [
                record for record in records
                if isinstance(record, dict) and record.get('id') != record_id
            ]
            
            if len(records) == original_count:
                logger.warning(f"Registro con id '{record_id}' no encontrado")
                return False
            
            self.write(records)
            logger.info(f"Registro eliminado con id: {record_id}")
            return True
            
        except (RuntimeError, ValueError) as e:
            logger.error(f"Error al eliminar registro: {e}")
            raise
    
    def get(self, record_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtener un registro específico por ID.
        
        Args:
            record_id: El ID del registro a recuperar
            
        Returns:
            El diccionario del registro si se encuentra, None en caso contrario
            
        Raises:
            ValueError: Si record_id está vacío
        """
        if not record_id or not isinstance(record_id, str):
            raise ValueError("record_id debe ser una cadena no vacía")
        
        try:
            records = self.read()
            
            for record in records:
                if isinstance(record, dict) and record.get('id') == record_id:
                    return record
            
            logger.debug(f"Registro con id '{record_id}' no encontrado")
            return None
            
        except RuntimeError as e:
            logger.error(f"Error al obtener registro: {e}")
            raise
