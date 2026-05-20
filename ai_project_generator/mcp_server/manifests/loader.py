"""
Manifest Loader - Cargador de manifiestos desde archivos
=========================================================
"""

import json
import yaml
import logging
from pathlib import Path
from typing import List, Dict, Any

from .models import ResourceManifest
from .registry import ManifestRegistry

logger = logging.getLogger(__name__)


class ManifestLoader:
    """
    Cargador de manifests desde archivos JSON/YAML
    
    Soporta:
    - Archivos individuales
    - Directorios con múltiples archivos
    - Validación automática con Pydantic
    """
    
    @staticmethod
    def load_from_file(file_path: str) -> ResourceManifest:
        """
        Carga un manifest desde un archivo
        
        Args:
            file_path: Ruta al archivo (JSON o YAML)
            
        Returns:
            ResourceManifest cargado y validado
            
        Raises:
            FileNotFoundError: Si el archivo no existe
            ValueError: Si el formato no es soportado o los datos son inválidos
            TypeError: Si file_path no es string
        """
        if not isinstance(file_path, str):
            raise TypeError(f"file_path must be a string, got {type(file_path)}")
        if not file_path or not file_path.strip():
            raise ValueError("file_path cannot be empty or whitespace")
        
        path = Path(file_path.strip())
        
        if not path.exists():
            raise FileNotFoundError(f"Manifest file not found: {file_path}")
        
        if not path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                if path.suffix.lower() in [".yaml", ".yml"]:
                    try:
                        data = yaml.safe_load(f)
                    except yaml.YAMLError as e:
                        raise ValueError(f"Invalid YAML in {file_path}: {e}") from e
                elif path.suffix.lower() == ".json":
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Invalid JSON in {file_path}: {e}") from e
                else:
                    raise ValueError(f"Unsupported file format: {path.suffix}. Supported: .json, .yaml, .yml")
            
            if not isinstance(data, dict):
                raise ValueError(f"Manifest file must contain a dictionary, got {type(data)}")
            
            # Intentar cargar usando Pydantic
            try:
                manifest = ResourceManifest(**data)
                logger.info(f"Loaded manifest: {manifest.resource_id} from {file_path}")
                return manifest
            except Exception as e:
                logger.error(f"Error validating manifest from {file_path}: {e}", exc_info=True)
                raise ValueError(f"Invalid manifest data in {file_path}: {e}") from e
        except (FileNotFoundError, ValueError, TypeError) as e:
            # Re-raise validation errors
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading manifest from {file_path}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load manifest from {file_path}: {e}") from e
    
    @staticmethod
    def load_from_directory(directory: str, pattern: str = "*.{json,yaml,yml}") -> List[ResourceManifest]:
        """
        Carga múltiples manifests desde un directorio
        
        Args:
            directory: Directorio con archivos de manifest
            pattern: Patrón de búsqueda (default: *.json, *.yaml, *.yml)
            
        Returns:
            Lista de ResourceManifest cargados
            
        Raises:
            ValueError: Si el directorio no existe o no es un directorio
            TypeError: Si directory o pattern no son strings
        """
        if not isinstance(directory, str):
            raise TypeError(f"directory must be a string, got {type(directory)}")
        if not directory or not directory.strip():
            raise ValueError("directory cannot be empty or whitespace")
        
        if not isinstance(pattern, str):
            raise TypeError(f"pattern must be a string, got {type(pattern)}")
        if not pattern or not pattern.strip():
            raise ValueError("pattern cannot be empty or whitespace")
        
        dir_path = Path(directory.strip())
        
        if not dir_path.exists():
            raise ValueError(f"Directory not found: {directory}")
        
        if not dir_path.is_dir():
            raise ValueError(f"Path is not a directory: {directory}")
        
        manifests = []
        errors = []
        
        # Buscar archivos que coincidan con el patrón
        try:
            files = list(dir_path.glob(pattern.strip()))
        except Exception as e:
            raise ValueError(f"Invalid glob pattern '{pattern}': {e}") from e
        
        if not files:
            logger.warning(f"No manifest files found in {directory} matching pattern '{pattern}'")
            return manifests
        
        for file_path in files:
            if not file_path.is_file():
                continue
            try:
                manifest = ManifestLoader.load_from_file(str(file_path))
                manifests.append(manifest)
            except Exception as e:
                error_msg = f"Skipping {file_path}: {e}"
                logger.warning(error_msg)
                errors.append(error_msg)
        
        logger.info(f"Loaded {len(manifests)} manifests from {directory}")
        if errors:
            logger.warning(f"Encountered {len(errors)} errors while loading manifests")
        
        return manifests
    
    @staticmethod
    def load_from_dict(data: Dict[str, Any]) -> ResourceManifest:
        """
        Carga un manifest desde un diccionario
        
        Args:
            data: Diccionario con datos del manifest
            
        Returns:
            ResourceManifest
            
        Raises:
            ValueError: Si los datos son inválidos
            TypeError: Si data no es un diccionario
        """
        if not isinstance(data, dict):
            raise TypeError(f"data must be a dictionary, got {type(data)}")
        if not data:
            raise ValueError("data cannot be empty")
        
        try:
            manifest = ResourceManifest(**data)
            logger.debug(f"Loaded manifest from dict: {manifest.resource_id}")
            return manifest
        except Exception as e:
            logger.error(f"Error loading manifest from dict: {e}", exc_info=True)
            raise ValueError(f"Invalid manifest data: {e}") from e
    
    @staticmethod
    def load_from_json(json_str: str) -> ResourceManifest:
        """
        Carga un manifest desde JSON string
        
        Args:
            json_str: String JSON
            
        Returns:
            ResourceManifest
            
        Raises:
            ValueError: Si el JSON es inválido o está vacío
            TypeError: Si json_str no es string
        """
        if not isinstance(json_str, str):
            raise TypeError(f"json_str must be a string, got {type(json_str)}")
        if not json_str or not json_str.strip():
            raise ValueError("json_str cannot be empty or whitespace")
        
        try:
            data = json.loads(json_str.strip())
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON string: {e}") from e
        
        if not isinstance(data, dict):
            raise ValueError(f"JSON must represent a dictionary, got {type(data)}")
        
        return ManifestLoader.load_from_dict(data)
    
    @staticmethod
    def register_from_file(registry: ManifestRegistry, file_path: str) -> None:
        """
        Carga y registra un manifest desde archivo
        
        Args:
            registry: Registry donde registrar (debe ser instancia de ManifestRegistry)
            file_path: Ruta al archivo
            
        Raises:
            ValueError: Si registry es None o file_path es inválido
            TypeError: Si registry no es ManifestRegistry
        """
        if registry is None:
            raise ValueError("registry cannot be None")
        if not hasattr(registry, 'register'):
            raise TypeError(f"registry must be a ManifestRegistry instance, got {type(registry)}")
        
        manifest = ManifestLoader.load_from_file(file_path)
        registry.register(manifest)
        logger.debug(f"Registered manifest {manifest.resource_id} from {file_path}")
    
    @staticmethod
    def register_from_directory(registry: ManifestRegistry, directory: str) -> None:
        """
        Carga y registra múltiples manifests desde directorio
        
        Args:
            registry: Registry donde registrar (debe ser instancia de ManifestRegistry)
            directory: Directorio con archivos
            
        Raises:
            ValueError: Si registry es None o directory es inválido
            TypeError: Si registry no es ManifestRegistry
        """
        if registry is None:
            raise ValueError("registry cannot be None")
        if not hasattr(registry, 'register'):
            raise TypeError(f"registry must be a ManifestRegistry instance, got {type(registry)}")
        
        manifests = ManifestLoader.load_from_directory(directory)
        registered_count = 0
        
        for manifest in manifests:
            try:
                registry.register(manifest)
                registered_count += 1
            except Exception as e:
                logger.warning(f"Failed to register manifest {manifest.resource_id}: {e}")
        
        logger.info(f"Registered {registered_count}/{len(manifests)} manifests from {directory}")

