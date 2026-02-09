"""Env Loader - Carga de variables de entorno mejorado"""
import os
from typing import Dict, Any, Optional, List, Union
from pathlib import Path


class EnvLoader:
    """Cargador de variables de entorno mejorado"""
    
    def __init__(self, prefix: Optional[str] = None, case_sensitive: bool = True):
        """
        Inicializar cargador.
        
        Args:
            prefix: Prefijo para filtrar variables (ej: "ROBOT_")
            case_sensitive: Si False, convierte keys a lowercase
        """
        self.prefix = prefix
        self.case_sensitive = case_sensitive
    
    def load(self, prefix: Optional[str] = None) -> Dict[str, Any]:
        """
        Carga variables de entorno.
        
        Args:
            prefix: Prefijo opcional (sobrescribe el del constructor)
        
        Returns:
            Diccionario con variables de entorno
        """
        prefix = prefix or self.prefix
        config = {}
        
        for key, value in os.environ.items():
            if prefix is None or key.startswith(prefix):
                final_key = key if self.case_sensitive else key.lower()
                config[final_key] = self._parse_value(value)
        
        return config
    
    def get(
        self,
        key: str,
        default: Any = None,
        required: bool = False,
        type: Optional[type] = None
    ) -> Any:
        """
        Obtener variable de entorno con parsing.
        
        Args:
            key: Nombre de la variable
            default: Valor por defecto
            required: Si True, lanza error si no existe
            type: Tipo a convertir (int, float, bool, str)
        
        Returns:
            Valor de la variable
        
        Raises:
            ValueError: Si required=True y la variable no existe
        """
        value = os.environ.get(key)
        
        if value is None:
            if required:
                raise ValueError(f"Required environment variable {key} not set")
            return default
        
        parsed = self._parse_value(value)
        
        if type is not None:
            try:
                return type(parsed)
            except (ValueError, TypeError):
                if required:
                    raise ValueError(f"Cannot convert {key} to {type.__name__}")
                return default
        
        return parsed
    
    def get_bool(self, key: str, default: bool = False) -> bool:
        """
        Obtener variable booleana.
        
        Args:
            key: Nombre de la variable
            default: Valor por defecto
        
        Returns:
            Valor booleano
        """
        value = os.environ.get(key, str(default)).lower()
        return value in ('true', '1', 'yes', 'on')
    
    def get_int(self, key: str, default: int = 0) -> int:
        """
        Obtener variable entera.
        
        Args:
            key: Nombre de la variable
            default: Valor por defecto
        
        Returns:
            Valor entero
        """
        try:
            return int(os.environ.get(key, str(default)))
        except ValueError:
            return default
    
    def get_float(self, key: str, default: float = 0.0) -> float:
        """
        Obtener variable flotante.
        
        Args:
            key: Nombre de la variable
            default: Valor por defecto
        
        Returns:
            Valor flotante
        """
        try:
            return float(os.environ.get(key, str(default)))
        except ValueError:
            return default
    
    def get_list(
        self,
        key: str,
        separator: str = ',',
        default: Optional[List[str]] = None
    ) -> List[str]:
        """
        Obtener variable como lista.
        
        Args:
            key: Nombre de la variable
            separator: Separador (default: ',')
            default: Valor por defecto
        
        Returns:
            Lista de strings
        """
        value = os.environ.get(key)
        if value is None:
            return default or []
        return [item.strip() for item in value.split(separator) if item.strip()]
    
    def load_from_file(
        self,
        file_path: Union[str, Path],
        override: bool = True
    ) -> Dict[str, Any]:
        """
        Cargar variables desde archivo .env.
        
        Args:
            file_path: Ruta al archivo .env
            override: Si True, sobrescribe variables existentes
        
        Returns:
            Diccionario con variables cargadas
        """
        try:
            from dotenv import dotenv_values
            file_path_obj = Path(file_path)
            
            if not file_path_obj.exists():
                return {}
            
            loaded = dotenv_values(file_path_obj)
            
            if override:
                for key, value in loaded.items():
                    if value is not None:
                        os.environ[key] = value
            
            return {k: self._parse_value(v) for k, v in loaded.items() if v is not None}
        except ImportError:
            return {}
        except Exception:
            return {}
    
    def _parse_value(self, value: str) -> Any:
        """
        Parsear valor de string a tipo apropiado.
        
        Args:
            value: Valor string
        
        Returns:
            Valor parseado
        """
        if not value:
            return value
        
        value = value.strip()
        
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        if value.isdigit():
            return int(value)
        
        try:
            if '.' in value:
                return float(value)
        except ValueError:
            pass
        
        return value
    
    def set(self, key: str, value: Any, override: bool = True):
        """
        Establecer variable de entorno.
        
        Args:
            key: Nombre de la variable
            value: Valor
            override: Si True, sobrescribe si existe
        """
        if key in os.environ and not override:
            return
        
        os.environ[key] = str(value)
    
    def unset(self, key: str):
        """
        Eliminar variable de entorno.
        
        Args:
            key: Nombre de la variable
        """
        os.environ.pop(key, None)
    
    def has(self, key: str) -> bool:
        """
        Verificar si variable existe.
        
        Args:
            key: Nombre de la variable
        
        Returns:
            True si existe
        """
        return key in os.environ

