"""
Generator Registry - Registro y Factory de Generadores
=======================================================

Implementa un patrón Registry/Factory para la creación y gestión de generadores
con lazy loading y cacheo inteligente.
Optimizado siguiendo mejores prácticas de Python y FastAPI.
"""

from typing import Dict, Any, Optional, Type
import importlib

from ...shared_utils import get_logger

logger = get_logger(__name__)


def _import_generator_module(module_path: str, class_name: str) -> Optional[Type]:
    """
    Importa un módulo de generador (función pura).
    
    Args:
        module_path: Ruta del módulo
        class_name: Nombre de la clase
        
    Returns:
        Clase del generador o None si falla
    """
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name, None)
    except (ImportError, AttributeError) as e:
        logger.warning(f"Failed to import {module_path}.{class_name}: {e}")
        return None


def _create_generator_instance(generator_class: Type) -> Optional[Any]:
    """
    Crea una instancia de generador (función pura).
    
    Args:
        generator_class: Clase del generador
        
    Returns:
        Instancia del generador o None si falla
    """
    if generator_class is None:
        return None
    
    try:
        return generator_class()
    except Exception as e:
        logger.warning(f"Failed to instantiate generator: {e}")
        return None


class GeneratorRegistry:
    """
    Registry para gestionar la creación y cacheo de generadores.
    Implementa lazy loading para optimizar memoria y tiempo de inicialización.
    Optimizado con funciones puras y mejor manejo de errores.
    """
    
    def __init__(self, base_module: str) -> None:
        """
        Inicializa el registry.
        
        Args:
            base_module: Módulo base para importar generadores (ej: "core.deep_learning")
            
        Raises:
            ValueError: Si base_module está vacío
        """
        if not base_module:
            raise ValueError("base_module cannot be empty")
        
        self.base_module = base_module
        self._generators: Dict[str, Any] = {}
        self._generator_config: Optional[Dict[str, tuple]] = None
    
    def set_config(self, config: Dict[str, tuple]) -> None:
        """
        Establece la configuración de generadores.
        
        Args:
            config: Diccionario con la configuración de generadores
            
        Raises:
            ValueError: Si config está vacío
        """
        if not config:
            raise ValueError("config cannot be empty")
        
        self._generator_config = config
    
    def get_generator(self, generator_key: str) -> Optional[Any]:
        """
        Obtiene un generador con lazy loading y cacheo.
        
        Args:
            generator_key: Clave del generador a obtener
            
        Returns:
            Instancia del generador o None si no se puede cargar
        """
        if not generator_key:
            return None
        
        if generator_key in self._generators:
            return self._generators[generator_key]
        
        if self._generator_config is None:
            logger.error("Generator configuration not set")
            return None
        
        if generator_key not in self._generator_config:
            logger.warning(f"Unknown generator: {generator_key}")
            return None
        
        generator = self._create_generator(generator_key)
        if generator is not None:
            self._generators[generator_key] = generator
        
        return generator
    
    def _create_generator(self, generator_key: str) -> Optional[Any]:
        """
        Crea una instancia de un generador.
        
        Args:
            generator_key: Clave del generador
            
        Returns:
            Instancia del generador o None si falla
        """
        if not self._generator_config:
            return None
        
        config = self._generator_config.get(generator_key)
        if not config:
            return None
        
        attr_name, class_name, _ = config
        module_path = f"{self.base_module}.{attr_name}"
        
        generator_class = _import_generator_module(module_path, class_name)
        if generator_class is None:
            return None
        
        return _create_generator_instance(generator_class)
    
    def clear_cache(self) -> None:
        """Limpia el cache de generadores."""
        self._generators.clear()
    
    def get_loaded_generators(self) -> List[str]:
        """
        Retorna la lista de generadores cargados.
        
        Returns:
            Lista de claves de generadores cargados
        """
        return list(self._generators.keys())


class GeneratorFactory:
    """
    Factory para crear instancias de GeneratorRegistry con configuración predefinida.
    Optimizado con funciones estáticas y mejor manejo de errores.
    """
    
    @staticmethod
    def create_registry(base_module: str, config: Dict[str, tuple]) -> GeneratorRegistry:
        """
        Crea un registry con configuración.
        
        Args:
            base_module: Módulo base para imports
            config: Configuración de generadores
            
        Returns:
            Instancia de GeneratorRegistry configurada
            
        Raises:
            ValueError: Si los parámetros son inválidos
        """
        if not base_module:
            raise ValueError("base_module cannot be empty")
        
        if not config:
            raise ValueError("config cannot be empty")
        
        registry = GeneratorRegistry(base_module)
        registry.set_config(config)
        return registry
