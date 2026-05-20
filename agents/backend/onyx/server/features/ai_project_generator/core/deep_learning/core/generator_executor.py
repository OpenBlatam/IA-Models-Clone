"""
Generator Executor - Ejecutor de generadores
============================================

Ejecuta generadores de forma modular y controlada.
Optimizado siguiendo mejores prácticas de Python y FastAPI.
"""

from pathlib import Path
from typing import Dict, Any, Optional

from ...shared_utils import get_logger, ensure_directory
from ..generator_config import GENERATOR_MAP
from ..utils.validators import (
    validate_generator_key,
    validate_project_path,
    get_target_directory
)
from ..utils.stats import GenerationStats

logger = get_logger(__name__)


def _should_skip_generation(
    generator: Any,
    keywords: Dict[str, Any]
) -> bool:
    """
    Determina si se debe omitir la generación (función pura).
    
    Args:
        generator: Generador a verificar
        keywords: Keywords del proyecto
        
    Returns:
        True si se debe omitir, False en caso contrario
    """
    if not hasattr(generator, 'should_generate'):
        return False
    
    try:
        return not generator.should_generate(keywords)
    except Exception as e:
        logger.warning(f"Error in should_generate: {e}")
        return False


def _execute_generator_with_validation(
    generator: Any,
    target_dir: Path,
    keywords: Dict[str, Any],
    project_info: Dict[str, Any]
) -> None:
    """
    Ejecuta generador con validación (función pura).
    
    Args:
        generator: Generador a ejecutar
        target_dir: Directorio destino
        keywords: Keywords del proyecto
        project_info: Información del proyecto
    """
    if hasattr(generator, 'generate_with_validation'):
        generator.generate_with_validation(target_dir, keywords, project_info)
    else:
        generator.generate(target_dir, keywords, project_info)


class GeneratorExecutor:
    """
    Ejecutor de generadores.
    
    Encapsula la lógica de ejecución de generadores con validación,
    manejo de errores y tracking de estadísticas.
    Optimizado con funciones puras y mejor manejo de errores.
    """
    
    def __init__(
        self,
        registry: Any,
        stats: Optional[GenerationStats] = None
    ) -> None:
        """
        Inicializar ejecutor.
        
        Args:
            registry: Registry de generadores
            stats: Estadísticas de generación (opcional)
            
        Raises:
            ValueError: Si registry es None
        """
        if registry is None:
            raise ValueError("registry cannot be None")
        
        self._registry = registry
        self._stats = stats or GenerationStats()
        self._generator_cache: Dict[str, Any] = {}
        self.logger = logging.getLogger(f"{__name__}.GeneratorExecutor")
    
    def get_generator(self, generator_key: str) -> Optional[Any]:
        """
        Obtener generador con cache.
        
        Args:
            generator_key: Clave del generador
            
        Returns:
            Generador o None si no se encuentra
        """
        if not generator_key:
            return None
        
        if generator_key in self._generator_cache:
            return self._generator_cache[generator_key]
        
        try:
            generator = self._registry.get_generator(generator_key)
            if generator is not None:
                self._generator_cache[generator_key] = generator
            return generator
        except Exception as e:
            self.logger.warning(f"Failed to get generator '{generator_key}': {e}")
            return None
    
    def execute_generator(
        self,
        generator_key: str,
        project_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
        skip_on_error: bool = False
    ) -> bool:
        """
        Ejecutar un generador.
        
        Args:
            generator_key: Clave del generador
            project_dir: Directorio del proyecto
            keywords: Keywords extraídos
            project_info: Información del proyecto
            skip_on_error: Si True, no lanza excepción en error
            
        Returns:
            True si se ejecutó exitosamente, False en caso contrario
            
        Raises:
            ValueError: Si los parámetros son inválidos
            RuntimeError: Si hay error y skip_on_error es False
        """
        validate_generator_key(generator_key)
        validate_project_path(project_dir)
        
        generator = self.get_generator(generator_key)
        if generator is None:
            self.logger.debug(f"Generator '{generator_key}' not found, skipping")
            self._stats.add_skipped(generator_key)
            return False
        
        if generator_key not in GENERATOR_MAP:
            self.logger.warning(f"Configuration not found for {generator_key}")
            self._stats.add_skipped(generator_key)
            return False
        
        if _should_skip_generation(generator, keywords):
            self.logger.debug(
                f"Generator '{generator_key}' skipped "
                "(should_generate returned False)"
            )
            self._stats.add_skipped(generator_key)
            return False
        
        target_dir = get_target_directory(project_dir, generator_key)
        if target_dir is None:
            self.logger.warning(
                f"Could not determine target directory for {generator_key}"
            )
            self._stats.add_skipped(generator_key)
            return False
        
        ensure_directory(target_dir)
        
        try:
            _execute_generator_with_validation(
                generator, target_dir, keywords, project_info
            )
            
            self._stats.add_success(generator_key)
            self.logger.debug(f"Successfully generated {generator_key}")
            return True
        except Exception as e:
            error_msg = str(e)
            self.logger.error(
                f"Failed to generate {generator_key}: {error_msg}",
                exc_info=True
            )
            self._stats.add_failure(generator_key, error_msg)
            
            if not skip_on_error:
                raise RuntimeError(
                    f"Failed to generate {generator_key}: {error_msg}"
                ) from e
            
            return False
    
    def get_stats(self) -> GenerationStats:
        """
        Obtener estadísticas.
        
        Returns:
            Estadísticas de generación
        """
        return self._stats
