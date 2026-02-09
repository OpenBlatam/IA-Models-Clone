"""
Category Generators - Generadores por categoría (optimizado)
============================================================

Generadores especializados agrupados por categoría funcional.
Reduce duplicación y mejora mantenibilidad.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Callable
from functools import partial

from .base_generator import BaseGenerator

logger = logging.getLogger(__name__)


class CategoryGeneratorRegistry:
    """
    Registry para generadores por categoría (optimizado).
    
    Agrupa generadores relacionados y proporciona acceso unificado.
    """
    
    def __init__(self):
        """Inicializar registry"""
        self._generators: Dict[str, List[BaseGenerator]] = {}
        self._logger = logging.getLogger(f"{__name__}.CategoryGeneratorRegistry")
    
    def register(
        self,
        category: str,
        generator: BaseGenerator
    ) -> None:
        """
        Registrar generador en categoría (optimizado).
        
        Args:
            category: Categoría del generador
            generator: Instancia del generador
        """
        if not category or not category.strip():
            raise ValueError("Category cannot be empty")
        
        if not isinstance(generator, BaseGenerator):
            raise TypeError("Generator must be instance of BaseGenerator")
        
        if category not in self._generators:
            self._generators[category] = []
        
        self._generators[category].append(generator)
        self._logger.debug(f"Registered {generator.name} in category {category}")
    
    def get_generators(self, category: str) -> List[BaseGenerator]:
        """
        Obtener generadores de una categoría (optimizado).
        
        Args:
            category: Categoría a obtener
            
        Returns:
            Lista de generadores en la categoría
        """
        return self._generators.get(category, [])
    
    def get_all_categories(self) -> List[str]:
        """
        Obtener todas las categorías (optimizado).
        
        Returns:
            Lista de categorías registradas
        """
        return list(self._generators.keys())


class TrainingCategoryGenerator:
    """
    Generador para categoría de entrenamiento (optimizado).
    
    Agrupa todos los generadores relacionados con entrenamiento.
    """
    
    CATEGORY = "training"
    
    def __init__(self, registry: CategoryGeneratorRegistry):
        """
        Inicializar generador de categoría.
        
        Args:
            registry: Registry de generadores
        """
        self.registry = registry
        self.logger = logging.getLogger(f"{__name__}.TrainingCategoryGenerator")
    
    def generate_all(
        self,
        project_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any]
    ) -> None:
        """
        Generar todos los componentes de entrenamiento (optimizado).
        
        Args:
            project_dir: Directorio del proyecto
            keywords: Keywords extraídos
            project_info: Información del proyecto
        """
        generators = self.registry.get_generators(self.CATEGORY)
        
        if not generators:
            self.logger.warning(f"No generators found for category {self.CATEGORY}")
            return
        
        target_dir = project_dir / "app" / "utils"
        target_dir.mkdir(parents=True, exist_ok=True)
        
        for generator in generators:
            try:
                generator.generate_with_validation(target_dir, keywords, project_info)
            except Exception as e:
                self.logger.error(
                    f"Error generating {generator.name}: {e}",
                    exc_info=True
                )


class DataCategoryGenerator:
    """Generador para categoría de datos (optimizado)"""
    
    CATEGORY = "data"
    
    def __init__(self, registry: CategoryGeneratorRegistry):
        self.registry = registry
        self.logger = logging.getLogger(f"{__name__}.DataCategoryGenerator")
    
    def generate_all(
        self,
        project_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any]
    ) -> None:
        """Generar todos los componentes de datos"""
        generators = self.registry.get_generators(self.CATEGORY)
        
        if not generators:
            self.logger.warning(f"No generators found for category {self.CATEGORY}")
            return
        
        target_dir = project_dir / "app" / "utils"
        target_dir.mkdir(parents=True, exist_ok=True)
        
        for generator in generators:
            try:
                generator.generate_with_validation(target_dir, keywords, project_info)
            except Exception as e:
                self.logger.error(
                    f"Error generating {generator.name}: {e}",
                    exc_info=True
                )


def create_category_generator_factory() -> Callable:
    """
    Factory para crear generadores de categoría (optimizado).
    
    Returns:
        Función factory para crear generadores
    """
    registry = CategoryGeneratorRegistry()
    
    def create_generator(category: str) -> Any:
        """Crear generador de categoría"""
        if category == "training":
            return TrainingCategoryGenerator(registry)
        elif category == "data":
            return DataCategoryGenerator(registry)
        else:
            raise ValueError(f"Unknown category: {category}")
    
    return create_generator

