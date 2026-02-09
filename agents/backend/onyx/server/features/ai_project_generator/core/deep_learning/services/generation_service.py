"""
Generation Service - Servicio de generación (optimizado)
========================================================

Servicio especializado para orquestar la generación de código.
Encapsula la lógica de alto nivel de generación.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

from ..generator_config import GENERATOR_MAP
from ..generation_strategy import StrategyOrchestrator
from ..utils.stats import GenerationStats
from .detection_service import DetectionService
from .validation_service import ValidationService
from .template_service import TemplateService

logger = logging.getLogger(__name__)


class GenerationService:
    """
    Servicio de generación (optimizado).
    
    Orquesta la generación de código de Deep Learning.
    Coordina entre diferentes servicios y componentes.
    """
    
    def __init__(
        self,
        registry: Any,
        executor: Any,
        stats: Optional[GenerationStats] = None
    ):
        """
        Inicializar servicio de generación (optimizado).
        
        Args:
            registry: Registry de generadores
            executor: Ejecutor de generadores
            stats: Estadísticas de generación (opcional)
        """
        self._registry = registry
        self._executor = executor
        self._stats = stats or GenerationStats()
        self._strategy_orchestrator = StrategyOrchestrator.create_default()
        
        # Servicios especializados
        self.detection_service = DetectionService()
        self.validation_service = ValidationService()
        self.template_service = TemplateService()
        
        self.logger = logging.getLogger(f"{__name__}.GenerationService")
    
    def prepare_generation(
        self,
        project_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Preparar generación (optimizado).
        
        Valida inputs, detecta características, y prepara project_info.
        
        Args:
            project_dir: Directorio del proyecto
            keywords: Keywords extraídos
            project_info: Información del proyecto
            
        Returns:
            project_info actualizado con características detectadas
            
        Raises:
            ValueError: Si la validación falla
        """
        # Validar request
        self.validation_service.validate_generation_request(
            project_dir, keywords, project_info
        )
        
        # Detectar características
        dl_features = self.detection_service.detect_all_features(keywords)
        
        # Actualizar project_info
        project_info.update(dl_features.to_dict())
        project_info["framework_type"] = dl_features.framework.value
        project_info["model_type"] = dl_features.model_type.value
        project_info["fine_tuning_technique"] = dl_features.fine_tuning_technique.value
        
        self.logger.info(
            f"Generation prepared: framework={dl_features.framework.value}, "
            f"model_type={dl_features.model_type.value}"
        )
        
        return project_info
    
    def get_generators_to_run(self, keywords: Dict[str, Any]) -> List[str]:
        """
        Obtener lista de generadores a ejecutar (optimizado).
        
        Args:
            keywords: Keywords extraídos
            
        Returns:
            Lista de claves de generadores
        """
        try:
            generators = self._strategy_orchestrator.get_generators_to_run(keywords)
            self.logger.debug(f"Generators to run: {generators}")
            return generators
        except Exception as e:
            self.logger.error(f"Error getting generators to run: {e}", exc_info=True)
            raise
    
    def generate_special_components(
        self,
        project_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
        generator_key: str
    ) -> bool:
        """
        Generar componentes especiales (interface, config) (optimizado).
        
        Args:
            project_dir: Directorio del proyecto
            keywords: Keywords extraídos
            project_info: Información del proyecto
            generator_key: Clave del generador especial
            
        Returns:
            True si se generó exitosamente
        """
        if generator_key == "interface":
            return self._generate_interface(project_dir, keywords, project_info)
        elif generator_key == "config":
            return self._generate_config(project_dir, keywords, project_info)
        else:
            return False
    
    def _generate_interface(
        self,
        project_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any]
    ) -> bool:
        """Generar interfaz Gradio"""
        if not keywords.get("requires_gradio") and not project_info.get("requires_gradio"):
            return False
        
        generator = self._executor.get_generator("interface")
        if generator is None:
            return False
        
        services_dir = project_dir / "app" / "services"
        services_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            if hasattr(generator, 'generate_with_validation'):
                generator.generate_with_validation(services_dir, keywords, project_info)
            else:
                generator.generate(services_dir, keywords, project_info)
            return True
        except Exception as e:
            self.logger.error(f"Failed to generate interface: {e}", exc_info=True)
            return False
    
    def _generate_config(
        self,
        project_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any]
    ) -> bool:
        """Generar configuración"""
        generator = self._executor.get_generator("config")
        if generator is None:
            return False
        
        config_dir = project_dir / "app" / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            if keywords.get("requires_training") or project_info.get("requires_training"):
                generator.generate_training_config(config_dir, keywords, project_info)
            
            generator.generate_model_config(config_dir, keywords, project_info)
            return True
        except Exception as e:
            self.logger.error(f"Failed to generate config: {e}", exc_info=True)
            return False
    
    def get_stats(self) -> GenerationStats:
        """
        Obtener estadísticas (optimizado).
        
        Returns:
            Estadísticas de generación
        """
        return self._stats

