"""
Generator Core - Núcleo del generador (optimizado)
====================================================

Clase principal que orquesta la generación de código de Deep Learning.
"""

from pathlib import Path
from typing import Dict, Any, Optional

from ...shared_utils import get_logger, ensure_directory
from ..generator_config import GENERATOR_MAP
from ..generator_registry import GeneratorFactory
from ..generation_strategy import StrategyOrchestrator
from ..utils.validators import validate_project_path
from ..utils.detectors import detect_framework, detect_model_type, FrameworkType, ModelType
from ..utils.stats import GenerationStats
from .generator_executor import GeneratorExecutor

logger = get_logger(__name__)

# Imports opcionales para funcionalidades avanzadas
try:
    from ...deep_learning.code_optimizer import get_optimizer
    CODE_OPTIMIZER_AVAILABLE = True
except ImportError:
    CODE_OPTIMIZER_AVAILABLE = False
    get_optimizer = None

try:
    from ...deep_learning.dependency_analyzer import get_analyzer
    DEPENDENCY_ANALYZER_AVAILABLE = True
except ImportError:
    DEPENDENCY_ANALYZER_AVAILABLE = False
    get_analyzer = None

try:
    from ...deep_learning.test_generator import get_test_generator
    TEST_GENERATOR_AVAILABLE = True
except ImportError:
    TEST_GENERATOR_AVAILABLE = False
    get_test_generator = None

try:
    from ...deep_learning.doc_generator import get_doc_generator
    DOC_GENERATOR_AVAILABLE = True
except ImportError:
    DOC_GENERATOR_AVAILABLE = False
    get_doc_generator = None

try:
    from ...deep_learning.docker_generator import get_docker_generator
    DOCKER_GENERATOR_AVAILABLE = True
except ImportError:
    DOCKER_GENERATOR_AVAILABLE = False
    get_docker_generator = None

try:
    from ...deep_learning.cicd_generator import get_cicd_generator
    CICD_GENERATOR_AVAILABLE = True
except ImportError:
    CICD_GENERATOR_AVAILABLE = False
    get_cicd_generator = None

try:
    from ...deep_learning.performance_analyzer import get_performance_analyzer
    PERFORMANCE_ANALYZER_AVAILABLE = True
except ImportError:
    PERFORMANCE_ANALYZER_AVAILABLE = False
    get_performance_analyzer = None


class DeepLearningGenerator:
    """
    Generador especializado para proyectos de Deep Learning (optimizado).
    
    Usa una arquitectura modular con generadores especializados para cada componente.
    Implementa lazy loading, registry pattern y strategy pattern para máxima modularidad.
    
    Características:
    - Soporte para múltiples frameworks (PyTorch, TensorFlow, JAX)
    - Generación de código siguiendo mejores prácticas de Deep Learning
    - Configuración YAML para hyperparameters
    - Experiment tracking integrado
    - Validación robusta y manejo de errores
    - Estadísticas de generación detalladas
    """
    
    def __init__(self) -> None:
        """
        Inicializa el generador de Deep Learning (optimizado).
        
        Raises:
            ImportError: Si no se pueden importar los módulos necesarios
        """
        try:
            base_module = __name__.rsplit('.', 1)[0].rsplit('.', 1)[0] + ".deep_learning"
            self._registry = GeneratorFactory.create_registry(base_module, GENERATOR_MAP)
            self._strategy_orchestrator = StrategyOrchestrator.create_default()
            
            # Estadísticas de generación
            self._stats = GenerationStats()
            
            # Ejecutor de generadores
            self._executor = GeneratorExecutor(self._registry, self._stats)
            
            self._optimizer = get_optimizer() if CODE_OPTIMIZER_AVAILABLE else None
            self._dependency_analyzer = get_analyzer() if DEPENDENCY_ANALYZER_AVAILABLE else None
            self._test_generator = get_test_generator() if TEST_GENERATOR_AVAILABLE else None
            self._doc_generator = get_doc_generator() if DOC_GENERATOR_AVAILABLE else None
            
            self._docker_generator = get_docker_generator() if DOCKER_GENERATOR_AVAILABLE else None
            self._cicd_generator = get_cicd_generator() if CICD_GENERATOR_AVAILABLE else None
            self._performance_analyzer = get_performance_analyzer() if PERFORMANCE_ANALYZER_AVAILABLE else None
            self._k8s_generator = get_k8s_generator() if KUBERNETES_GENERATOR_AVAILABLE else None
            self._api_generator = get_api_generator() if API_GENERATOR_AVAILABLE else None
            self._monitoring_generator = get_monitoring_generator() if MONITORING_GENERATOR_AVAILABLE else None
            self._training_generator = get_training_generator() if TRAINING_SCRIPT_GENERATOR_AVAILABLE else None
            self._versioning = get_versioning() if MODEL_VERSIONING_AVAILABLE else None
            self._dashboard_generator = get_dashboard_generator() if DASHBOARD_GENERATOR_AVAILABLE else None
            
            logger.info("DeepLearningGenerator initialized successfully")
        except Exception as e:
            logger.error(
                f"Failed to initialize DeepLearningGenerator: {e}",
                exc_info=True
            )
            raise ImportError(f"Failed to initialize generator: {e}") from e
    
    def _generate_component(
        self,
        generator_key: str,
        project_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """
        Método genérico para generar componentes.
        
        Args:
            generator_key: Clave del generador
            project_dir: Directorio del proyecto
            keywords: Keywords extraídos
            project_info: Información del proyecto
        """
        self._executor.execute_generator(generator_key, project_dir, keywords, project_info)
    
    def generate_model_architecture(
        self,
        project_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """Genera la arquitectura del modelo según el tipo detectado."""
        self._generate_component("model", project_dir, keywords, project_info)
    
    def generate_training_utils(
        self,
        project_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """Genera utilidades para entrenamiento."""
        self._generate_component("training", project_dir, keywords, project_info)
    
    def generate_data_utils(
        self,
        project_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """Genera utilidades para manejo de datos."""
        self._generate_component("data", project_dir, keywords, project_info)
    
    def generate_evaluation_utils(
        self,
        project_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """Genera utilidades para evaluación."""
        self._generate_component("evaluation", project_dir, keywords, project_info)
    
    def generate_performance_utils(
        self,
        project_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """Genera utilidades para performance y profiling."""
        self._generate_component("performance", project_dir, keywords, project_info)
    
    def generate_additional_utils(
        self,
        project_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """Genera utilidades adicionales (debugging, visualización, etc.)."""
        self._generate_component("utils", project_dir, keywords, project_info)
    
    def generate_deployment_utils(
        self,
        project_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """Genera utilidades de deployment y serving."""
        self._generate_component("deployment", project_dir, keywords, project_info)
    
    def generate_testing_utils(
        self,
        project_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """Genera utilidades de testing."""
        self._generate_component("testing", project_dir, keywords, project_info)
    
    def generate_conversion_utils(
        self,
        project_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """Genera utilidades de conversión de modelos."""
        self._generate_component("conversion", project_dir, keywords, project_info)
    
    def generate_monitoring_utils(
        self,
        project_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """Genera utilidades de monitoreo."""
        self._generate_component("monitoring", project_dir, keywords, project_info)
    
    def generate_analysis_utils(
        self,
        project_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """Genera utilidades de análisis."""
        self._generate_component("analysis", project_dir, keywords, project_info)
    
    def generate_experimentation_utils(
        self,
        project_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """Genera utilidades de experimentación."""
        self._generate_component("experimentation", project_dir, keywords, project_info)
    
    def generate_serialization_utils(
        self,
        project_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """Genera utilidades de serialización."""
        self._generate_component("serialization", project_dir, keywords, project_info)
    
    def generate_validation_utils(
        self,
        project_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """Genera utilidades de validación."""
        self._generate_component("validation", project_dir, keywords, project_info)
    
    def generate_security_utils(
        self,
        project_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """Genera utilidades de seguridad."""
        self._generate_component("security", project_dir, keywords, project_info)
    
    def generate_data_io_utils(
        self,
        project_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """Genera utilidades de I/O de datos."""
        self._generate_component("data_io", project_dir, keywords, project_info)
    
    def generate_reporting_utils(
        self,
        project_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """Genera utilidades de reportes."""
        self._generate_component("reporting", project_dir, keywords, project_info)
    
    def generate_preprocessing_utils(
        self,
        project_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """Genera utilidades de preprocesamiento."""
        self._generate_component("preprocessing", project_dir, keywords, project_info)
    
    def generate_postprocessing_utils(
        self,
        project_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """Genera utilidades de postprocesamiento."""
        self._generate_component("postprocessing", project_dir, keywords, project_info)
    
    def generate_gradio_interface(
        self,
        project_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """
        Genera interfaz Gradio si se requiere.
        
        Args:
            project_dir: Directorio del proyecto
            keywords: Keywords extraídos
            project_info: Información del proyecto
        """
        if not keywords.get("requires_gradio"):
            return
        
        generator = self._executor.get_generator("interface")
        if generator is None:
            return
        
        services_dir = project_dir / "app" / "services"
        ensure_directory(services_dir)
        
        try:
            if hasattr(generator, 'generate_with_validation'):
                generator.generate_with_validation(services_dir, keywords, project_info)
            else:
                generator.generate(services_dir, keywords, project_info)
        except Exception as e:
            logger.error(f"Failed to generate Gradio interface: {e}", exc_info=True)
            raise
    
    def generate_configs(
        self,
        project_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """
        Genera archivos de configuración.
        
        Args:
            project_dir: Directorio del proyecto
            keywords: Keywords extraídos
            project_info: Información del proyecto
        """
        generator = self._executor.get_generator("config")
        if generator is None:
            return
        
        config_dir = project_dir / "app" / "config"
        ensure_directory(config_dir)
        
        try:
            if keywords.get("requires_training"):
                generator.generate_training_config(config_dir, keywords, project_info)
            
            generator.generate_model_config(config_dir, keywords, project_info)
        except Exception as e:
            logger.error(f"Failed to generate configs: {e}", exc_info=True)
            raise
    
    def generate_all(
        self,
        project_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
        continue_on_error: bool = False
    ) -> GenerationStats:
        """
        Genera todos los componentes de Deep Learning usando estrategias modulares (optimizado).
        
        Args:
            project_dir: Directorio del proyecto
            keywords: Keywords extraídos
            project_info: Información del proyecto
            continue_on_error: Si True, continúa generando aunque haya errores
            
        Returns:
            Estadísticas de generación
            
        Raises:
            ValueError: Si los parámetros son inválidos
        """
        validate_project_path(project_dir)
        
        if not keywords:
            raise ValueError("keywords cannot be empty")
        
        if not project_info:
            raise ValueError("project_info cannot be empty")
        
        # Resetear estadísticas
        self._stats.reset()
        
        # Detectar framework y tipo de modelo
        framework = detect_framework(keywords)
        model_type = detect_model_type(keywords)
        
        logger.info(
            f"Generating Deep Learning components "
            f"(framework: {framework.value}, model_type: {model_type.value})..."
        )
        
        # Agregar información detectada a project_info
        project_info["framework"] = framework.value
        project_info["model_type"] = model_type.value
        
        try:
            generators_to_run = self._strategy_orchestrator.get_generators_to_run(keywords)
        except Exception as e:
            logger.error(f"Failed to get generators to run: {e}", exc_info=True)
            raise
        
        # Generar componentes
        for generator_key in generators_to_run:
            try:
                if generator_key == "interface":
                    self.generate_gradio_interface(project_dir, keywords, project_info)
                elif generator_key == "config":
                    self.generate_configs(project_dir, keywords, project_info)
                else:
                    self._executor.execute_generator(
                        generator_key,
                        project_dir,
                        keywords,
                        project_info,
                        skip_on_error=continue_on_error
                    )
            except Exception as e:
                if continue_on_error:
                    logger.warning(f"Error generating {generator_key}, continuing...")
                    self._stats.add_failure(generator_key, str(e))
                else:
                    logger.error(f"Failed to generate {generator_key}: {e}", exc_info=True)
                    raise
        
        # Log resumen
        summary = self._stats.get_summary()
        logger.info(
            f"Deep Learning components generation completed: "
            f"{summary['successful']} successful, {summary['failed']} failed, "
            f"{summary['skipped']} skipped (success rate: {summary['success_rate']:.2%})"
        )
        
        return self._stats
    
    def get_generation_stats(self) -> GenerationStats:
        """
        Obtener estadísticas de generación (optimizado).
        
        Returns:
            Estadísticas de generación
        """
        return self._stats
    
    def analyze_dependencies(
        self,
        project_dir: Path
    ) -> Dict[str, Any]:
        """
        Analizar dependencias del proyecto.
        
        Args:
            project_dir: Directorio del proyecto
            
        Returns:
            Diccionario con información de dependencias
        """
        if not self._dependency_analyzer:
            return {'error': 'Dependency analyzer no disponible'}
        
        deps = self._dependency_analyzer.analyze_project(project_dir)
        
        return {
            'stdlib': sorted(deps.stdlib),
            'third_party': sorted(deps.third_party),
            'local': sorted(deps.local),
            'missing': sorted(deps.missing),
            'by_file': {
                path: [
                    {
                        'module': d.module,
                        'import_type': d.import_type,
                        'alias': d.alias
                    }
                    for d in deps_list
                ]
                for path, deps_list in deps.by_file.items()
            }
        }
    
    def generate_requirements_txt(
        self,
        project_dir: Path,
        output_file: Optional[Path] = None
    ) -> str:
        """
        Generar requirements.txt basado en dependencias.
        
        Args:
            project_dir: Directorio del proyecto
            output_file: Archivo de salida (opcional)
            
        Returns:
            Contenido de requirements.txt
        """
        if not self._dependency_analyzer:
            return "# Dependency analyzer no disponible\n"
        
        if output_file is None:
            output_file = project_dir / "requirements.txt"
        
        return self._dependency_analyzer.generate_requirements(project_dir, output_file)
    
    def generate_tests(
        self,
        project_dir: Path,
        output_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Generar tests automáticos para el proyecto.
        
        Args:
            project_dir: Directorio del proyecto
            output_dir: Directorio de salida (opcional)
            
        Returns:
            Diccionario con información de tests generados
        """
        if not self._test_generator:
            return {'error': 'Test generator no disponible'}
        
        if output_dir is None:
            output_dir = project_dir / "tests"
        
        results = self._test_generator.generate_tests_for_project(project_dir, output_dir)
        
        return {
            'total_files': len(results),
            'total_tests': sum(len(tests) for tests in results.values()),
            'by_file': {
                path: len(tests)
                for path, tests in results.items()
            }
        }
    
    def generate_documentation(
        self,
        project_dir: Path,
        output_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Generar documentación automática para el proyecto.
        
        Args:
            project_dir: Directorio del proyecto
            output_dir: Directorio de salida (opcional)
            
        Returns:
            Diccionario con información de documentación generada
        """
        if not self._doc_generator:
            return {'error': 'Documentation generator no disponible'}
        
        if output_dir is None:
            output_dir = project_dir / "docs" / "api"
        
        results = self._doc_generator.generate_docs_for_project(project_dir, output_dir)
        
        return {
            'total_files': len(results),
            'output_dir': str(output_dir),
            'files': list(results.keys())
        }
    
    def optimize_generated_code(
        self,
        project_dir: Path,
        generator_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Optimizar código generado.
        
        Args:
            project_dir: Directorio del proyecto
            generator_key: Filtrar por generador (opcional)
            
        Returns:
            Diccionario con resultados de optimización
        """
        if not self._optimizer:
            return {'error': 'Code optimizer no disponible'}
        
        # Obtener directorio objetivo
        if generator_key:
            from ..generator_config import GENERATOR_MAP
            if generator_key in GENERATOR_MAP:
                _, _, subdir = GENERATOR_MAP[generator_key]
                target_dir = project_dir / "app" / subdir
            else:
                return {'error': f'Generador {generator_key} no encontrado'}
        else:
            target_dir = project_dir / "app"
        
        if not target_dir.exists():
            return {'error': f'Directorio no encontrado: {target_dir}'}
        
        files_to_optimize = list(target_dir.rglob("*.py"))
        
        results = {}
        total_optimizations = 0
        
        for py_file in files_to_optimize:
            optimization = self._optimizer.optimize_file(py_file)
            if optimization.optimizations_applied:
                results[str(py_file.relative_to(project_dir))] = {
                    'optimizations': optimization.optimizations_applied,
                    'size_reduction': optimization.original_size - optimization.optimized_size
                }
                total_optimizations += len(optimization.optimizations_applied)
        
        return {
            'total_files_optimized': len(results),
            'total_optimizations': total_optimizations,
            'results': results
        }
    
    def generate_docker_files(
        self,
        project_dir: Path,
        framework: str = "pytorch",
        use_gpu: bool = False,
        expose_port: Optional[int] = 8000
    ) -> Dict[str, str]:
        """
        Generar archivos Docker.
        
        Args:
            project_dir: Directorio del proyecto
            framework: Framework a usar
            use_gpu: Si usar GPU
            expose_port: Puerto a exponer (opcional)
            
        Returns:
            Diccionario con archivos generados
        """
        if not self._docker_generator:
            return {'error': 'Docker generator no disponible'}
        
        return self._docker_generator.generate_all(
            project_dir,
            framework=framework,
            use_gpu=use_gpu,
            expose_port=expose_port
        )
    
    def generate_cicd_config(
        self,
        project_dir: Path,
        platform: str = "github",
        config: Optional[Any] = None
    ) -> Dict[str, str]:
        """
        Generar configuraciones CI/CD.
        
        Args:
            project_dir: Directorio del proyecto
            platform: Plataforma CI/CD ('github', 'gitlab', 'jenkins')
            config: Configuración personalizada (opcional)
            
        Returns:
            Diccionario con archivos generados
        """
        if not self._cicd_generator:
            return {'error': 'CI/CD generator no disponible'}
        
        return self._cicd_generator.generate_all(project_dir, platform=platform, config=config)
    
    def analyze_performance(
        self,
        project_dir: Path
    ) -> Dict[str, Any]:
        """
        Analizar performance del código generado.
        
        Args:
            project_dir: Directorio del proyecto
            
        Returns:
            Diccionario con métricas de performance
        """
        if not self._performance_analyzer:
            return {'error': 'Performance analyzer no disponible'}
        
        return self._performance_analyzer.analyze_project(project_dir)

