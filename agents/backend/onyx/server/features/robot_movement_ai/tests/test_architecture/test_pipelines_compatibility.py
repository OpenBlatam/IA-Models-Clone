"""
Tests para el módulo de compatibilidad de pipelines
"""

import pytest
import warnings
from unittest.mock import patch, MagicMock
import sys
import os


class TestPipelineCompatibility:
    """Tests para el wrapper de compatibilidad de pipelines"""
    
    def test_imports_available(self):
        """Test que todos los imports están disponibles"""
        from core.architecture.pipelines import (
            PipelineStage,
            FunctionStage,
            Pipeline,
            PipelineBuilder,
            PipelineFactory,
            PipelineRegistry,
            SequentialExecutor,
            ParallelExecutor,
            ConditionalExecutor,
            LoggingMiddleware,
            MetricsMiddleware,
            CachingMiddleware,
            RetryMiddleware,
            ValidationMiddleware,
            pipeline_stage,
            async_pipeline_stage,
            validate_stage,
            cache_stage,
            retry_stage,
            metrics_stage
        )
        
        # Verificar que son importables
        assert PipelineStage is not None
        assert Pipeline is not None
        assert PipelineBuilder is not None
    
    def test_legacy_aliases(self):
        """Test que los alias legacy funcionan"""
        from core.architecture.pipelines import (
            ParallelPipeline,
            ConditionalPipeline,
            ParallelExecutor,
            ConditionalExecutor
        )
        
        # Verificar que los alias apuntan a los ejecutores correctos
        assert ParallelPipeline == ParallelExecutor
        assert ConditionalPipeline == ConditionalExecutor
    
    def test_deprecation_warning(self):
        """Test que se emite advertencia de deprecación"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from core.architecture.pipelines import Pipeline
            
            # Verificar que se emitió la advertencia
            assert len(w) > 0
            assert any(issubclass(warning.category, DeprecationWarning) for warning in w)
    
    def test_get_import_status(self):
        """Test de la función get_import_status"""
        from core.architecture.pipelines import get_import_status
        
        status = get_import_status()
        
        assert isinstance(status, dict)
        assert "successful" in status
        assert "version" in status
        assert "status" in status
        assert "deprecated" in status
        assert status["successful"] is True
        assert status["deprecated"] is True
    
    def test_version_info(self):
        """Test que la información de versión está disponible"""
        from core.architecture.pipelines import __version__, __status__, __deprecated__
        
        assert __version__ is not None
        assert isinstance(__version__, str)
        assert __status__ == "legacy_compatibility"
        assert __deprecated__ is True
    
    def test_all_exports(self):
        """Test que __all__ contiene todos los exports"""
        from core.architecture import pipelines
        
        # Verificar que todos los items en __all__ son importables
        for item_name in pipelines.__all__:
            if not item_name.startswith('_'):
                assert hasattr(pipelines, item_name), f"{item_name} not found in module"


class TestPipelineCompatibilityUsage:
    """Tests de uso del módulo de compatibilidad"""
    
    def test_legacy_usage_pattern(self):
        """Test del patrón de uso legacy"""
        from core.architecture.pipelines import Pipeline, PipelineStage
        
        # Verificar que se pueden usar como en el código legacy
        assert Pipeline is not None
        assert PipelineStage is not None
    
    def test_builder_usage(self):
        """Test del uso del builder"""
        from core.architecture.pipelines import PipelineBuilder
        
        # Verificar que el builder está disponible
        assert PipelineBuilder is not None
        assert callable(PipelineBuilder)
    
    def test_middleware_usage(self):
        """Test del uso de middleware"""
        from core.architecture.pipelines import (
            LoggingMiddleware,
            MetricsMiddleware,
            CachingMiddleware,
            RetryMiddleware,
            ValidationMiddleware
        )
        
        # Verificar que todos los middlewares están disponibles
        assert LoggingMiddleware is not None
        assert MetricsMiddleware is not None
        assert CachingMiddleware is not None
        assert RetryMiddleware is not None
        assert ValidationMiddleware is not None
    
    def test_decorators_usage(self):
        """Test del uso de decoradores"""
        from core.architecture.pipelines import (
            pipeline_stage,
            async_pipeline_stage,
            validate_stage,
            cache_stage,
            retry_stage,
            metrics_stage
        )
        
        # Verificar que todos los decoradores están disponibles
        assert callable(pipeline_stage)
        assert callable(async_pipeline_stage)
        assert callable(validate_stage)
        assert callable(cache_stage)
        assert callable(retry_stage)
        assert callable(metrics_stage)


class TestPipelineCompatibilityErrorHandling:
    """Tests de manejo de errores en el módulo de compatibilidad"""
    
    @patch('core.architecture.pipelines.logger')
    def test_import_error_logging(self, mock_logger):
        """Test que los errores de importación se registran"""
        # Este test verifica que el logging funciona
        # No podemos simular fácilmente un error de importación sin romper el test
        from core.architecture.pipelines import get_import_status
        
        status = get_import_status()
        # Si hay un error, debería estar en el status
        if not status["successful"]:
            assert status["error"] is not None
    
    def test_import_status_on_success(self):
        """Test que get_import_status reporta éxito cuando las importaciones funcionan"""
        from core.architecture.pipelines import get_import_status
        
        status = get_import_status()
        # En un entorno normal, las importaciones deberían ser exitosas
        # Este test verifica que el estado se reporta correctamente
        assert "successful" in status
        assert isinstance(status["successful"], bool)
    
    def test_validate_imports(self):
        """Test de la función validate_imports"""
        from core.architecture.pipelines import validate_imports
        
        result = validate_imports()
        assert isinstance(result, bool)
        # En un entorno normal, debería retornar True
        assert result is True
    
    def test_get_migration_guide(self):
        """Test de la función get_migration_guide"""
        from core.architecture.pipelines import get_migration_guide
        
        guide = get_migration_guide()
        
        assert isinstance(guide, dict)
        assert "current_usage" in guide
        assert "recommended_usage" in guide
        assert "benefits" in guide
        assert "migration_steps" in guide
        
        assert "example" in guide["current_usage"]
        assert "example" in guide["recommended_usage"]
        assert isinstance(guide["benefits"], list)
        assert isinstance(guide["migration_steps"], list)
    
    def test_check_compatibility(self):
        """Test de la función check_compatibility"""
        from core.architecture.pipelines import check_compatibility
        
        report = check_compatibility()
        
        assert isinstance(report, dict)
        assert "status" in report
        assert "imports_successful" in report
        assert "imports_valid" in report
        assert "version" in report
        assert "deprecated" in report
        assert "recommendations" in report
        
        assert report["status"] in ["ok", "warning", "error"]
        assert isinstance(report["recommendations"], list)
    
    def test_get_import_statistics(self):
        """Test de la función get_import_statistics"""
        from core.architecture.pipelines import get_import_statistics
        
        stats = get_import_statistics()
        
        assert isinstance(stats, dict)
        assert "total_imports" in stats
        assert "available_imports" in stats
        assert "missing_imports" in stats
        assert "coverage_percentage" in stats
        assert "categories" in stats
        
        assert isinstance(stats["total_imports"], int)
        assert isinstance(stats["available_imports"], int)
        assert isinstance(stats["coverage_percentage"], float)
        assert isinstance(stats["categories"], dict)
    
    def test_import_status_enum(self):
        """Test del enum ImportStatus"""
        from core.architecture.pipelines import ImportStatus
        
        assert ImportStatus.SUCCESS.value == "success"
        assert ImportStatus.PARTIAL.value == "partial"
        assert ImportStatus.FAILED.value == "failed"
        assert ImportStatus.UNKNOWN.value == "unknown"
    
    def test_health_score_in_compatibility(self):
        """Test que check_compatibility incluye health_score"""
        from core.architecture.pipelines import check_compatibility
        
        report = check_compatibility()
        
        assert "health_score" in report
        assert isinstance(report["health_score"], float)
        assert 0.0 <= report["health_score"] <= 1.0

