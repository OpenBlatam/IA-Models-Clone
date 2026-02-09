"""
Tests de integración para el sistema completo de pipelines
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch


class TestPipelineSystem:
    """Tests para PipelineSystem"""
    
    def test_system_initialization(self):
        """Test de inicialización del sistema"""
        from core.architecture.pipelines_integration import PipelineSystem
        
        system = PipelineSystem(
            enable_cache=True,
            enable_metrics=True,
            enable_monitoring=False
        )
        
        assert system.enable_cache is True
        assert system.enable_metrics is True
        assert system.enable_monitoring is False
    
    def test_check_health(self):
        """Test de check de salud"""
        from core.architecture.pipelines_integration import PipelineSystem
        
        system = PipelineSystem(
            enable_cache=False,
            enable_metrics=False,
            enable_monitoring=False
        )
        
        health = system.check_health()
        
        assert "compatibility" in health
        assert "statistics" in health
        assert "cache_enabled" in health
    
    def test_get_comprehensive_report(self):
        """Test de reporte completo"""
        from core.architecture.pipelines_integration import PipelineSystem
        
        system = PipelineSystem(
            enable_cache=False,
            enable_metrics=False,
            enable_monitoring=False
        )
        
        report = system.get_comprehensive_report()
        
        assert "health" in report
        assert "components" in report
        assert "cache" in report["components"]
        assert "metrics" in report["components"]
        assert "monitoring" in report["components"]
    
    def test_get_status(self):
        """Test de estado del sistema"""
        from core.architecture.pipelines_integration import PipelineSystem
        
        system = PipelineSystem(
            enable_cache=False,
            enable_metrics=False,
            enable_monitoring=False
        )
        
        status = system.get_status()
        
        assert "cache" in status
        assert "metrics" in status
        assert "monitoring" in status
    
    def test_quick_system_check(self):
        """Test de check rápido"""
        from core.architecture.pipelines_integration import quick_system_check
        
        result = quick_system_check()
        
        assert "status" in result
        assert "health_score" in result
        assert "coverage" in result


class TestCacheIntegration:
    """Tests de integración con cache"""
    
    def test_cache_in_system(self):
        """Test que el cache funciona en el sistema"""
        from core.architecture.pipelines_integration import PipelineSystem
        
        system = PipelineSystem(
            enable_cache=True,
            enable_metrics=False,
            enable_monitoring=False
        )
        
        # Primera llamada (miss)
        health1 = system.check_health(use_cache=True)
        
        # Segunda llamada (hit)
        health2 = system.check_health(use_cache=True)
        
        # Deben ser iguales (del cache)
        assert health1 == health2
        
        # Verificar estadísticas de cache
        if system._cache:
            stats = system._cache.get_stats()
            assert stats["hits"] > 0 or stats["misses"] > 0


class TestMetricsIntegration:
    """Tests de integración con métricas"""
    
    def test_metrics_in_system(self):
        """Test que las métricas funcionan en el sistema"""
        from core.architecture.pipelines_integration import PipelineSystem
        
        system = PipelineSystem(
            enable_cache=False,
            enable_metrics=True,
            enable_monitoring=False
        )
        
        # Realizar varias operaciones
        for _ in range(3):
            system.check_health()
        
        # Verificar métricas
        if system._metrics:
            stats = system._metrics.get_stats()
            assert "health_check_total" in stats.get("counters", {})
            assert stats["counters"]["health_check_total"] >= 3


class TestMonitoringIntegration:
    """Tests de integración con monitoreo"""
    
    def test_monitoring_in_system(self):
        """Test que el monitoreo funciona en el sistema"""
        from core.architecture.pipelines_integration import PipelineSystem
        
        system = PipelineSystem(
            enable_cache=False,
            enable_metrics=False,
            enable_monitoring=True
        )
        
        # Iniciar monitoreo
        system.start_monitoring()
        
        # Verificar que está activo
        status = system.get_status()
        assert status["monitoring"]["enabled"] is True
        
        # Detener monitoreo
        system.stop_monitoring()
        
        # Verificar que se detuvo
        status = system.get_status()
        assert status["monitoring"]["active"] is False


class TestFullIntegration:
    """Tests de integración completa"""
    
    def test_full_system(self):
        """Test del sistema completo con todos los componentes"""
        from core.architecture.pipelines_integration import PipelineSystem
        
        system = PipelineSystem(
            enable_cache=True,
            enable_metrics=True,
            enable_monitoring=False  # No iniciar monitoreo en tests
        )
        
        # Realizar operaciones
        health = system.check_health()
        report = system.get_comprehensive_report()
        status = system.get_status()
        
        # Verificar que todo funciona
        assert health is not None
        assert report is not None
        assert status is not None
        
        # Verificar componentes
        assert system._cache is not None
        assert system._metrics is not None
    
    def test_export_report(self, tmp_path):
        """Test de exportación de reporte"""
        from core.architecture.pipelines_integration import PipelineSystem
        
        system = PipelineSystem(
            enable_cache=False,
            enable_metrics=False,
            enable_monitoring=False
        )
        
        output_file = tmp_path / "report.json"
        system.export_report(output_file, format="json")
        
        assert output_file.exists()
        content = output_file.read_text()
        assert len(content) > 0

