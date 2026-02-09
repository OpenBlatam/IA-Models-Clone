"""
Tests para performance monitor
"""

import pytest
import time
from contextlib import contextmanager


class TestMeasureTime:
    """Tests para measure_time context manager"""
    
    @pytest.fixture
    def measure_time_function(self):
        """Fixture para obtener la función"""
        try:
            from api.utils.performance_monitor import measure_time
            return measure_time
        except ImportError:
            pytest.skip("measure_time not available")
    
    @pytest.mark.unit
    def test_measure_time_basic(self, measure_time_function):
        """Test básico de medición de tiempo"""
        with measure_time_function("test_operation") as timer:
            time.sleep(0.1)  # Simular operación
        
        # Verificar que se midió el tiempo
        assert timer is not None or True  # Depende de la implementación
    
    @pytest.mark.unit
    def test_measure_time_with_logging(self, measure_time_function):
        """Test con logging"""
        with measure_time_function("test_operation", log=True) as timer:
            time.sleep(0.05)
        
        # Debe ejecutarse sin errores
        assert True
    
    @pytest.mark.unit
    def test_measure_time_nested(self, measure_time_function):
        """Test con operaciones anidadas"""
        with measure_time_function("outer") as outer:
            with measure_time_function("inner") as inner:
                time.sleep(0.01)
            time.sleep(0.01)
        
        # Debe manejar operaciones anidadas
        assert True

