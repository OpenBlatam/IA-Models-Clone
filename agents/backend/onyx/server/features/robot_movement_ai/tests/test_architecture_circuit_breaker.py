"""
Tests para Circuit Breaker - Arquitectura Mejorada
==================================================
"""

import pytest
import asyncio
from unittest.mock import AsyncMock

from core.architecture.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerManager,
    CircuitState,
    InfrastructureError,
    ErrorCode
)


class TestCircuitBreakerConfig:
    """Tests para CircuitBreakerConfig."""
    
    def test_create_valid_config(self):
        """Test crear configuración válida."""
        config = CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout=60.0
        )
        assert config.failure_threshold == 5
        assert config.recovery_timeout == 60.0
    
    def test_config_validation(self):
        """Test validación de configuración."""
        # Threshold negativo
        with pytest.raises(ValueError):
            CircuitBreakerConfig(failure_threshold=-1)
        
        # Timeout negativo
        with pytest.raises(ValueError):
            CircuitBreakerConfig(recovery_timeout=-1)


class TestCircuitBreaker:
    """Tests para CircuitBreaker."""
    
    @pytest.fixture
    def config(self):
        """Configuración para tests."""
        return CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=1.0,  # Corto para tests
            success_threshold=1
        )
    
    @pytest.fixture
    def circuit(self, config):
        """Crear circuit breaker."""
        return CircuitBreaker(name="test_circuit", config=config)
    
    @pytest.mark.asyncio
    async def test_call_success(self, circuit):
        """Test llamada exitosa."""
        async def success_func():
            return "success"
        
        result = await circuit.call(success_func)
        assert result == "success"
        assert circuit.state == CircuitState.CLOSED
    
    @pytest.mark.asyncio
    async def test_call_failure(self, circuit):
        """Test llamada con fallo."""
        async def failing_func():
            raise Exception("Error")
        
        with pytest.raises(Exception):
            await circuit.call(failing_func)
        
        assert circuit.metrics.failed_requests == 1
    
    @pytest.mark.asyncio
    async def test_circuit_opens_after_threshold(self, circuit):
        """Test que el circuit se abre después del threshold."""
        async def failing_func():
            raise Exception("Error")
        
        # Primer fallo
        try:
            await circuit.call(failing_func)
        except:
            pass
        
        # Segundo fallo - debería abrir
        try:
            await circuit.call(failing_func)
        except:
            pass
        
        assert circuit.state == CircuitState.OPEN
    
    @pytest.mark.asyncio
    async def test_circuit_rejects_when_open(self, circuit):
        """Test que el circuit rechaza llamadas cuando está abierto."""
        async def failing_func():
            raise Exception("Error")
        
        # Abrir circuit
        try:
            await circuit.call(failing_func)
            await circuit.call(failing_func)
        except:
            pass
        
        assert circuit.state == CircuitState.OPEN
        
        # Intentar llamar debería rechazar
        with pytest.raises(InfrastructureError) as exc_info:
            await circuit.call(failing_func)
        
        assert exc_info.value.code == ErrorCode.INFRASTRUCTURE_SERVICE_UNAVAILABLE
        assert circuit.metrics.rejected_requests > 0
    
    @pytest.mark.asyncio
    async def test_circuit_half_open_recovery(self, circuit):
        """Test recuperación desde half-open."""
        async def failing_func():
            raise Exception("Error")
        
        async def success_func():
            return "success"
        
        # Abrir circuit
        try:
            await circuit.call(failing_func)
            await circuit.call(failing_func)
        except:
            pass
        
        assert circuit.state == CircuitState.OPEN
        
        # Esperar recovery timeout
        await asyncio.sleep(1.1)
        
        # Intentar llamada exitosa - debería ir a half-open
        result = await circuit.call(success_func)
        assert result == "success"
        
        # Si hay más éxitos, debería cerrar
        await circuit.call(success_func)
        assert circuit.state == CircuitState.CLOSED
    
    @pytest.mark.asyncio
    async def test_circuit_timeout(self, circuit):
        """Test timeout de llamada."""
        config = CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout=60.0,
            call_timeout=0.1  # Timeout corto
        )
        circuit = CircuitBreaker(name="test", config=config)
        
        async def slow_func():
            await asyncio.sleep(1.0)
            return "done"
        
        with pytest.raises(InfrastructureError) as exc_info:
            await circuit.call(slow_func)
        
        assert exc_info.value.code == ErrorCode.INFRASTRUCTURE_TIMEOUT
    
    @pytest.mark.asyncio
    async def test_circuit_metrics(self, circuit):
        """Test métricas del circuit."""
        async def success_func():
            return "success"
        
        async def failing_func():
            raise Exception("Error")
        
        # Llamadas exitosas
        await circuit.call(success_func)
        await circuit.call(success_func)
        
        # Llamadas fallidas
        try:
            await circuit.call(failing_func)
        except:
            pass
        
        metrics = circuit.metrics
        assert metrics.total_requests == 3
        assert metrics.successful_requests == 2
        assert metrics.failed_requests == 1
        assert metrics.success_rate > 0
    
    @pytest.mark.asyncio
    async def test_circuit_domain_events(self, circuit):
        """Test domain events."""
        async def failing_func():
            raise Exception("Error")
        
        # Abrir circuit
        try:
            await circuit.call(failing_func)
            await circuit.call(failing_func)
        except:
            pass
        
        events = circuit.get_domain_events()
        assert len(events) > 0
        assert any(e.circuit_name == "test_circuit" for e in events)
    
    def test_circuit_reset(self, circuit):
        """Test reset manual."""
        # Simular estado abierto
        circuit._state = CircuitState.OPEN
        
        circuit.reset()
        assert circuit.state == CircuitState.CLOSED


class TestCircuitBreakerManager:
    """Tests para CircuitBreakerManager."""
    
    @pytest.fixture
    async def manager(self):
        """Crear manager."""
        return CircuitBreakerManager()
    
    @pytest.mark.asyncio
    async def test_get_or_create(self, manager):
        """Test obtener o crear circuit breaker."""
        config = CircuitBreakerConfig(failure_threshold=5)
        
        circuit1 = await manager.get_or_create("test", config)
        circuit2 = await manager.get_or_create("test", config)
        
        # Debería ser la misma instancia
        assert circuit1 is circuit2
    
    @pytest.mark.asyncio
    async def test_get_nonexistent(self, manager):
        """Test obtener circuit breaker que no existe."""
        circuit = await manager.get("nonexistent")
        assert circuit is None
    
    @pytest.mark.asyncio
    async def test_list_all(self, manager):
        """Test listar todos los circuit breakers."""
        await manager.get_or_create("circuit1")
        await manager.get_or_create("circuit2")
        
        all_circuits = await manager.list_all()
        assert len(all_circuits) == 2
    
    @pytest.mark.asyncio
    async def test_reset_circuit(self, manager):
        """Test resetear circuit breaker."""
        circuit = await manager.get_or_create("test")
        circuit._state = CircuitState.OPEN
        
        await manager.reset("test")
        assert circuit.state == CircuitState.CLOSED
    
    @pytest.mark.asyncio
    async def test_get_metrics(self, manager):
        """Test obtener métricas."""
        circuit = await manager.get_or_create("test")
        
        metrics = await manager.get_metrics("test")
        assert metrics is not None
        assert metrics == circuit.metrics




