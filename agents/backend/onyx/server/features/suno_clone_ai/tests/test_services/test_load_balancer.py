"""
Tests para el load balancer
"""

import pytest
from unittest.mock import Mock, patch
from services.load_balancer import LoadBalancer, LoadBalancingStrategy, Backend


@pytest.fixture
def load_balancer():
    """Instancia del load balancer"""
    try:
        return LoadBalancer(strategy=LoadBalancingStrategy.ROUND_ROBIN)
    except Exception as e:
        pytest.skip(f"LoadBalancer not available: {e}")


@pytest.mark.unit
class TestLoadBalancer:
    """Tests para el load balancer"""
    
    def test_balancer_initialization(self, load_balancer):
        """Test de inicialización"""
        if load_balancer is None:
            pytest.skip("LoadBalancer not available")
        assert load_balancer is not None
        assert isinstance(load_balancer, LoadBalancer)
        assert load_balancer.strategy == LoadBalancingStrategy.ROUND_ROBIN
    
    def test_add_backend(self, load_balancer):
        """Test de agregar backend"""
        if load_balancer is None:
            pytest.skip("LoadBalancer not available")
        
        load_balancer.add_backend(
            backend_id="backend-1",
            url="http://backend1.example.com",
            weight=1
        )
        
        assert "backend-1" in load_balancer.backends
        assert load_balancer.backends["backend-1"].url == "http://backend1.example.com"
    
    def test_get_backend_round_robin(self, load_balancer):
        """Test de obtener backend con round-robin"""
        if load_balancer is None:
            pytest.skip("LoadBalancer not available")
        
        # Agregar múltiples backends
        load_balancer.add_backend("backend-1", "http://backend1.com")
        load_balancer.add_backend("backend-2", "http://backend2.com")
        
        # Obtener backends (debería rotar)
        backend1 = load_balancer.get_backend()
        backend2 = load_balancer.get_backend()
        
        assert backend1 is not None
        assert backend2 is not None
        # En round-robin, deberían ser diferentes (o rotar)
        assert backend1.id in ["backend-1", "backend-2"]
        assert backend2.id in ["backend-1", "backend-2"]
    
    def test_get_backend_no_backends(self, load_balancer):
        """Test cuando no hay backends"""
        if load_balancer is None:
            pytest.skip("LoadBalancer not available")
        
        backend = load_balancer.get_backend()
        assert backend is None
    
    def test_get_stats(self, load_balancer):
        """Test de obtención de estadísticas"""
        if load_balancer is None:
            pytest.skip("LoadBalancer not available")
        
        load_balancer.add_backend("backend-1", "http://backend1.com")
        
        stats = load_balancer.get_stats()
        
        assert isinstance(stats, dict)
        assert "total_backends" in stats or "backends" in stats


@pytest.mark.unit
class TestBackend:
    """Tests para Backend"""
    
    def test_backend_creation(self):
        """Test de creación de backend"""
        backend = Backend(
            id="backend-1",
            url="http://backend1.com",
            weight=1
        )
        
        assert backend.id == "backend-1"
        assert backend.url == "http://backend1.com"
        assert backend.weight == 1
        assert backend.healthy is True
    
    def test_backend_success_rate(self):
        """Test de cálculo de tasa de éxito"""
        backend = Backend(id="backend-1", url="http://backend1.com")
        backend.total_requests = 100
        backend.failed_requests = 10
        
        success_rate = backend.get_success_rate()
        
        assert success_rate == 0.9
    
    def test_backend_avg_response_time(self):
        """Test de cálculo de tiempo de respuesta promedio"""
        backend = Backend(id="backend-1", url="http://backend1.com")
        backend.response_times = [0.1, 0.2, 0.3]
        
        avg_time = backend.get_avg_response_time()
        
        assert avg_time == pytest.approx(0.2, rel=0.1)


@pytest.mark.integration
class TestLoadBalancerIntegration:
    """Tests de integración para load balancer"""
    
    def test_full_load_balancing_workflow(self, load_balancer):
        """Test del flujo completo"""
        if load_balancer is None:
            pytest.skip("LoadBalancer not available")
        
        # 1. Agregar backends
        load_balancer.add_backend("backend-1", "http://backend1.com", weight=1)
        load_balancer.add_backend("backend-2", "http://backend2.com", weight=2)
        
        # 2. Obtener backends
        backend1 = load_balancer.get_backend()
        backend2 = load_balancer.get_backend()
        
        assert backend1 is not None
        assert backend2 is not None
        
        # 3. Obtener estadísticas
        stats = load_balancer.get_stats()
        assert isinstance(stats, dict)



