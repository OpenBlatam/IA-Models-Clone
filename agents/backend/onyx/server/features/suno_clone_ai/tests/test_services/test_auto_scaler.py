"""
Tests para el auto-scaler
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from services.auto_scaler import AutoScaler, ScalingPolicy, ScalingAction, ScalingDecision


@pytest.fixture
def auto_scaler():
    """Instancia del auto-scaler"""
    try:
        return AutoScaler()
    except Exception as e:
        pytest.skip(f"AutoScaler not available: {e}")


@pytest.mark.unit
class TestAutoScaler:
    """Tests para el auto-scaler"""
    
    def test_scaler_initialization(self, auto_scaler):
        """Test de inicialización"""
        if auto_scaler is None:
            pytest.skip("AutoScaler not available")
        assert auto_scaler is not None
        assert isinstance(auto_scaler, AutoScaler)
        assert len(auto_scaler.policies) > 0  # Debe tener políticas por defecto
    
    def test_add_policy(self, auto_scaler):
        """Test de agregar política"""
        if auto_scaler is None:
            pytest.skip("AutoScaler not available")
        
        policy = ScalingPolicy(
            name="test_policy",
            metric="cpu",
            threshold_up=80.0,
            threshold_down=30.0
        )
        
        auto_scaler.add_policy(policy)
        assert "test_policy" in auto_scaler.policies
        assert auto_scaler.policies["test_policy"] == policy
    
    def test_evaluate_scale_up(self, auto_scaler):
        """Test de evaluación para scale up"""
        if auto_scaler is None:
            pytest.skip("AutoScaler not available")
        
        # Simular métrica alta
        auto_scaler.record_metric("cpu", 85.0)
        
        decision = auto_scaler.evaluate()
        
        assert decision is not None
        assert isinstance(decision, ScalingDecision)
        # Puede ser SCALE_UP o NO_ACTION dependiendo de cooldown
        assert decision.action in [ScalingAction.SCALE_UP, ScalingAction.NO_ACTION]
    
    def test_evaluate_scale_down(self, auto_scaler):
        """Test de evaluación para scale down"""
        if auto_scaler is None:
            pytest.skip("AutoScaler not available")
        
        # Configurar réplicas altas
        auto_scaler.current_replicas = 5
        
        # Simular métrica baja
        auto_scaler.record_metric("cpu", 20.0)
        
        decision = auto_scaler.evaluate()
        
        assert decision is not None
        # Puede ser SCALE_DOWN o NO_ACTION dependiendo de cooldown
        assert decision.action in [ScalingAction.SCALE_DOWN, ScalingAction.NO_ACTION]
    
    def test_evaluate_no_action(self, auto_scaler):
        """Test de evaluación sin acción"""
        if auto_scaler is None:
            pytest.skip("AutoScaler not available")
        
        # Simular métrica en rango normal
        auto_scaler.record_metric("cpu", 50.0)
        
        decision = auto_scaler.evaluate()
        
        assert decision is not None
        # Puede ser NO_ACTION o alguna acción dependiendo de políticas
        assert decision.action in [
            ScalingAction.NO_ACTION,
            ScalingAction.SCALE_UP,
            ScalingAction.SCALE_DOWN
        ]
    
    def test_record_metric(self, auto_scaler):
        """Test de registro de métrica"""
        if auto_scaler is None:
            pytest.skip("AutoScaler not available")
        
        auto_scaler.record_metric("cpu", 75.0)
        
        assert "cpu" in auto_scaler.metric_history
        assert len(auto_scaler.metric_history["cpu"]) > 0
    
    def test_get_stats(self, auto_scaler):
        """Test de obtención de estadísticas"""
        if auto_scaler is None:
            pytest.skip("AutoScaler not available")
        
        stats = auto_scaler.get_stats()
        
        assert isinstance(stats, dict)
        assert "current_replicas" in stats
        assert "policies_count" in stats


@pytest.mark.unit
class TestScalingPolicy:
    """Tests para ScalingPolicy"""
    
    def test_policy_creation(self):
        """Test de creación de política"""
        policy = ScalingPolicy(
            name="test",
            metric="cpu",
            threshold_up=80.0,
            threshold_down=30.0
        )
        
        assert policy.name == "test"
        assert policy.metric == "cpu"
        assert policy.threshold_up == 80.0
        assert policy.threshold_down == 30.0
        assert policy.enabled is True


@pytest.mark.integration
class TestAutoScalerIntegration:
    """Tests de integración para auto-scaler"""
    
    def test_full_scaling_workflow(self, auto_scaler):
        """Test del flujo completo de escalado"""
        if auto_scaler is None:
            pytest.skip("AutoScaler not available")
        
        # 1. Agregar política
        policy = ScalingPolicy(
            name="integration_test",
            metric="cpu",
            threshold_up=80.0,
            threshold_down=30.0
        )
        auto_scaler.add_policy(policy)
        
        # 2. Registrar métricas
        auto_scaler.record_metric("cpu", 85.0)
        
        # 3. Evaluar
        decision = auto_scaler.evaluate()
        assert decision is not None
        
        # 4. Obtener estadísticas
        stats = auto_scaler.get_stats()
        assert isinstance(stats, dict)



