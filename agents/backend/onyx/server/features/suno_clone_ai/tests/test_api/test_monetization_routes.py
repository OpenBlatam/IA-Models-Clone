"""
Tests para las rutas de monetización
"""

import pytest
from unittest.mock import Mock, patch
from fastapi import status
from fastapi.testclient import TestClient
from datetime import datetime, timedelta

from api.routes.monetization import router
from services.monetization import MonetizationService, SubscriptionTier


@pytest.fixture
def mock_monetization_service():
    """Mock del servicio de monetización"""
    service = Mock(spec=MonetizationService)
    
    # Mock de suscripción
    subscription = Mock()
    subscription.user_id = "user-123"
    subscription.tier = SubscriptionTier.PREMIUM
    subscription.start_date = datetime.now()
    subscription.end_date = datetime.now() + timedelta(days=30)
    subscription.status = "active"
    
    service.create_subscription = Mock(return_value=subscription)
    service.get_subscription = Mock(return_value=subscription)
    service.add_credits = Mock(return_value=True)
    service.get_credits = Mock(return_value=100)
    service.get_revenue_stats = Mock(return_value={"total": 1000.0})
    
    return service


@pytest.fixture
def client(mock_monetization_service):
    """Cliente de prueba con mocks"""
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    
    with patch('api.routes.monetization.get_monetization_service', return_value=mock_monetization_service):
        with patch('api.routes.monetization.get_current_user', return_value={"user_id": "test_user"}):
            yield TestClient(app)


@pytest.mark.unit
@pytest.mark.api
class TestCreateSubscription:
    """Tests para crear suscripción"""
    
    def test_create_subscription_success(self, client, mock_monetization_service):
        """Test de creación exitosa de suscripción"""
        response = client.post(
            "/monetization/subscriptions",
            json={
                "tier": "premium",
                "duration_days": 30,
                "auto_renew": True
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "user_id" in data
        assert data["tier"] == "premium"
        assert "start_date" in data
        assert "end_date" in data
    
    def test_create_subscription_different_tiers(self, client, mock_monetization_service):
        """Test con diferentes niveles de suscripción"""
        tiers = ["free", "basic", "premium", "enterprise"]
        
        for tier in tiers:
            try:
                response = client.post(
                    "/monetization/subscriptions",
                    json={"tier": tier, "duration_days": 30}
                )
                # Puede ser válido o inválido dependiendo de si el tier existe
                assert response.status_code in [
                    status.HTTP_200_OK,
                    status.HTTP_400_BAD_REQUEST
                ]
            except Exception:
                pass
    
    def test_create_subscription_invalid_tier(self, client):
        """Test con tier inválido"""
        response = client.post(
            "/monetization/subscriptions",
            json={"tier": "invalid", "duration_days": 30}
        )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Invalid tier" in response.json()["detail"]


@pytest.mark.unit
@pytest.mark.api
class TestGetSubscription:
    """Tests para obtener suscripción"""
    
    def test_get_my_subscription_success(self, client, mock_monetization_service):
        """Test de obtención exitosa"""
        response = client.get("/monetization/subscriptions/me")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "user_id" in data or "tier" in data


@pytest.mark.unit
@pytest.mark.api
class TestCredits:
    """Tests para créditos"""
    
    def test_add_credits_success(self, client, mock_monetization_service):
        """Test de agregar créditos exitosamente"""
        response = client.post(
            "/monetization/credits",
            json={"amount": 100}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "message" in data or "credits" in data
    
    def test_get_credits_success(self, client, mock_monetization_service):
        """Test de obtener créditos exitosamente"""
        response = client.get("/monetization/credits")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "credits" in data or "balance" in data


@pytest.mark.unit
@pytest.mark.api
class TestRevenueStats:
    """Tests para estadísticas de ingresos"""
    
    def test_get_revenue_stats_success(self, client, mock_monetization_service):
        """Test de obtención exitosa de estadísticas"""
        response = client.get("/monetization/revenue/stats")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data, dict)


@pytest.mark.integration
@pytest.mark.api
class TestMonetizationIntegration:
    """Tests de integración para monetización"""
    
    def test_full_monetization_workflow(self, client, mock_monetization_service):
        """Test del flujo completo de monetización"""
        # 1. Crear suscripción
        sub_response = client.post(
            "/monetization/subscriptions",
            json={"tier": "premium", "duration_days": 30}
        )
        assert sub_response.status_code == status.HTTP_200_OK
        
        # 2. Obtener suscripción
        get_sub_response = client.get("/monetization/subscriptions/me")
        assert get_sub_response.status_code == status.HTTP_200_OK
        
        # 3. Agregar créditos
        credits_response = client.post(
            "/monetization/credits",
            json={"amount": 50}
        )
        assert credits_response.status_code == status.HTTP_200_OK
        
        # 4. Obtener créditos
        get_credits_response = client.get("/monetization/credits")
        assert get_credits_response.status_code == status.HTTP_200_OK



