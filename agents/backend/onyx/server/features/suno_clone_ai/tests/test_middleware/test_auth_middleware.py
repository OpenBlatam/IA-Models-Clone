"""
Tests para el middleware de autenticación
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from fastapi import HTTPException, status, Header
from fastapi.testclient import TestClient
from fastapi import FastAPI, Depends

from middleware.auth_middleware import get_current_user, require_role


@pytest.fixture
def app_with_auth():
    """App FastAPI con rutas de autenticación"""
    app = FastAPI()
    
    @app.get("/public")
    async def public_endpoint():
        return {"message": "public"}
    
    @app.get("/protected")
    async def protected_endpoint(user: dict = Depends(get_current_user)):
        return {"user": user, "message": "protected"}
    
    @app.get("/admin")
    async def admin_endpoint(user: dict = Depends(require_role("admin"))):
        return {"user": user, "message": "admin"}
    
    return app


@pytest.mark.unit
class TestGetCurrentUser:
    """Tests para get_current_user"""
    
    def test_get_current_user_with_valid_token(self, app_with_auth):
        """Test con token válido"""
        with patch('middleware.auth_middleware.verify_token') as mock_verify:
            mock_verify.return_value = {"user_id": "user-123", "email": "test@example.com"}
            
            client = TestClient(app_with_auth)
            response = client.get(
                "/protected",
                headers={"authorization": "Bearer valid-token"}
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert "user" in data
            assert data["user"]["user_id"] == "user-123"
    
    def test_get_current_user_without_token(self, app_with_auth):
        """Test sin token"""
        client = TestClient(app_with_auth)
        response = client.get("/protected")
        
        # Puede ser 401 o 422 dependiendo de la implementación
        assert response.status_code in [
            status.HTTP_401_UNAUTHORIZED,
            status.HTTP_422_UNPROCESSABLE_ENTITY
        ]
    
    def test_get_current_user_invalid_token(self, app_with_auth):
        """Test con token inválido"""
        with patch('middleware.auth_middleware.verify_token') as mock_verify:
            mock_verify.side_effect = HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
            
            client = TestClient(app_with_auth)
            response = client.get(
                "/protected",
                headers={"authorization": "Bearer invalid-token"}
            )
            
            assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    def test_get_current_user_expired_token(self, app_with_auth):
        """Test con token expirado"""
        with patch('middleware.auth_middleware.verify_token') as mock_verify:
            mock_verify.side_effect = HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token expired"
            )
            
            client = TestClient(app_with_auth)
            response = client.get(
                "/protected",
                headers={"authorization": "Bearer expired-token"}
            )
            
            assert response.status_code == status.HTTP_401_UNAUTHORIZED


@pytest.mark.unit
class TestRequireRole:
    """Tests para require_role"""
    
    def test_require_role_admin_success(self, app_with_auth):
        """Test con rol admin válido"""
        with patch('middleware.auth_middleware.verify_token') as mock_verify:
            with patch('middleware.auth_middleware.check_role') as mock_check:
                mock_verify.return_value = {"user_id": "user-123", "role": "admin"}
                mock_check.return_value = True
                
                client = TestClient(app_with_auth)
                response = client.get(
                    "/admin",
                    headers={"authorization": "Bearer admin-token"}
                )
                
                assert response.status_code == status.HTTP_200_OK
                data = response.json()
                assert data["message"] == "admin"
    
    def test_require_role_insufficient_permissions(self, app_with_auth):
        """Test con permisos insuficientes"""
        with patch('middleware.auth_middleware.verify_token') as mock_verify:
            with patch('middleware.auth_middleware.check_role') as mock_check:
                mock_verify.return_value = {"user_id": "user-123", "role": "user"}
                mock_check.return_value = False
                
                client = TestClient(app_with_auth)
                response = client.get(
                    "/admin",
                    headers={"authorization": "Bearer user-token"}
                )
                
                assert response.status_code == status.HTTP_403_FORBIDDEN
    
    def test_require_role_different_roles(self, app_with_auth):
        """Test con diferentes roles"""
        roles = ["admin", "moderator", "user"]
        
        for role in roles:
            with patch('middleware.auth_middleware.verify_token') as mock_verify:
                with patch('middleware.auth_middleware.check_role') as mock_check:
                    mock_verify.return_value = {"user_id": "user-123", "role": role}
                    mock_check.return_value = (role == "admin")
                    
                    client = TestClient(app_with_auth)
                    response = client.get(
                        "/admin",
                        headers={"authorization": f"Bearer {role}-token"}
                    )
                    
                    if role == "admin":
                        assert response.status_code == status.HTTP_200_OK
                    else:
                        assert response.status_code == status.HTTP_403_FORBIDDEN


@pytest.mark.unit
class TestPublicEndpoints:
    """Tests para endpoints públicos"""
    
    def test_public_endpoint_no_auth_required(self, app_with_auth):
        """Test que endpoints públicos no requieren autenticación"""
        client = TestClient(app_with_auth)
        response = client.get("/public")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["message"] == "public"


@pytest.mark.integration
class TestAuthMiddlewareIntegration:
    """Tests de integración para middleware de autenticación"""
    
    def test_auth_workflow(self, app_with_auth):
        """Test del flujo completo de autenticación"""
        with patch('middleware.auth_middleware.verify_token') as mock_verify:
            with patch('middleware.auth_middleware.check_role') as mock_check:
                mock_verify.return_value = {"user_id": "user-123", "role": "admin"}
                mock_check.return_value = True
                
                client = TestClient(app_with_auth)
                
                # 1. Endpoint público (sin auth)
                public_response = client.get("/public")
                assert public_response.status_code == status.HTTP_200_OK
                
                # 2. Endpoint protegido (con auth)
                protected_response = client.get(
                    "/protected",
                    headers={"authorization": "Bearer valid-token"}
                )
                assert protected_response.status_code == status.HTTP_200_OK
                
                # 3. Endpoint admin (con rol)
                admin_response = client.get(
                    "/admin",
                    headers={"authorization": "Bearer admin-token"}
                )
                assert admin_response.status_code == status.HTTP_200_OK



