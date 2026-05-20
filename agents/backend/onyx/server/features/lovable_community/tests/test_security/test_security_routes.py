"""
Tests de seguridad para endpoints de Lovable Community
"""

import pytest
from fastapi import status
from tests.helpers.security_helpers import (
    SecurityTestHelper,
    generate_sql_injection_payloads,
    generate_xss_payloads,
    generate_path_traversal_payloads,
    generate_large_inputs
)


class TestSQLInjectionProtection:
    """Tests de protección contra SQL Injection"""
    
    @pytest.mark.security
    def test_sql_injection_in_chat_id(self, test_client, chat_service):
        """Test que los IDs de chat están protegidos contra SQL injection"""
        payloads = SecurityTestHelper.generate_sql_injection_payloads()
        
        for payload in payloads:
            response = test_client.get(f"/community/chats/{payload}")
            
            # Debe retornar 404 o 400, no 500 (error de SQL)
            assert response.status_code in [
                status.HTTP_404_NOT_FOUND,
                status.HTTP_400_BAD_REQUEST
            ], f"SQL injection payload '{payload}' should be rejected"
    
    @pytest.mark.security
    def test_sql_injection_in_search_query(self, test_client):
        """Test que las queries de búsqueda están protegidas"""
        payloads = SecurityTestHelper.generate_sql_injection_payloads()
        
        for payload in payloads:
            response = test_client.get(
                "/community/search",
                params={"query": payload}
            )
            
            # Debe manejar el payload de forma segura
            assert response.status_code != status.HTTP_500_INTERNAL_SERVER_ERROR, \
                f"SQL injection in query '{payload}' should not cause server error"


class TestXSSProtection:
    """Tests de protección contra XSS"""
    
    @pytest.mark.security
    def test_xss_in_title(self, test_client, sample_user_id):
        """Test que los títulos están sanitizados contra XSS"""
        payloads = SecurityTestHelper.generate_xss_payloads()
        
        for payload in payloads:
            response = test_client.post(
                "/community/publish",
                json={
                    "title": payload,
                    "chat_content": "{}"
                },
                headers={"X-User-ID": sample_user_id}
            )
            
            # Debe validar y rechazar o sanitizar
            assert response.status_code in [
                status.HTTP_201_CREATED,
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                status.HTTP_400_BAD_REQUEST
            ], f"XSS payload in title '{payload}' should be handled safely"
            
            if response.status_code == status.HTTP_201_CREATED:
                # Si se acepta, debe estar sanitizado
                data = response.json()
                assert "<script>" not in data.get("title", "").lower()
    
    @pytest.mark.security
    def test_xss_in_description(self, test_client, sample_user_id):
        """Test que las descripciones están sanitizadas"""
        payloads = SecurityTestHelper.generate_xss_payloads()
        
        for payload in payloads:
            response = test_client.post(
                "/community/publish",
                json={
                    "title": "Test",
                    "description": payload,
                    "chat_content": "{}"
                },
                headers={"X-User-ID": sample_user_id}
            )
            
            assert response.status_code != status.HTTP_500_INTERNAL_SERVER_ERROR


class TestPathTraversalProtection:
    """Tests de protección contra Path Traversal"""
    
    @pytest.mark.security
    def test_path_traversal_in_chat_id(self, test_client):
        """Test que los IDs no permiten path traversal"""
        payloads = SecurityTestHelper.generate_path_traversal_payloads()
        
        for payload in payloads:
            response = test_client.get(f"/community/chats/{payload}")
            
            # Debe rechazar o no encontrar
            assert response.status_code in [
                status.HTTP_404_NOT_FOUND,
                status.HTTP_400_BAD_REQUEST
            ], f"Path traversal payload '{payload}' should be rejected"


class TestInputSizeLimits:
    """Tests de límites de tamaño de input"""
    
    @pytest.mark.security
    def test_large_title(self, test_client, sample_user_id):
        """Test que los títulos grandes son rechazados"""
        large_title = "A" * 201  # Más del límite de 200
        
        response = test_client.post(
            "/community/publish",
            json={
                "title": large_title,
                "chat_content": "{}"
            },
            headers={"X-User-ID": sample_user_id}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    @pytest.mark.security
    def test_large_content(self, test_client, sample_user_id):
        """Test que el contenido grande es rechazado"""
        large_content = "A" * 50001  # Más del límite de 50000
        
        response = test_client.post(
            "/community/publish",
            json={
                "title": "Test",
                "chat_content": large_content
            },
            headers={"X-User-ID": sample_user_id}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    @pytest.mark.security
    def test_large_description(self, test_client, sample_user_id):
        """Test que las descripciones grandes son rechazadas"""
        large_description = "A" * 1001  # Más del límite de 1000
        
        response = test_client.post(
            "/community/publish",
            json={
                "title": "Test",
                "description": large_description,
                "chat_content": "{}"
            },
            headers={"X-User-ID": sample_user_id}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestRateLimiting:
    """Tests de rate limiting (si está implementado)"""
    
    @pytest.mark.security
    @pytest.mark.slow
    def test_rapid_requests(self, test_client, sample_user_id):
        """Test de múltiples requests rápidos"""
        # Hacer muchos requests rápidos
        responses = []
        for i in range(20):
            response = test_client.get("/community/chats")
            responses.append(response.status_code)
        
        # Todos deben ser válidos (o algunos rechazados si hay rate limiting)
        # No debe haber errores 500
        assert all(
            code != status.HTTP_500_INTERNAL_SERVER_ERROR
            for code in responses
        )


class TestAuthorization:
    """Tests de autorización"""
    
    @pytest.mark.security
    def test_update_chat_unauthorized(self, test_client, chat_service, sample_user_id):
        """Test que no se puede actualizar chat de otro usuario"""
        # Crear chat
        chat = chat_service.publish_chat(
            user_id=sample_user_id,
            title="Owner's Chat",
            chat_content="{}"
        )
        
        # Intentar actualizar como otro usuario
        response = test_client.put(
            f"/community/chats/{chat.id}",
            json={"title": "Hacked Title"},
            headers={"X-User-ID": "other-user"}
        )
        
        # Debe rechazar
        assert response.status_code in [
            status.HTTP_403_FORBIDDEN,
            status.HTTP_401_UNAUTHORIZED,
            status.HTTP_400_BAD_REQUEST
        ]
    
    @pytest.mark.security
    def test_delete_chat_unauthorized(self, test_client, chat_service, sample_user_id):
        """Test que no se puede eliminar chat de otro usuario"""
        chat = chat_service.publish_chat(
            user_id=sample_user_id,
            title="Owner's Chat",
            chat_content="{}"
        )
        
        response = test_client.delete(
            f"/community/chats/{chat.id}",
            headers={"X-User-ID": "other-user"}
        )
        
        assert response.status_code in [
            status.HTTP_403_FORBIDDEN,
            status.HTTP_401_UNAUTHORIZED,
            status.HTTP_400_BAD_REQUEST
        ]


class TestInputValidation:
    """Tests de validación de inputs"""
    
    @pytest.mark.security
    def test_empty_strings_rejected(self, test_client, sample_user_id):
        """Test que strings vacíos son rechazados"""
        response = test_client.post(
            "/community/publish",
            json={
                "title": "",
                "chat_content": "{}"
            },
            headers={"X-User-ID": sample_user_id}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    @pytest.mark.security
    def test_invalid_vote_type(self, test_client, chat_service, sample_user_id):
        """Test que tipos de voto inválidos son rechazados"""
        chat = chat_service.publish_chat(
            user_id=sample_user_id,
            title="Test",
            chat_content="{}"
        )
        
        response = test_client.post(
            f"/community/chats/{chat.id}/vote",
            json={
                "chat_id": chat.id,
                "vote_type": "invalid_type"
            },
            headers={"X-User-ID": sample_user_id}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    @pytest.mark.security
    def test_too_many_tags(self, test_client, sample_user_id):
        """Test que demasiados tags son rechazados"""
        response = test_client.post(
            "/community/publish",
            json={
                "title": "Test",
                "chat_content": "{}",
                "tags": [f"tag{i}" for i in range(11)]  # Más del límite de 10
            },
            headers={"X-User-ID": sample_user_id}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

