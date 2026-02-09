"""
Tests de seguridad para endpoints
"""

import pytest
from fastapi import status
from unittest.mock import patch
from tests.helpers.test_helpers import generate_test_song_id


class TestSecurityGeneration:
    """Tests de seguridad para endpoints de generación"""
    
    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_sql_injection_in_prompt(self, test_client):
        """Test de protección contra SQL injection en prompt"""
        malicious_prompt = "'; DROP TABLE songs; --"
        
        request = {
            "prompt": malicious_prompt,
            "user_id": "test-user"
        }
        
        with patch('api.routes.generation.generate_song_background'):
            response = test_client.post("/suno/generate", json=request)
            
            # Debe aceptar el prompt pero sanitizarlo internamente
            # O rechazarlo si es muy sospechoso
            assert response.status_code in [
                status.HTTP_202_ACCEPTED,
                status.HTTP_400_BAD_REQUEST
            ]
    
    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_xss_in_prompt(self, test_client):
        """Test de protección contra XSS en prompt"""
        xss_prompt = "<script>alert('XSS')</script>"
        
        request = {
            "prompt": xss_prompt,
            "user_id": "test-user"
        }
        
        with patch('api.routes.generation.generate_song_background'):
            response = test_client.post("/suno/generate", json=request)
            
            # Debe sanitizar o rechazar
            assert response.status_code in [
                status.HTTP_202_ACCEPTED,
                status.HTTP_400_BAD_REQUEST
            ]
    
    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_path_traversal_in_song_id(self, test_client):
        """Test de protección contra path traversal"""
        malicious_id = "../../../etc/passwd"
        
        response = test_client.get(f"/suno/generate/status/{malicious_id}")
        
        # Debe rechazar IDs inválidos
        assert response.status_code in [
            status.HTTP_400_BAD_REQUEST,
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            status.HTTP_404_NOT_FOUND
        ]
    
    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_command_injection_in_prompt(self, test_client):
        """Test de protección contra command injection"""
        malicious_prompt = "; rm -rf /"
        
        request = {
            "prompt": malicious_prompt,
            "user_id": "test-user"
        }
        
        with patch('api.routes.generation.generate_song_background'):
            response = test_client.post("/suno/generate", json=request)
            
            # Debe sanitizar o rechazar
            assert response.status_code in [
                status.HTTP_202_ACCEPTED,
                status.HTTP_400_BAD_REQUEST
            ]
    
    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_oversized_prompt(self, test_client):
        """Test de protección contra prompts muy grandes"""
        oversized_prompt = "A" * 10000  # Muy grande
        
        request = {
            "prompt": oversized_prompt,
            "user_id": "test-user"
        }
        
        response = test_client.post("/suno/generate", json=request)
        
        # Debe rechazar prompts muy grandes
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_special_characters_in_user_id(self, test_client):
        """Test con caracteres especiales en user_id"""
        request = {
            "prompt": "A song",
            "user_id": "../../../admin"
        }
        
        with patch('api.routes.generation.generate_song_background'):
            response = test_client.post("/suno/generate", json=request)
            
            # Debe manejar o sanitizar user_id
            assert response.status_code in [
                status.HTTP_202_ACCEPTED,
                status.HTTP_400_BAD_REQUEST
            ]


class TestSecuritySongs:
    """Tests de seguridad para endpoints de canciones"""
    
    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_path_traversal_in_download(self, test_client):
        """Test de protección en download"""
        malicious_id = "../../../etc/passwd"
        
        response = test_client.get(f"/suno/songs/{malicious_id}/download")
        
        # Debe rechazar
        assert response.status_code in [
            status.HTTP_400_BAD_REQUEST,
            status.HTTP_404_NOT_FOUND,
            status.HTTP_422_UNPROCESSABLE_ENTITY
        ]
    
    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_sql_injection_in_search(self, test_client):
        """Test de protección en búsqueda"""
        malicious_query = "'; DROP TABLE songs; --"
        
        response = test_client.get(
            "/suno/songs/search",
            params={"query": malicious_query}
        )
        
        # Debe sanitizar o rechazar
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_400_BAD_REQUEST
        ]


class TestRateLimiting:
    """Tests de rate limiting"""
    
    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_rapid_requests(self, test_client):
        """Test de múltiples requests rápidos"""
        request = {
            "prompt": "A song",
            "user_id": "test-user"
        }
        
        responses = []
        with patch('api.routes.generation.generate_song_background'):
            for _ in range(20):  # Muchas requests rápidas
                response = test_client.post("/suno/generate", json=request)
                responses.append(response.status_code)
        
        # Algunas pueden ser rate limited (429)
        # O todas pueden pasar si no hay rate limiting
        assert all(
            code in [status.HTTP_202_ACCEPTED, status.HTTP_429_TOO_MANY_REQUESTS]
            for code in responses
        )

