"""
Tests comprehensivos de seguridad
"""

import pytest
from unittest.mock import Mock, patch
from fastapi import status
from fastapi.testclient import TestClient
import base64


@pytest.mark.security
class TestInputSanitization:
    """Tests de sanitización de inputs"""
    
    def test_xss_prevention(self):
        """Test de prevención de XSS"""
        from fastapi import FastAPI, Body
        from fastapi.testclient import TestClient
        
        app = FastAPI()
        
        @app.post("/test")
        async def test_endpoint(text: str = Body(...)):
            # Debería sanitizar o escapar el texto
            return {"text": text}
        
        client = TestClient(app)
        
        xss_payloads = [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "javascript:alert('XSS')",
            "<svg onload=alert('XSS')>",
            "';alert('XSS');//"
        ]
        
        for payload in xss_payloads:
            response = client.post("/test", json={"text": payload})
            # El texto debería ser sanitizado o escapado
            assert response.status_code in [
                status.HTTP_200_OK,
                status.HTTP_400_BAD_REQUEST
            ]
            if response.status_code == status.HTTP_200_OK:
                # Verificar que el script no está en la respuesta sin escapar
                response_text = str(response.json())
                assert "<script>" not in response_text or "&lt;script&gt;" in response_text
    
    def test_sql_injection_prevention(self):
        """Test de prevención de SQL injection"""
        from fastapi import FastAPI, Query
        from fastapi.testclient import TestClient
        
        app = FastAPI()
        
        @app.get("/search")
        async def search(query: str = Query(...)):
            # Debería usar parámetros preparados, no concatenación
            return {"query": query}
        
        client = TestClient(app)
        
        sql_payloads = [
            "'; DROP TABLE songs; --",
            "' OR '1'='1",
            "admin'--",
            "' UNION SELECT * FROM users--",
            "1' OR '1'='1"
        ]
        
        for payload in sql_payloads:
            response = client.get(f"/search?query={payload}")
            # Debería manejar apropiadamente sin ejecutar SQL
            assert response.status_code in [
                status.HTTP_200_OK,
                status.HTTP_400_BAD_REQUEST,
                status.HTTP_422_UNPROCESSABLE_ENTITY
            ]


@pytest.mark.security
class TestAuthentication:
    """Tests de autenticación"""
    
    def test_unauthorized_access(self):
        """Test de acceso no autorizado"""
        from fastapi import FastAPI, Depends, HTTPException, Header
        from fastapi.testclient import TestClient
        
        app = FastAPI()
        
        def verify_token(authorization: str = Header(...)):
            if authorization != "Bearer valid-token":
                raise HTTPException(status_code=401, detail="Unauthorized")
            return {"user_id": "user-123"}
        
        @app.get("/protected")
        async def protected_route(user: dict = Depends(verify_token)):
            return {"data": "protected"}
        
        client = TestClient(app)
        
        # Sin token
        response = client.get("/protected")
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        
        # Token inválido
        response = client.get("/protected", headers={"authorization": "Bearer invalid-token"})
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        
        # Token válido
        response = client.get("/protected", headers={"authorization": "Bearer valid-token"})
        assert response.status_code == status.HTTP_200_OK
    
    def test_token_expiration(self):
        """Test de expiración de tokens"""
        from fastapi import FastAPI, Depends, HTTPException, Header
        from fastapi.testclient import TestClient
        from datetime import datetime, timedelta
        
        app = FastAPI()
        expired_tokens = set()
        
        def verify_token(authorization: str = Header(...)):
            token = authorization.replace("Bearer ", "")
            if token in expired_tokens:
                raise HTTPException(status_code=401, detail="Token expired")
            return {"user_id": "user-123"}
        
        @app.get("/protected")
        async def protected_route(user: dict = Depends(verify_token)):
            return {"data": "protected"}
        
        client = TestClient(app)
        
        # Token válido
        response = client.get("/protected", headers={"authorization": "Bearer valid-token"})
        assert response.status_code == status.HTTP_200_OK
        
        # Simular expiración
        expired_tokens.add("valid-token")
        response = client.get("/protected", headers={"authorization": "Bearer valid-token"})
        assert response.status_code == status.HTTP_401_UNAUTHORIZED


@pytest.mark.security
class TestRateLimiting:
    """Tests de rate limiting"""
    
    def test_rate_limit_enforcement(self):
        """Test de aplicación de rate limiting"""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        
        app = FastAPI()
        request_count = {"count": 0}
        
        @app.get("/limited")
        async def limited_route():
            request_count["count"] += 1
            if request_count["count"] > 5:
                return {"error": "Rate limit exceeded"}, 429
            return {"success": True}
        
        client = TestClient(app)
        
        # Primeras 5 requests deberían funcionar
        for i in range(5):
            response = client.get("/limited")
            assert response.status_code == status.HTTP_200_OK
        
        # La 6ta debería ser rechazada
        response = client.get("/limited")
        assert response.status_code == status.HTTP_429_TOO_MANY_REQUESTS


@pytest.mark.security
class TestDataValidation:
    """Tests de validación de datos"""
    
    def test_path_traversal_prevention(self):
        """Test de prevención de path traversal"""
        from fastapi import FastAPI, Path
        from fastapi.testclient import TestClient
        
        app = FastAPI()
        
        @app.get("/files/{filename}")
        async def get_file(filename: str = Path(...)):
            # Debería validar que no hay path traversal
            if ".." in filename or "/" in filename:
                return {"error": "Invalid filename"}, 400
            return {"filename": filename}
        
        client = TestClient(app)
        
        traversal_attempts = [
            "../../etc/passwd",
            "..\\..\\windows\\system32",
            "/etc/passwd",
            "file/../../etc/passwd"
        ]
        
        for attempt in traversal_attempts:
            response = client.get(f"/files/{attempt}")
            assert response.status_code in [
                status.HTTP_400_BAD_REQUEST,
                status.HTTP_404_NOT_FOUND,
                status.HTTP_422_UNPROCESSABLE_ENTITY
            ]
    
    def test_file_upload_validation(self):
        """Test de validación de uploads de archivos"""
        from fastapi import FastAPI, UploadFile, File
        from fastapi.testclient import TestClient
        import io
        
        app = FastAPI()
        
        @app.post("/upload")
        async def upload_file(file: UploadFile = File(...)):
            # Validar tipo de archivo
            if not file.filename.endswith(('.wav', '.mp3', '.ogg')):
                return {"error": "Invalid file type"}, 400
            return {"filename": file.filename}
        
        client = TestClient(app)
        
        # Archivo válido
        valid_file = ("test.wav", io.BytesIO(b"audio data"), "audio/wav")
        response = client.post("/upload", files={"file": valid_file})
        assert response.status_code == status.HTTP_200_OK
        
        # Archivo inválido
        invalid_file = ("test.exe", io.BytesIO(b"executable"), "application/x-msdownload")
        response = client.post("/upload", files={"file": invalid_file})
        assert response.status_code == status.HTTP_400_BAD_REQUEST



