"""
MCP Request Signing - Firma y verificación de requests
========================================================
"""

import logging
import hmac
import hashlib
import base64
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

from .exceptions import MCPError

logger = logging.getLogger(__name__)


class RequestSigner:
    """
    Firmador de requests
    
    Permite firmar y verificar requests para autenticación.
    """
    
    def __init__(self, secret_key: str, algorithm: str = "sha256"):
        """
        Args:
            secret_key: Clave secreta para firmar
            algorithm: Algoritmo de hash (sha256, sha512)
        """
        self.secret_key = secret_key.encode() if isinstance(secret_key, str) else secret_key
        self.algorithm = algorithm
        self._hash_func = getattr(hashlib, algorithm)
    
    def sign_request(
        self,
        method: str,
        path: str,
        body: Optional[bytes] = None,
        timestamp: Optional[datetime] = None,
        nonce: Optional[str] = None,
    ) -> str:
        """
        Firma un request
        
        Args:
            method: Método HTTP
            path: Path del request
            body: Body del request (opcional)
            timestamp: Timestamp (opcional)
            nonce: Nonce (opcional)
            
        Returns:
            Firma del request
        """
        import uuid
        
        timestamp = timestamp or datetime.utcnow()
        nonce = nonce or str(uuid.uuid4())
        
        # Construir string a firmar
        body_hash = ""
        if body:
            body_hash = base64.b64encode(
                self._hash_func(body).digest()
            ).decode()
        
        string_to_sign = f"{method}\n{path}\n{timestamp.isoformat()}\n{nonce}\n{body_hash}"
        
        # Firmar
        signature = hmac.new(
            self.secret_key,
            string_to_sign.encode(),
            self._hash_func
        ).digest()
        
        signature_b64 = base64.b64encode(signature).decode()
        
        return f"{self.algorithm}:{signature_b64}:{timestamp.isoformat()}:{nonce}"
    
    def verify_request(
        self,
        method: str,
        path: str,
        signature: str,
        body: Optional[bytes] = None,
        max_age_seconds: int = 300,
    ) -> bool:
        """
        Verifica la firma de un request
        
        Args:
            method: Método HTTP
            path: Path del request
            signature: Firma recibida
            body: Body del request (opcional)
            max_age_seconds: Edad máxima permitida en segundos
            
        Returns:
            True si la firma es válida
        """
        try:
            parts = signature.split(":")
            if len(parts) != 4:
                return False
            
            algo, sig_b64, timestamp_str, nonce = parts
            
            if algo != self.algorithm:
                return False
            
            # Verificar timestamp
            timestamp = datetime.fromisoformat(timestamp_str)
            age = (datetime.utcnow() - timestamp).total_seconds()
            
            if age > max_age_seconds:
                logger.warning(f"Request signature expired: {age}s > {max_age_seconds}s")
                return False
            
            # Reconstruir firma esperada
            expected_signature = self.sign_request(
                method=method,
                path=path,
                body=body,
                timestamp=timestamp,
                nonce=nonce,
            )
            
            # Comparar
            return hmac.compare_digest(signature, expected_signature)
            
        except Exception as e:
            logger.error(f"Error verifying signature: {e}")
            return False


class RequestVerifier:
    """
    Verificador de requests
    
    Middleware para verificar firmas de requests.
    """
    
    def __init__(self, signer: RequestSigner):
        """
        Args:
            signer: RequestSigner
        """
        self.signer = signer
    
    async def verify(self, request: Any) -> bool:
        """
        Verifica un request
        
        Args:
            request: Request de FastAPI
            
        Returns:
            True si es válido
        """
        signature = request.headers.get("X-Request-Signature")
        
        if not signature:
            return False
        
        body = await request.body()
        
        return self.signer.verify_request(
            method=request.method,
            path=request.url.path,
            signature=signature,
            body=body,
        )

