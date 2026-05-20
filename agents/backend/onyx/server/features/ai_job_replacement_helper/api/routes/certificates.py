"""
Certificates endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.certificates import CertificatesService

router = APIRouter()
certificates_service = CertificatesService()


@router.get("/{user_id}")
async def get_user_certificates(user_id: str) -> Dict[str, Any]:
    """Obtener certificados del usuario"""
    try:
        certificates = certificates_service.get_user_certificates(user_id)
        return {
            "certificates": [
                {
                    "id": cert.id,
                    "type": cert.certificate_type,
                    "title": cert.title,
                    "description": cert.description,
                    "issued_date": cert.issued_date.isoformat(),
                    "verification_code": cert.verification_code,
                }
                for cert in certificates
            ],
            "total": len(certificates),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/verify/{verification_code}")
async def verify_certificate(verification_code: str) -> Dict[str, Any]:
    """Verificar certificado"""
    try:
        certificate = certificates_service.verify_certificate(verification_code)
        if not certificate:
            raise HTTPException(status_code=404, detail="Certificate not found")
        
        return {
            "valid": True,
            "certificate": {
                "id": certificate.id,
                "title": certificate.title,
                "issued_date": certificate.issued_date.isoformat(),
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




