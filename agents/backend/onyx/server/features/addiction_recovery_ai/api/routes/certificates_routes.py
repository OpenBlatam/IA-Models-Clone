"""
Certificates routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body, Query
from fastapi.responses import JSONResponse
from typing import Optional

try:
    from services.certificate_service import CertificateService
except ImportError:
    from ...services.certificate_service import CertificateService

router = APIRouter()

certificates = CertificateService()


@router.post("/certificates/generate")
async def generate_certificate(
    user_id: str = Body(...),
    certificate_type: str = Body(...),
    title: str = Body(...),
    description: str = Body(...)
):
    """Genera un certificado"""
    try:
        certificate = certificates.generate_certificate(
            user_id, certificate_type, title, description
        )
        return JSONResponse(content=certificate)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generando certificado: {str(e)}")


@router.get("/certificates/{user_id}")
async def get_user_certificates(
    user_id: str,
    certificate_type: Optional[str] = Query(None)
):
    """Obtiene certificados del usuario"""
    try:
        certs = certificates.get_user_certificates(user_id, certificate_type)
        return JSONResponse(content={
            "user_id": user_id,
            "certificates": certs,
            "total": len(certs),
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo certificados: {str(e)}")



