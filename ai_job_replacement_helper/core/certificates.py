"""
Certificates Service - Sistema de certificados
==============================================

Sistema para generar certificados de logros y completación.
"""

import logging
from datetime import datetime
from typing import Dict, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class CertificateType(str):
    """Tipos de certificados"""
    COURSE_COMPLETION = "course_completion"
    SKILL_MASTERY = "skill_mastery"
    ROADMAP_COMPLETION = "roadmap_completion"
    ACHIEVEMENT = "achievement"


@dataclass
class Certificate:
    """Certificado"""
    id: str
    user_id: str
    certificate_type: str
    title: str
    description: str
    issued_date: datetime = field(default_factory=datetime.now)
    certificate_url: Optional[str] = None
    verification_code: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class CertificatesService:
    """Servicio de certificados"""
    
    def __init__(self):
        """Inicializar servicio"""
        self.certificates: Dict[str, List[Certificate]] = {}  # user_id -> [certificates]
        logger.info("CertificatesService initialized")
    
    def generate_certificate(
        self,
        user_id: str,
        certificate_type: str,
        title: str,
        description: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Certificate:
        """Generar certificado"""
        import secrets
        
        certificate = Certificate(
            id=f"cert_{user_id}_{int(datetime.now().timestamp())}",
            user_id=user_id,
            certificate_type=certificate_type,
            title=title,
            description=description,
            verification_code=secrets.token_hex(8).upper(),
            metadata=metadata or {},
            certificate_url=f"/certificates/{user_id}/{certificate.id}",  # URL simulado
        )
        
        if user_id not in self.certificates:
            self.certificates[user_id] = []
        
        self.certificates[user_id].append(certificate)
        
        logger.info(f"Certificate generated for user {user_id}: {title}")
        return certificate
    
    def get_user_certificates(self, user_id: str) -> list:
        """Obtener certificados del usuario"""
        return self.certificates.get(user_id, [])
    
    def verify_certificate(self, verification_code: str) -> Optional[Certificate]:
        """Verificar certificado por código"""
        for certificates_list in self.certificates.values():
            for cert in certificates_list:
                if cert.verification_code == verification_code:
                    return cert
        return None
    
    def generate_roadmap_completion_certificate(
        self,
        user_id: str,
        roadmap_name: str
    ) -> Certificate:
        """Generar certificado de completación de roadmap"""
        return self.generate_certificate(
            user_id=user_id,
            certificate_type=CertificateType.ROADMAP_COMPLETION,
            title=f"Completación de Roadmap: {roadmap_name}",
            description=f"Has completado exitosamente el roadmap '{roadmap_name}'",
            metadata={"roadmap_name": roadmap_name}
        )
    
    def generate_skill_certificate(
        self,
        user_id: str,
        skill_name: str,
        skill_level: str
    ) -> Certificate:
        """Generar certificado de habilidad"""
        return self.generate_certificate(
            user_id=user_id,
            certificate_type=CertificateType.SKILL_MASTERY,
            title=f"Dominio de {skill_name}",
            description=f"Has demostrado dominio en {skill_name} (Nivel: {skill_level})",
            metadata={"skill_name": skill_name, "skill_level": skill_level}
        )




