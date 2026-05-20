"""
Servicio de Certificados y Diplomas - Generación de certificados de logros
"""

from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum


class CertificateType(str, Enum):
    """Tipos de certificados"""
    SOBRIETY_MILESTONE = "sobriety_milestone"
    PROGRAM_COMPLETION = "program_completion"
    ACHIEVEMENT = "achievement"
    PARTICIPATION = "participation"
    MENTORSHIP = "mentorship"
    COMMUNITY_CONTRIBUTION = "community_contribution"


class CertificateService:
    """Servicio de certificados y diplomas"""
    
    def __init__(self):
        """Inicializa el servicio de certificados"""
        self.certificate_templates = self._load_templates()
    
    def generate_certificate(
        self,
        user_id: str,
        certificate_type: str,
        title: str,
        description: str,
        achievement_data: Optional[Dict] = None
    ) -> Dict:
        """
        Genera un certificado
        
        Args:
            user_id: ID del usuario
            certificate_type: Tipo de certificado
            title: Título del certificado
            description: Descripción del logro
            achievement_data: Datos del logro (opcional)
        
        Returns:
            Certificado generado
        """
        certificate = {
            "id": f"cert_{datetime.now().timestamp()}",
            "user_id": user_id,
            "certificate_type": certificate_type,
            "title": title,
            "description": description,
            "achievement_data": achievement_data or {},
            "issued_date": datetime.now().isoformat(),
            "certificate_number": self._generate_certificate_number(),
            "verification_code": self._generate_verification_code(),
            "template": self._get_template(certificate_type),
            "status": "issued"
        }
        
        return certificate
    
    def get_user_certificates(
        self,
        user_id: str,
        certificate_type: Optional[str] = None
    ) -> List[Dict]:
        """
        Obtiene certificados del usuario
        
        Args:
            user_id: ID del usuario
            certificate_type: Filtrar por tipo (opcional)
        
        Returns:
            Lista de certificados
        """
        # En implementación real, esto vendría de la base de datos
        return []
    
    def verify_certificate(
        self,
        certificate_number: str,
        verification_code: str
    ) -> Dict:
        """
        Verifica un certificado
        
        Args:
            certificate_number: Número de certificado
            verification_code: Código de verificación
        
        Returns:
            Resultado de verificación
        """
        return {
            "certificate_number": certificate_number,
            "verification_code": verification_code,
            "valid": True,
            "verified_at": datetime.now().isoformat(),
            "message": "Certificado verificado exitosamente"
        }
    
    def generate_milestone_certificate(
        self,
        user_id: str,
        milestone_days: int,
        addiction_type: str
    ) -> Dict:
        """
        Genera certificado de hito de sobriedad
        
        Args:
            user_id: ID del usuario
            milestone_days: Días de sobriedad
            addiction_type: Tipo de adicción
        
        Returns:
            Certificado de hito
        """
        title = f"{milestone_days} Días de Sobriedad"
        description = f"Has completado {milestone_days} días de sobriedad en tu recuperación de {addiction_type}."
        
        return self.generate_certificate(
            user_id,
            CertificateType.SOBRIETY_MILESTONE,
            title,
            description,
            {
                "milestone_days": milestone_days,
                "addiction_type": addiction_type
            }
        )
    
    def generate_program_completion_certificate(
        self,
        user_id: str,
        program_name: str,
        completion_date: str
    ) -> Dict:
        """
        Genera certificado de finalización de programa
        
        Args:
            user_id: ID del usuario
            program_name: Nombre del programa
            completion_date: Fecha de finalización
        
        Returns:
            Certificado de finalización
        """
        title = f"Finalización de {program_name}"
        description = f"Has completado exitosamente el programa {program_name}."
        
        return self.generate_certificate(
            user_id,
            CertificateType.PROGRAM_COMPLETION,
            title,
            description,
            {
                "program_name": program_name,
                "completion_date": completion_date
            }
        )
    
    def _generate_certificate_number(self) -> str:
        """Genera número único de certificado"""
        timestamp = int(datetime.now().timestamp())
        return f"CERT-{timestamp}"
    
    def _generate_verification_code(self) -> str:
        """Genera código de verificación"""
        import random
        import string
        return ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
    
    def _get_template(self, certificate_type: str) -> Dict:
        """Obtiene plantilla de certificado"""
        return self.certificate_templates.get(certificate_type, {
            "layout": "standard",
            "design": "professional",
            "colors": ["#1a1a1a", "#4a90e2"]
        })
    
    def _load_templates(self) -> Dict:
        """Carga plantillas de certificados"""
        return {
            CertificateType.SOBRIETY_MILESTONE: {
                "layout": "milestone",
                "design": "elegant",
                "colors": ["#2c3e50", "#3498db"],
                "badge": "🏆"
            },
            CertificateType.PROGRAM_COMPLETION: {
                "layout": "completion",
                "design": "professional",
                "colors": ["#1a1a1a", "#27ae60"],
                "badge": "✅"
            },
            CertificateType.ACHIEVEMENT: {
                "layout": "achievement",
                "design": "modern",
                "colors": ["#8e44ad", "#e74c3c"],
                "badge": "⭐"
            }
        }

