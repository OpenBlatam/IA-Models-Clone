"""
Biometrics Service - Sistema de biometría y reconocimiento
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class BiometricType(str, Enum):
    """Tipos biométricos"""
    FACE = "face"
    FINGERPRINT = "fingerprint"
    VOICE = "voice"
    IRIS = "iris"
    PALM = "palm"


class BiometricsService:
    """Servicio para biometría y reconocimiento"""
    
    def __init__(self):
        self.enrollments: Dict[str, Dict[str, Any]] = {}
        self.verifications: Dict[str, List[Dict[str, Any]]] = {}
        self.access_logs: Dict[str, List[Dict[str, Any]]] = {}
    
    def enroll_biometric(
        self,
        user_id: str,
        biometric_type: BiometricType,
        biometric_data: str,  # En producción, sería hash o template
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Registrar biometría"""
        
        enrollment_id = f"bio_{user_id}_{biometric_type.value}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        enrollment = {
            "enrollment_id": enrollment_id,
            "user_id": user_id,
            "biometric_type": biometric_type.value,
            "biometric_hash": f"hash_{biometric_data[:20]}",  # Placeholder
            "metadata": metadata or {},
            "enrolled_at": datetime.now().isoformat(),
            "is_active": True,
            "note": "En producción, esto almacenaría templates biométricos seguros"
        }
        
        if user_id not in self.enrollments:
            self.enrollments[user_id] = []
        
        self.enrollments[user_id].append(enrollment)
        
        return enrollment
    
    def verify_biometric(
        self,
        user_id: str,
        biometric_type: BiometricType,
        biometric_data: str,
        threshold: float = 0.8
    ) -> Dict[str, Any]:
        """Verificar biometría"""
        
        user_enrollments = self.enrollments.get(user_id, [])
        matching_enrollment = next(
            (e for e in user_enrollments if e["biometric_type"] == biometric_type.value and e["is_active"]),
            None
        )
        
        if not matching_enrollment:
            return {
                "verified": False,
                "reason": "No enrollment found",
                "confidence": 0.0
            }
        
        # Simular verificación (en producción, comparar templates)
        confidence = 0.85  # Placeholder
        
        verification = {
            "verification_id": f"verify_{user_id}_{len(self.verifications.get(user_id, [])) + 1}",
            "user_id": user_id,
            "biometric_type": biometric_type.value,
            "verified": confidence >= threshold,
            "confidence": confidence,
            "threshold": threshold,
            "verified_at": datetime.now().isoformat()
        }
        
        if user_id not in self.verifications:
            self.verifications[user_id] = []
        
        self.verifications[user_id].append(verification)
        
        return verification
    
    def record_access(
        self,
        user_id: str,
        location: str,
        access_granted: bool,
        method: str = "biometric"
    ) -> Dict[str, Any]:
        """Registrar acceso"""
        
        access_log = {
            "log_id": f"access_{user_id}_{len(self.access_logs.get(user_id, [])) + 1}",
            "user_id": user_id,
            "location": location,
            "access_granted": access_granted,
            "method": method,
            "timestamp": datetime.now().isoformat()
        }
        
        if user_id not in self.access_logs:
            self.access_logs[user_id] = []
        
        self.access_logs[user_id].append(access_log)
        
        return access_log
    
    def get_access_history(
        self,
        user_id: str,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """Obtener historial de acceso"""
        from datetime import timedelta
        
        logs = self.access_logs.get(user_id, [])
        cutoff_date = datetime.now() - timedelta(days=days)
        
        recent_logs = [
            log for log in logs
            if datetime.fromisoformat(log["timestamp"]) >= cutoff_date
        ]
        
        return recent_logs




