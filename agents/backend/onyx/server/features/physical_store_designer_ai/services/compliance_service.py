"""
Compliance Service - Sistema de compliance y regulaciones
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ComplianceType(str, Enum):
    """Tipos de compliance"""
    BUILDING_CODE = "building_code"
    FIRE_SAFETY = "fire_safety"
    ACCESSIBILITY = "accessibility"
    ENVIRONMENTAL = "environmental"
    HEALTH_SAFETY = "health_safety"
    ZONING = "zoning"


class ComplianceStatus(str, Enum):
    """Estados de compliance"""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PENDING_REVIEW = "pending_review"
    EXEMPT = "exempt"


class ComplianceService:
    """Servicio para compliance y regulaciones"""
    
    def __init__(self):
        self.requirements: Dict[str, List[Dict[str, Any]]] = {}
        self.assessments: Dict[str, Dict[str, Any]] = {}
        self.certifications: Dict[str, List[Dict[str, Any]]] = {}
    
    def assess_compliance(
        self,
        store_id: str,
        design_data: Dict[str, Any],
        location: Optional[str] = None
    ) -> Dict[str, Any]:
        """Evaluar compliance del diseño"""
        
        assessment_id = f"assess_{store_id}_{datetime.now().strftime('%Y%m%d')}"
        
        # Evaluar diferentes tipos de compliance
        building_code = self._assess_building_code(design_data)
        fire_safety = self._assess_fire_safety(design_data)
        accessibility = self._assess_accessibility(design_data)
        environmental = self._assess_environmental(design_data)
        
        assessment = {
            "assessment_id": assessment_id,
            "store_id": store_id,
            "location": location,
            "assessed_at": datetime.now().isoformat(),
            "compliance_checks": {
                "building_code": building_code,
                "fire_safety": fire_safety,
                "accessibility": accessibility,
                "environmental": environmental
            },
            "overall_status": self._calculate_overall_status([
                building_code, fire_safety, accessibility, environmental
            ]),
            "issues": self._identify_issues([
                building_code, fire_safety, accessibility, environmental
            ]),
            "recommendations": self._generate_recommendations([
                building_code, fire_safety, accessibility, environmental
            ])
        }
        
        self.assessments[assessment_id] = assessment
        
        return assessment
    
    def _assess_building_code(self, design_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluar código de construcción"""
        layout = design_data.get("layout", {})
        dimensions = layout.get("dimensions", {})
        
        # Verificaciones básicas
        width = dimensions.get("width", 0)
        length = dimensions.get("length", 0)
        height = dimensions.get("height", 0)
        
        issues = []
        
        if width < 3:  # Mínimo 3m
            issues.append("Ancho mínimo no cumplido")
        
        if height < 2.5:  # Mínimo 2.5m
            issues.append("Altura mínima no cumplida")
        
        return {
            "type": ComplianceType.BUILDING_CODE.value,
            "status": ComplianceStatus.COMPLIANT.value if not issues else ComplianceStatus.NON_COMPLIANT.value,
            "issues": issues,
            "checked_at": datetime.now().isoformat()
        }
    
    def _assess_fire_safety(self, design_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluar seguridad contra incendios"""
        layout = design_data.get("layout", {})
        zones = layout.get("zones", [])
        
        issues = []
        
        # Verificar salidas de emergencia
        if len(zones) > 1:
            # Debería haber múltiples salidas
            issues.append("Verificar múltiples salidas de emergencia")
        
        return {
            "type": ComplianceType.FIRE_SAFETY.value,
            "status": ComplianceStatus.PENDING_REVIEW.value,
            "issues": issues,
            "checked_at": datetime.now().isoformat()
        }
    
    def _assess_accessibility(self, design_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluar accesibilidad"""
        issues = []
        
        # Verificar accesibilidad básica
        layout = design_data.get("layout", {})
        zones = layout.get("zones", [])
        
        # Debería haber rampas/ascensores si hay múltiples niveles
        if len(zones) > 3:
            issues.append("Verificar accesibilidad para múltiples zonas")
        
        return {
            "type": ComplianceType.ACCESSIBILITY.value,
            "status": ComplianceStatus.COMPLIANT.value if not issues else ComplianceStatus.PENDING_REVIEW.value,
            "issues": issues,
            "checked_at": datetime.now().isoformat()
        }
    
    def _assess_environmental(self, design_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluar compliance ambiental"""
        issues = []
        
        # Verificar aspectos ambientales
        style = design_data.get("style", "")
        
        if "eco" not in style.lower():
            issues.append("Considerar aspectos ambientales en el diseño")
        
        return {
            "type": ComplianceType.ENVIRONMENTAL.value,
            "status": ComplianceStatus.COMPLIANT.value if not issues else ComplianceStatus.PENDING_REVIEW.value,
            "issues": issues,
            "checked_at": datetime.now().isoformat()
        }
    
    def _calculate_overall_status(
        self,
        checks: List[Dict[str, Any]]
    ) -> str:
        """Calcular estado general"""
        statuses = [c["status"] for c in checks]
        
        if ComplianceStatus.NON_COMPLIANT.value in statuses:
            return ComplianceStatus.NON_COMPLIANT.value
        elif ComplianceStatus.PENDING_REVIEW.value in statuses:
            return ComplianceStatus.PENDING_REVIEW.value
        else:
            return ComplianceStatus.COMPLIANT.value
    
    def _identify_issues(
        self,
        checks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Identificar todos los issues"""
        all_issues = []
        
        for check in checks:
            for issue in check.get("issues", []):
                all_issues.append({
                    "type": check["type"],
                    "issue": issue,
                    "severity": "high" if check["status"] == ComplianceStatus.NON_COMPLIANT.value else "medium"
                })
        
        return all_issues
    
    def _generate_recommendations(
        self,
        checks: List[Dict[str, Any]]
    ) -> List[str]:
        """Generar recomendaciones"""
        recommendations = []
        
        for check in checks:
            if check["status"] != ComplianceStatus.COMPLIANT.value:
                if check["type"] == ComplianceType.BUILDING_CODE.value:
                    recommendations.append("Consultar con arquitecto sobre código de construcción")
                elif check["type"] == ComplianceType.FIRE_SAFETY.value:
                    recommendations.append("Revisar plan de seguridad contra incendios")
                elif check["type"] == ComplianceType.ACCESSIBILITY.value:
                    recommendations.append("Asegurar cumplimiento de ADA/leyes de accesibilidad")
                elif check["type"] == ComplianceType.ENVIRONMENTAL.value:
                    recommendations.append("Considerar certificaciones ambientales (LEED, etc.)")
        
        return recommendations
    
    def get_compliance_certificate(
        self,
        assessment_id: str
    ) -> Optional[Dict[str, Any]]:
        """Obtener certificado de compliance"""
        assessment = self.assessments.get(assessment_id)
        
        if not assessment:
            return None
        
        if assessment["overall_status"] != ComplianceStatus.COMPLIANT.value:
            return None
        
        certificate = {
            "certificate_id": f"cert_{assessment_id}",
            "assessment_id": assessment_id,
            "store_id": assessment["store_id"],
            "issued_at": datetime.now().isoformat(),
            "valid_until": (datetime.now().replace(year=datetime.now().year + 1)).isoformat(),
            "status": "valid"
        }
        
        store_id = assessment["store_id"]
        if store_id not in self.certifications:
            self.certifications[store_id] = []
        
        self.certifications[store_id].append(certificate)
        
        return certificate




