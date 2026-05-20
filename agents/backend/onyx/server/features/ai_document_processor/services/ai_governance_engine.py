"""
Motor de Gobernanza AI
=====================

Motor para gobernanza de IA, cumplimiento normativo y gestión de políticas.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import numpy as np
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)

class ComplianceFramework(str, Enum):
    """Marcos de cumplimiento"""
    GDPR = "gdpr"
    CCPA = "ccpa"
    HIPAA = "hipaa"
    SOX = "sox"
    ISO27001 = "iso27001"
    NIST = "nist"
    AI_ACT = "ai_act"
    CUSTOM = "custom"

class PolicyType(str, Enum):
    """Tipos de políticas"""
    DATA_PROTECTION = "data_protection"
    PRIVACY = "privacy"
    SECURITY = "security"
    ETHICS = "ethics"
    TRANSPARENCY = "transparency"
    ACCOUNTABILITY = "accountability"
    FAIRNESS = "fairness"
    HUMAN_RIGHTS = "human_rights"

class ComplianceStatus(str, Enum):
    """Estados de cumplimiento"""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    UNDER_REVIEW = "under_review"
    NOT_APPLICABLE = "not_applicable"

class RiskCategory(str, Enum):
    """Categorías de riesgo"""
    LEGAL = "legal"
    REGULATORY = "regulatory"
    REPUTATIONAL = "reputational"
    FINANCIAL = "financial"
    OPERATIONAL = "operational"
    TECHNICAL = "technical"

@dataclass
class Policy:
    """Política de gobernanza"""
    id: str
    name: str
    policy_type: PolicyType
    framework: ComplianceFramework
    description: str
    requirements: List[str] = field(default_factory=list)
    controls: List[str] = field(default_factory=list)
    metrics: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    is_active: bool = True

@dataclass
class ComplianceAssessment:
    """Evaluación de cumplimiento"""
    id: str
    framework: ComplianceFramework
    assessment_date: datetime
    status: ComplianceStatus
    score: float
    findings: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    next_review_date: Optional[datetime] = None
    assessor: str = "system"

@dataclass
class RiskAssessment:
    """Evaluación de riesgo"""
    id: str
    risk_category: RiskCategory
    risk_level: str
    probability: float
    impact: float
    risk_score: float
    description: str
    mitigation_measures: List[str] = field(default_factory=list)
    owner: str = ""
    status: str = "open"
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class GovernanceReport:
    """Reporte de gobernanza"""
    id: str
    report_type: str
    period_start: datetime
    period_end: datetime
    compliance_summary: Dict[str, Any]
    risk_summary: Dict[str, Any]
    policy_updates: List[Dict[str, Any]]
    recommendations: List[str]
    generated_at: datetime = field(default_factory=datetime.now)

class AIGovernanceEngine:
    """Motor de gobernanza AI"""
    
    def __init__(self):
        self.policies: Dict[str, Policy] = {}
        self.compliance_assessments: Dict[str, ComplianceAssessment] = {}
        self.risk_assessments: Dict[str, RiskAssessment] = {}
        self.governance_reports: Dict[str, GovernanceReport] = {}
        self.audit_trails: List[Dict[str, Any]] = []
        
        # Configuración de marcos de cumplimiento
        self.compliance_frameworks = {}
        
    async def initialize(self):
        """Inicializa el motor de gobernanza"""
        logger.info("Inicializando motor de gobernanza AI...")
        
        # Cargar políticas por defecto
        await self._load_default_policies()
        
        # Cargar marcos de cumplimiento
        await self._load_compliance_frameworks()
        
        # Cargar evaluaciones previas
        await self._load_previous_assessments()
        
        logger.info("Motor de gobernanza AI inicializado")
    
    async def _load_default_policies(self):
        """Carga políticas por defecto"""
        try:
            # Política de Protección de Datos (GDPR)
            gdpr_policy = Policy(
                id="gdpr_data_protection",
                name="Protección de Datos Personales",
                policy_type=PolicyType.DATA_PROTECTION,
                framework=ComplianceFramework.GDPR,
                description="Política de protección de datos personales conforme al GDPR",
                requirements=[
                    "Consentimiento explícito para el procesamiento de datos",
                    "Derecho al olvido y portabilidad de datos",
                    "Notificación de violaciones de datos en 72 horas",
                    "Evaluación de impacto en la protección de datos (DPIA)",
                    "Designación de un Delegado de Protección de Datos (DPO)"
                ],
                controls=[
                    "Encriptación de datos en tránsito y en reposo",
                    "Controles de acceso basados en roles",
                    "Registro de actividades de procesamiento",
                    "Auditorías regulares de seguridad",
                    "Capacitación del personal en protección de datos"
                ],
                metrics=[
                    "Tiempo de respuesta a solicitudes de datos",
                    "Número de violaciones de datos reportadas",
                    "Porcentaje de empleados capacitados en GDPR",
                    "Tiempo de notificación de violaciones"
                ]
            )
            
            # Política de Privacidad
            privacy_policy = Policy(
                id="privacy_policy",
                name="Política de Privacidad",
                policy_type=PolicyType.PRIVACY,
                framework=ComplianceFramework.GDPR,
                description="Política de privacidad para el procesamiento de datos con IA",
                requirements=[
                    "Transparencia en el uso de datos",
                    "Minimización de datos recolectados",
                    "Limitación del propósito del procesamiento",
                    "Precisión y actualización de datos",
                    "Limitación del tiempo de almacenamiento"
                ],
                controls=[
                    "Anonimización y pseudonimización de datos",
                    "Controles de retención de datos",
                    "Procesos de eliminación segura",
                    "Monitoreo de acceso a datos",
                    "Revisión regular de políticas de privacidad"
                ],
                metrics=[
                    "Porcentaje de datos anonimizados",
                    "Tiempo promedio de retención de datos",
                    "Número de solicitudes de privacidad procesadas",
                    "Tiempo de respuesta a solicitudes de privacidad"
                ]
            )
            
            # Política de Ética AI
            ethics_policy = Policy(
                id="ai_ethics_policy",
                name="Política de Ética en IA",
                policy_type=PolicyType.ETHICS,
                framework=ComplianceFramework.AI_ACT,
                description="Política de ética para el desarrollo y uso de sistemas de IA",
                requirements=[
                    "Transparencia en algoritmos y decisiones",
                    "Equidad y no discriminación",
                    "Responsabilidad humana en decisiones críticas",
                    "Privacidad y protección de datos",
                    "Beneficio social y bienestar humano"
                ],
                controls=[
                    "Evaluación de impacto ético de sistemas de IA",
                    "Pruebas de sesgo en algoritmos",
                    "Revisión ética por comité independiente",
                    "Capacitación en ética AI para desarrolladores",
                    "Monitoreo continuo de sesgos"
                ],
                metrics=[
                    "Número de evaluaciones éticas realizadas",
                    "Porcentaje de sistemas con evaluación ética",
                    "Tiempo promedio de revisión ética",
                    "Número de sesgos detectados y corregidos"
                ]
            )
            
            # Política de Seguridad
            security_policy = Policy(
                id="security_policy",
                name="Política de Seguridad de la Información",
                policy_type=PolicyType.SECURITY,
                framework=ComplianceFramework.ISO27001,
                description="Política de seguridad para sistemas de IA y datos",
                requirements=[
                    "Clasificación y etiquetado de datos",
                    "Controles de acceso y autenticación",
                    "Protección contra amenazas cibernéticas",
                    "Respuesta a incidentes de seguridad",
                    "Continuidad del negocio y recuperación"
                ],
                controls=[
                    "Firewalls y sistemas de detección de intrusiones",
                    "Antivirus y protección contra malware",
                    "Copias de seguridad regulares",
                    "Plan de respuesta a incidentes",
                    "Capacitación en seguridad del personal"
                ],
                metrics=[
                    "Número de incidentes de seguridad",
                    "Tiempo de detección de amenazas",
                    "Tiempo de respuesta a incidentes",
                    "Porcentaje de empleados capacitados en seguridad"
                ]
            )
            
            # Guardar políticas
            self.policies["gdpr_data_protection"] = gdpr_policy
            self.policies["privacy_policy"] = privacy_policy
            self.policies["ai_ethics_policy"] = ethics_policy
            self.policies["security_policy"] = security_policy
            
            logger.info(f"Cargadas {len(self.policies)} políticas por defecto")
            
        except Exception as e:
            logger.error(f"Error cargando políticas por defecto: {e}")
    
    async def _load_compliance_frameworks(self):
        """Carga marcos de cumplimiento"""
        try:
            # Marco GDPR
            self.compliance_frameworks[ComplianceFramework.GDPR] = {
                "name": "General Data Protection Regulation",
                "jurisdiction": "European Union",
                "requirements": [
                    "Lawfulness, fairness and transparency",
                    "Purpose limitation",
                    "Data minimisation",
                    "Accuracy",
                    "Storage limitation",
                    "Integrity and confidentiality",
                    "Accountability"
                ],
                "penalties": {
                    "minor": "Up to €10 million or 2% of annual turnover",
                    "major": "Up to €20 million or 4% of annual turnover"
                },
                "key_articles": [
                    "Article 5: Principles relating to processing",
                    "Article 6: Lawfulness of processing",
                    "Article 7: Conditions for consent",
                    "Article 25: Data protection by design and by default",
                    "Article 32: Security of processing"
                ]
            }
            
            # Marco AI Act
            self.compliance_frameworks[ComplianceFramework.AI_ACT] = {
                "name": "AI Act (Proposed EU Regulation)",
                "jurisdiction": "European Union",
                "requirements": [
                    "Risk-based approach to AI regulation",
                    "Transparency and explainability",
                    "Human oversight",
                    "Robustness and accuracy",
                    "Privacy and data governance",
                    "Diversity, non-discrimination and fairness"
                ],
                "penalties": {
                    "minor": "Up to €10 million or 2% of annual turnover",
                    "major": "Up to €30 million or 6% of annual turnover"
                },
                "risk_categories": [
                    "Minimal risk",
                    "Limited risk",
                    "High risk",
                    "Unacceptable risk"
                ]
            }
            
            # Marco ISO 27001
            self.compliance_frameworks[ComplianceFramework.ISO27001] = {
                "name": "ISO/IEC 27001 Information Security Management",
                "jurisdiction": "International",
                "requirements": [
                    "Information security policies",
                    "Organization of information security",
                    "Human resource security",
                    "Asset management",
                    "Access control",
                    "Cryptography",
                    "Physical and environmental security",
                    "Operations security",
                    "Communications security",
                    "System acquisition, development and maintenance",
                    "Supplier relationships",
                    "Information security incident management",
                    "Information security aspects of business continuity management",
                    "Compliance"
                ],
                "certification": "Third-party certification available",
                "controls": "114 security controls in Annex A"
            }
            
            logger.info(f"Cargados {len(self.compliance_frameworks)} marcos de cumplimiento")
            
        except Exception as e:
            logger.error(f"Error cargando marcos de cumplimiento: {e}")
    
    async def _load_previous_assessments(self):
        """Carga evaluaciones previas"""
        try:
            assessments_file = Path("data/governance_assessments.json")
            if assessments_file.exists():
                with open(assessments_file, 'r', encoding='utf-8') as f:
                    assessments_data = json.load(f)
                
                for assessment_data in assessments_data:
                    assessment = ComplianceAssessment(
                        id=assessment_data["id"],
                        framework=ComplianceFramework(assessment_data["framework"]),
                        assessment_date=datetime.fromisoformat(assessment_data["assessment_date"]),
                        status=ComplianceStatus(assessment_data["status"]),
                        score=assessment_data["score"],
                        findings=assessment_data["findings"],
                        recommendations=assessment_data["recommendations"],
                        next_review_date=datetime.fromisoformat(assessment_data["next_review_date"]) if assessment_data.get("next_review_date") else None,
                        assessor=assessment_data.get("assessor", "system")
                    )
                    self.compliance_assessments[assessment.id] = assessment
                
                logger.info(f"Cargadas {len(self.compliance_assessments)} evaluaciones previas")
            
        except Exception as e:
            logger.error(f"Error cargando evaluaciones previas: {e}")
    
    async def create_policy(
        self,
        name: str,
        policy_type: PolicyType,
        framework: ComplianceFramework,
        description: str,
        requirements: List[str] = None,
        controls: List[str] = None,
        metrics: List[str] = None
    ) -> str:
        """Crea una nueva política"""
        try:
            policy_id = f"{policy_type.value}_{framework.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            policy = Policy(
                id=policy_id,
                name=name,
                policy_type=policy_type,
                framework=framework,
                description=description,
                requirements=requirements or [],
                controls=controls or [],
                metrics=metrics or []
            )
            
            self.policies[policy_id] = policy
            
            # Registrar en auditoría
            await self._log_audit_event(
                "policy_created",
                policy_id,
                {"policy_name": name, "framework": framework.value}
            )
            
            logger.info(f"Política creada: {policy_id}")
            return policy_id
            
        except Exception as e:
            logger.error(f"Error creando política: {e}")
            raise
    
    async def update_policy(
        self,
        policy_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """Actualiza una política existente"""
        try:
            if policy_id not in self.policies:
                return False
            
            policy = self.policies[policy_id]
            
            # Actualizar campos permitidos
            if "name" in updates:
                policy.name = updates["name"]
            if "description" in updates:
                policy.description = updates["description"]
            if "requirements" in updates:
                policy.requirements = updates["requirements"]
            if "controls" in updates:
                policy.controls = updates["controls"]
            if "metrics" in updates:
                policy.metrics = updates["metrics"]
            if "is_active" in updates:
                policy.is_active = updates["is_active"]
            
            policy.updated_at = datetime.now()
            
            # Registrar en auditoría
            await self._log_audit_event(
                "policy_updated",
                policy_id,
                {"updates": list(updates.keys())}
            )
            
            logger.info(f"Política actualizada: {policy_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error actualizando política: {e}")
            return False
    
    async def conduct_compliance_assessment(
        self,
        framework: ComplianceFramework,
        assessor: str = "system"
    ) -> str:
        """Realiza evaluación de cumplimiento"""
        try:
            assessment_id = f"assessment_{framework.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Crear evaluación
            assessment = ComplianceAssessment(
                id=assessment_id,
                framework=framework,
                assessment_date=datetime.now(),
                status=ComplianceStatus.UNDER_REVIEW,
                score=0.0,
                assessor=assessor
            )
            
            # Realizar evaluación
            await self._perform_compliance_evaluation(assessment)
            
            # Guardar evaluación
            self.compliance_assessments[assessment_id] = assessment
            await self._save_assessment(assessment)
            
            # Registrar en auditoría
            await self._log_audit_event(
                "compliance_assessment",
                assessment_id,
                {"framework": framework.value, "score": assessment.score}
            )
            
            logger.info(f"Evaluación de cumplimiento completada: {assessment_id}")
            return assessment_id
            
        except Exception as e:
            logger.error(f"Error realizando evaluación de cumplimiento: {e}")
            raise
    
    async def _perform_compliance_evaluation(self, assessment: ComplianceAssessment):
        """Realiza evaluación de cumplimiento específica"""
        try:
            framework = assessment.framework
            findings = []
            recommendations = []
            total_score = 0.0
            max_score = 0.0
            
            if framework == ComplianceFramework.GDPR:
                # Evaluar cumplimiento GDPR
                gdpr_requirements = self.compliance_frameworks[framework]["requirements"]
                
                for requirement in gdpr_requirements:
                    max_score += 1.0
                    # Simular evaluación (en implementación real, verificar controles)
                    compliance_score = np.random.uniform(0.6, 1.0)
                    total_score += compliance_score
                    
                    if compliance_score < 0.8:
                        findings.append({
                            "requirement": requirement,
                            "status": "non_compliant" if compliance_score < 0.6 else "partially_compliant",
                            "score": compliance_score,
                            "evidence": f"Evidence for {requirement}",
                            "gap": f"Gap identified in {requirement}"
                        })
                        
                        recommendations.append(f"Improve compliance with {requirement}")
            
            elif framework == ComplianceFramework.AI_ACT:
                # Evaluar cumplimiento AI Act
                ai_requirements = self.compliance_frameworks[framework]["requirements"]
                
                for requirement in ai_requirements:
                    max_score += 1.0
                    compliance_score = np.random.uniform(0.5, 1.0)
                    total_score += compliance_score
                    
                    if compliance_score < 0.7:
                        findings.append({
                            "requirement": requirement,
                            "status": "non_compliant" if compliance_score < 0.5 else "partially_compliant",
                            "score": compliance_score,
                            "evidence": f"Evidence for {requirement}",
                            "gap": f"Gap identified in {requirement}"
                        })
                        
                        recommendations.append(f"Enhance {requirement} implementation")
            
            elif framework == ComplianceFramework.ISO27001:
                # Evaluar cumplimiento ISO 27001
                iso_requirements = self.compliance_frameworks[framework]["requirements"]
                
                for requirement in iso_requirements:
                    max_score += 1.0
                    compliance_score = np.random.uniform(0.7, 1.0)
                    total_score += compliance_score
                    
                    if compliance_score < 0.8:
                        findings.append({
                            "requirement": requirement,
                            "status": "non_compliant" if compliance_score < 0.6 else "partially_compliant",
                            "score": compliance_score,
                            "evidence": f"Evidence for {requirement}",
                            "gap": f"Gap identified in {requirement}"
                        })
                        
                        recommendations.append(f"Strengthen {requirement} controls")
            
            # Calcular puntuación final
            final_score = total_score / max_score if max_score > 0 else 0.0
            
            # Determinar estado
            if final_score >= 0.9:
                assessment.status = ComplianceStatus.COMPLIANT
            elif final_score >= 0.7:
                assessment.status = ComplianceStatus.PARTIALLY_COMPLIANT
            else:
                assessment.status = ComplianceStatus.NON_COMPLIANT
            
            assessment.score = final_score
            assessment.findings = findings
            assessment.recommendations = recommendations
            assessment.next_review_date = datetime.now() + timedelta(days=90)
            
        except Exception as e:
            logger.error(f"Error realizando evaluación de cumplimiento: {e}")
            assessment.status = ComplianceStatus.NON_COMPLIANT
            assessment.score = 0.0
    
    async def assess_risk(
        self,
        risk_category: RiskCategory,
        description: str,
        probability: float,
        impact: float,
        owner: str = ""
    ) -> str:
        """Evalúa un riesgo"""
        try:
            risk_id = f"risk_{risk_category.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Calcular puntuación de riesgo
            risk_score = probability * impact
            
            # Determinar nivel de riesgo
            if risk_score >= 0.8:
                risk_level = "critical"
            elif risk_score >= 0.6:
                risk_level = "high"
            elif risk_score >= 0.4:
                risk_level = "medium"
            else:
                risk_level = "low"
            
            # Generar medidas de mitigación
            mitigation_measures = await self._generate_mitigation_measures(risk_category, risk_level)
            
            risk_assessment = RiskAssessment(
                id=risk_id,
                risk_category=risk_category,
                risk_level=risk_level,
                probability=probability,
                impact=impact,
                risk_score=risk_score,
                description=description,
                mitigation_measures=mitigation_measures,
                owner=owner
            )
            
            self.risk_assessments[risk_id] = risk_assessment
            
            # Registrar en auditoría
            await self._log_audit_event(
                "risk_assessed",
                risk_id,
                {"category": risk_category.value, "level": risk_level, "score": risk_score}
            )
            
            logger.info(f"Riesgo evaluado: {risk_id}")
            return risk_id
            
        except Exception as e:
            logger.error(f"Error evaluando riesgo: {e}")
            raise
    
    async def _generate_mitigation_measures(self, risk_category: RiskCategory, risk_level: str) -> List[str]:
        """Genera medidas de mitigación"""
        try:
            mitigation_measures = {
                RiskCategory.LEGAL: [
                    "Revisar y actualizar términos y condiciones",
                    "Implementar procedimientos de cumplimiento legal",
                    "Capacitar al personal en aspectos legales",
                    "Establecer consultas legales regulares",
                    "Mantener documentación legal actualizada"
                ],
                RiskCategory.REGULATORY: [
                    "Monitorear cambios regulatorios",
                    "Implementar controles de cumplimiento",
                    "Realizar auditorías regulatorias",
                    "Capacitar en regulaciones aplicables",
                    "Establecer procesos de reporte regulatorio"
                ],
                RiskCategory.REPUTATIONAL: [
                    "Implementar gestión de crisis",
                    "Establecer comunicación transparente",
                    "Monitorear reputación en línea",
                    "Desarrollar políticas de responsabilidad social",
                    "Capacitar en comunicación corporativa"
                ],
                RiskCategory.FINANCIAL: [
                    "Implementar controles financieros",
                    "Establecer presupuestos de contingencia",
                    "Realizar análisis de costo-beneficio",
                    "Implementar seguros apropiados",
                    "Monitorear indicadores financieros"
                ],
                RiskCategory.OPERATIONAL: [
                    "Implementar procedimientos operativos",
                    "Establecer planes de continuidad",
                    "Capacitar al personal operativo",
                    "Implementar controles de calidad",
                    "Monitorear métricas operativas"
                ],
                RiskCategory.TECHNICAL: [
                    "Implementar controles de seguridad técnica",
                    "Establecer arquitectura robusta",
                    "Realizar pruebas de penetración",
                    "Implementar monitoreo de sistemas",
                    "Capacitar en seguridad técnica"
                ]
            }
            
            base_measures = mitigation_measures.get(risk_category, [
                "Identificar y evaluar el riesgo",
                "Implementar controles apropiados",
                "Monitorear y revisar regularmente",
                "Capacitar al personal relevante",
                "Documentar procesos y procedimientos"
            ])
            
            # Ajustar medidas según nivel de riesgo
            if risk_level == "critical":
                return base_measures + [
                    "Implementar medidas inmediatas",
                    "Establecer monitoreo continuo",
                    "Asignar recursos adicionales",
                    "Revisar semanalmente"
                ]
            elif risk_level == "high":
                return base_measures + [
                    "Implementar medidas prioritarias",
                    "Establecer monitoreo regular",
                    "Asignar responsable específico",
                    "Revisar mensualmente"
                ]
            else:
                return base_measures[:3]  # Medidas básicas para riesgos bajos/medios
            
        except Exception as e:
            logger.error(f"Error generando medidas de mitigación: {e}")
            return ["Implementar controles básicos de riesgo"]
    
    async def generate_governance_report(
        self,
        report_type: str,
        period_start: datetime,
        period_end: datetime
    ) -> str:
        """Genera reporte de gobernanza"""
        try:
            report_id = f"report_{report_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Generar resumen de cumplimiento
            compliance_summary = await self._generate_compliance_summary(period_start, period_end)
            
            # Generar resumen de riesgos
            risk_summary = await self._generate_risk_summary(period_start, period_end)
            
            # Obtener actualizaciones de políticas
            policy_updates = await self._get_policy_updates(period_start, period_end)
            
            # Generar recomendaciones
            recommendations = await self._generate_governance_recommendations(compliance_summary, risk_summary)
            
            report = GovernanceReport(
                id=report_id,
                report_type=report_type,
                period_start=period_start,
                period_end=period_end,
                compliance_summary=compliance_summary,
                risk_summary=risk_summary,
                policy_updates=policy_updates,
                recommendations=recommendations
            )
            
            self.governance_reports[report_id] = report
            
            # Registrar en auditoría
            await self._log_audit_event(
                "governance_report_generated",
                report_id,
                {"report_type": report_type, "period": f"{period_start} to {period_end}"}
            )
            
            logger.info(f"Reporte de gobernanza generado: {report_id}")
            return report_id
            
        except Exception as e:
            logger.error(f"Error generando reporte de gobernanza: {e}")
            raise
    
    async def _generate_compliance_summary(self, period_start: datetime, period_end: datetime) -> Dict[str, Any]:
        """Genera resumen de cumplimiento"""
        try:
            # Filtrar evaluaciones del período
            period_assessments = [
                assessment for assessment in self.compliance_assessments.values()
                if period_start <= assessment.assessment_date <= period_end
            ]
            
            if not period_assessments:
                return {
                    "total_assessments": 0,
                    "average_score": 0.0,
                    "compliance_status": "no_data",
                    "frameworks_assessed": [],
                    "findings_summary": {}
                }
            
            # Calcular estadísticas
            total_assessments = len(period_assessments)
            average_score = np.mean([a.score for a in period_assessments])
            
            # Distribución por estado
            status_distribution = {}
            for assessment in period_assessments:
                status = assessment.status.value
                status_distribution[status] = status_distribution.get(status, 0) + 1
            
            # Marcos evaluados
            frameworks_assessed = list(set(a.framework.value for a in period_assessments))
            
            # Resumen de hallazgos
            findings_summary = {}
            for assessment in period_assessments:
                for finding in assessment.findings:
                    status = finding.get("status", "unknown")
                    findings_summary[status] = findings_summary.get(status, 0) + 1
            
            return {
                "total_assessments": total_assessments,
                "average_score": average_score,
                "compliance_status": "compliant" if average_score >= 0.8 else "needs_improvement",
                "status_distribution": status_distribution,
                "frameworks_assessed": frameworks_assessed,
                "findings_summary": findings_summary
            }
            
        except Exception as e:
            logger.error(f"Error generando resumen de cumplimiento: {e}")
            return {"error": str(e)}
    
    async def _generate_risk_summary(self, period_start: datetime, period_end: datetime) -> Dict[str, Any]:
        """Genera resumen de riesgos"""
        try:
            # Filtrar riesgos del período
            period_risks = [
                risk for risk in self.risk_assessments.values()
                if period_start <= risk.created_at <= period_end
            ]
            
            if not period_risks:
                return {
                    "total_risks": 0,
                    "risk_distribution": {},
                    "average_risk_score": 0.0,
                    "critical_risks": 0
                }
            
            # Calcular estadísticas
            total_risks = len(period_risks)
            average_risk_score = np.mean([r.risk_score for r in period_risks])
            
            # Distribución por categoría
            category_distribution = {}
            level_distribution = {}
            critical_risks = 0
            
            for risk in period_risks:
                category = risk.risk_category.value
                level = risk.risk_level
                
                category_distribution[category] = category_distribution.get(category, 0) + 1
                level_distribution[level] = level_distribution.get(level, 0) + 1
                
                if level == "critical":
                    critical_risks += 1
            
            return {
                "total_risks": total_risks,
                "risk_distribution": category_distribution,
                "level_distribution": level_distribution,
                "average_risk_score": average_risk_score,
                "critical_risks": critical_risks
            }
            
        except Exception as e:
            logger.error(f"Error generando resumen de riesgos: {e}")
            return {"error": str(e)}
    
    async def _get_policy_updates(self, period_start: datetime, period_end: datetime) -> List[Dict[str, Any]]:
        """Obtiene actualizaciones de políticas"""
        try:
            updates = []
            
            for policy in self.policies.values():
                if period_start <= policy.updated_at <= period_end:
                    updates.append({
                        "policy_id": policy.id,
                        "policy_name": policy.name,
                        "updated_at": policy.updated_at.isoformat(),
                        "framework": policy.framework.value
                    })
            
            return updates
            
        except Exception as e:
            logger.error(f"Error obteniendo actualizaciones de políticas: {e}")
            return []
    
    async def _generate_governance_recommendations(
        self,
        compliance_summary: Dict[str, Any],
        risk_summary: Dict[str, Any]
    ) -> List[str]:
        """Genera recomendaciones de gobernanza"""
        try:
            recommendations = []
            
            # Recomendaciones basadas en cumplimiento
            if compliance_summary.get("average_score", 0) < 0.8:
                recommendations.append("Mejorar puntuación de cumplimiento general")
                recommendations.append("Implementar plan de acción para hallazgos no conformes")
            
            if compliance_summary.get("findings_summary", {}).get("non_compliant", 0) > 0:
                recommendations.append("Abordar hallazgos de no conformidad prioritariamente")
                recommendations.append("Establecer cronograma de corrección")
            
            # Recomendaciones basadas en riesgos
            if risk_summary.get("critical_risks", 0) > 0:
                recommendations.append("Abordar riesgos críticos inmediatamente")
                recommendations.append("Implementar medidas de mitigación urgentes")
            
            if risk_summary.get("average_risk_score", 0) > 0.6:
                recommendations.append("Revisar y fortalecer controles de riesgo")
                recommendations.append("Implementar programa de gestión de riesgos")
            
            # Recomendaciones generales
            recommendations.extend([
                "Realizar evaluaciones de cumplimiento regulares",
                "Mantener políticas actualizadas",
                "Capacitar al personal en gobernanza",
                "Establecer métricas de gobernanza",
                "Implementar monitoreo continuo"
            ])
            
            return recommendations[:10]  # Limitar a 10 recomendaciones
            
        except Exception as e:
            logger.error(f"Error generando recomendaciones de gobernanza: {e}")
            return ["Revisar aspectos de gobernanza"]
    
    async def _log_audit_event(self, event_type: str, entity_id: str, details: Dict[str, Any]):
        """Registra evento de auditoría"""
        try:
            audit_event = {
                "timestamp": datetime.now().isoformat(),
                "event_type": event_type,
                "entity_id": entity_id,
                "details": details,
                "user": "system"
            }
            
            self.audit_trails.append(audit_event)
            
            # Mantener solo los últimos 1000 eventos
            if len(self.audit_trails) > 1000:
                self.audit_trails = self.audit_trails[-1000:]
            
        except Exception as e:
            logger.error(f"Error registrando evento de auditoría: {e}")
    
    async def _save_assessment(self, assessment: ComplianceAssessment):
        """Guarda evaluación de cumplimiento"""
        try:
            # Crear directorio de datos
            data_dir = Path("data")
            data_dir.mkdir(exist_ok=True)
            
            # Convertir a formato serializable
            assessment_data = {
                "id": assessment.id,
                "framework": assessment.framework.value,
                "assessment_date": assessment.assessment_date.isoformat(),
                "status": assessment.status.value,
                "score": assessment.score,
                "findings": assessment.findings,
                "recommendations": assessment.recommendations,
                "next_review_date": assessment.next_review_date.isoformat() if assessment.next_review_date else None,
                "assessor": assessment.assessor
            }
            
            # Cargar evaluaciones existentes
            assessments_file = data_dir / "governance_assessments.json"
            assessments = []
            if assessments_file.exists():
                with open(assessments_file, 'r', encoding='utf-8') as f:
                    assessments = json.load(f)
            
            # Agregar nueva evaluación
            assessments.append(assessment_data)
            
            # Mantener solo las últimas 100 evaluaciones
            assessments = assessments[-100:]
            
            # Guardar archivo
            with open(assessments_file, 'w', encoding='utf-8') as f:
                json.dump(assessments, f, indent=2, ensure_ascii=False, default=str)
            
        except Exception as e:
            logger.error(f"Error guardando evaluación de cumplimiento: {e}")
    
    async def get_governance_dashboard_data(self) -> Dict[str, Any]:
        """Obtiene datos para dashboard de gobernanza"""
        try:
            # Estadísticas de políticas
            total_policies = len(self.policies)
            active_policies = len([p for p in self.policies.values() if p.is_active])
            
            # Estadísticas de cumplimiento
            total_assessments = len(self.compliance_assessments)
            recent_assessments = [
                a for a in self.compliance_assessments.values()
                if a.assessment_date >= datetime.now() - timedelta(days=30)
            ]
            
            avg_compliance_score = np.mean([a.score for a in self.compliance_assessments.values()]) if self.compliance_assessments else 0.0
            
            # Estadísticas de riesgos
            total_risks = len(self.risk_assessments)
            critical_risks = len([r for r in self.risk_assessments.values() if r.risk_level == "critical"])
            high_risks = len([r for r in self.risk_assessments.values() if r.risk_level == "high"])
            
            # Distribución por marco de cumplimiento
            framework_distribution = {}
            for assessment in self.compliance_assessments.values():
                framework = assessment.framework.value
                framework_distribution[framework] = framework_distribution.get(framework, 0) + 1
            
            return {
                "policies": {
                    "total": total_policies,
                    "active": active_policies,
                    "inactive": total_policies - active_policies
                },
                "compliance": {
                    "total_assessments": total_assessments,
                    "recent_assessments": len(recent_assessments),
                    "average_score": avg_compliance_score,
                    "framework_distribution": framework_distribution
                },
                "risks": {
                    "total_risks": total_risks,
                    "critical_risks": critical_risks,
                    "high_risks": high_risks,
                    "risk_levels": {
                        "critical": critical_risks,
                        "high": high_risks,
                        "medium": len([r for r in self.risk_assessments.values() if r.risk_level == "medium"]),
                        "low": len([r for r in self.risk_assessments.values() if r.risk_level == "low"])
                    }
                },
                "audit_trails": {
                    "total_events": len(self.audit_trails),
                    "recent_events": len([e for e in self.audit_trails if datetime.fromisoformat(e["timestamp"]) >= datetime.now() - timedelta(days=7)])
                },
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo datos del dashboard de gobernanza: {e}")
            return {"error": str(e)}
    
    async def get_policy_by_id(self, policy_id: str) -> Optional[Policy]:
        """Obtiene política por ID"""
        try:
            return self.policies.get(policy_id)
        except Exception as e:
            logger.error(f"Error obteniendo política: {e}")
            return None
    
    async def get_assessment_by_id(self, assessment_id: str) -> Optional[ComplianceAssessment]:
        """Obtiene evaluación por ID"""
        try:
            return self.compliance_assessments.get(assessment_id)
        except Exception as e:
            logger.error(f"Error obteniendo evaluación: {e}")
            return None
    
    async def get_risk_by_id(self, risk_id: str) -> Optional[RiskAssessment]:
        """Obtiene riesgo por ID"""
        try:
            return self.risk_assessments.get(risk_id)
        except Exception as e:
            logger.error(f"Error obteniendo riesgo: {e}")
            return None
    
    async def get_audit_trail(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Obtiene rastro de auditoría"""
        try:
            return self.audit_trails[-limit:] if self.audit_trails else []
        except Exception as e:
            logger.error(f"Error obteniendo rastro de auditoría: {e}")
            return []


