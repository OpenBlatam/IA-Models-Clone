"""
Motor de Ética AI
================

Motor para evaluación ética, detección de sesgos y cumplimiento de principios de IA responsable.
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
import re

logger = logging.getLogger(__name__)

class EthicsPrinciple(str, Enum):
    """Principios éticos"""
    FAIRNESS = "fairness"
    TRANSPARENCY = "transparency"
    ACCOUNTABILITY = "accountability"
    PRIVACY = "privacy"
    NON_MALEFICENCE = "non_maleficence"
    BENEFICENCE = "beneficence"
    AUTONOMY = "autonomy"
    JUSTICE = "justice"

class BiasType(str, Enum):
    """Tipos de sesgo"""
    GENDER_BIAS = "gender_bias"
    RACIAL_BIAS = "racial_bias"
    AGE_BIAS = "age_bias"
    RELIGIOUS_BIAS = "religious_bias"
    SOCIOECONOMIC_BIAS = "socioeconomic_bias"
    CULTURAL_BIAS = "cultural_bias"
    LINGUISTIC_BIAS = "linguistic_bias"
    CONFIRMATION_BIAS = "confirmation_bias"
    SELECTION_BIAS = "selection_bias"
    ALGORITHMIC_BIAS = "algorithmic_bias"

class RiskLevel(str, Enum):
    """Niveles de riesgo"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class EthicsAssessment:
    """Evaluación ética"""
    id: str
    content: str
    principles_scores: Dict[EthicsPrinciple, float] = field(default_factory=dict)
    bias_detected: List[Dict[str, Any]] = field(default_factory=list)
    risk_level: RiskLevel = RiskLevel.LOW
    recommendations: List[str] = field(default_factory=list)
    compliance_score: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class BiasDetection:
    """Detección de sesgo"""
    bias_type: BiasType
    severity: float
    evidence: List[str]
    affected_groups: List[str]
    mitigation_suggestions: List[str]
    confidence: float

@dataclass
class EthicsReport:
    """Reporte ético"""
    assessment_id: str
    overall_score: float
    principle_scores: Dict[str, float]
    bias_summary: Dict[str, int]
    risk_assessment: Dict[str, Any]
    recommendations: List[str]
    compliance_status: str
    generated_at: datetime = field(default_factory=datetime.now)

class AIEthicsEngine:
    """Motor de ética AI"""
    
    def __init__(self):
        self.ethics_assessments: Dict[str, EthicsAssessment] = {}
        self.bias_patterns: Dict[BiasType, List[str]] = {}
        self.ethics_guidelines: Dict[EthicsPrinciple, List[str]] = {}
        self.risk_thresholds: Dict[RiskLevel, float] = {
            RiskLevel.LOW: 0.0,
            RiskLevel.MEDIUM: 0.3,
            RiskLevel.HIGH: 0.6,
            RiskLevel.CRITICAL: 0.8
        }
        
    async def initialize(self):
        """Inicializa el motor de ética"""
        logger.info("Inicializando motor de ética AI...")
        
        # Cargar patrones de sesgo
        await self._load_bias_patterns()
        
        # Cargar guías éticas
        await self._load_ethics_guidelines()
        
        # Cargar evaluaciones previas
        await self._load_previous_assessments()
        
        logger.info("Motor de ética AI inicializado")
    
    async def _load_bias_patterns(self):
        """Carga patrones de sesgo"""
        try:
            # Patrones de sesgo de género
            self.bias_patterns[BiasType.GENDER_BIAS] = [
                r'\b(?:he|him|his)\b.*\b(?:engineer|programmer|scientist|doctor|lawyer)\b',
                r'\b(?:she|her|hers)\b.*\b(?:nurse|teacher|secretary|assistant)\b',
                r'\b(?:man|men)\b.*\b(?:strong|aggressive|leader|boss)\b',
                r'\b(?:woman|women)\b.*\b(?:emotional|sensitive|caregiver|nurturing)\b',
                r'\b(?:masculine|feminine)\b.*\b(?:appropriate|inappropriate)\b'
            ]
            
            # Patrones de sesgo racial
            self.bias_patterns[BiasType.RACIAL_BIAS] = [
                r'\b(?:black|white|asian|hispanic|latino)\b.*\b(?:criminal|terrorist|illegal)\b',
                r'\b(?:african|european|asian|american)\b.*\b(?:inferior|superior|primitive|advanced)\b',
                r'\b(?:minority|majority)\b.*\b(?:problem|solution|threat|benefit)\b'
            ]
            
            # Patrones de sesgo de edad
            self.bias_patterns[BiasType.AGE_BIAS] = [
                r'\b(?:young|old|elderly|senior|junior)\b.*\b(?:inexperienced|outdated|slow|fast)\b',
                r'\b(?:millennial|boomer|gen[xyz])\b.*\b(?:lazy|entitled|stubborn|tech-savvy)\b',
                r'\b(?:age|aging)\b.*\b(?:decline|improvement|deterioration|enhancement)\b'
            ]
            
            # Patrones de sesgo religioso
            self.bias_patterns[BiasType.RELIGIOUS_BIAS] = [
                r'\b(?:christian|muslim|jewish|hindu|buddhist|atheist)\b.*\b(?:terrorist|extremist|fanatic)\b',
                r'\b(?:religion|religious)\b.*\b(?:backward|primitive|outdated|progressive)\b',
                r'\b(?:faith|belief)\b.*\b(?:irrational|logical|superstitious|scientific)\b'
            ]
            
            # Patrones de sesgo socioeconómico
            self.bias_patterns[BiasType.SOCIOECONOMIC_BIAS] = [
                r'\b(?:rich|poor|wealthy|poverty)\b.*\b(?:deserving|undeserving|lazy|hardworking)\b',
                r'\b(?:upper|lower|middle)\s+class\b.*\b(?:superior|inferior|better|worse)\b',
                r'\b(?:privilege|disadvantage)\b.*\b(?:earned|unearned|deserved|undeserved)\b'
            ]
            
            # Patrones de sesgo cultural
            self.bias_patterns[BiasType.CULTURAL_BIAS] = [
                r'\b(?:western|eastern|european|asian|african|american)\b.*\b(?:civilized|uncivilized|advanced|backward)\b',
                r'\b(?:culture|cultural)\b.*\b(?:superior|inferior|better|worse)\b',
                r'\b(?:tradition|modern)\b.*\b(?:outdated|progressive|backward|forward)\b'
            ]
            
            # Patrones de sesgo lingüístico
            self.bias_patterns[BiasType.LINGUISTIC_BIAS] = [
                r'\b(?:accent|dialect|language)\b.*\b(?:uneducated|educated|intelligent|stupid)\b',
                r'\b(?:grammar|pronunciation)\b.*\b(?:correct|incorrect|proper|improper)\b',
                r'\b(?:native|non-native)\b.*\b(?:speaker|language)\b.*\b(?:better|worse|superior|inferior)\b'
            ]
            
            logger.info(f"Cargados patrones de sesgo para {len(self.bias_patterns)} tipos")
            
        except Exception as e:
            logger.error(f"Error cargando patrones de sesgo: {e}")
    
    async def _load_ethics_guidelines(self):
        """Carga guías éticas"""
        try:
            # Principio de Equidad
            self.ethics_guidelines[EthicsPrinciple.FAIRNESS] = [
                "El sistema debe tratar a todos los usuarios de manera justa e imparcial",
                "No debe discriminar basándose en características protegidas",
                "Debe proporcionar oportunidades iguales para todos los grupos",
                "Los algoritmos deben ser probados para detectar sesgos",
                "Debe haber mecanismos de corrección para sesgos detectados"
            ]
            
            # Principio de Transparencia
            self.ethics_guidelines[EthicsPrinciple.TRANSPARENCY] = [
                "El funcionamiento del sistema debe ser explicable",
                "Los usuarios deben entender cómo se toman las decisiones",
                "Debe haber documentación clara de los procesos",
                "Los criterios de decisión deben ser públicos",
                "Debe haber canales para obtener explicaciones"
            ]
            
            # Principio de Responsabilidad
            self.ethics_guidelines[EthicsPrinciple.ACCOUNTABILITY] = [
                "Debe haber responsables claros por las decisiones del sistema",
                "Debe haber mecanismos de auditoría y supervisión",
                "Debe haber procesos de apelación y corrección",
                "Los errores deben ser documentados y corregidos",
                "Debe haber consecuencias por mal uso del sistema"
            ]
            
            # Principio de Privacidad
            self.ethics_guidelines[EthicsPrinciple.PRIVACY] = [
                "Los datos personales deben ser protegidos",
                "Debe haber consentimiento informado para el uso de datos",
                "Los datos deben ser minimizados y anonimizados cuando sea posible",
                "Debe haber controles de acceso apropiados",
                "Los usuarios deben tener control sobre sus datos"
            ]
            
            # Principio de No Maleficencia
            self.ethics_guidelines[EthicsPrinciple.NON_MALEFICENCE] = [
                "El sistema no debe causar daño a los usuarios",
                "Debe haber salvaguardas contra usos maliciosos",
                "Los riesgos deben ser identificados y mitigados",
                "Debe haber mecanismos de detección de abuso",
                "El sistema debe ser robusto contra ataques"
            ]
            
            # Principio de Beneficencia
            self.ethics_guidelines[EthicsPrinciple.BENEFICENCE] = [
                "El sistema debe beneficiar a los usuarios",
                "Debe mejorar la calidad de vida o trabajo",
                "Debe ser accesible para todos los grupos",
                "Debe promover el bienestar social",
                "Los beneficios deben superar los riesgos"
            ]
            
            # Principio de Autonomía
            self.ethics_guidelines[EthicsPrinciple.AUTONOMY] = [
                "Los usuarios deben mantener control sobre sus decisiones",
                "El sistema debe respetar la libertad de elección",
                "No debe manipular o coaccionar a los usuarios",
                "Debe haber opciones de opt-out",
                "Los usuarios deben poder modificar o eliminar sus datos"
            ]
            
            # Principio de Justicia
            self.ethics_guidelines[EthicsPrinciple.JUSTICE] = [
                "El sistema debe distribuir beneficios y cargas de manera justa",
                "No debe perpetuar desigualdades existentes",
                "Debe considerar las necesidades de grupos vulnerables",
                "Debe promover la equidad social",
                "Los recursos deben ser asignados de manera justa"
            ]
            
            logger.info(f"Cargadas guías éticas para {len(self.ethics_guidelines)} principios")
            
        except Exception as e:
            logger.error(f"Error cargando guías éticas: {e}")
            raise
    
    async def _load_previous_assessments(self):
        """Carga evaluaciones previas"""
        try:
            assessments_file = Path("data/ethics_assessments.json")
            if assessments_file.exists():
                with open(assessments_file, 'r', encoding='utf-8') as f:
                    assessments_data = json.load(f)
                
                for assessment_data in assessments_data:
                    assessment = EthicsAssessment(
                        id=assessment_data["id"],
                        content=assessment_data["content"],
                        principles_scores={
                            EthicsPrinciple(k): v for k, v in assessment_data["principles_scores"].items()
                        },
                        bias_detected=assessment_data["bias_detected"],
                        risk_level=RiskLevel(assessment_data["risk_level"]),
                        recommendations=assessment_data["recommendations"],
                        compliance_score=assessment_data["compliance_score"],
                        created_at=datetime.fromisoformat(assessment_data["created_at"])
                    )
                    self.ethics_assessments[assessment.id] = assessment
                
                logger.info(f"Cargadas {len(self.ethics_assessments)} evaluaciones previas")
            
        except Exception as e:
            logger.error(f"Error cargando evaluaciones previas: {e}")
    
    async def assess_ethics(self, content: str, context: Dict[str, Any] = None) -> str:
        """Evalúa aspectos éticos del contenido"""
        try:
            assessment_id = f"ethics_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Crear evaluación
            assessment = EthicsAssessment(
                id=assessment_id,
                content=content
            )
            
            # Evaluar principios éticos
            await self._evaluate_ethics_principles(assessment)
            
            # Detectar sesgos
            await self._detect_biases(assessment)
            
            # Calcular nivel de riesgo
            await self._calculate_risk_level(assessment)
            
            # Generar recomendaciones
            await self._generate_recommendations(assessment)
            
            # Calcular puntuación de cumplimiento
            await self._calculate_compliance_score(assessment)
            
            # Guardar evaluación
            self.ethics_assessments[assessment_id] = assessment
            await self._save_assessment(assessment)
            
            logger.info(f"Evaluación ética completada: {assessment_id}")
            return assessment_id
            
        except Exception as e:
            logger.error(f"Error evaluando ética: {e}")
            raise
    
    async def _evaluate_ethics_principles(self, assessment: EthicsAssessment):
        """Evalúa principios éticos"""
        try:
            content_lower = assessment.content.lower()
            
            # Evaluar cada principio
            for principle in EthicsPrinciple:
                score = await self._evaluate_principle(principle, content_lower, assessment.content)
                assessment.principles_scores[principle] = score
            
        except Exception as e:
            logger.error(f"Error evaluando principios éticos: {e}")
    
    async def _evaluate_principle(self, principle: EthicsPrinciple, content_lower: str, original_content: str) -> float:
        """Evalúa un principio específico"""
        try:
            score = 0.5  # Puntuación base neutral
            
            if principle == EthicsPrinciple.FAIRNESS:
                # Buscar indicadores de equidad
                fairness_indicators = [
                    "equal", "fair", "impartial", "unbiased", "equitable",
                    "justice", "rights", "opportunity", "access"
                ]
                bias_indicators = [
                    "discriminate", "bias", "unfair", "unequal", "prejudice"
                ]
                
                fairness_count = sum(1 for indicator in fairness_indicators if indicator in content_lower)
                bias_count = sum(1 for indicator in bias_indicators if indicator in content_lower)
                
                score = 0.5 + (fairness_count * 0.1) - (bias_count * 0.2)
                
            elif principle == EthicsPrinciple.TRANSPARENCY:
                # Buscar indicadores de transparencia
                transparency_indicators = [
                    "explain", "clear", "transparent", "understandable",
                    "document", "disclose", "open", "visible"
                ]
                opacity_indicators = [
                    "hidden", "secret", "confidential", "black box", "unclear"
                ]
                
                transparency_count = sum(1 for indicator in transparency_indicators if indicator in content_lower)
                opacity_count = sum(1 for indicator in opacity_indicators if indicator in content_lower)
                
                score = 0.5 + (transparency_count * 0.1) - (opacity_count * 0.2)
                
            elif principle == EthicsPrinciple.ACCOUNTABILITY:
                # Buscar indicadores de responsabilidad
                accountability_indicators = [
                    "responsible", "accountable", "liable", "answerable",
                    "oversight", "audit", "review", "monitor"
                ]
                irresponsibility_indicators = [
                    "unaccountable", "irresponsible", "unmonitored", "unchecked"
                ]
                
                accountability_count = sum(1 for indicator in accountability_indicators if indicator in content_lower)
                irresponsibility_count = sum(1 for indicator in irresponsibility_indicators if indicator in content_lower)
                
                score = 0.5 + (accountability_count * 0.1) - (irresponsibility_count * 0.2)
                
            elif principle == EthicsPrinciple.PRIVACY:
                # Buscar indicadores de privacidad
                privacy_indicators = [
                    "privacy", "confidential", "secure", "protect", "anonymize",
                    "consent", "permission", "control", "personal data"
                ]
                privacy_violation_indicators = [
                    "surveillance", "tracking", "monitoring", "intrusion", "breach"
                ]
                
                privacy_count = sum(1 for indicator in privacy_indicators if indicator in content_lower)
                violation_count = sum(1 for indicator in privacy_violation_indicators if indicator in content_lower)
                
                score = 0.5 + (privacy_count * 0.1) - (violation_count * 0.2)
                
            elif principle == EthicsPrinciple.NON_MALEFICENCE:
                # Buscar indicadores de no maleficencia
                safety_indicators = [
                    "safe", "secure", "protect", "prevent", "harmless",
                    "beneficial", "helpful", "supportive"
                ]
                harm_indicators = [
                    "harm", "damage", "hurt", "dangerous", "risky", "threat"
                ]
                
                safety_count = sum(1 for indicator in safety_indicators if indicator in content_lower)
                harm_count = sum(1 for indicator in harm_indicators if indicator in content_lower)
                
                score = 0.5 + (safety_count * 0.1) - (harm_count * 0.2)
                
            elif principle == EthicsPrinciple.BENEFICENCE:
                # Buscar indicadores de beneficencia
                benefit_indicators = [
                    "benefit", "help", "improve", "enhance", "support",
                    "positive", "good", "useful", "valuable"
                ]
                detriment_indicators = [
                    "harmful", "negative", "bad", "useless", "detrimental"
                ]
                
                benefit_count = sum(1 for indicator in benefit_indicators if indicator in content_lower)
                detriment_count = sum(1 for indicator in detriment_indicators if indicator in content_lower)
                
                score = 0.5 + (benefit_count * 0.1) - (detriment_count * 0.2)
                
            elif principle == EthicsPrinciple.AUTONOMY:
                # Buscar indicadores de autonomía
                autonomy_indicators = [
                    "choice", "freedom", "autonomy", "control", "decision",
                    "voluntary", "consent", "option", "preference"
                ]
                coercion_indicators = [
                    "force", "coerce", "manipulate", "control", "restrict"
                ]
                
                autonomy_count = sum(1 for indicator in autonomy_indicators if indicator in content_lower)
                coercion_count = sum(1 for indicator in coercion_indicators if indicator in content_lower)
                
                score = 0.5 + (autonomy_count * 0.1) - (coercion_count * 0.2)
                
            elif principle == EthicsPrinciple.JUSTICE:
                # Buscar indicadores de justicia
                justice_indicators = [
                    "just", "fair", "equal", "equitable", "right",
                    "deserve", "merit", "appropriate", "balanced"
                ]
                injustice_indicators = [
                    "unjust", "unfair", "unequal", "wrong", "inappropriate"
                ]
                
                justice_count = sum(1 for indicator in justice_indicators if indicator in content_lower)
                injustice_count = sum(1 for indicator in injustice_indicators if indicator in content_lower)
                
                score = 0.5 + (justice_count * 0.1) - (injustice_count * 0.2)
            
            # Normalizar puntuación entre 0 y 1
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"Error evaluando principio {principle}: {e}")
            return 0.5
    
    async def _detect_biases(self, assessment: EthicsAssessment):
        """Detecta sesgos en el contenido"""
        try:
            content_lower = assessment.content.lower()
            
            for bias_type, patterns in self.bias_patterns.items():
                bias_detections = []
                
                for pattern in patterns:
                    matches = re.findall(pattern, content_lower, re.IGNORECASE)
                    if matches:
                        bias_detection = BiasDetection(
                            bias_type=bias_type,
                            severity=len(matches) / len(content_lower.split()) * 100,  # Porcentaje de palabras
                            evidence=matches,
                            affected_groups=self._identify_affected_groups(bias_type, matches),
                            mitigation_suggestions=self._generate_bias_mitigation(bias_type),
                            confidence=min(0.9, len(matches) * 0.2)
                        )
                        bias_detections.append(bias_detection)
                
                if bias_detections:
                    # Combinar detecciones del mismo tipo
                    combined_bias = {
                        "bias_type": bias_type.value,
                        "severity": max(d.severity for d in bias_detections),
                        "evidence": [evidence for d in bias_detections for evidence in d.evidence],
                        "affected_groups": list(set(group for d in bias_detections for group in d.affected_groups)),
                        "mitigation_suggestions": list(set(suggestion for d in bias_detections for suggestion in d.mitigation_suggestions)),
                        "confidence": max(d.confidence for d in bias_detections)
                    }
                    assessment.bias_detected.append(combined_bias)
            
        except Exception as e:
            logger.error(f"Error detectando sesgos: {e}")
    
    def _identify_affected_groups(self, bias_type: BiasType, evidence: List[str]) -> List[str]:
        """Identifica grupos afectados por el sesgo"""
        try:
            affected_groups = []
            
            if bias_type == BiasType.GENDER_BIAS:
                affected_groups = ["women", "men", "non-binary", "transgender"]
            elif bias_type == BiasType.RACIAL_BIAS:
                affected_groups = ["racial minorities", "ethnic groups", "people of color"]
            elif bias_type == BiasType.AGE_BIAS:
                affected_groups = ["young people", "elderly", "seniors", "millennials"]
            elif bias_type == BiasType.RELIGIOUS_BIAS:
                affected_groups = ["religious minorities", "atheists", "agnostics"]
            elif bias_type == BiasType.SOCIOECONOMIC_BIAS:
                affected_groups = ["low-income", "high-income", "working class", "middle class"]
            elif bias_type == BiasType.CULTURAL_BIAS:
                affected_groups = ["cultural minorities", "immigrants", "indigenous people"]
            elif bias_type == BiasType.LINGUISTIC_BIAS:
                affected_groups = ["non-native speakers", "people with accents", "multilingual individuals"]
            else:
                affected_groups = ["vulnerable groups"]
            
            return affected_groups
            
        except Exception as e:
            logger.error(f"Error identificando grupos afectados: {e}")
            return []
    
    def _generate_bias_mitigation(self, bias_type: BiasType) -> List[str]:
        """Genera sugerencias de mitigación de sesgo"""
        try:
            mitigation_suggestions = {
                BiasType.GENDER_BIAS: [
                    "Usar lenguaje inclusivo y neutral en cuanto al género",
                    "Evitar estereotipos de género en ejemplos y descripciones",
                    "Incluir representación diversa en casos de uso",
                    "Revisar algoritmos para detectar sesgos de género"
                ],
                BiasType.RACIAL_BIAS: [
                    "Evitar referencias a características raciales o étnicas",
                    "Usar ejemplos culturalmente diversos",
                    "Implementar pruebas de sesgo racial en algoritmos",
                    "Incluir perspectivas de múltiples culturas"
                ],
                BiasType.AGE_BIAS: [
                    "Evitar estereotipos relacionados con la edad",
                    "Incluir ejemplos de diferentes grupos de edad",
                    "Considerar las necesidades de usuarios de todas las edades",
                    "Evitar lenguaje que discrimine por edad"
                ],
                BiasType.RELIGIOUS_BIAS: [
                    "Respetar todas las creencias religiosas",
                    "Evitar referencias a religiones específicas",
                    "Usar ejemplos seculares cuando sea posible",
                    "Incluir perspectivas de diferentes tradiciones religiosas"
                ],
                BiasType.SOCIOECONOMIC_BIAS: [
                    "Evitar referencias a estatus socioeconómico",
                    "Considerar la accesibilidad para diferentes niveles de ingresos",
                    "Usar ejemplos que no asuman privilegios económicos",
                    "Implementar características de accesibilidad"
                ],
                BiasType.CULTURAL_BIAS: [
                    "Incluir perspectivas culturales diversas",
                    "Evitar centrarse en una sola cultura",
                    "Usar ejemplos culturalmente inclusivos",
                    "Considerar diferencias culturales en el diseño"
                ],
                BiasType.LINGUISTIC_BIAS: [
                    "Evitar juzgar por acentos o dialectos",
                    "Proporcionar soporte para múltiples idiomas",
                    "Usar lenguaje claro y accesible",
                    "Incluir opciones de accesibilidad lingüística"
                ]
            }
            
            return mitigation_suggestions.get(bias_type, [
                "Revisar el contenido para detectar sesgos",
                "Incluir perspectivas diversas",
                "Implementar pruebas de sesgo",
                "Consultar con grupos afectados"
            ])
            
        except Exception as e:
            logger.error(f"Error generando mitigación de sesgo: {e}")
            return ["Revisar contenido para sesgos"]
    
    async def _calculate_risk_level(self, assessment: EthicsAssessment):
        """Calcula nivel de riesgo"""
        try:
            # Calcular puntuación de riesgo basada en sesgos detectados
            risk_score = 0.0
            
            for bias in assessment.bias_detected:
                risk_score += bias["severity"] * bias["confidence"]
            
            # Determinar nivel de riesgo
            if risk_score >= self.risk_thresholds[RiskLevel.CRITICAL]:
                assessment.risk_level = RiskLevel.CRITICAL
            elif risk_score >= self.risk_thresholds[RiskLevel.HIGH]:
                assessment.risk_level = RiskLevel.HIGH
            elif risk_score >= self.risk_thresholds[RiskLevel.MEDIUM]:
                assessment.risk_level = RiskLevel.MEDIUM
            else:
                assessment.risk_level = RiskLevel.LOW
            
        except Exception as e:
            logger.error(f"Error calculando nivel de riesgo: {e}")
            assessment.risk_level = RiskLevel.MEDIUM
    
    async def _generate_recommendations(self, assessment: EthicsAssessment):
        """Genera recomendaciones éticas"""
        try:
            recommendations = []
            
            # Recomendaciones basadas en principios éticos
            for principle, score in assessment.principles_scores.items():
                if score < 0.4:  # Puntuación baja
                    principle_recommendations = self.ethics_guidelines[principle]
                    recommendations.extend(principle_recommendations[:2])  # Tomar las primeras 2
            
            # Recomendaciones basadas en sesgos detectados
            for bias in assessment.bias_detected:
                recommendations.extend(bias["mitigation_suggestions"])
            
            # Recomendaciones generales basadas en nivel de riesgo
            if assessment.risk_level == RiskLevel.CRITICAL:
                recommendations.extend([
                    "Revisión urgente requerida antes del despliegue",
                    "Consultar con expertos en ética AI",
                    "Implementar salvaguardas adicionales",
                    "Considerar retrasar el lanzamiento"
                ])
            elif assessment.risk_level == RiskLevel.HIGH:
                recommendations.extend([
                    "Revisión detallada recomendada",
                    "Implementar medidas de mitigación",
                    "Monitoreo continuo requerido",
                    "Capacitación del equipo en ética AI"
                ])
            elif assessment.risk_level == RiskLevel.MEDIUM:
                recommendations.extend([
                    "Revisión periódica recomendada",
                    "Implementar controles básicos",
                    "Monitoreo regular",
                    "Actualizar políticas de ética"
                ])
            else:
                recommendations.extend([
                    "Mantener buenas prácticas actuales",
                    "Revisión rutinaria",
                    "Monitoreo básico",
                    "Continuar con el desarrollo"
                ])
            
            # Eliminar duplicados y limitar número
            assessment.recommendations = list(set(recommendations))[:10]
            
        except Exception as e:
            logger.error(f"Error generando recomendaciones: {e}")
            assessment.recommendations = ["Revisar aspectos éticos del contenido"]
    
    async def _calculate_compliance_score(self, assessment: EthicsAssessment):
        """Calcula puntuación de cumplimiento"""
        try:
            # Puntuación basada en principios éticos (70% del peso)
            principles_score = np.mean(list(assessment.principles_scores.values()))
            
            # Puntuación basada en sesgos detectados (30% del peso)
            bias_penalty = 0.0
            for bias in assessment.bias_detected:
                bias_penalty += bias["severity"] * bias["confidence"] * 0.1
            
            # Calcular puntuación final
            compliance_score = (principles_score * 0.7) - bias_penalty
            
            # Normalizar entre 0 y 1
            assessment.compliance_score = max(0.0, min(1.0, compliance_score))
            
        except Exception as e:
            logger.error(f"Error calculando puntuación de cumplimiento: {e}")
            assessment.compliance_score = 0.5
    
    async def _save_assessment(self, assessment: EthicsAssessment):
        """Guarda evaluación ética"""
        try:
            # Crear directorio de datos
            data_dir = Path("data")
            data_dir.mkdir(exist_ok=True)
            
            # Convertir a formato serializable
            assessment_data = {
                "id": assessment.id,
                "content": assessment.content,
                "principles_scores": {k.value: v for k, v in assessment.principles_scores.items()},
                "bias_detected": assessment.bias_detected,
                "risk_level": assessment.risk_level.value,
                "recommendations": assessment.recommendations,
                "compliance_score": assessment.compliance_score,
                "created_at": assessment.created_at.isoformat()
            }
            
            # Cargar evaluaciones existentes
            assessments_file = data_dir / "ethics_assessments.json"
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
            logger.error(f"Error guardando evaluación ética: {e}")
    
    async def generate_ethics_report(self, assessment_id: str) -> Optional[EthicsReport]:
        """Genera reporte ético"""
        try:
            if assessment_id not in self.ethics_assessments:
                return None
            
            assessment = self.ethics_assessments[assessment_id]
            
            # Calcular puntuación general
            overall_score = assessment.compliance_score
            
            # Convertir puntuaciones de principios
            principle_scores = {k.value: v for k, v in assessment.principles_scores.items()}
            
            # Resumen de sesgos
            bias_summary = {}
            for bias in assessment.bias_detected:
                bias_type = bias["bias_type"]
                bias_summary[bias_type] = bias_summary.get(bias_type, 0) + 1
            
            # Evaluación de riesgo
            risk_assessment = {
                "level": assessment.risk_level.value,
                "factors": len(assessment.bias_detected),
                "severity": max([b["severity"] for b in assessment.bias_detected], default=0.0)
            }
            
            # Estado de cumplimiento
            if overall_score >= 0.8:
                compliance_status = "excellent"
            elif overall_score >= 0.6:
                compliance_status = "good"
            elif overall_score >= 0.4:
                compliance_status = "fair"
            else:
                compliance_status = "poor"
            
            report = EthicsReport(
                assessment_id=assessment_id,
                overall_score=overall_score,
                principle_scores=principle_scores,
                bias_summary=bias_summary,
                risk_assessment=risk_assessment,
                recommendations=assessment.recommendations,
                compliance_status=compliance_status
            )
            
            return report
            
        except Exception as e:
            logger.error(f"Error generando reporte ético: {e}")
            return None
    
    async def get_ethics_dashboard_data(self) -> Dict[str, Any]:
        """Obtiene datos para dashboard de ética"""
        try:
            # Estadísticas generales
            total_assessments = len(self.ethics_assessments)
            
            # Distribución por nivel de riesgo
            risk_distribution = {level.value: 0 for level in RiskLevel}
            for assessment in self.ethics_assessments.values():
                risk_distribution[assessment.risk_level.value] += 1
            
            # Puntuación promedio de cumplimiento
            avg_compliance = np.mean([a.compliance_score for a in self.ethics_assessments.values()]) if self.ethics_assessments else 0.0
            
            # Distribución de sesgos
            bias_distribution = {}
            for assessment in self.ethics_assessments.values():
                for bias in assessment.bias_detected:
                    bias_type = bias["bias_type"]
                    bias_distribution[bias_type] = bias_distribution.get(bias_type, 0) + 1
            
            # Puntuaciones promedio por principio
            principle_averages = {}
            for principle in EthicsPrinciple:
                scores = [a.principles_scores.get(principle, 0.5) for a in self.ethics_assessments.values()]
                principle_averages[principle.value] = np.mean(scores) if scores else 0.5
            
            return {
                "total_assessments": total_assessments,
                "risk_distribution": risk_distribution,
                "average_compliance": avg_compliance,
                "bias_distribution": bias_distribution,
                "principle_averages": principle_averages,
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo datos del dashboard de ética: {e}")
            return {"error": str(e)}
    
    async def get_assessment_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Obtiene historial de evaluaciones"""
        try:
            assessments = list(self.ethics_assessments.values())
            assessments.sort(key=lambda x: x.created_at, reverse=True)
            
            return [
                {
                    "id": assessment.id,
                    "compliance_score": assessment.compliance_score,
                    "risk_level": assessment.risk_level.value,
                    "bias_count": len(assessment.bias_detected),
                    "created_at": assessment.created_at.isoformat()
                }
                for assessment in assessments[:limit]
            ]
            
        except Exception as e:
            logger.error(f"Error obteniendo historial de evaluaciones: {e}")
            return []


