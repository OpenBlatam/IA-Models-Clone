"""
Feasibility Analyzer - Análisis de viabilidad de prototipos
===========================================================
"""

import logging
from typing import Dict, List, Any, Optional
from ..models.schemas import PrototypeResponse, Material, ProductType

logger = logging.getLogger(__name__)


class FeasibilityAnalyzer:
    """Analizador de viabilidad de prototipos"""
    
    def __init__(self):
        self.risk_factors = self._load_risk_factors()
    
    def _load_risk_factors(self) -> Dict[str, Any]:
        """Carga factores de riesgo por tipo de producto"""
        return {
            "licuadora": {
                "safety_concerns": ["Conexión eléctrica", "Cuchillas expuestas", "Sobrecarga del motor"],
                "technical_challenges": ["Alineación del motor", "Sellado del vaso", "Balance de cuchillas"],
                "regulatory": ["Certificación eléctrica", "Normas de seguridad"]
            },
            "estufa": {
                "safety_concerns": ["Fugas de gas", "Llama descontrolada", "Quemaduras"],
                "technical_challenges": ["Sellado de válvulas", "Regulación de flujo", "Distribución de calor"],
                "regulatory": ["Certificación de gas", "Normas de construcción", "Inspección profesional requerida"]
            },
            "maquina": {
                "safety_concerns": ["Partes móviles", "Cortes", "Proyección de material"],
                "technical_challenges": ["Precisión", "Vibraciones", "Desgaste"],
                "regulatory": ["Certificación de seguridad", "Normas industriales"]
            }
        }
    
    def analyze_feasibility(self, response: PrototypeResponse, 
                           user_experience: Optional[str] = None,
                           available_tools: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analiza la viabilidad de un prototipo
        
        Args:
            response: Respuesta del prototipo generado
            user_experience: Nivel de experiencia del usuario (principiante, intermedio, avanzado)
            available_tools: Lista de herramientas disponibles
            
        Returns:
            Análisis de viabilidad completo
        """
        product_type = response.specifications.get("tipo", "otro")
        difficulty = response.difficulty_level
        
        # Análisis de complejidad
        complexity_score = self._calculate_complexity_score(response)
        
        # Análisis de costo
        cost_analysis = self._analyze_costs(response)
        
        # Análisis de tiempo
        time_analysis = self._analyze_time(response)
        
        # Análisis de materiales
        material_analysis = self._analyze_materials(response.materials)
        
        # Análisis de seguridad
        safety_analysis = self._analyze_safety(product_type, response)
        
        # Análisis de viabilidad técnica
        technical_feasibility = self._analyze_technical_feasibility(
            response, user_experience, available_tools
        )
        
        # Score general de viabilidad
        feasibility_score = self._calculate_feasibility_score(
            complexity_score, cost_analysis, time_analysis, 
            material_analysis, safety_analysis, technical_feasibility
        )
        
        # Recomendaciones
        recommendations = self._generate_recommendations(
            feasibility_score, complexity_score, cost_analysis,
            safety_analysis, user_experience
        )
        
        return {
            "feasibility_score": feasibility_score,
            "feasibility_level": self._get_feasibility_level(feasibility_score),
            "complexity": {
                "score": complexity_score,
                "level": self._get_complexity_level(complexity_score),
                "factors": {
                    "num_parts": len(response.cad_parts),
                    "num_materials": len(response.materials),
                    "difficulty": difficulty
                }
            },
            "cost_analysis": cost_analysis,
            "time_analysis": time_analysis,
            "material_analysis": material_analysis,
            "safety_analysis": safety_analysis,
            "technical_feasibility": technical_feasibility,
            "recommendations": recommendations,
            "risk_factors": self.risk_factors.get(product_type, {}),
            "overall_assessment": self._get_overall_assessment(feasibility_score)
        }
    
    def _calculate_complexity_score(self, response: PrototypeResponse) -> float:
        """Calcula el score de complejidad (0-100)"""
        score = 0
        
        # Factor: número de partes
        score += min(len(response.cad_parts) * 5, 30)
        
        # Factor: número de materiales
        score += min(len(response.materials) * 3, 25)
        
        # Factor: dificultad
        difficulty_multiplier = {
            "Fácil": 1.0,
            "Media": 1.5,
            "Difícil": 2.0
        }
        score *= difficulty_multiplier.get(response.difficulty_level, 1.5)
        
        # Factor: número de pasos de ensamblaje
        score += min(len(response.assembly_instructions) * 2, 20)
        
        return min(score, 100)
    
    def _analyze_costs(self, response: PrototypeResponse) -> Dict[str, Any]:
        """Analiza los costos del prototipo"""
        total_cost = response.total_cost_estimate
        material_costs = [m.total_price for m in response.materials]
        
        return {
            "total_cost": total_cost,
            "average_material_cost": sum(material_costs) / len(material_costs) if material_costs else 0,
            "max_material_cost": max(material_costs) if material_costs else 0,
            "cost_distribution": {
                "low": sum(c for c in material_costs if c < 20),
                "medium": sum(c for c in material_costs if 20 <= c < 50),
                "high": sum(c for c in material_costs if c >= 50)
            },
            "cost_rating": "bajo" if total_cost < 100 else "medio" if total_cost < 300 else "alto"
        }
    
    def _analyze_time(self, response: PrototypeResponse) -> Dict[str, Any]:
        """Analiza el tiempo estimado"""
        build_time = response.estimated_build_time
        
        # Extraer horas del tiempo estimado
        hours = 0
        if "hora" in build_time.lower():
            try:
                hours = int(build_time.split("-")[0].split()[0])
            except:
                hours = 3
        
        return {
            "estimated_time": build_time,
            "estimated_hours": hours,
            "time_rating": "rápido" if hours < 3 else "moderado" if hours < 6 else "lento",
            "complexity_time_factor": "simple" if hours < 3 else "moderado" if hours < 6 else "complejo"
        }
    
    def _analyze_materials(self, materials: List[Material]) -> Dict[str, Any]:
        """Analiza la disponibilidad y complejidad de materiales"""
        total_sources = sum(len(m.sources) for m in materials)
        avg_sources = total_sources / len(materials) if materials else 0
        
        # Categorizar materiales por disponibilidad
        high_availability = sum(1 for m in materials if len(m.sources) >= 3)
        medium_availability = sum(1 for m in materials if 1 <= len(m.sources) < 3)
        low_availability = sum(1 for m in materials if len(m.sources) < 1)
        
        return {
            "total_materials": len(materials),
            "average_sources_per_material": avg_sources,
            "availability_rating": "alta" if avg_sources >= 3 else "media" if avg_sources >= 1 else "baja",
            "availability_distribution": {
                "high": high_availability,
                "medium": medium_availability,
                "low": low_availability
            }
        }
    
    def _analyze_safety(self, product_type: str, response: PrototypeResponse) -> Dict[str, Any]:
        """Analiza aspectos de seguridad"""
        risk_factors = self.risk_factors.get(product_type, {})
        
        safety_concerns = risk_factors.get("safety_concerns", [])
        regulatory_requirements = risk_factors.get("regulatory", [])
        
        # Determinar nivel de riesgo
        risk_level = "bajo"
        if len(safety_concerns) >= 3:
            risk_level = "alto"
        elif len(safety_concerns) >= 1:
            risk_level = "medio"
        
        return {
            "risk_level": risk_level,
            "safety_concerns": safety_concerns,
            "regulatory_requirements": regulatory_requirements,
            "requires_professional_help": len(regulatory_requirements) > 0,
            "safety_rating": "seguro" if risk_level == "bajo" else "precaución" if risk_level == "medio" else "riesgoso"
        }
    
    def _analyze_technical_feasibility(self, response: PrototypeResponse,
                                       user_experience: Optional[str],
                                       available_tools: Optional[List[str]]) -> Dict[str, Any]:
        """Analiza la viabilidad técnica según experiencia y herramientas"""
        experience_levels = {
            "principiante": 0.3,
            "intermedio": 0.6,
            "avanzado": 0.9
        }
        
        experience_score = experience_levels.get(user_experience or "intermedio", 0.5)
        
        # Analizar herramientas necesarias
        all_tools = []
        for step in response.assembly_instructions:
            all_tools.extend(step.tools_needed)
        
        required_tools = list(set(all_tools))
        available_tools_list = available_tools or []
        
        tools_available = sum(1 for tool in required_tools if tool.lower() in [t.lower() for t in available_tools_list])
        tools_coverage = tools_available / len(required_tools) if required_tools else 1.0
        
        return {
            "user_experience": user_experience or "intermedio",
            "experience_score": experience_score,
            "required_tools": required_tools,
            "tools_available": tools_available,
            "tools_coverage": tools_coverage,
            "feasibility_for_user": "alta" if experience_score >= 0.7 and tools_coverage >= 0.8 else \
                                   "media" if experience_score >= 0.4 and tools_coverage >= 0.5 else "baja"
        }
    
    def _calculate_feasibility_score(self, complexity: float, cost_analysis: Dict,
                                    time_analysis: Dict, material_analysis: Dict,
                                    safety_analysis: Dict, technical: Dict) -> float:
        """Calcula el score general de viabilidad (0-100)"""
        score = 100
        
        # Reducir por complejidad
        score -= complexity * 0.3
        
        # Reducir por costo
        if cost_analysis["cost_rating"] == "alto":
            score -= 15
        elif cost_analysis["cost_rating"] == "medio":
            score -= 8
        
        # Reducir por tiempo
        if time_analysis["time_rating"] == "lento":
            score -= 10
        
        # Reducir por disponibilidad de materiales
        if material_analysis["availability_rating"] == "baja":
            score -= 15
        
        # Reducir por seguridad
        if safety_analysis["risk_level"] == "alto":
            score -= 20
        elif safety_analysis["risk_level"] == "medio":
            score -= 10
        
        # Reducir por viabilidad técnica
        if technical["feasibility_for_user"] == "baja":
            score -= 15
        
        return max(0, min(100, score))
    
    def _get_feasibility_level(self, score: float) -> str:
        """Obtiene el nivel de viabilidad"""
        if score >= 80:
            return "Muy Alta"
        elif score >= 60:
            return "Alta"
        elif score >= 40:
            return "Media"
        elif score >= 20:
            return "Baja"
        else:
            return "Muy Baja"
    
    def _get_complexity_level(self, score: float) -> str:
        """Obtiene el nivel de complejidad"""
        if score >= 70:
            return "Muy Alta"
        elif score >= 50:
            return "Alta"
        elif score >= 30:
            return "Media"
        else:
            return "Baja"
    
    def _generate_recommendations(self, feasibility_score: float, complexity: float,
                                cost_analysis: Dict, safety_analysis: Dict,
                                user_experience: Optional[str]) -> List[str]:
        """Genera recomendaciones basadas en el análisis"""
        recommendations = []
        
        if feasibility_score < 50:
            recommendations.append("Considera simplificar el diseño o dividir el proyecto en fases")
        
        if complexity > 70:
            recommendations.append("Este proyecto es muy complejo. Considera buscar ayuda de un experto")
        
        if cost_analysis["cost_rating"] == "alto":
            recommendations.append("El costo es alto. Considera buscar materiales alternativos más económicos")
        
        if safety_analysis["risk_level"] == "alto":
            recommendations.append("⚠️ ALTA PRIORIDAD: Este proyecto tiene riesgos de seguridad. Consulta con un profesional")
        
        if user_experience == "principiante" and complexity > 50:
            recommendations.append("Como principiante, considera empezar con proyectos más simples")
        
        if material_analysis["availability_rating"] == "baja":
            recommendations.append("Algunos materiales pueden ser difíciles de encontrar. Verifica disponibilidad antes de comenzar")
        
        if not recommendations:
            recommendations.append("El proyecto parece viable. ¡Adelante con precaución!")
        
        return recommendations
    
    def _get_overall_assessment(self, score: float) -> str:
        """Obtiene la evaluación general"""
        if score >= 80:
            return "Proyecto muy viable. Puedes proceder con confianza."
        elif score >= 60:
            return "Proyecto viable con algunas consideraciones."
        elif score >= 40:
            return "Proyecto factible pero requiere planificación cuidadosa."
        elif score >= 20:
            return "Proyecto desafiante. Considera simplificaciones o ayuda profesional."
        else:
            return "Proyecto muy desafiante. Se recomienda reconsiderar el enfoque o buscar ayuda experta."




