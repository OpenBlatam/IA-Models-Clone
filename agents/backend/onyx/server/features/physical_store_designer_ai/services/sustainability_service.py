"""
Sustainability Service - Sistema de sostenibilidad y huella de carbono
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from ..services.llm_service import LLMService

logger = logging.getLogger(__name__)


class SustainabilityService:
    """Servicio para sostenibilidad y huella de carbono"""
    
    def __init__(self, llm_service: Optional[LLMService] = None):
        self.llm_service = llm_service or LLMService()
        self.assessments: Dict[str, Dict[str, Any]] = {}
        self.carbon_footprints: Dict[str, Dict[str, Any]] = {}
    
    async def calculate_carbon_footprint(
        self,
        store_id: str,
        design_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calcular huella de carbono"""
        
        # Factores de emisión (simplificado)
        materials = design_data.get("decoration_plan", {}).get("materials", [])
        energy_usage = design_data.get("technical_plans", {}).get("electrical", {}).get("estimated_power", 0)
        
        # Calcular emisiones de materiales
        material_emissions = sum(
            material.get("carbon_factor", 0) * material.get("quantity", 0)
            for material in materials
        )
        
        # Calcular emisiones de energía (kg CO2 por kWh)
        energy_emissions = energy_usage * 0.5 * 365  # 0.5 kg CO2/kWh, anual
        
        total_emissions = material_emissions + energy_emissions
        
        footprint = {
            "store_id": store_id,
            "total_co2_kg": round(total_emissions, 2),
            "breakdown": {
                "materials": round(material_emissions, 2),
                "energy": round(energy_emissions, 2)
            },
            "calculated_at": datetime.now().isoformat(),
            "offset_required": round(total_emissions * 1.1, 2)  # 10% adicional para offset
        }
        
        self.carbon_footprints[store_id] = footprint
        
        return footprint
    
    async def assess_sustainability(
        self,
        store_id: str,
        design_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluar sostenibilidad del diseño"""
        
        assessment_id = f"sustain_{store_id}_{datetime.now().strftime('%Y%m%d')}"
        
        # Evaluar diferentes aspectos
        materials_score = self._assess_materials(design_data)
        energy_score = self._assess_energy(design_data)
        water_score = self._assess_water(design_data)
        waste_score = self._assess_waste(design_data)
        
        overall_score = (materials_score + energy_score + water_score + waste_score) / 4
        
        assessment = {
            "assessment_id": assessment_id,
            "store_id": store_id,
            "overall_score": round(overall_score, 2),
            "scores": {
                "materials": materials_score,
                "energy": energy_score,
                "water": water_score,
                "waste": waste_score
            },
            "rating": self._get_rating(overall_score),
            "recommendations": await self._generate_recommendations(design_data, {
                "materials": materials_score,
                "energy": energy_score,
                "water": water_score,
                "waste": waste_score
            }),
            "assessed_at": datetime.now().isoformat()
        }
        
        self.assessments[assessment_id] = assessment
        
        return assessment
    
    def _assess_materials(self, design_data: Dict[str, Any]) -> float:
        """Evaluar materiales"""
        materials = design_data.get("decoration_plan", {}).get("materials", [])
        
        if not materials:
            return 5.0  # Neutral
        
        eco_materials = sum(1 for m in materials if m.get("eco_friendly", False))
        total = len(materials)
        
        score = (eco_materials / total * 10) if total > 0 else 5.0
        return round(score, 2)
    
    def _assess_energy(self, design_data: Dict[str, Any]) -> float:
        """Evaluar energía"""
        technical = design_data.get("technical_plans", {})
        electrical = technical.get("electrical", {})
        
        # Verificar uso de energía renovable
        renewable = electrical.get("renewable_energy", False)
        led_lighting = electrical.get("led_lighting", False)
        
        score = 5.0
        if renewable:
            score += 3.0
        if led_lighting:
            score += 2.0
        
        return min(10.0, round(score, 2))
    
    def _assess_water(self, design_data: Dict[str, Any]) -> float:
        """Evaluar agua"""
        technical = design_data.get("technical_plans", {})
        plumbing = technical.get("plumbing", {})
        
        water_efficient = plumbing.get("water_efficient_fixtures", False)
        recycling = plumbing.get("water_recycling", False)
        
        score = 5.0
        if water_efficient:
            score += 3.0
        if recycling:
            score += 2.0
        
        return min(10.0, round(score, 2))
    
    def _assess_waste(self, design_data: Dict[str, Any]) -> float:
        """Evaluar residuos"""
        waste_management = design_data.get("waste_management", {})
        recycling = waste_management.get("recycling_program", False)
        composting = waste_management.get("composting", False)
        
        score = 5.0
        if recycling:
            score += 2.5
        if composting:
            score += 2.5
        
        return min(10.0, round(score, 2))
    
    def _get_rating(self, score: float) -> str:
        """Obtener rating"""
        if score >= 8.0:
            return "excellent"
        elif score >= 6.0:
            return "good"
        elif score >= 4.0:
            return "fair"
        else:
            return "needs_improvement"
    
    async def _generate_recommendations(
        self,
        design_data: Dict[str, Any],
        scores: Dict[str, float]
    ) -> List[str]:
        """Generar recomendaciones"""
        recommendations = []
        
        if scores["materials"] < 6.0:
            recommendations.append("Usar más materiales eco-friendly y reciclados")
        
        if scores["energy"] < 6.0:
            recommendations.append("Considerar energía renovable y LED lighting")
        
        if scores["water"] < 6.0:
            recommendations.append("Instalar fixtures de bajo consumo de agua")
        
        if scores["waste"] < 6.0:
            recommendations.append("Implementar programa de reciclaje y compostaje")
        
        if self.llm_service.client:
            try:
                llm_recommendations = await self._get_llm_recommendations(design_data, scores)
                recommendations.extend(llm_recommendations)
            except Exception as e:
                logger.error(f"Error obteniendo recomendaciones LLM: {e}")
        
        return recommendations[:10]  # Top 10
    
    async def _get_llm_recommendations(
        self,
        design_data: Dict[str, Any],
        scores: Dict[str, float]
    ) -> List[str]:
        """Obtener recomendaciones usando LLM"""
        prompt = f"""Basado en estos scores de sostenibilidad:
- Materials: {scores['materials']}/10
- Energy: {scores['energy']}/10
- Water: {scores['water']}/10
- Waste: {scores['waste']}/10

Genera recomendaciones específicas para mejorar la sostenibilidad del diseño."""
        
        result = await self.llm_service.generate_structured(
            prompt=prompt,
            system_prompt="Eres un experto en sostenibilidad y diseño ecológico."
        )
        
        if result and isinstance(result, dict) and "recommendations" in result:
            return result["recommendations"]
        
        return []
    
    def get_sustainability_certification(
        self,
        assessment_id: str
    ) -> Optional[Dict[str, Any]]:
        """Obtener certificación de sostenibilidad"""
        assessment = self.assessments.get(assessment_id)
        
        if not assessment:
            return None
        
        if assessment["overall_score"] < 7.0:
            return None
        
        certification = {
            "certificate_id": f"cert_sustain_{assessment_id}",
            "assessment_id": assessment_id,
            "store_id": assessment["store_id"],
            "rating": assessment["rating"],
            "score": assessment["overall_score"],
            "issued_at": datetime.now().isoformat(),
            "valid_until": (datetime.now().replace(year=datetime.now().year + 1)).isoformat(),
            "certification_type": "sustainability"
        }
        
        return certification




