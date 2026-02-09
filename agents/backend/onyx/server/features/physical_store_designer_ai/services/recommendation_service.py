"""
Recommendation Service - Recomendaciones inteligentes
"""

import logging
from typing import Dict, Any, Optional, List
from ..core.models import StoreType, DesignStyle, StoreDesign

logger = logging.getLogger(__name__)


class RecommendationService:
    """Servicio para generar recomendaciones inteligentes"""
    
    def __init__(self, llm_service: Optional[Any] = None):
        self.llm_service = llm_service
    
    def generate_recommendations(
        self,
        design: StoreDesign,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generar recomendaciones inteligentes para un diseño"""
        
        recommendations = {
            "immediate_actions": self._get_immediate_actions(design),
            "optimization_suggestions": self._get_optimization_suggestions(design),
            "risk_alerts": self._get_risk_alerts(design),
            "opportunities": self._get_opportunities(design),
            "best_practices": self._get_best_practices(design.store_type, design.style),
            "next_steps": self._get_next_steps(design)
        }
        
        return recommendations
    
    def _get_immediate_actions(self, design: StoreDesign) -> List[Dict[str, Any]]:
        """Obtener acciones inmediatas"""
        actions = []
        
        # Verificar información faltante
        if not design.financial_analysis:
            actions.append({
                "action": "Completar análisis financiero",
                "priority": "high",
                "reason": "Necesario para evaluación completa"
            })
        
        if not design.competitor_analysis:
            actions.append({
                "action": "Realizar análisis de competencia",
                "priority": "medium",
                "reason": "Importante para posicionamiento"
            })
        
        # Verificar viabilidad financiera
        if design.financial_analysis:
            monthly_profit = design.financial_analysis.get("profitability", {}).get("monthly_profit", 0)
            if monthly_profit < 0:
                actions.append({
                    "action": "Revisar estrategia de precios y costos",
                    "priority": "high",
                    "reason": "El negocio no es rentable con proyecciones actuales"
                })
        
        # Verificar presupuesto
        if design.decoration_plan:
            total_cost = sum(design.decoration_plan.budget_estimate.values())
            if total_cost > 50000:
                actions.append({
                    "action": "Considerar opciones de financiamiento",
                    "priority": "medium",
                    "reason": "Inversión inicial alta"
                })
        
        return actions
    
    def _get_optimization_suggestions(self, design: StoreDesign) -> List[Dict[str, Any]]:
        """Obtener sugerencias de optimización"""
        suggestions = []
        
        # Optimización de layout
        if design.layout:
            zones_count = len(design.layout.zones)
            if zones_count > 6:
                suggestions.append({
                    "area": "Layout",
                    "suggestion": "Considerar consolidar zonas para mejor flujo",
                    "impact": "medium"
                })
        
        # Optimización de marketing
        if design.marketing_plan:
            strategies_count = len(design.marketing_plan.marketing_strategy)
            if strategies_count < 5:
                suggestions.append({
                    "area": "Marketing",
                    "suggestion": "Agregar más estrategias de marketing",
                    "impact": "high"
                })
        
        # Optimización financiera
        if design.financial_analysis:
            break_even = design.financial_analysis.get("break_even", {}).get("months")
            if break_even and break_even > 18:
                suggestions.append({
                    "area": "Finanzas",
                    "suggestion": "Reducir costos iniciales o aumentar ingresos proyectados",
                    "impact": "high"
                })
        
        return suggestions
    
    def _get_risk_alerts(self, design: StoreDesign) -> List[Dict[str, Any]]:
        """Obtener alertas de riesgo"""
        alerts = []
        
        # Riesgo financiero
        if design.financial_analysis:
            monthly_profit = design.financial_analysis.get("profitability", {}).get("monthly_profit", 0)
            if monthly_profit < 1000:
                alerts.append({
                    "type": "financial",
                    "severity": "high",
                    "message": "Margen de ganancia muy bajo, riesgo alto de pérdidas",
                    "recommendation": "Revisar costos operativos y estrategia de precios"
                })
        
        # Riesgo de competencia
        if design.competitor_analysis:
            threats = design.competitor_analysis.get("threats", [])
            if len(threats) > 3:
                alerts.append({
                    "type": "market",
                    "severity": "medium",
                    "message": "Múltiples amenazas identificadas en el mercado",
                    "recommendation": "Desarrollar estrategias de diferenciación"
                })
        
        # Riesgo de inversión
        if design.financial_analysis:
            initial_investment = design.financial_analysis.get("initial_investment", {}).get("total", 0)
            if initial_investment > 100000:
                alerts.append({
                    "type": "investment",
                    "severity": "medium",
                    "message": "Inversión inicial alta",
                    "recommendation": "Asegurar financiamiento adecuado y plan de contingencia"
                })
        
        return alerts
    
    def _get_opportunities(self, design: StoreDesign) -> List[Dict[str, Any]]:
        """Obtener oportunidades identificadas"""
        opportunities = []
        
        if design.competitor_analysis:
            opps = design.competitor_analysis.get("opportunities", [])
            for opp in opps[:3]:  # Top 3
                opportunities.append({
                    "description": opp,
                    "potential_impact": "high",
                    "feasibility": "medium"
                })
        
        # Oportunidades basadas en tipo de tienda
        if design.store_type == StoreType.CAFE:
            opportunities.append({
                "description": "Eventos y actividades para crear comunidad",
                "potential_impact": "high",
                "feasibility": "high"
            })
        
        if design.store_type == StoreType.BOUTIQUE:
            opportunities.append({
                "description": "Colaboraciones con influencers locales",
                "potential_impact": "medium",
                "feasibility": "high"
            })
        
        return opportunities
    
    def _get_best_practices(
        self,
        store_type: StoreType,
        style: DesignStyle
    ) -> List[str]:
        """Obtener mejores prácticas"""
        practices = [
            "Realizar investigación de mercado antes de abrir",
            "Tener un plan de marketing sólido desde el inicio",
            "Mantener reservas de efectivo para primeros 6 meses",
            "Establecer relaciones con proveedores locales",
            "Invertir en experiencia del cliente"
        ]
        
        if store_type in [StoreType.RESTAURANT, StoreType.CAFE]:
            practices.append("Enfocarse en calidad de productos y servicio")
            practices.append("Crear ambiente único y memorable")
        
        if store_type in [StoreType.BOUTIQUE, StoreType.CLOTHING]:
            practices.append("Rotación frecuente de inventario")
            practices.append("Atención personalizada al cliente")
        
        return practices
    
    def _get_next_steps(self, design: StoreDesign) -> List[Dict[str, Any]]:
        """Obtener próximos pasos recomendados"""
        steps = [
            {
                "step": 1,
                "action": "Revisar y validar análisis financiero",
                "timeline": "Inmediato",
                "priority": "high"
            },
            {
                "step": 2,
                "action": "Obtener cotizaciones de proveedores",
                "timeline": "1-2 semanas",
                "priority": "high"
            },
            {
                "step": 3,
                "action": "Consultar con arquitecto/contratista",
                "timeline": "2-3 semanas",
                "priority": "high"
            },
            {
                "step": 4,
                "action": "Solicitar permisos y licencias",
                "timeline": "1 mes",
                "priority": "high"
            },
            {
                "step": 5,
                "action": "Desarrollar plan de marketing detallado",
                "timeline": "2-3 semanas",
                "priority": "medium"
            },
            {
                "step": 6,
                "action": "Buscar financiamiento si es necesario",
                "timeline": "1-2 meses",
                "priority": "high" if design.financial_analysis and design.financial_analysis.get("initial_investment", {}).get("total", 0) > 50000 else "medium"
            }
        ]
        
        return steps




