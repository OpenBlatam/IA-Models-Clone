"""
Predictive Analysis Service - Análisis predictivo y ML
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from ..core.models import StoreType, DesignStyle

logger = logging.getLogger(__name__)


class PredictiveAnalysisService:
    """Servicio para análisis predictivo"""
    
    def predict_success_probability(
        self,
        design: StoreDesign,
        historical_data: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Predecir probabilidad de éxito"""
        
        factors = self._analyze_success_factors(design)
        score = self._calculate_success_score(factors)
        probability = self._calculate_probability(score)
        
        return {
            "success_probability": probability,
            "success_score": score,
            "factors": factors,
            "key_indicators": self._identify_key_indicators(design),
            "recommendations": self._generate_success_recommendations(probability, factors)
        }
    
    def _analyze_success_factors(self, design: StoreDesign) -> Dict[str, Any]:
        """Analizar factores de éxito"""
        factors = {
            "financial_viability": 0,
            "market_positioning": 0,
            "location_potential": 0,
            "design_quality": 0,
            "marketing_strength": 0
        }
        
        # Factor financiero
        if design.financial_analysis:
            monthly_profit = design.financial_analysis.get("profitability", {}).get("monthly_profit", 0)
            if monthly_profit > 5000:
                factors["financial_viability"] = 10
            elif monthly_profit > 2000:
                factors["financial_viability"] = 7
            elif monthly_profit > 0:
                factors["financial_viability"] = 5
            else:
                factors["financial_viability"] = 2
        
        # Factor de posicionamiento
        if design.competitor_analysis:
            opportunities = len(design.competitor_analysis.get("opportunities", []))
            threats = len(design.competitor_analysis.get("threats", []))
            
            if opportunities > threats * 1.5:
                factors["market_positioning"] = 9
            elif opportunities > threats:
                factors["market_positioning"] = 7
            else:
                factors["market_positioning"] = 4
        
        # Factor de diseño
        if design.decoration_plan and design.layout:
            zones_count = len(design.layout.zones)
            furniture_count = len(design.decoration_plan.furniture_recommendations)
            
            if zones_count >= 3 and furniture_count >= 4:
                factors["design_quality"] = 8
            elif zones_count >= 2 and furniture_count >= 3:
                factors["design_quality"] = 6
            else:
                factors["design_quality"] = 4
        
        # Factor de marketing
        if design.marketing_plan:
            strategies_count = len(design.marketing_plan.marketing_strategy)
            tactics_count = len(design.marketing_plan.sales_tactics)
            
            if strategies_count >= 5 and tactics_count >= 5:
                factors["marketing_strength"] = 9
            elif strategies_count >= 3 and tactics_count >= 3:
                factors["marketing_strength"] = 6
            else:
                factors["marketing_strength"] = 4
        
        # Factor de ubicación (asumir medio si no hay análisis)
        factors["location_potential"] = 6
        
        return factors
    
    def _calculate_success_score(self, factors: Dict[str, Any]) -> float:
        """Calcular score de éxito"""
        weights = {
            "financial_viability": 0.30,
            "market_positioning": 0.25,
            "location_potential": 0.20,
            "design_quality": 0.15,
            "marketing_strength": 0.10
        }
        
        score = sum(factors[key] * weights[key] for key in weights.keys())
        return round(score, 2)
    
    def _calculate_probability(self, score: float) -> float:
        """Calcular probabilidad de éxito"""
        # Mapear score (0-10) a probabilidad (0-100%)
        probability = (score / 10) * 100
        return round(probability, 1)
    
    def _identify_key_indicators(self, design: StoreDesign) -> List[Dict[str, Any]]:
        """Identificar indicadores clave"""
        indicators = []
        
        if design.financial_analysis:
            monthly_profit = design.financial_analysis.get("profitability", {}).get("monthly_profit", 0)
            indicators.append({
                "indicator": "Rentabilidad Mensual",
                "value": f"${monthly_profit:,.0f}",
                "status": "positive" if monthly_profit > 0 else "negative"
            })
            
            break_even = design.financial_analysis.get("break_even", {}).get("months")
            if break_even:
                indicators.append({
                    "indicator": "Punto de Equilibrio",
                    "value": f"{break_even} meses",
                    "status": "positive" if break_even < 12 else "warning" if break_even < 24 else "negative"
                })
        
        if design.competitor_analysis:
            opportunities = len(design.competitor_analysis.get("opportunities", []))
            indicators.append({
                "indicator": "Oportunidades de Mercado",
                "value": f"{opportunities} identificadas",
                "status": "positive" if opportunities >= 3 else "neutral"
            })
        
        return indicators
    
    def _generate_success_recommendations(
        self,
        probability: float,
        factors: Dict[str, Any]
    ) -> List[str]:
        """Generar recomendaciones para mejorar éxito"""
        recommendations = []
        
        if probability < 50:
            recommendations.append("Probabilidad de éxito baja - Revisar estrategia completa")
            recommendations.append("Considerar reducir costos o aumentar proyecciones de ingresos")
        
        if factors["financial_viability"] < 5:
            recommendations.append("Mejorar viabilidad financiera - Revisar costos y precios")
        
        if factors["market_positioning"] < 5:
            recommendations.append("Fortalecer posicionamiento - Desarrollar diferenciación clara")
        
        if factors["marketing_strength"] < 5:
            recommendations.append("Reforzar estrategia de marketing - Más tácticas y canales")
        
        if probability >= 70:
            recommendations.append("Alta probabilidad de éxito - Proceder con confianza")
            recommendations.append("Mantener enfoque en ejecución y calidad")
        
        return recommendations
    
    def predict_revenue(
        self,
        design: StoreDesign,
        months: int = 12
    ) -> Dict[str, Any]:
        """Predecir ingresos futuros"""
        
        if not design.financial_analysis:
            return {"error": "Análisis financiero no disponible"}
        
        base_revenue = design.financial_analysis.get("revenue_estimate", {}).get("monthly", 0)
        
        # Proyección con crecimiento
        projection = []
        cumulative_revenue = 0
        
        for month in range(1, months + 1):
            # Crecimiento gradual en primeros meses, luego estabilización
            if month <= 3:
                multiplier = 0.6 + (month * 0.15)  # 60%, 75%, 90%
            elif month <= 6:
                multiplier = 0.9 + ((month - 3) * 0.03)  # 90% a 100%
            else:
                # Crecimiento del 2% mensual después del mes 6
                multiplier = 1.0 + ((month - 6) * 0.02)
            
            month_revenue = base_revenue * multiplier
            cumulative_revenue += month_revenue
            
            projection.append({
                "month": month,
                "revenue": round(month_revenue, 2),
                "cumulative": round(cumulative_revenue, 2)
            })
        
        return {
            "base_monthly_revenue": base_revenue,
            "projection_months": months,
            "projection": projection,
            "total_projected_revenue": round(cumulative_revenue, 2),
            "average_monthly": round(cumulative_revenue / months, 2)
        }
    
    def predict_customer_traffic(
        self,
        design: StoreDesign,
        location_score: Optional[float] = None
    ) -> Dict[str, Any]:
        """Predecir tráfico de clientes"""
        
        # Score de ubicación (1-10)
        location = location_score or 7
        
        # Factor de tipo de tienda
        store_type_factors = {
            StoreType.RESTAURANT: 1.2,
            StoreType.CAFE: 1.5,
            StoreType.BOUTIQUE: 0.8,
            StoreType.RETAIL: 1.0
        }
        
        factor = store_type_factors.get(design.store_type, 1.0)
        
        # Cálculo base de tráfico diario
        base_daily = location * 50 * factor
        
        # Ajuste por marketing
        if design.marketing_plan:
            marketing_boost = len(design.marketing_plan.marketing_strategy) * 0.1
            base_daily *= (1 + marketing_boost)
        
        return {
            "estimated_daily_customers": round(base_daily),
            "estimated_weekly_customers": round(base_daily * 7),
            "estimated_monthly_customers": round(base_daily * 30),
            "peak_hours": "12:00-14:00, 18:00-20:00",
            "factors": {
                "location_score": location,
                "store_type_factor": factor,
                "marketing_boost": marketing_boost if design.marketing_plan else 0
            }
        }




