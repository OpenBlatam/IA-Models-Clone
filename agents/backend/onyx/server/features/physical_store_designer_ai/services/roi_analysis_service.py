"""
ROI Analysis Service - Análisis de ROI avanzado
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from ..services.llm_service import LLMService

logger = logging.getLogger(__name__)


class ROIAnalysisService:
    """Servicio para análisis de ROI avanzado"""
    
    def __init__(self, llm_service: Optional[LLMService] = None):
        self.llm_service = llm_service or LLMService()
    
    def calculate_roi(
        self,
        initial_investment: float,
        monthly_revenue: float,
        monthly_costs: float,
        months: int = 12
    ) -> Dict[str, Any]:
        """Calcular ROI"""
        
        monthly_profit = monthly_revenue - monthly_costs
        total_profit = monthly_profit * months
        roi_percentage = (total_profit / initial_investment * 100) if initial_investment > 0 else 0
        payback_period = (initial_investment / monthly_profit) if monthly_profit > 0 else float('inf')
        
        return {
            "initial_investment": initial_investment,
            "monthly_revenue": monthly_revenue,
            "monthly_costs": monthly_costs,
            "monthly_profit": monthly_profit,
            "total_profit": total_profit,
            "roi_percentage": round(roi_percentage, 2),
            "payback_period_months": round(payback_period, 2),
            "analysis_period_months": months,
            "net_profit": total_profit - initial_investment
        }
    
    def calculate_npv(
        self,
        initial_investment: float,
        cash_flows: List[float],
        discount_rate: float = 0.1
    ) -> Dict[str, Any]:
        """Calcular NPV (Net Present Value)"""
        
        npv = -initial_investment
        
        for i, cash_flow in enumerate(cash_flows):
            period = i + 1
            discounted_value = cash_flow / ((1 + discount_rate) ** period)
            npv += discounted_value
        
        return {
            "initial_investment": initial_investment,
            "cash_flows": cash_flows,
            "discount_rate": discount_rate,
            "npv": round(npv, 2),
            "is_profitable": npv > 0,
            "periods": len(cash_flows)
        }
    
    def calculate_irr(
        self,
        initial_investment: float,
        cash_flows: List[float],
        max_iterations: int = 100
    ) -> Dict[str, Any]:
        """Calcular IRR (Internal Rate of Return)"""
        
        # Método de aproximación
        low_rate = -0.99
        high_rate = 10.0
        tolerance = 0.0001
        
        for _ in range(max_iterations):
            mid_rate = (low_rate + high_rate) / 2
            npv = -initial_investment
            
            for i, cash_flow in enumerate(cash_flows):
                period = i + 1
                npv += cash_flow / ((1 + mid_rate) ** period)
            
            if abs(npv) < tolerance:
                irr = mid_rate
                break
            elif npv > 0:
                low_rate = mid_rate
            else:
                high_rate = mid_rate
        else:
            irr = (low_rate + high_rate) / 2
        
        return {
            "initial_investment": initial_investment,
            "cash_flows": cash_flows,
            "irr": round(irr * 100, 2),  # Como porcentaje
            "is_acceptable": irr > 0.1,  # 10% mínimo
            "periods": len(cash_flows)
        }
    
    async def generate_roi_report(
        self,
        store_id: str,
        financial_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generar reporte completo de ROI"""
        
        initial_investment = financial_data.get("initial_investment", {}).get("total", 0)
        monthly_revenue = financial_data.get("projected_revenue", {}).get("monthly", 0)
        monthly_costs = financial_data.get("operating_costs", {}).get("monthly", 0)
        
        # Calcular métricas básicas
        roi = self.calculate_roi(initial_investment, monthly_revenue, monthly_costs)
        
        # Calcular NPV (12 meses)
        cash_flows = [monthly_revenue - monthly_costs] * 12
        npv = self.calculate_npv(initial_investment, cash_flows)
        
        # Calcular IRR
        irr = self.calculate_irr(initial_investment, cash_flows)
        
        # Generar análisis con LLM
        analysis = await self._generate_roi_analysis(roi, npv, irr)
        
        return {
            "store_id": store_id,
            "roi": roi,
            "npv": npv,
            "irr": irr,
            "analysis": analysis,
            "generated_at": datetime.now().isoformat(),
            "recommendations": self._generate_recommendations(roi, npv, irr)
        }
    
    async def _generate_roi_analysis(
        self,
        roi: Dict[str, Any],
        npv: Dict[str, Any],
        irr: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generar análisis de ROI usando LLM"""
        
        if not self.llm_service.client:
            return self._generate_basic_analysis(roi, npv, irr)
        
        try:
            prompt = f"""Analiza estos resultados financieros y proporciona un análisis profesional:
- ROI: {roi['roi_percentage']}%
- Payback Period: {roi['payback_period_months']} meses
- NPV: ${npv['npv']}
- IRR: {irr['irr']}%

Proporciona:
1. Evaluación general de viabilidad
2. Fortalezas del proyecto
3. Riesgos identificados
4. Recomendaciones"""
            
            result = await self.llm_service.generate_structured(
                prompt=prompt,
                system_prompt="Eres un analista financiero experto en evaluación de proyectos."
            )
            
            return result if result else self._generate_basic_analysis(roi, npv, irr)
        except Exception as e:
            logger.error(f"Error generando análisis ROI: {e}")
            return self._generate_basic_analysis(roi, npv, irr)
    
    def _generate_basic_analysis(
        self,
        roi: Dict[str, Any],
        npv: Dict[str, Any],
        irr: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generar análisis básico"""
        return {
            "viability": "viable" if roi["roi_percentage"] > 0 else "not_viable",
            "strengths": [
                f"ROI positivo: {roi['roi_percentage']}%",
                f"Payback en {roi['payback_period_months']} meses"
            ] if roi["roi_percentage"] > 0 else [],
            "risks": [
                "Depende de proyecciones de ingresos",
                "Costos operativos pueden variar"
            ]
        }
    
    def _generate_recommendations(
        self,
        roi: Dict[str, Any],
        npv: Dict[str, Any],
        irr: Dict[str, Any]
    ) -> List[str]:
        """Generar recomendaciones"""
        recommendations = []
        
        if roi["roi_percentage"] < 10:
            recommendations.append("Considerar optimizar costos o aumentar ingresos")
        
        if roi["payback_period_months"] > 24:
            recommendations.append("Payback period largo - considerar financiamiento")
        
        if npv["npv"] < 0:
            recommendations.append("NPV negativo - revisar proyecciones")
        
        if irr["irr"] < 10:
            recommendations.append("IRR bajo - considerar alternativas de inversión")
        
        if not recommendations:
            recommendations.append("Proyecto financieramente viable")
        
        return recommendations
    
    def compare_scenarios(
        self,
        scenarios: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Comparar múltiples escenarios"""
        
        comparisons = []
        
        for scenario in scenarios:
            roi = self.calculate_roi(
                scenario["initial_investment"],
                scenario["monthly_revenue"],
                scenario["monthly_costs"]
            )
            comparisons.append({
                "scenario_name": scenario.get("name", "Scenario"),
                "roi": roi
            })
        
        # Encontrar mejor escenario
        best_scenario = max(comparisons, key=lambda x: x["roi"]["roi_percentage"])
        
        return {
            "scenarios": comparisons,
            "best_scenario": best_scenario,
            "comparison_date": datetime.now().isoformat()
        }




