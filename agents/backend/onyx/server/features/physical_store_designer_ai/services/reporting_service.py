"""
Reporting Service - Generación de reportes avanzados
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from ..core.models import StoreDesign

logger = logging.getLogger(__name__)


class ReportingService:
    """Servicio para generar reportes avanzados"""
    
    def generate_comprehensive_report(
        self,
        design: StoreDesign
    ) -> Dict[str, Any]:
        """Generar reporte completo del diseño"""
        
        return {
            "report_id": f"report_{design.store_id}_{datetime.now().strftime('%Y%m%d')}",
            "generated_at": datetime.now().isoformat(),
            "store_info": {
                "store_id": design.store_id,
                "store_name": design.store_name,
                "store_type": design.store_type.value,
                "style": design.style.value,
                "created_at": design.created_at.isoformat() if isinstance(design.created_at, datetime) else str(design.created_at)
            },
            "executive_summary": self._generate_executive_summary(design),
            "financial_summary": self._generate_financial_summary(design),
            "design_summary": self._generate_design_summary(design),
            "marketing_summary": self._generate_marketing_summary(design),
            "risk_assessment": self._generate_risk_assessment(design),
            "recommendations": self._generate_report_recommendations(design),
            "next_steps": self._generate_report_next_steps(design),
            "appendix": self._generate_appendix(design)
        }
    
    def _generate_executive_summary(self, design: StoreDesign) -> Dict[str, Any]:
        """Generar resumen ejecutivo"""
        total_cost = 0
        monthly_profit = 0
        break_even = None
        
        if design.financial_analysis:
            total_cost = design.financial_analysis.get("initial_investment", {}).get("total", 0)
            monthly_profit = design.financial_analysis.get("profitability", {}).get("monthly_profit", 0)
            break_even = design.financial_analysis.get("break_even", {}).get("months")
        
        return {
            "overview": f"Diseño completo para {design.store_name}, una {design.store_type.value} con estilo {design.style.value}",
            "key_highlights": [
                f"Inversión inicial estimada: ${total_cost:,.0f}",
                f"Ganancia mensual proyectada: ${monthly_profit:,.0f}",
                f"Punto de equilibrio: {break_even} meses" if break_even else "Punto de equilibrio: No calculado"
            ],
            "viability": "Alta" if monthly_profit > 5000 else "Media" if monthly_profit > 0 else "Baja"
        }
    
    def _generate_financial_summary(self, design: StoreDesign) -> Dict[str, Any]:
        """Generar resumen financiero"""
        if not design.financial_analysis:
            return {"message": "Análisis financiero no disponible"}
        
        fin = design.financial_analysis
        
        return {
            "initial_investment": fin.get("initial_investment", {}),
            "monthly_costs": fin.get("monthly_costs", {}),
            "revenue_estimate": fin.get("revenue_estimate", {}),
            "profitability": fin.get("profitability", {}),
            "break_even": fin.get("break_even", {}),
            "12_month_projection": fin.get("12_month_projection", [])[:6],  # Primeros 6 meses
            "key_metrics": {
                "total_investment": fin.get("initial_investment", {}).get("total", 0),
                "monthly_profit": fin.get("profitability", {}).get("monthly_profit", 0),
                "annual_profit": fin.get("profitability", {}).get("annual_profit", 0),
                "profit_margin": fin.get("profitability", {}).get("profit_margin", 0)
            }
        }
    
    def _generate_design_summary(self, design: StoreDesign) -> Dict[str, Any]:
        """Generar resumen de diseño"""
        return {
            "layout": {
                "dimensions": design.layout.dimensions,
                "zones": len(design.layout.zones),
                "zones_detail": design.layout.zones
            },
            "decoration": {
                "color_scheme": design.decoration_plan.color_scheme,
                "furniture_count": len(design.decoration_plan.furniture_recommendations),
                "decoration_elements": len(design.decoration_plan.decoration_elements),
                "total_budget": sum(design.decoration_plan.budget_estimate.values())
            },
            "visualizations": {
                "count": len(design.visualizations),
                "types": [v.view_type for v in design.visualizations]
            }
        }
    
    def _generate_marketing_summary(self, design: StoreDesign) -> Dict[str, Any]:
        """Generar resumen de marketing"""
        if not design.marketing_plan:
            return {"message": "Plan de marketing no disponible"}
        
        mp = design.marketing_plan
        
        return {
            "target_audience": mp.target_audience,
            "strategies_count": len(mp.marketing_strategy),
            "tactics_count": len(mp.sales_tactics),
            "promotions_count": len(mp.promotion_ideas),
            "social_media_plan": mp.social_media_plan,
            "opening_strategy": mp.opening_strategy
        }
    
    def _generate_risk_assessment(self, design: StoreDesign) -> Dict[str, Any]:
        """Generar evaluación de riesgos"""
        risks = []
        
        if design.financial_analysis:
            monthly_profit = design.financial_analysis.get("profitability", {}).get("monthly_profit", 0)
            if monthly_profit < 0:
                risks.append({
                    "type": "Financial",
                    "severity": "High",
                    "description": "Proyección de pérdidas mensuales",
                    "mitigation": "Revisar costos y estrategia de precios"
                })
        
        if design.competitor_analysis:
            threats = design.competitor_analysis.get("threats", [])
            if len(threats) > 3:
                risks.append({
                    "type": "Market",
                    "severity": "Medium",
                    "description": "Múltiples amenazas competitivas",
                    "mitigation": "Desarrollar diferenciación clara"
                })
        
        return {
            "total_risks": len(risks),
            "risks": risks,
            "overall_risk_level": "High" if any(r["severity"] == "High" for r in risks) else "Medium" if risks else "Low"
        }
    
    def _generate_report_recommendations(self, design: StoreDesign) -> List[str]:
        """Generar recomendaciones para el reporte"""
        recommendations = []
        
        if design.financial_analysis:
            break_even = design.financial_analysis.get("break_even", {}).get("months")
            if break_even and break_even > 18:
                recommendations.append("Considerar reducir costos iniciales o aumentar proyecciones de ingresos")
        
        if design.competitor_analysis:
            opportunities = design.competitor_analysis.get("opportunities", [])
            if opportunities:
                recommendations.append(f"Explorar oportunidades: {', '.join(opportunities[:2])}")
        
        recommendations.extend([
            "Validar proyecciones financieras con datos reales del mercado",
            "Obtener cotizaciones de múltiples proveedores",
            "Consultar con profesionales (arquitecto, contador, abogado)",
            "Desarrollar plan de contingencia"
        ])
        
        return recommendations
    
    def _generate_report_next_steps(self, design: StoreDesign) -> List[Dict[str, Any]]:
        """Generar próximos pasos para el reporte"""
        return [
            {
                "step": 1,
                "action": "Revisar y validar análisis financiero",
                "priority": "High",
                "timeline": "1 semana"
            },
            {
                "step": 2,
                "action": "Obtener cotizaciones de proveedores",
                "priority": "High",
                "timeline": "2 semanas"
            },
            {
                "step": 3,
                "action": "Consultar con arquitecto/contratista",
                "priority": "High",
                "timeline": "2-3 semanas"
            },
            {
                "step": 4,
                "action": "Solicitar permisos y licencias",
                "priority": "High",
                "timeline": "1 mes"
            }
        ]
    
    def _generate_appendix(self, design: StoreDesign) -> Dict[str, Any]:
        """Generar apéndice del reporte"""
        return {
            "detailed_financials": design.financial_analysis,
            "competitor_details": design.competitor_analysis,
            "inventory_details": design.inventory_recommendations,
            "kpis_details": design.kpis,
            "full_layout": design.layout.dict(),
            "full_decoration_plan": design.decoration_plan.dict(),
            "full_marketing_plan": design.marketing_plan.dict()
        }
    
    def generate_pdf_report(self, design: StoreDesign) -> str:
        """Generar reporte en formato PDF (placeholder)"""
        # En producción, usar biblioteca como reportlab o weasyprint
        report = self.generate_comprehensive_report(design)
        return f"PDF report generated for {design.store_name} (ID: {report['report_id']})"
    
    def generate_excel_report(self, design: StoreDesign) -> str:
        """Generar reporte en formato Excel (placeholder)"""
        # En producción, usar biblioteca como openpyxl o pandas
        report = self.generate_comprehensive_report(design)
        return f"Excel report generated for {design.store_name} (ID: {report['report_id']})"




