"""
Financial Analysis Service - Análisis de viabilidad financiera
"""

import logging
from typing import Dict, Any, Optional
from ..core.models import StoreType, DecorationPlan

logger = logging.getLogger(__name__)


class FinancialAnalysisService:
    """Servicio para análisis financiero"""
    
    def generate_financial_analysis(
        self,
        store_type: StoreType,
        decoration_plan: DecorationPlan,
        location: Optional[str] = None,
        estimated_revenue: Optional[float] = None
    ) -> Dict[str, Any]:
        """Generar análisis financiero"""
        
        # Costos iniciales
        initial_costs = {
            "decoration": sum(decoration_plan.budget_estimate.values()),
            "equipment": self._estimate_equipment_cost(store_type),
            "inventory": self._estimate_initial_inventory(store_type),
            "licenses": 2000,  # Licencias y permisos
            "marketing_launch": 5000,  # Marketing de apertura
            "contingency": 10000  # Contingencia
        }
        
        total_initial_investment = sum(initial_costs.values())
        
        # Costos operativos mensuales
        monthly_costs = {
            "rent": self._estimate_rent(store_type, location),
            "utilities": 800,
            "staff": self._estimate_staff_cost(store_type),
            "inventory_replenishment": self._estimate_monthly_inventory(store_type),
            "marketing": 2000,
            "insurance": 500,
            "maintenance": 1000,
            "other": 1000
        }
        
        total_monthly_costs = sum(monthly_costs.values())
        
        # Estimación de ingresos
        if not estimated_revenue:
            estimated_revenue = self._estimate_revenue(store_type)
        
        monthly_revenue = estimated_revenue
        monthly_profit = monthly_revenue - total_monthly_costs
        
        # Análisis de punto de equilibrio
        break_even_months = total_initial_investment / monthly_profit if monthly_profit > 0 else None
        
        # Proyección de 12 meses
        projection = self._generate_12_month_projection(
            total_initial_investment,
            monthly_revenue,
            total_monthly_costs,
            monthly_profit
        )
        
        return {
            "initial_investment": {
                "breakdown": initial_costs,
                "total": total_initial_investment
            },
            "monthly_costs": {
                "breakdown": monthly_costs,
                "total": total_monthly_costs
            },
            "revenue_estimate": {
                "monthly": monthly_revenue,
                "annual": monthly_revenue * 12
            },
            "profitability": {
                "monthly_profit": monthly_profit,
                "annual_profit": monthly_profit * 12,
                "profit_margin": (monthly_profit / monthly_revenue * 100) if monthly_revenue > 0 else 0
            },
            "break_even": {
                "months": break_even_months,
                "revenue_needed_monthly": total_monthly_costs
            },
            "12_month_projection": projection,
            "recommendations": self._generate_financial_recommendations(
                total_initial_investment,
                monthly_profit,
                break_even_months
            )
        }
    
    def _estimate_equipment_cost(self, store_type: StoreType) -> float:
        """Estimar costo de equipamiento"""
        costs = {
            StoreType.RESTAURANT: 30000,
            StoreType.CAFE: 25000,
            StoreType.BOUTIQUE: 15000,
            StoreType.RETAIL: 20000,
            StoreType.SUPERMARKET: 50000,
            StoreType.PHARMACY: 30000,
            StoreType.ELECTRONICS: 40000,
            StoreType.CLOTHING: 18000,
            StoreType.FURNITURE: 25000
        }
        return costs.get(store_type, 20000)
    
    def _estimate_initial_inventory(self, store_type: StoreType) -> float:
        """Estimar inventario inicial"""
        costs = {
            StoreType.RESTAURANT: 10000,
            StoreType.CAFE: 8000,
            StoreType.BOUTIQUE: 25000,
            StoreType.RETAIL: 30000,
            StoreType.SUPERMARKET: 50000,
            StoreType.PHARMACY: 35000,
            StoreType.ELECTRONICS: 40000,
            StoreType.CLOTHING: 30000,
            StoreType.FURNITURE: 35000
        }
        return costs.get(store_type, 20000)
    
    def _estimate_rent(self, store_type: StoreType, location: Optional[str]) -> float:
        """Estimar alquiler mensual"""
        # Estimación básica, en producción usar datos reales
        base_rents = {
            StoreType.RESTAURANT: 5000,
            StoreType.CAFE: 3000,
            StoreType.BOUTIQUE: 4000,
            StoreType.RETAIL: 3500,
            StoreType.SUPERMARKET: 8000,
            StoreType.PHARMACY: 4500,
            StoreType.ELECTRONICS: 5000,
            StoreType.CLOTHING: 4000,
            StoreType.FURNITURE: 6000
        }
        return base_rents.get(store_type, 4000)
    
    def _estimate_staff_cost(self, store_type: StoreType) -> float:
        """Estimar costo de personal mensual"""
        costs = {
            StoreType.RESTAURANT: 12000,  # 3-4 empleados
            StoreType.CAFE: 6000,  # 2-3 empleados
            StoreType.BOUTIQUE: 5000,  # 2 empleados
            StoreType.RETAIL: 7000,  # 2-3 empleados
            StoreType.SUPERMARKET: 15000,  # 5-6 empleados
            StoreType.PHARMACY: 8000,  # 2-3 empleados
            StoreType.ELECTRONICS: 8000,
            StoreType.CLOTHING: 6000,
            StoreType.FURNITURE: 7000
        }
        return costs.get(store_type, 6000)
    
    def _estimate_monthly_inventory(self, store_type: StoreType) -> float:
        """Estimar reposición de inventario mensual"""
        return self._estimate_initial_inventory(store_type) * 0.3
    
    def _estimate_revenue(self, store_type: StoreType) -> float:
        """Estimar ingresos mensuales"""
        revenues = {
            StoreType.RESTAURANT: 25000,
            StoreType.CAFE: 15000,
            StoreType.BOUTIQUE: 20000,
            StoreType.RETAIL: 30000,
            StoreType.SUPERMARKET: 80000,
            StoreType.PHARMACY: 40000,
            StoreType.ELECTRONICS: 50000,
            StoreType.CLOTHING: 25000,
            StoreType.FURNITURE: 35000
        }
        return revenues.get(store_type, 20000)
    
    def _generate_12_month_projection(
        self,
        initial_investment: float,
        monthly_revenue: float,
        monthly_costs: float,
        monthly_profit: float
    ) -> List[Dict[str, Any]]:
        """Generar proyección de 12 meses"""
        projection = []
        cumulative_profit = -initial_investment
        
        for month in range(1, 13):
            # Crecimiento gradual en primeros meses
            if month <= 3:
                revenue_multiplier = 0.6 + (month * 0.15)  # 60%, 75%, 90%
            elif month <= 6:
                revenue_multiplier = 0.9 + ((month - 3) * 0.03)  # 90% a 100%
            else:
                revenue_multiplier = 1.0 + ((month - 6) * 0.02)  # Crecimiento del 2% mensual
            
            month_revenue = monthly_revenue * revenue_multiplier
            month_profit = month_revenue - monthly_costs
            cumulative_profit += month_profit
            
            projection.append({
                "month": month,
                "revenue": round(month_revenue, 2),
                "costs": round(monthly_costs, 2),
                "profit": round(month_profit, 2),
                "cumulative_profit": round(cumulative_profit, 2)
            })
        
        return projection
    
    def _generate_financial_recommendations(
        self,
        initial_investment: float,
        monthly_profit: float,
        break_even_months: Optional[float]
    ) -> List[str]:
        """Generar recomendaciones financieras"""
        recommendations = []
        
        if initial_investment > 100000:
            recommendations.append("Considera buscar financiamiento o inversores para el capital inicial")
        
        if monthly_profit < 0:
            recommendations.append("Revisa costos operativos y estrategia de precios para alcanzar rentabilidad")
            recommendations.append("Considera reducir costos fijos o aumentar ingresos")
        
        if break_even_months and break_even_months > 24:
            recommendations.append("El punto de equilibrio es a largo plazo, asegura capital suficiente para 24+ meses")
        
        if monthly_profit > 0 and break_even_months and break_even_months < 12:
            recommendations.append("Excelente proyección financiera, considera reinvertir utilidades en crecimiento")
        
        recommendations.append("Mantén un fondo de emergencia de al menos 3 meses de costos operativos")
        recommendations.append("Monitorea métricas clave mensualmente y ajusta estrategia según resultados")
        
        return recommendations




