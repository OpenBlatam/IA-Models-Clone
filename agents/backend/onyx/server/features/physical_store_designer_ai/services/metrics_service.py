"""
Metrics Service - Métricas y KPIs para el negocio
"""

import logging
from typing import Dict, Any, List, Optional
from ..core.models import StoreType

logger = logging.getLogger(__name__)


class MetricsService:
    """Servicio para métricas y KPIs"""
    
    def generate_kpis(
        self,
        store_type: StoreType,
        financial_analysis: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generar KPIs recomendados"""
        
        base_kpis = {
            "sales": {
                "revenue_per_day": "Ingresos diarios",
                "revenue_per_month": "Ingresos mensuales",
                "average_transaction_value": "Valor promedio de transacción",
                "transactions_per_day": "Transacciones diarias"
            },
            "customers": {
                "new_customers_per_month": "Nuevos clientes mensuales",
                "returning_customers_rate": "Tasa de clientes recurrentes",
                "customer_lifetime_value": "Valor de vida del cliente",
                "customer_acquisition_cost": "Costo de adquisición de cliente"
            },
            "operations": {
                "inventory_turnover": "Rotación de inventario",
                "cost_of_goods_sold": "Costo de bienes vendidos",
                "gross_margin": "Margen bruto",
                "operating_expenses_ratio": "Ratio de gastos operativos"
            },
            "financial": {
                "break_even_point": "Punto de equilibrio",
                "cash_flow": "Flujo de caja",
                "profit_margin": "Margen de ganancia",
                "roi": "Retorno de inversión"
            }
        }
        
        # KPIs específicos por tipo de tienda
        store_specific_kpis = {
            StoreType.RESTAURANT: {
                "tables_turnover": "Rotación de mesas",
                "average_dining_time": "Tiempo promedio de comida",
                "food_cost_percentage": "Porcentaje de costo de comida",
                "labor_cost_percentage": "Porcentaje de costo de personal"
            },
            StoreType.CAFE: {
                "cups_per_day": "Tazas vendidas por día",
                "average_order_value": "Valor promedio de orden",
                "peak_hours": "Horas pico",
                "product_mix": "Mix de productos"
            },
            StoreType.BOUTIQUE: {
                "units_sold_per_day": "Unidades vendidas por día",
                "average_item_price": "Precio promedio de artículo",
                "sell_through_rate": "Tasa de venta",
                "markdown_percentage": "Porcentaje de descuentos"
            }
        }
        
        kpis = base_kpis.copy()
        if store_type in store_specific_kpis:
            kpis["store_specific"] = store_specific_kpis[store_type]
        
        # Agregar valores si hay análisis financiero
        if financial_analysis:
            kpis["calculated_values"] = {
                "break_even_point": financial_analysis.get("break_even", {}).get("months"),
                "profit_margin": financial_analysis.get("profitability", {}).get("profit_margin"),
                "monthly_revenue": financial_analysis.get("revenue_estimate", {}).get("monthly"),
                "monthly_profit": financial_analysis.get("profitability", {}).get("monthly_profit")
            }
        
        return {
            "kpis": kpis,
            "tracking_recommendations": self._generate_tracking_recommendations(store_type),
            "reporting_frequency": {
                "daily": ["Revenue", "Transactions", "Customer count"],
                "weekly": ["Inventory levels", "Marketing performance", "Staff performance"],
                "monthly": ["Financial statements", "KPI dashboard", "Competitor analysis"]
            }
        }
    
    def _generate_tracking_recommendations(self, store_type: StoreType) -> List[str]:
        """Generar recomendaciones de seguimiento"""
        recommendations = [
            "Implementa un sistema de punto de venta (POS) con análisis integrado",
            "Revisa métricas diarias para identificar tendencias rápidamente",
            "Compara resultados con proyecciones financieras mensualmente",
            "Establece alertas para métricas críticas (ej: stock bajo, ventas bajas)",
            "Realiza análisis de tendencias semanales para ajustar estrategia"
        ]
        
        if store_type in [StoreType.RESTAURANT, StoreType.CAFE]:
            recommendations.append("Monitorea productos más vendidos para optimizar menú")
            recommendations.append("Analiza horas pico para optimizar personal")
        
        if store_type in [StoreType.BOUTIQUE, StoreType.CLOTHING, StoreType.RETAIL]:
            recommendations.append("Rastrea rotación de inventario por categoría")
            recommendations.append("Monitorea tasas de conversión por producto")
        
        return recommendations
    
    def generate_dashboard_metrics(
        self,
        store_type: StoreType
    ) -> Dict[str, Any]:
        """Generar métricas para dashboard"""
        return {
            "key_metrics": {
                "today": {
                    "revenue": "Ingresos de hoy",
                    "transactions": "Transacciones de hoy",
                    "customers": "Clientes de hoy"
                },
                "this_week": {
                    "revenue": "Ingresos de la semana",
                    "growth": "Crecimiento vs semana anterior",
                    "top_products": "Productos más vendidos"
                },
                "this_month": {
                    "revenue": "Ingresos del mes",
                    "profit": "Ganancia del mes",
                    "new_customers": "Nuevos clientes"
                }
            },
            "alerts": [
                "Stock bajo en productos clave",
                "Ventas por debajo del objetivo",
                "Gastos por encima del presupuesto"
            ],
            "trends": [
                "Tendencia de ventas",
                "Tendencia de clientes",
                "Tendencia de costos"
            ]
        }

