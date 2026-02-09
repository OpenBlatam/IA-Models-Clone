"""
Optimization Service - Optimización de costos y presupuesto
"""

import logging
from typing import Dict, Any, List, Optional
from ..core.models import StoreDesign, StoreType

logger = logging.getLogger(__name__)


class OptimizationService:
    """Servicio para optimización de costos y presupuesto"""
    
    def optimize_budget(
        self,
        design: StoreDesign,
        target_budget: Optional[float] = None
    ) -> Dict[str, Any]:
        """Optimizar presupuesto del diseño"""
        
        if not design.financial_analysis:
            return {"error": "Análisis financiero no disponible"}
        
        current_budget = design.financial_analysis.get("initial_investment", {}).get("total", 0)
        
        if not target_budget:
            # Reducir 15% como objetivo por defecto
            target_budget = current_budget * 0.85
        
        optimizations = self._generate_optimizations(design, current_budget, target_budget)
        
        return {
            "current_budget": current_budget,
            "target_budget": target_budget,
            "savings": current_budget - target_budget,
            "optimizations": optimizations,
            "recommendations": self._generate_optimization_recommendations(optimizations)
        }
    
    def _generate_optimizations(
        self,
        design: StoreDesign,
        current_budget: float,
        target_budget: float
    ) -> List[Dict[str, Any]]:
        """Generar optimizaciones"""
        optimizations = []
        savings_needed = current_budget - target_budget
        
        # Optimización de decoración
        if design.decoration_plan:
            deco_budget = sum(design.decoration_plan.budget_estimate.values())
            if deco_budget > savings_needed * 0.3:
                optimizations.append({
                    "area": "Decoración",
                    "current_cost": deco_budget,
                    "suggested_cost": deco_budget * 0.85,
                    "savings": deco_budget * 0.15,
                    "suggestions": [
                        "Usar materiales más económicos",
                        "Reducir elementos decorativos no esenciales",
                        "Priorizar áreas visibles al cliente"
                    ]
                })
        
        # Optimización de equipamiento
        if design.financial_analysis:
            equipment_cost = design.financial_analysis.get("initial_investment", {}).get("equipment", 0)
            if equipment_cost > savings_needed * 0.3:
                optimizations.append({
                    "area": "Equipamiento",
                    "current_cost": equipment_cost,
                    "suggested_cost": equipment_cost * 0.90,
                    "savings": equipment_cost * 0.10,
                    "suggestions": [
                        "Considerar equipos usados o reacondicionados",
                        "Priorizar equipos esenciales",
                        "Negociar descuentos por volumen",
                        "Leasing en lugar de compra"
                    ]
                })
        
        # Optimización de inventario inicial
        if design.financial_analysis:
            inventory_cost = design.financial_analysis.get("initial_investment", {}).get("inventory", 0)
            if inventory_cost > savings_needed * 0.2:
                optimizations.append({
                    "area": "Inventario Inicial",
                    "current_cost": inventory_cost,
                    "suggested_cost": inventory_cost * 0.80,
                    "savings": inventory_cost * 0.20,
                    "suggestions": [
                        "Reducir inventario inicial",
                        "Enfocarse en productos de rotación rápida",
                        "Establecer relación con proveedores para reposición rápida"
                    ]
                })
        
        # Optimización de marketing de apertura
        if design.financial_analysis:
            marketing_cost = design.financial_analysis.get("initial_investment", {}).get("marketing_launch", 0)
            if marketing_cost > 0:
                optimizations.append({
                    "area": "Marketing de Apertura",
                    "current_cost": marketing_cost,
                    "suggested_cost": marketing_cost * 0.75,
                    "savings": marketing_cost * 0.25,
                    "suggestions": [
                        "Enfocarse en marketing digital (más económico)",
                        "Colaboraciones con influencers locales",
                        "Marketing orgánico en redes sociales",
                        "Eventos de apertura más pequeños"
                    ]
                })
        
        return optimizations
    
    def _generate_optimization_recommendations(
        self,
        optimizations: List[Dict[str, Any]]
    ) -> List[str]:
        """Generar recomendaciones de optimización"""
        recommendations = []
        
        total_savings = sum(opt.get("savings", 0) for opt in optimizations)
        
        if total_savings > 0:
            recommendations.append(f"Optimizaciones pueden ahorrar ${total_savings:,.0f}")
        
        recommendations.extend([
            "Priorizar optimizaciones de alto impacto",
            "Mantener calidad en áreas críticas (cocina, servicio al cliente)",
            "Revisar cada optimización antes de implementar",
            "Considerar impacto en experiencia del cliente"
        ])
        
        return recommendations
    
    def optimize_layout(self, design: StoreDesign) -> Dict[str, Any]:
        """Optimizar layout del local"""
        layout = design.layout
        zones = layout.zones
        
        optimizations = []
        
        # Verificar eficiencia de zonas
        total_zones = len(zones)
        if total_zones > 6:
            optimizations.append({
                "type": "consolidation",
                "suggestion": "Considerar consolidar zonas para mejor flujo",
                "impact": "medium"
            })
        
        # Verificar áreas de almacenamiento
        storage_zones = [z for z in zones if "almacén" in z.get("name", "").lower() or "storage" in z.get("name", "").lower()]
        if len(storage_zones) > 2:
            optimizations.append({
                "type": "storage",
                "suggestion": "Consolidar áreas de almacenamiento",
                "impact": "high"
            })
        
        # Verificar flujo de tráfico
        if not layout.traffic_flow or not layout.traffic_flow.get("flow"):
            optimizations.append({
                "type": "traffic",
                "suggestion": "Optimizar flujo de tráfico para maximizar exposición",
                "impact": "high"
            })
        
        return {
            "current_layout": {
                "zones_count": total_zones,
                "zones": zones
            },
            "optimizations": optimizations,
            "recommendations": [
                "Maximizar espacio de ventas",
                "Minimizar áreas no productivas",
                "Optimizar flujo de clientes y personal"
            ]
        }
    
    def optimize_marketing_budget(
        self,
        design: StoreDesign,
        monthly_marketing_budget: float
    ) -> Dict[str, Any]:
        """Optimizar presupuesto de marketing mensual"""
        
        allocation = {
            "digital_marketing": monthly_marketing_budget * 0.40,
            "social_media": monthly_marketing_budget * 0.25,
            "local_advertising": monthly_marketing_budget * 0.20,
            "events_promotions": monthly_marketing_budget * 0.15
        }
        
        strategies = {
            "digital_marketing": [
                "Google Ads con presupuesto diario",
                "Facebook/Instagram Ads",
                "Email marketing",
                "SEO local"
            ],
            "social_media": [
                "Contenido orgánico diario",
                "Stories y reels",
                "Colaboraciones con influencers",
                "Community management"
            ],
            "local_advertising": [
                "Publicidad en medios locales",
                "Flyers y material impreso",
                "Patrocinios locales",
                "Publicidad en transporte"
            ],
            "events_promotions": [
                "Eventos mensuales",
                "Promociones especiales",
                "Programas de fidelización",
                "Colaboraciones con otros negocios"
            ]
        }
        
        return {
            "monthly_budget": monthly_marketing_budget,
            "allocation": allocation,
            "strategies": strategies,
            "recommendations": [
                "Medir ROI de cada canal",
                "Ajustar presupuesto según resultados",
                "Enfocarse en canales de mayor conversión",
                "Mantener presencia constante"
            ]
        }




