"""
Dashboard Service - Dashboard y visualizaciones
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from ..core.models import StoreType

logger = logging.getLogger(__name__)


class DashboardService:
    """Servicio para generar dashboards"""
    
    def generate_dashboard(
        self,
        designs: List[Dict[str, Any]],
        time_range: str = "all"  # "all", "week", "month", "year"
    ) -> Dict[str, Any]:
        """Generar dashboard completo"""
        
        return {
            "overview": self._generate_overview(designs),
            "statistics": self._generate_statistics(designs, time_range),
            "trends": self._generate_trends(designs, time_range),
            "breakdown": self._generate_breakdown(designs),
            "recent_activity": self._generate_recent_activity(designs),
            "insights": self._generate_insights(designs)
        }
    
    def _generate_overview(self, designs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generar resumen general"""
        return {
            "total_designs": len(designs),
            "active_designs": len([d for d in designs if d.get("status") == "active"]),
            "total_investment": sum(
                d.get("financial_analysis", {}).get("initial_investment", {}).get("total", 0)
                for d in designs
            ),
            "average_profit": sum(
                d.get("financial_analysis", {}).get("profitability", {}).get("monthly_profit", 0)
                for d in designs
            ) / len(designs) if designs else 0
        }
    
    def _generate_statistics(
        self,
        designs: List[Dict[str, Any]],
        time_range: str
    ) -> Dict[str, Any]:
        """Generar estadísticas"""
        
        # Filtrar por rango de tiempo
        filtered_designs = self._filter_by_time_range(designs, time_range)
        
        # Estadísticas por tipo de tienda
        store_types = {}
        for design in filtered_designs:
            store_type = design.get("store_type", "unknown")
            store_types[store_type] = store_types.get(store_type, 0) + 1
        
        # Estadísticas por estilo
        styles = {}
        for design in filtered_designs:
            style = design.get("style", "unknown")
            styles[style] = styles.get(style, 0) + 1
        
        return {
            "designs_in_period": len(filtered_designs),
            "by_store_type": store_types,
            "by_style": styles,
            "total_investment": sum(
                d.get("financial_analysis", {}).get("initial_investment", {}).get("total", 0)
                for d in filtered_designs
            ),
            "average_profit": sum(
                d.get("financial_analysis", {}).get("profitability", {}).get("monthly_profit", 0)
                for d in filtered_designs
            ) / len(filtered_designs) if filtered_designs else 0
        }
    
    def _filter_by_time_range(
        self,
        designs: List[Dict[str, Any]],
        time_range: str
    ) -> List[Dict[str, Any]]:
        """Filtrar diseños por rango de tiempo"""
        now = datetime.now()
        
        if time_range == "all":
            return designs
        elif time_range == "week":
            cutoff = now - timedelta(days=7)
        elif time_range == "month":
            cutoff = now - timedelta(days=30)
        elif time_range == "year":
            cutoff = now - timedelta(days=365)
        else:
            return designs
        
        filtered = []
        for design in designs:
            created_at = design.get("created_at")
            if created_at:
                try:
                    if isinstance(created_at, str):
                        created_dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    else:
                        created_dt = created_at
                    
                    if created_dt >= cutoff:
                        filtered.append(design)
                except:
                    pass
        
        return filtered
    
    def _generate_trends(
        self,
        designs: List[Dict[str, Any]],
        time_range: str
    ) -> Dict[str, Any]:
        """Generar tendencias"""
        
        # Tendencias de creación
        creation_trends = {}
        for design in designs:
            created_at = design.get("created_at")
            if created_at:
                try:
                    if isinstance(created_at, str):
                        date = datetime.fromisoformat(created_at.replace('Z', '+00:00')).date()
                    else:
                        date = created_at.date()
                    
                    week_key = date.isoformat()[:7]  # YYYY-MM
                    creation_trends[week_key] = creation_trends.get(week_key, 0) + 1
                except:
                    pass
        
        return {
            "creation_trend": creation_trends,
            "popular_store_types": self._get_popular_items(designs, "store_type"),
            "popular_styles": self._get_popular_items(designs, "style")
        }
    
    def _get_popular_items(
        self,
        designs: List[Dict[str, Any]],
        field: str
    ) -> List[Dict[str, Any]]:
        """Obtener items más populares"""
        counts = {}
        for design in designs:
            value = design.get(field)
            if value:
                counts[value] = counts.get(value, 0) + 1
        
        sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        return [{"item": item, "count": count} for item, count in sorted_items[:5]]
    
    def _generate_breakdown(self, designs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generar desglose"""
        return {
            "by_store_type": self._breakdown_by_field(designs, "store_type"),
            "by_style": self._breakdown_by_field(designs, "style"),
            "by_viability": self._breakdown_by_viability(designs)
        }
    
    def _breakdown_by_field(
        self,
        designs: List[Dict[str, Any]],
        field: str
    ) -> Dict[str, int]:
        """Desglose por campo"""
        breakdown = {}
        for design in designs:
            value = design.get(field)
            if value:
                breakdown[value] = breakdown.get(value, 0) + 1
        return breakdown
    
    def _breakdown_by_viability(self, designs: List[Dict[str, Any]]) -> Dict[str, int]:
        """Desglose por viabilidad"""
        breakdown = {"high": 0, "medium": 0, "low": 0}
        
        for design in designs:
            monthly_profit = design.get("financial_analysis", {}).get("profitability", {}).get("monthly_profit", 0)
            if monthly_profit > 5000:
                breakdown["high"] += 1
            elif monthly_profit > 0:
                breakdown["medium"] += 1
            else:
                breakdown["low"] += 1
        
        return breakdown
    
    def _generate_recent_activity(self, designs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generar actividad reciente"""
        # Ordenar por fecha de creación
        sorted_designs = sorted(
            designs,
            key=lambda d: d.get("created_at", ""),
            reverse=True
        )[:10]
        
        return [
            {
                "store_id": d.get("store_id"),
                "store_name": d.get("store_name"),
                "store_type": d.get("store_type"),
                "created_at": d.get("created_at")
            }
            for d in sorted_designs
        ]
    
    def _generate_insights(self, designs: List[Dict[str, Any]]) -> List[str]:
        """Generar insights"""
        insights = []
        
        if not designs:
            return ["No hay diseños para analizar"]
        
        # Insight sobre tipo más popular
        store_types = self._breakdown_by_field(designs, "store_type")
        if store_types:
            most_common = max(store_types.items(), key=lambda x: x[1])
            insights.append(f"El tipo de tienda más común es {most_common[0]} ({most_common[1]} diseños)")
        
        # Insight sobre viabilidad
        viability = self._breakdown_by_viability(designs)
        if viability["high"] > 0:
            insights.append(f"{viability['high']} diseños tienen alta viabilidad financiera")
        
        # Insight sobre inversión
        total_investment = sum(
            d.get("financial_analysis", {}).get("initial_investment", {}).get("total", 0)
            for d in designs
        )
        if total_investment > 0:
            avg_investment = total_investment / len(designs)
            insights.append(f"Inversión promedio: ${avg_investment:,.0f}")
        
        return insights




