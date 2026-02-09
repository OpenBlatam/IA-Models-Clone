"""
Trends Service - Análisis de tendencias
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from ..core.models import StoreType, DesignStyle

logger = logging.getLogger(__name__)


class TrendsService:
    """Servicio para análisis de tendencias"""
    
    def analyze_trends(
        self,
        designs: List[Dict[str, Any]],
        period: str = "month"  # "week", "month", "quarter", "year"
    ) -> Dict[str, Any]:
        """Analizar tendencias en los diseños"""
        
        return {
            "store_type_trends": self._analyze_store_type_trends(designs, period),
            "style_trends": self._analyze_style_trends(designs, period),
            "budget_trends": self._analyze_budget_trends(designs, period),
            "popular_combinations": self._analyze_popular_combinations(designs),
            "emerging_trends": self._identify_emerging_trends(designs),
            "predictions": self._generate_predictions(designs)
        }
    
    def _analyze_store_type_trends(
        self,
        designs: List[Dict[str, Any]],
        period: str
    ) -> Dict[str, Any]:
        """Analizar tendencias de tipos de tienda"""
        trends = {}
        
        for design in designs:
            store_type = design.get("store_type")
            created_at = design.get("created_at")
            
            if store_type and created_at:
                try:
                    if isinstance(created_at, str):
                        date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    else:
                        date = created_at
                    
                    period_key = self._get_period_key(date, period)
                    
                    if store_type not in trends:
                        trends[store_type] = {}
                    
                    trends[store_type][period_key] = trends[store_type].get(period_key, 0) + 1
                except:
                    pass
        
        # Calcular crecimiento
        growth = {}
        for store_type, periods in trends.items():
            if len(periods) >= 2:
                sorted_periods = sorted(periods.items())
                recent = sorted_periods[-1][1]
                previous = sorted_periods[-2][1] if len(sorted_periods) > 1 else recent
                
                if previous > 0:
                    growth[store_type] = ((recent - previous) / previous) * 100
                else:
                    growth[store_type] = 100 if recent > 0 else 0
        
        return {
            "distribution": trends,
            "growth": growth,
            "most_popular": max(trends.items(), key=lambda x: sum(x[1].values()))[0] if trends else None
        }
    
    def _analyze_style_trends(
        self,
        designs: List[Dict[str, Any]],
        period: str
    ) -> Dict[str, Any]:
        """Analizar tendencias de estilos"""
        trends = {}
        
        for design in designs:
            style = design.get("style")
            created_at = design.get("created_at")
            
            if style and created_at:
                try:
                    if isinstance(created_at, str):
                        date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    else:
                        date = created_at
                    
                    period_key = self._get_period_key(date, period)
                    
                    if style not in trends:
                        trends[style] = {}
                    
                    trends[style][period_key] = trends[style].get(period_key, 0) + 1
                except:
                    pass
        
        return {
            "distribution": trends,
            "most_popular": max(trends.items(), key=lambda x: sum(x[1].values()))[0] if trends else None,
            "trending": self._get_trending_items(trends)
        }
    
    def _analyze_budget_trends(
        self,
        designs: List[Dict[str, Any]],
        period: str
    ) -> Dict[str, Any]:
        """Analizar tendencias de presupuesto"""
        budgets_by_period = {}
        
        for design in designs:
            financial = design.get("financial_analysis", {})
            initial_investment = financial.get("initial_investment", {}).get("total", 0)
            created_at = design.get("created_at")
            
            if initial_investment > 0 and created_at:
                try:
                    if isinstance(created_at, str):
                        date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    else:
                        date = created_at
                    
                    period_key = self._get_period_key(date, period)
                    
                    if period_key not in budgets_by_period:
                        budgets_by_period[period_key] = []
                    
                    budgets_by_period[period_key].append(initial_investment)
                except:
                    pass
        
        # Calcular promedios
        averages = {}
        for period_key, budgets in budgets_by_period.items():
            averages[period_key] = sum(budgets) / len(budgets) if budgets else 0
        
        return {
            "averages_by_period": averages,
            "overall_average": sum(sum(budgets) for budgets in budgets_by_period.values()) / sum(len(budgets) for budgets in budgets_by_period.values()) if budgets_by_period else 0,
            "trend": "increasing" if len(averages) >= 2 and list(averages.values())[-1] > list(averages.values())[-2] else "decreasing" if len(averages) >= 2 else "stable"
        }
    
    def _get_period_key(self, date: datetime, period: str) -> str:
        """Obtener clave de período"""
        if period == "week":
            return date.strftime("%Y-W%W")
        elif period == "month":
            return date.strftime("%Y-%m")
        elif period == "quarter":
            quarter = (date.month - 1) // 3 + 1
            return f"{date.year}-Q{quarter}"
        elif period == "year":
            return str(date.year)
        else:
            return date.strftime("%Y-%m")
    
    def _analyze_popular_combinations(self, designs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analizar combinaciones populares"""
        combinations = {}
        
        for design in designs:
            store_type = design.get("store_type")
            style = design.get("style")
            
            if store_type and style:
                key = f"{store_type}_{style}"
                combinations[key] = combinations.get(key, 0) + 1
        
        sorted_combinations = sorted(combinations.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {
                "combination": key,
                "count": count,
                "store_type": key.split("_")[0],
                "style": key.split("_")[1] if "_" in key else ""
            }
            for key, count in sorted_combinations[:10]
        ]
    
    def _identify_emerging_trends(self, designs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identificar tendencias emergentes"""
        # Comparar últimos 30 días vs anteriores
        now = datetime.now()
        recent_cutoff = now - timedelta(days=30)
        older_cutoff = now - timedelta(days=60)
        
        recent_designs = []
        older_designs = []
        
        for design in designs:
            created_at = design.get("created_at")
            if created_at:
                try:
                    if isinstance(created_at, str):
                        date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    else:
                        date = created_at
                    
                    if date >= recent_cutoff:
                        recent_designs.append(design)
                    elif date >= older_cutoff:
                        older_designs.append(design)
                except:
                    pass
        
        emerging = []
        
        # Analizar estilos emergentes
        recent_styles = {}
        older_styles = {}
        
        for design in recent_designs:
            style = design.get("style")
            if style:
                recent_styles[style] = recent_styles.get(style, 0) + 1
        
        for design in older_designs:
            style = design.get("style")
            if style:
                older_styles[style] = older_styles.get(style, 0) + 1
        
        for style, recent_count in recent_styles.items():
            older_count = older_styles.get(style, 0)
            if recent_count > older_count * 1.5:  # 50% más que antes
                emerging.append({
                    "type": "style",
                    "item": style,
                    "growth": ((recent_count - older_count) / older_count * 100) if older_count > 0 else 100
                })
        
        return emerging
    
    def _get_trending_items(self, trends: Dict[str, Dict[str, Any]]) -> List[str]:
        """Obtener items en tendencia"""
        trending = []
        
        for item, periods in trends.items():
            if len(periods) >= 2:
                sorted_periods = sorted(periods.items())
                recent = sorted_periods[-1][1]
                previous = sorted_periods[-2][1] if len(sorted_periods) > 1 else 0
                
                if recent > previous * 1.2:  # 20% de crecimiento
                    trending.append(item)
        
        return trending
    
    def _generate_predictions(self, designs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generar predicciones basadas en tendencias"""
        predictions = {
            "next_popular_store_type": None,
            "next_popular_style": None,
            "budget_trend": "stable"
        }
        
        store_type_trends = self._analyze_store_type_trends(designs, "month")
        style_trends = self._analyze_style_trends(designs, "month")
        budget_trends = self._analyze_budget_trends(designs, "month")
        
        # Predecir tipo de tienda más popular
        if store_type_trends.get("growth"):
            growth = store_type_trends["growth"]
            if growth:
                predictions["next_popular_store_type"] = max(growth.items(), key=lambda x: x[1])[0]
        
        # Predecir estilo más popular
        if style_trends.get("trending"):
            trending = style_trends["trending"]
            if trending:
                predictions["next_popular_style"] = trending[0]
        
        # Predecir tendencia de presupuesto
        predictions["budget_trend"] = budget_trends.get("trend", "stable")
        
        return predictions




