"""
Servicio de Análisis de Patrones de Compra - Sistema completo de análisis de compras
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict


class PurchasePatternAnalysisService:
    """Servicio de análisis de patrones de compra"""
    
    def __init__(self):
        """Inicializa el servicio de análisis de compras"""
        pass
    
    def record_purchase(
        self,
        user_id: str,
        purchase_data: Dict
    ) -> Dict:
        """
        Registra una compra
        
        Args:
            user_id: ID del usuario
            purchase_data: Datos de la compra
        
        Returns:
            Compra registrada
        """
        purchase = {
            "id": f"purchase_{datetime.now().timestamp()}",
            "user_id": user_id,
            "purchase_data": purchase_data,
            "amount": purchase_data.get("amount", 0),
            "category": purchase_data.get("category", "other"),
            "merchant": purchase_data.get("merchant", "unknown"),
            "timestamp": purchase_data.get("timestamp", datetime.now().isoformat()),
            "recorded_at": datetime.now().isoformat()
        }
        
        return purchase
    
    def analyze_purchase_patterns(
        self,
        user_id: str,
        purchases: List[Dict],
        days: int = 30
    ) -> Dict:
        """
        Analiza patrones de compra
        
        Args:
            user_id: ID del usuario
            purchases: Lista de compras
            days: Número de días
        
        Returns:
            Análisis de patrones
        """
        if not purchases:
            return {
                "user_id": user_id,
                "analysis": "no_data"
            }
        
        return {
            "user_id": user_id,
            "period_days": days,
            "total_purchases": len(purchases),
            "total_spent": sum(p.get("amount", 0) for p in purchases),
            "category_breakdown": self._analyze_categories(purchases),
            "temporal_patterns": self._analyze_temporal_patterns(purchases),
            "risk_indicators": self._detect_purchase_risk_indicators(purchases),
            "generated_at": datetime.now().isoformat()
        }
    
    def detect_trigger_purchases(
        self,
        user_id: str,
        purchases: List[Dict]
    ) -> Dict:
        """
        Detecta compras relacionadas con triggers
        
        Args:
            user_id: ID del usuario
            purchases: Lista de compras
        
        Returns:
            Compras de riesgo detectadas
        """
        trigger_categories = ["alcohol", "tobacco", "pharmacy"]
        trigger_merchants = ["bar", "liquor_store", "tobacco_shop"]
        
        trigger_purchases = []
        
        for purchase in purchases:
            category = purchase.get("category", "").lower()
            merchant = purchase.get("merchant", "").lower()
            
            if any(trigger in category for trigger in trigger_categories) or \
               any(trigger in merchant for trigger in trigger_merchants):
                trigger_purchases.append({
                    "purchase_id": purchase.get("id"),
                    "trigger_type": "substance_related",
                    "severity": "high",
                    "detected_at": datetime.now().isoformat()
                })
        
        return {
            "user_id": user_id,
            "trigger_purchases": trigger_purchases,
            "total_triggers": len(trigger_purchases),
            "recommendations": self._generate_purchase_recommendations(trigger_purchases),
            "generated_at": datetime.now().isoformat()
        }
    
    def _analyze_categories(self, purchases: List[Dict]) -> Dict:
        """Analiza categorías de compras"""
        category_totals = defaultdict(float)
        
        for purchase in purchases:
            category = purchase.get("category", "other")
            amount = purchase.get("amount", 0)
            category_totals[category] += amount
        
        return dict(category_totals)
    
    def _analyze_temporal_patterns(self, purchases: List[Dict]) -> Dict:
        """Analiza patrones temporales"""
        hour_counts = defaultdict(int)
        
        for purchase in purchases:
            timestamp = purchase.get("timestamp")
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp)
                    hour_counts[dt.hour] += 1
                except:
                    pass
        
        return {
            "peak_hours": sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)[:3],
            "pattern": "evening" if max(hour_counts.keys() or [0]) > 18 else "daytime"
        }
    
    def _detect_purchase_risk_indicators(self, purchases: List[Dict]) -> List[str]:
        """Detecta indicadores de riesgo en compras"""
        indicators = []
        
        recent_purchases = [p for p in purchases if self._is_recent(p.get("timestamp"))]
        if len(recent_purchases) > 10:
            indicators.append("high_frequency_purchases")
        
        return indicators
    
    def _is_recent(self, timestamp: Optional[str]) -> bool:
        """Verifica si timestamp es reciente"""
        if not timestamp:
            return False
        try:
            dt = datetime.fromisoformat(timestamp)
            return (datetime.now() - dt).days <= 7
        except:
            return False
    
    def _generate_purchase_recommendations(self, triggers: List[Dict]) -> List[str]:
        """Genera recomendaciones basadas en compras"""
        if triggers:
            return [
                "Considera evitar lugares donde se venden sustancias",
                "Usa aplicaciones de bloqueo de compras si es necesario"
            ]
        return []

