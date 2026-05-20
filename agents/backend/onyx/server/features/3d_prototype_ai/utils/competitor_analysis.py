"""
Competitor Analysis - Sistema de análisis de competencia
=========================================================
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


class CompetitorAnalysis:
    """Sistema de análisis de competencia"""
    
    def __init__(self):
        self.competitors: Dict[str, Dict[str, Any]] = {}
        self.competitor_data: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.market_share: Dict[str, float] = {}
    
    def add_competitor(self, competitor_id: str, name: str,
                      category: str, market_share: float = 0.0):
        """Agrega un competidor"""
        competitor = {
            "id": competitor_id,
            "name": name,
            "category": category,
            "market_share": market_share,
            "added_at": datetime.now().isoformat(),
            "features": [],
            "pricing": {},
            "strengths": [],
            "weaknesses": []
        }
        
        self.competitors[competitor_id] = competitor
        self.market_share[competitor_id] = market_share
        
        logger.info(f"Competidor agregado: {competitor_id} - {name}")
        return competitor
    
    def record_competitor_data(self, competitor_id: str, data_type: str,
                              data: Dict[str, Any]):
        """Registra datos de competidor"""
        record = {
            "competitor_id": competitor_id,
            "data_type": data_type,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        
        self.competitor_data[competitor_id].append(record)
        
        # Mantener solo últimos 1000 registros por competidor
        if len(self.competitor_data[competitor_id]) > 1000:
            self.competitor_data[competitor_id] = self.competitor_data[competitor_id][-1000:]
    
    def analyze_competitor(self, competitor_id: str) -> Dict[str, Any]:
        """Analiza un competidor"""
        competitor = self.competitors.get(competitor_id)
        if not competitor:
            raise ValueError(f"Competidor no encontrado: {competitor_id}")
        
        recent_data = self.competitor_data.get(competitor_id, [])
        
        analysis = {
            "competitor": competitor,
            "data_points": len(recent_data),
            "market_position": self._calculate_market_position(competitor["market_share"]),
            "threat_level": self._calculate_threat_level(competitor, recent_data),
            "recommendations": self._generate_recommendations(competitor, recent_data)
        }
        
        return analysis
    
    def _calculate_market_position(self, market_share: float) -> str:
        """Calcula posición de mercado"""
        if market_share > 30:
            return "market_leader"
        elif market_share > 15:
            return "strong_competitor"
        elif market_share > 5:
            return "moderate_competitor"
        else:
            return "minor_competitor"
    
    def _calculate_threat_level(self, competitor: Dict[str, Any],
                               recent_data: List[Dict[str, Any]]) -> str:
        """Calcula nivel de amenaza"""
        threat_score = 0
        
        # Basado en market share
        threat_score += competitor["market_share"] / 10
        
        # Basado en actividad reciente
        if len(recent_data) > 50:
            threat_score += 2
        
        if threat_score > 5:
            return "high"
        elif threat_score > 2:
            return "medium"
        else:
            return "low"
    
    def _generate_recommendations(self, competitor: Dict[str, Any],
                                 recent_data: List[Dict[str, Any]]) -> List[str]:
        """Genera recomendaciones"""
        recommendations = []
        
        if competitor["market_share"] > 20:
            recommendations.append("Monitorear de cerca - competidor importante")
        
        if len(recent_data) > 100:
            recommendations.append("Alta actividad - considerar respuesta competitiva")
        
        if competitor.get("pricing", {}).get("lower_than_us"):
            recommendations.append("Revisar estrategia de precios")
        
        return recommendations
    
    def compare_with_competitors(self, our_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Compara nuestras métricas con competidores"""
        comparison = {
            "our_metrics": our_metrics,
            "competitors": [],
            "market_average": {},
            "our_position": "unknown"
        }
        
        for competitor_id, competitor in self.competitors.items():
            comp_metrics = {
                "name": competitor["name"],
                "market_share": competitor["market_share"],
                "threat_level": self._calculate_threat_level(competitor, self.competitor_data.get(competitor_id, []))
            }
            comparison["competitors"].append(comp_metrics)
        
        # Calcular promedio de mercado
        if self.competitors:
            avg_market_share = sum(c["market_share"] for c in self.competitors.values()) / len(self.competitors)
            comparison["market_average"] = {
                "market_share": avg_market_share
            }
        
        return comparison
    
    def get_competitive_landscape(self) -> Dict[str, Any]:
        """Obtiene panorama competitivo"""
        return {
            "total_competitors": len(self.competitors),
            "market_leaders": [
                c for c in self.competitors.values()
                if self._calculate_market_position(c["market_share"]) == "market_leader"
            ],
            "high_threat_competitors": [
                {
                    "id": cid,
                    "name": c["name"],
                    "threat_level": self._calculate_threat_level(c, self.competitor_data.get(cid, []))
                }
                for cid, c in self.competitors.items()
                if self._calculate_threat_level(c, self.competitor_data.get(cid, [])) == "high"
            ],
            "market_share_distribution": {
                cid: c["market_share"]
                for cid, c in self.competitors.items()
            }
        }




