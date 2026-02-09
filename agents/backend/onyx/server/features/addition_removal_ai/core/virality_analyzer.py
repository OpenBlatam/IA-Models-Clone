"""
Virality Analyzer - Sistema de análisis de viralidad
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


@dataclass
class ShareEvent:
    """Evento de compartir"""
    content_id: str
    user_id: str
    share_type: str  # social, email, link, etc.
    timestamp: datetime
    metadata: Dict[str, Any] = None


class ViralityAnalyzer:
    """Analizador de viralidad"""

    def __init__(self):
        """Inicializar analizador"""
        self.share_events: List[ShareEvent] = []
        self.content_shares: Dict[str, List[ShareEvent]] = defaultdict(list)
        self.user_shares: Dict[str, List[ShareEvent]] = defaultdict(list)

    def record_share(
        self,
        content_id: str,
        user_id: str,
        share_type: str = "social",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Registrar compartir de contenido.

        Args:
            content_id: ID del contenido
            user_id: ID del usuario
            share_type: Tipo de compartir (social, email, link, etc.)
            metadata: Metadatos adicionales
        """
        share_event = ShareEvent(
            content_id=content_id,
            user_id=user_id,
            share_type=share_type,
            timestamp=datetime.utcnow(),
            metadata=metadata or {}
        )
        
        self.share_events.append(share_event)
        self.content_shares[content_id].append(share_event)
        self.user_shares[user_id].append(share_event)
        
        logger.debug(f"Compartir registrado: {user_id} - {content_id} - {share_type}")

    def calculate_virality_score(
        self,
        content_id: str,
        period_days: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Calcular score de viralidad de un contenido.

        Args:
            content_id: ID del contenido
            period_days: Período en días (opcional)

        Returns:
            Análisis de viralidad
        """
        shares = self.content_shares.get(content_id, [])
        
        if period_days:
            cutoff_date = datetime.utcnow() - timedelta(days=period_days)
            shares = [s for s in shares if s.timestamp >= cutoff_date]
        
        if not shares:
            return {
                "content_id": content_id,
                "virality_score": 0.0,
                "total_shares": 0,
                "message": "No hay compartidos registrados"
            }
        
        total_shares = len(shares)
        
        # Usuarios únicos que compartieron
        unique_sharers = len(set(s.user_id for s in shares))
        
        # Distribución por tipo de compartir
        share_type_dist = Counter(s.share_type for s in shares)
        
        # Velocidad de compartir (shares por día)
        if len(shares) >= 2:
            first_share = min(s.timestamp for s in shares)
            last_share = max(s.timestamp for s in shares)
            days_span = (last_share - first_share).days + 1
            shares_per_day = total_shares / days_span if days_span > 0 else total_shares
        else:
            shares_per_day = total_shares
        
        # Calcular score de viralidad (0-1)
        # Factores: total de shares, usuarios únicos, velocidad, diversidad de tipos
        base_score = min(1.0, total_shares / 100)  # Normalizar a 100 shares
        unique_score = min(1.0, unique_sharers / 50)  # Normalizar a 50 usuarios únicos
        velocity_score = min(1.0, shares_per_day / 10)  # Normalizar a 10 shares/día
        diversity_score = min(1.0, len(share_type_dist) / 5)  # Normalizar a 5 tipos
        
        virality_score = (
            base_score * 0.4 +
            unique_score * 0.3 +
            velocity_score * 0.2 +
            diversity_score * 0.1
        )
        
        return {
            "content_id": content_id,
            "virality_score": virality_score,
            "total_shares": total_shares,
            "unique_sharers": unique_sharers,
            "shares_per_day": shares_per_day,
            "share_type_distribution": dict(share_type_dist),
            "period_days": period_days
        }

    def analyze_viral_content(
        self,
        limit: int = 10,
        period_days: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Analizar contenidos más virales.

        Args:
            limit: Límite de contenidos
            period_days: Período en días (opcional)

        Returns:
            Lista de contenidos virales
        """
        viral_content = []
        
        for content_id in self.content_shares.keys():
            analysis = self.calculate_virality_score(content_id, period_days)
            if "error" not in analysis and analysis.get("virality_score", 0) > 0:
                viral_content.append(analysis)
        
        # Ordenar por score de viralidad
        viral_content.sort(key=lambda x: x.get("virality_score", 0), reverse=True)
        
        return viral_content[:limit]

    def get_sharing_trends(
        self,
        content_id: Optional[str] = None,
        period_days: int = 30
    ) -> Dict[str, Any]:
        """
        Obtener tendencias de compartir.

        Args:
            content_id: ID del contenido (opcional)
            period_days: Período en días

        Returns:
            Tendencias de compartir
        """
        shares = self.share_events
        
        if content_id:
            shares = [s for s in shares if s.content_id == content_id]
        
        cutoff_date = datetime.utcnow() - timedelta(days=period_days)
        shares = [s for s in shares if s.timestamp >= cutoff_date]
        
        if not shares:
            return {"error": "No hay compartidos en el período"}
        
        # Agrupar por día
        daily_shares = defaultdict(int)
        for share in shares:
            day_key = share.timestamp.date().isoformat()
            daily_shares[day_key] += 1
        
        # Calcular tendencia
        days = sorted(daily_shares.keys())
        if len(days) >= 2:
            first_half = sum(daily_shares[d] for d in days[:len(days)//2])
            second_half = sum(daily_shares[d] for d in days[len(days)//2:])
            trend = "increasing" if second_half > first_half else "decreasing" if second_half < first_half else "stable"
        else:
            trend = "insufficient_data"
        
        return {
            "content_id": content_id,
            "period_days": period_days,
            "total_shares": len(shares),
            "daily_shares": dict(daily_shares),
            "trend": trend,
            "average_shares_per_day": len(shares) / period_days if period_days > 0 else 0
        }

    def get_influencer_analysis(
        self,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Obtener análisis de usuarios más influyentes (que más comparten).

        Args:
            limit: Límite de usuarios

        Returns:
            Lista de usuarios influyentes
        """
        user_share_counts = Counter(s.user_id for s in self.share_events)
        
        influencers = []
        for user_id, share_count in user_share_counts.most_common(limit):
            user_shares = self.user_shares[user_id]
            
            # Tipos de contenido compartido
            content_types = Counter(s.content_id for s in user_shares)
            
            # Tipos de compartir preferidos
            share_types = Counter(s.share_type for s in user_shares)
            
            influencers.append({
                "user_id": user_id,
                "total_shares": share_count,
                "unique_content_shared": len(content_types),
                "preferred_share_types": dict(share_types),
                "top_shared_content": [
                    {"content_id": content_id, "times": count}
                    for content_id, count in content_types.most_common(5)
                ]
            })
        
        return influencers






