"""
Sistema de Análisis de Tendencias Musicales

Proporciona:
- Análisis de tendencias de géneros
- Tendencias de BPM
- Análisis de keys populares
- Tendencias temporales
- Predicción de tendencias
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


@dataclass
class TrendData:
    """Datos de tendencia"""
    metric: str
    value: Any
    timestamp: datetime
    change: float = 0.0  # Cambio porcentual


@dataclass
class GenreTrend:
    """Tendencia de género"""
    genre: str
    popularity: float
    growth_rate: float
    sample_count: int


@dataclass
class TrendReport:
    """Reporte de tendencias"""
    period_start: datetime
    period_end: datetime
    genre_trends: List[GenreTrend]
    bpm_trends: Dict[str, float]
    key_trends: Dict[str, float]
    top_tags: List[str]
    generated_at: datetime = field(default_factory=datetime.now)


class TrendAnalysisService:
    """Servicio de análisis de tendencias"""
    
    def __init__(self):
        self.trend_history: List[Dict[str, Any]] = []
        logger.info("TrendAnalysisService initialized")
    
    def analyze_trends(
        self,
        songs: List[Dict[str, Any]],
        period_days: int = 30
    ) -> TrendReport:
        """
        Analiza tendencias de canciones
        
        Args:
            songs: Lista de canciones con metadata
            period_days: Período de análisis en días
        
        Returns:
            TrendReport
        """
        if not songs:
            return TrendReport(
                period_start=datetime.now(),
                period_end=datetime.now(),
                genre_trends=[],
                bpm_trends={},
                key_trends={},
                top_tags=[]
            )
        
        # Filtrar por período
        cutoff_date = datetime.now() - timedelta(days=period_days)
        recent_songs = [
            s for s in songs
            if isinstance(s.get("created_at"), datetime) and s.get("created_at") >= cutoff_date
        ]
        
        # Análisis de géneros
        genre_trends = self._analyze_genre_trends(recent_songs)
        
        # Análisis de BPM
        bpm_trends = self._analyze_bpm_trends(recent_songs)
        
        # Análisis de keys
        key_trends = self._analyze_key_trends(recent_songs)
        
        # Tags más populares
        top_tags = self._analyze_tags(recent_songs)
        
        report = TrendReport(
            period_start=cutoff_date,
            period_end=datetime.now(),
            genre_trends=genre_trends,
            bpm_trends=bpm_trends,
            key_trends=key_trends,
            top_tags=top_tags
        )
        
        logger.info(f"Trend analysis completed: {len(recent_songs)} songs")
        return report
    
    def _analyze_genre_trends(self, songs: List[Dict[str, Any]]) -> List[GenreTrend]:
        """Analiza tendencias de géneros"""
        genre_counts = Counter()
        genre_views = defaultdict(int)
        
        for song in songs:
            genre = song.get("genre", "unknown")
            genre_counts[genre] += 1
            genre_views[genre] += song.get("views", 0)
        
        trends = []
        total = sum(genre_counts.values())
        
        for genre, count in genre_counts.most_common(10):
            popularity = count / total if total > 0 else 0
            avg_views = genre_views[genre] / count if count > 0 else 0
            
            # Calcular growth rate (simplificado)
            growth_rate = popularity * 0.1  # Placeholder
            
            trends.append(GenreTrend(
                genre=genre,
                popularity=popularity,
                growth_rate=growth_rate,
                sample_count=count
            ))
        
        return trends
    
    def _analyze_bpm_trends(self, songs: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analiza tendencias de BPM"""
        bpm_ranges = {
            "60-80": 0,   # Lento
            "80-100": 0,  # Moderado
            "100-120": 0, # Medio
            "120-140": 0, # Rápido
            "140+": 0     # Muy rápido
        }
        
        total = 0
        for song in songs:
            bpm = song.get("bpm", 0)
            if bpm == 0:
                continue
            
            total += 1
            if 60 <= bpm < 80:
                bpm_ranges["60-80"] += 1
            elif 80 <= bpm < 100:
                bpm_ranges["80-100"] += 1
            elif 100 <= bpm < 120:
                bpm_ranges["100-120"] += 1
            elif 120 <= bpm < 140:
                bpm_ranges["120-140"] += 1
            elif bpm >= 140:
                bpm_ranges["140+"] += 1
        
        # Convertir a porcentajes
        if total > 0:
            return {k: v / total for k, v in bpm_ranges.items()}
        
        return bpm_ranges
    
    def _analyze_key_trends(self, songs: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analiza tendencias de keys"""
        key_counts = Counter()
        
        for song in songs:
            key = song.get("key", "unknown")
            if key != "unknown":
                key_counts[key] += 1
        
        total = sum(key_counts.values())
        
        if total > 0:
            return {k: v / total for k, v in key_counts.most_common(12)}
        
        return {}
    
    def _analyze_tags(self, songs: List[Dict[str, Any]], limit: int = 20) -> List[str]:
        """Analiza tags más populares"""
        tag_counts = Counter()
        
        for song in songs:
            tags = song.get("tags", [])
            if isinstance(tags, list):
                for tag in tags:
                    tag_counts[tag] += 1
        
        return [tag for tag, _ in tag_counts.most_common(limit)]
    
    def predict_trends(
        self,
        current_trends: TrendReport,
        historical_data: Optional[List[TrendReport]] = None
    ) -> Dict[str, Any]:
        """
        Predice tendencias futuras
        
        Args:
            current_trends: Tendencias actuales
            historical_data: Datos históricos (opcional)
        
        Returns:
            Predicciones
        """
        predictions = {
            "predicted_genres": [],
            "predicted_bpm_range": "100-120",
            "predicted_keys": [],
            "confidence": 0.5
        }
        
        # Predicción simplificada basada en crecimiento
        if current_trends.genre_trends:
            # Géneros con mayor crecimiento
            growing_genres = [
                gt for gt in current_trends.genre_trends
                if gt.growth_rate > 0
            ]
            growing_genres.sort(key=lambda x: x.growth_rate, reverse=True)
            predictions["predicted_genres"] = [gt.genre for gt in growing_genres[:3]]
        
        # BPM más popular
        if current_trends.bpm_trends:
            predicted_bpm = max(
                current_trends.bpm_trends.items(),
                key=lambda x: x[1]
            )[0]
            predictions["predicted_bpm_range"] = predicted_bpm
        
        # Keys más populares
        if current_trends.key_trends:
            top_keys = sorted(
                current_trends.key_trends.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            predictions["predicted_keys"] = [k for k, _ in top_keys]
        
        return predictions
    
    def get_trend_comparison(
        self,
        period1: TrendReport,
        period2: TrendReport
    ) -> Dict[str, Any]:
        """
        Compara dos períodos de tendencias
        
        Args:
            period1: Primer período
            period2: Segundo período
        
        Returns:
            Comparación
        """
        comparison = {
            "genre_changes": {},
            "bpm_changes": {},
            "key_changes": {},
            "overall_growth": 0.0
        }
        
        # Comparar géneros
        genre1 = {gt.genre: gt.popularity for gt in period1.genre_trends}
        genre2 = {gt.genre: gt.popularity for gt in period2.genre_trends}
        
        all_genres = set(genre1.keys()) | set(genre2.keys())
        for genre in all_genres:
            pop1 = genre1.get(genre, 0)
            pop2 = genre2.get(genre, 0)
            change = pop2 - pop1
            if abs(change) > 0.01:  # Cambio significativo
                comparison["genre_changes"][genre] = change
        
        # Comparar BPM
        for range_name in period1.bpm_trends:
            if range_name in period2.bpm_trends:
                change = period2.bpm_trends[range_name] - period1.bpm_trends[range_name]
                if abs(change) > 0.01:
                    comparison["bpm_changes"][range_name] = change
        
        return comparison


# Instancia global
_trend_analysis_service: Optional[TrendAnalysisService] = None


def get_trend_analysis_service() -> TrendAnalysisService:
    """Obtiene la instancia global del servicio de análisis de tendencias"""
    global _trend_analysis_service
    if _trend_analysis_service is None:
        _trend_analysis_service = TrendAnalysisService()
    return _trend_analysis_service

