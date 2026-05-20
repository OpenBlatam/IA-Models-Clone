"""
Analytics Service - Servicio de Analytics
==========================================

Servicio para análisis y métricas de publicaciones.
"""

import logging
import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)


class AnalyticsService:
    """Servicio de analytics para publicaciones"""
    
    def __init__(self, storage_path: str = "data/analytics"):
        """
        Inicializar servicio de analytics
        
        Args:
            storage_path: Ruta para almacenar analytics persistentes
        """
        self.metrics: Dict[str, Dict[str, Any]] = {}
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.storage_path / "metrics.json"
        self._cache = None
        self._load_metrics()
        logger.info(f"Analytics Service inicializado (storage: {storage_path})")
    
    def _init_cache(self):
        """Inicializar caché si está disponible"""
        try:
            from ..utils.cache_manager import CacheManager
            self._cache = CacheManager(default_ttl=300)
        except Exception as e:
            logger.debug(f"No se pudo inicializar caché: {e}")
    
    def _load_metrics(self):
        """Cargar métricas desde archivo"""
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, 'r', encoding='utf-8') as f:
                    self.metrics = json.load(f)
                logger.debug(f"Métricas cargadas desde {self.metrics_file}")
            except Exception as e:
                logger.error(f"Error cargando métricas: {e}")
                self.metrics = {}
        else:
            logger.info(f"No se encontró archivo de métricas, iniciando vacío")
    
    def _save_metrics(self):
        """Guardar métricas en archivo"""
        try:
            with open(self.metrics_file, 'w', encoding='utf-8') as f:
                json.dump(self.metrics, f, indent=2, default=str)
            logger.debug(f"Métricas guardadas en {self.metrics_file}")
        except Exception as e:
            logger.error(f"Error guardando métricas: {e}")
    
    def record_engagement(
        self,
        post_id: str,
        platform: str,
        metrics: Dict[str, Any]
    ):
        """
        Registrar engagement de un post
        
        Args:
            post_id: ID del post
            platform: Plataforma
            metrics: Métricas (likes, comments, shares, etc.)
        """
        key = f"{post_id}_{platform}"
        
        self.metrics[key] = {
            "post_id": post_id,
            "platform": platform,
            "metrics": metrics,
            "recorded_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            **metrics
        }
        
        self._save_metrics()
        
        if self._cache:
            cache_key = f"analytics:{key}"
            self._cache.delete(cache_key)
        
        logger.info(f"Métricas registradas para {key}")
    
    def get_post_analytics(
        self,
        post_id: str,
        platform: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Obtener analytics de un post
        
        Args:
            post_id: ID del post
            platform: Plataforma específica (opcional)
            
        Returns:
            Dict con analytics
        """
        if not self._cache:
            self._init_cache()
        
        cache_key = f"analytics:post:{post_id}:{platform or 'all'}"
        
        if self._cache:
            cached_result = self._cache.get(cache_key)
            if cached_result is not None:
                return cached_result
        
        if platform:
            key = f"{post_id}_{platform}"
            result = self.metrics.get(key, {})
        else:
            results = {}
            for key, data in self.metrics.items():
                if key.startswith(f"{post_id}_"):
                    platform_name = data.get("platform")
                    if platform_name:
                        results[platform_name] = data
            result = results
        
        if self._cache:
            self._cache.set(cache_key, result, ttl=300)
        
        return result
    
    def get_platform_analytics(
        self,
        platform: str,
        days: int = 7
    ) -> Dict[str, Any]:
        """
        Obtener analytics de una plataforma
        
        Args:
            platform: Nombre de la plataforma
            days: Número de días hacia atrás
            
        Returns:
            Dict con analytics agregados
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        platform_metrics = [
            m for m in self.metrics.values()
            if m.get("platform") == platform
            and datetime.fromisoformat(m.get("recorded_at", "")) >= cutoff_date
        ]
        
        if not platform_metrics:
            return {
                "platform": platform,
                "total_posts": 0,
                "total_engagement": 0,
                "average_engagement_rate": 0
            }
        
        total_likes = sum(m.get("likes", 0) for m in platform_metrics)
        total_comments = sum(m.get("comments", 0) for m in platform_metrics)
        total_shares = sum(
            m.get("shares", 0) or m.get("retweets", 0) for m in platform_metrics
        )
        total_reach = sum(m.get("reach", 0) or m.get("impressions", 0) for m in platform_metrics)
        
        total_engagement = total_likes + total_comments + total_shares
        avg_engagement_rate = (total_engagement / total_reach * 100) if total_reach > 0 else 0
        
        return {
            "platform": platform,
            "period_days": days,
            "total_posts": len(platform_metrics),
            "total_likes": total_likes,
            "total_comments": total_comments,
            "total_shares": total_shares,
            "total_reach": total_reach,
            "total_engagement": total_engagement,
            "average_engagement_rate": round(avg_engagement_rate, 2)
        }
    
    def get_best_performing_posts(
        self,
        platform: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Obtener posts con mejor performance
        
        Args:
            platform: Plataforma específica (opcional)
            limit: Número de posts a retornar
            
        Returns:
            Lista de posts ordenados por engagement
        """
        metrics_list = list(self.metrics.values())
        
        if platform:
            metrics_list = [m for m in metrics_list if m.get("platform") == platform]
        
        # Calcular engagement total para cada post
        for metric in metrics_list:
            likes = metric.get("likes", 0)
            comments = metric.get("comments", 0)
            shares = metric.get("shares", 0) or metric.get("retweets", 0)
            metric["total_engagement"] = likes + comments + shares
        
        # Ordenar por engagement
        metrics_list.sort(key=lambda x: x.get("total_engagement", 0), reverse=True)
        
        return metrics_list[:limit]
    
    def get_engagement_trends(
        self,
        platform: str,
        days: int = 30
    ) -> Dict[str, List[float]]:
        """
        Obtener tendencias de engagement
        
        Args:
            platform: Plataforma
            days: Número de días
            
        Returns:
            Dict con tendencias por día
        """
        if not self._cache:
            self._init_cache()
        
        cache_key = f"analytics:trends:{platform}:{days}"
        
        if self._cache:
            cached_result = self._cache.get(cache_key)
            if cached_result is not None:
                return cached_result
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        platform_metrics = [
            m for m in self.metrics.values()
            if m.get("platform") == platform
            and datetime.fromisoformat(m.get("recorded_at", "")) >= cutoff_date
        ]
        
        daily_engagement = defaultdict(float)
        daily_likes = defaultdict(int)
        daily_comments = defaultdict(int)
        daily_shares = defaultdict(int)
        
        for metric in platform_metrics:
            recorded_at = datetime.fromisoformat(metric.get("recorded_at", ""))
            day_key = recorded_at.date().isoformat()
            
            likes = metric.get("likes", 0)
            comments = metric.get("comments", 0)
            shares = metric.get("shares", 0) or metric.get("retweets", 0)
            
            daily_engagement[day_key] += likes + comments + shares
            daily_likes[day_key] += likes
            daily_comments[day_key] += comments
            daily_shares[day_key] += shares
        
        sorted_dates = sorted(daily_engagement.keys())
        
        result = {
            "dates": sorted_dates,
            "engagement": [daily_engagement[date] for date in sorted_dates],
            "likes": [daily_likes[date] for date in sorted_dates],
            "comments": [daily_comments[date] for date in sorted_dates],
            "shares": [daily_shares[date] for date in sorted_dates]
        }
        
        if self._cache:
            self._cache.set(cache_key, result, ttl=600)
        
        return result
    
    def get_comparison(
        self,
        platform1: str,
        platform2: str,
        days: int = 7
    ) -> Dict[str, Any]:
        """
        Comparar analytics entre dos plataformas
        
        Args:
            platform1: Primera plataforma
            platform2: Segunda plataforma
            days: Número de días
            
        Returns:
            Dict con comparación
        """
        analytics1 = self.get_platform_analytics(platform1, days)
        analytics2 = self.get_platform_analytics(platform2, days)
        
        return {
            platform1: analytics1,
            platform2: analytics2,
            "comparison": {
                "engagement_diff": analytics1["total_engagement"] - analytics2["total_engagement"],
                "posts_diff": analytics1["total_posts"] - analytics2["total_posts"],
                "rate_diff": analytics1["average_engagement_rate"] - analytics2["average_engagement_rate"]
            }
        }
    
    def get_summary(
        self,
        days: int = 7
    ) -> Dict[str, Any]:
        """
        Obtener resumen general de analytics
        
        Args:
            days: Número de días
            
        Returns:
            Dict con resumen
        """
        if not self._cache:
            self._init_cache()
        
        cache_key = f"analytics:summary:{days}"
        
        if self._cache:
            cached_result = self._cache.get(cache_key)
            if cached_result is not None:
                return cached_result
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        all_metrics = [
            m for m in self.metrics.values()
            if datetime.fromisoformat(m.get("recorded_at", "")) >= cutoff_date
        ]
        
        platforms = set(m.get("platform") for m in all_metrics)
        
        platform_summaries = {}
        for platform in platforms:
            platform_summaries[platform] = self.get_platform_analytics(platform, days)
        
        total_engagement = sum(s["total_engagement"] for s in platform_summaries.values())
        total_posts = sum(s["total_posts"] for s in platform_summaries.values())
        
        result = {
            "period_days": days,
            "total_posts": total_posts,
            "total_engagement": total_engagement,
            "platforms": platform_summaries,
            "best_platform": max(
                platform_summaries.items(),
                key=lambda x: x[1]["total_engagement"]
            )[0] if platform_summaries else None
        }
        
        if self._cache:
            self._cache.set(cache_key, result, ttl=300)
        
        return result



