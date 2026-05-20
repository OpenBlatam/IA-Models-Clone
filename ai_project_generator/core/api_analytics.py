"""
API Analytics - Analytics de API
================================

Analytics avanzado de API:
- Request analytics
- Response time tracking
- Error rate tracking
- User behavior analytics
- Endpoint popularity
- Performance metrics
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


class APIRequest:
    """Request de API"""
    
    def __init__(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        response_time: float,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None
    ) -> None:
        self.endpoint = endpoint
        self.method = method
        self.status_code = status_code
        self.response_time = response_time
        self.user_id = user_id
        self.ip_address = ip_address
        self.timestamp = datetime.now()


class APIAnalytics:
    """
    Analytics de API.
    """
    
    def __init__(self) -> None:
        self.requests: List[APIRequest] = []
        self.max_requests = 100000  # Mantener últimos 100k requests
        self.endpoint_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "count": 0,
            "total_time": 0.0,
            "errors": 0,
            "status_codes": defaultdict(int)
        })
        self.user_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "requests": 0,
            "total_time": 0.0,
            "endpoints": set()
        })
    
    def record_request(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        response_time: float,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None
    ) -> None:
        """Registra request"""
        request = APIRequest(endpoint, method, status_code, response_time, user_id, ip_address)
        self.requests.append(request)
        
        # Limpiar requests antiguos
        if len(self.requests) > self.max_requests:
            self.requests = self.requests[-self.max_requests:]
        
        # Actualizar estadísticas de endpoint
        key = f"{method}:{endpoint}"
        stats = self.endpoint_stats[key]
        stats["count"] += 1
        stats["total_time"] += response_time
        if status_code >= 400:
            stats["errors"] += 1
        stats["status_codes"][status_code] += 1
        
        # Actualizar estadísticas de usuario
        if user_id:
            user_stat = self.user_stats[user_id]
            user_stat["requests"] += 1
            user_stat["total_time"] += response_time
            user_stat["endpoints"].add(key)
    
    def get_endpoint_stats(
        self,
        endpoint: Optional[str] = None,
        method: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Obtiene estadísticas de endpoint"""
        if endpoint and method:
            key = f"{method}:{endpoint}"
            stats = self.endpoint_stats.get(key, {})
            return {
                "endpoint": endpoint,
                "method": method,
                "total_requests": stats.get("count", 0),
                "avg_response_time": (
                    stats.get("total_time", 0) / stats.get("count", 1)
                    if stats.get("count", 0) > 0 else 0
                ),
                "error_rate": (
                    stats.get("errors", 0) / stats.get("count", 1) * 100
                    if stats.get("count", 0) > 0 else 0
                ),
                "status_codes": dict(stats.get("status_codes", {}))
            }
        
        # Estadísticas agregadas
        all_stats = {}
        for key, stats in self.endpoint_stats.items():
            method, endpoint = key.split(":", 1)
            all_stats[key] = {
                "endpoint": endpoint,
                "method": method,
                "total_requests": stats["count"],
                "avg_response_time": stats["total_time"] / stats["count"] if stats["count"] > 0 else 0,
                "error_rate": stats["errors"] / stats["count"] * 100 if stats["count"] > 0 else 0
            }
        
        return all_stats
    
    def get_user_stats(
        self,
        user_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Optional[Dict[str, Any]]:
        """Obtiene estadísticas de usuario"""
        user_stat = self.user_stats.get(user_id)
        if not user_stat:
            return None
        
        filtered_requests = [
            r for r in self.requests
            if r.user_id == user_id
            and (not start_date or r.timestamp >= start_date)
            and (not end_date or r.timestamp <= end_date)
        ]
        
        return {
            "user_id": user_id,
            "total_requests": len(filtered_requests),
            "avg_response_time": (
                sum(r.response_time for r in filtered_requests) / len(filtered_requests)
                if filtered_requests else 0
            ),
            "unique_endpoints": len(user_stat["endpoints"]),
            "endpoints": list(user_stat["endpoints"])
        }
    
    def get_popular_endpoints(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Obtiene endpoints más populares"""
        endpoint_counts = [
            {
                "endpoint": key.split(":", 1)[1],
                "method": key.split(":", 1)[0],
                "requests": stats["count"]
            }
            for key, stats in self.endpoint_stats.items()
        ]
        
        return sorted(endpoint_counts, key=lambda x: x["requests"], reverse=True)[:limit]
    
    def get_performance_metrics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Obtiene métricas de performance"""
        filtered_requests = self.requests
        if start_date or end_date:
            filtered_requests = [
                r for r in self.requests
                if (not start_date or r.timestamp >= start_date)
                and (not end_date or r.timestamp <= end_date)
            ]
        
        if not filtered_requests:
            return {}
        
        response_times = [r.response_time for r in filtered_requests]
        error_count = sum(1 for r in filtered_requests if r.status_code >= 400)
        
        return {
            "total_requests": len(filtered_requests),
            "avg_response_time": sum(response_times) / len(response_times),
            "min_response_time": min(response_times),
            "max_response_time": max(response_times),
            "error_rate": (error_count / len(filtered_requests)) * 100,
            "success_rate": ((len(filtered_requests) - error_count) / len(filtered_requests)) * 100
        }
    
    def get_trends(
        self,
        period: str = "hour",
        limit: int = 24
    ) -> List[Dict[str, Any]]:
        """Obtiene tendencias"""
        now = datetime.now()
        trends = []
        
        for i in range(limit):
            if period == "hour":
                start = now - timedelta(hours=i+1)
                end = now - timedelta(hours=i)
            elif period == "day":
                start = now - timedelta(days=i+1)
                end = now - timedelta(days=i)
            else:
                start = now - timedelta(minutes=(i+1)*5)
                end = now - timedelta(minutes=i*5)
            
            period_requests = [
                r for r in self.requests
                if start <= r.timestamp < end
            ]
            
            if period_requests:
                response_times = [r.response_time for r in period_requests]
                trends.append({
                    "period": start.isoformat(),
                    "requests": len(period_requests),
                    "avg_response_time": sum(response_times) / len(response_times),
                    "errors": sum(1 for r in period_requests if r.status_code >= 400)
                })
        
        return list(reversed(trends))


def get_api_analytics() -> APIAnalytics:
    """Obtiene analytics de API"""
    return APIAnalytics()















