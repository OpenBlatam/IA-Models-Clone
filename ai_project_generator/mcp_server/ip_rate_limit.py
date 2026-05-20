"""
MCP IP Rate Limiting - Rate limiting por IP
============================================
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class IPRateLimiter:
    """
    Rate limiter por IP
    
    Limita requests por dirección IP.
    """
    
    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        block_duration_minutes: int = 60,
    ):
        """
        Args:
            requests_per_minute: Requests por minuto
            requests_per_hour: Requests por hora
            block_duration_minutes: Duración de bloqueo en minutos
        """
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.block_duration_minutes = block_duration_minutes
        
        self._minute_requests: Dict[str, deque] = defaultdict(lambda: deque())
        self._hour_requests: Dict[str, deque] = defaultdict(lambda: deque())
        self._blocked_ips: Dict[str, datetime] = {}
    
    def check_rate_limit(self, ip_address: str) -> tuple[bool, Optional[str], Optional[datetime]]:
        """
        Verifica rate limit para una IP
        
        Args:
            ip_address: Dirección IP
            
        Returns:
            Tuple (allowed, error_message, unblock_at)
        """
        now = datetime.utcnow()
        
        # Verificar si está bloqueada
        if ip_address in self._blocked_ips:
            unblock_at = self._blocked_ips[ip_address]
            if now < unblock_at:
                return False, f"IP {ip_address} is blocked", unblock_at
            else:
                # Desbloquear
                del self._blocked_ips[ip_address]
        
        # Limpiar requests antiguos
        minute_ago = now - timedelta(minutes=1)
        hour_ago = now - timedelta(hours=1)
        
        minute_requests = self._minute_requests[ip_address]
        while minute_requests and minute_requests[0] < minute_ago:
            minute_requests.popleft()
        
        hour_requests = self._hour_requests[ip_address]
        while hour_requests and hour_requests[0] < hour_ago:
            hour_requests.popleft()
        
        # Verificar límites
        if len(minute_requests) >= self.requests_per_minute:
            # Bloquear IP
            unblock_at = now + timedelta(minutes=self.block_duration_minutes)
            self._blocked_ips[ip_address] = unblock_at
            return False, f"Rate limit exceeded for IP {ip_address}", unblock_at
        
        if len(hour_requests) >= self.requests_per_hour:
            # Bloquear IP
            unblock_at = now + timedelta(minutes=self.block_duration_minutes)
            self._blocked_ips[ip_address] = unblock_at
            return False, f"Hourly rate limit exceeded for IP {ip_address}", unblock_at
        
        # Registrar request
        minute_requests.append(now)
        hour_requests.append(now)
        
        return True, None, None
    
    def block_ip(self, ip_address: str, duration_minutes: Optional[int] = None):
        """
        Bloquea una IP manualmente
        
        Args:
            ip_address: Dirección IP
            duration_minutes: Duración en minutos (usa default si None)
        """
        duration = duration_minutes or self.block_duration_minutes
        unblock_at = datetime.utcnow() + timedelta(minutes=duration)
        self._blocked_ips[ip_address] = unblock_at
        logger.warning(f"Manually blocked IP: {ip_address} until {unblock_at}")
    
    def unblock_ip(self, ip_address: str):
        """
        Desbloquea una IP
        
        Args:
            ip_address: Dirección IP
        """
        if ip_address in self._blocked_ips:
            del self._blocked_ips[ip_address]
            logger.info(f"Unblocked IP: {ip_address}")
    
    def get_ip_stats(self, ip_address: str) -> Dict[str, Any]:
        """
        Obtiene estadísticas para una IP
        
        Args:
            ip_address: Dirección IP
            
        Returns:
            Diccionario con estadísticas
        """
        now = datetime.utcnow()
        minute_ago = now - timedelta(minutes=1)
        hour_ago = now - timedelta(hours=1)
        
        minute_requests = self._minute_requests[ip_address]
        hour_requests = self._hour_requests[ip_address]
        
        recent_minute = sum(1 for t in minute_requests if t >= minute_ago)
        recent_hour = sum(1 for t in hour_requests if t >= hour_ago)
        
        is_blocked = ip_address in self._blocked_ips
        unblock_at = self._blocked_ips.get(ip_address)
        
        return {
            "ip_address": ip_address,
            "requests_last_minute": recent_minute,
            "requests_last_hour": recent_hour,
            "limits": {
                "per_minute": self.requests_per_minute,
                "per_hour": self.requests_per_hour,
            },
            "blocked": is_blocked,
            "unblock_at": unblock_at.isoformat() if unblock_at else None,
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas generales"""
        return {
            "total_tracked_ips": len(self._minute_requests),
            "blocked_ips": len(self._blocked_ips),
            "limits": {
                "per_minute": self.requests_per_minute,
                "per_hour": self.requests_per_hour,
            },
        }

