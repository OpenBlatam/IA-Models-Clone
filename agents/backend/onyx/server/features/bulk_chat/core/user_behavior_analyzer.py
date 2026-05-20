"""
User Behavior Analyzer - Analizador de Comportamiento de Usuarios
==================================================================

Sistema de análisis de comportamiento de usuarios con detección de patrones y anomalías.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import statistics

logger = logging.getLogger(__name__)


class BehaviorType(Enum):
    """Tipo de comportamiento."""
    NORMAL = "normal"
    SUSPICIOUS = "suspicious"
    ANOMALOUS = "anomalous"
    FRAUDULENT = "fraudulent"


@dataclass
class UserAction:
    """Acción de usuario."""
    action_id: str
    user_id: str
    action_type: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


@dataclass
class UserProfile:
    """Perfil de usuario."""
    user_id: str
    total_actions: int = 0
    action_frequency: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    avg_session_duration: float = 0.0
    preferred_hours: List[int] = field(default_factory=list)
    common_actions: List[str] = field(default_factory=list)
    last_seen: Optional[datetime] = None
    risk_score: float = 0.0
    behavior_type: BehaviorType = BehaviorType.NORMAL


class UserBehaviorAnalyzer:
    """Analizador de comportamiento de usuarios."""
    
    def __init__(self, history_size: int = 100000):
        self.history_size = history_size
        self.actions: deque = deque(maxlen=history_size)
        self.user_profiles: Dict[str, UserProfile] = {}
        self.anomalies: List[Dict[str, Any]] = []
        self._lock = asyncio.Lock()
    
    def record_action(
        self,
        user_id: str,
        action_type: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Registrar acción de usuario."""
        action_id = f"action_{user_id}_{datetime.now().timestamp()}"
        
        action = UserAction(
            action_id=action_id,
            user_id=user_id,
            action_type=action_type,
            timestamp=datetime.now(),
            ip_address=ip_address,
            user_agent=user_agent,
            metadata=metadata or {},
        )
        
        self.actions.append(action)
        
        # Actualizar perfil
        asyncio.create_task(self._update_user_profile(user_id, action))
        
        # Detectar anomalías
        asyncio.create_task(self._detect_anomalies(user_id, action))
        
        return action_id
    
    async def _update_user_profile(self, user_id: str, action: UserAction):
        """Actualizar perfil de usuario."""
        async with self._lock:
            if user_id not in self.user_profiles:
                self.user_profiles[user_id] = UserProfile(user_id=user_id)
            
            profile = self.user_profiles[user_id]
            profile.total_actions += 1
            profile.action_frequency[action.action_type] += 1
            profile.last_seen = action.timestamp
            
            # Actualizar acciones comunes
            sorted_actions = sorted(
                profile.action_frequency.items(),
                key=lambda x: x[1],
                reverse=True,
            )
            profile.common_actions = [a[0] for a in sorted_actions[:10]]
            
            # Actualizar horas preferidas
            hour = action.timestamp.hour
            if hour not in profile.preferred_hours:
                profile.preferred_hours.append(hour)
                profile.preferred_hours.sort()
    
    async def _detect_anomalies(self, user_id: str, action: UserAction):
        """Detectar anomalías en comportamiento."""
        profile = self.user_profiles.get(user_id)
        if not profile:
            return
        
        # Obtener acciones recientes del usuario
        recent_actions = [
            a for a in self.actions
            if a.user_id == user_id
            and (datetime.now() - a.timestamp).total_seconds() < 3600  # Última hora
        ]
        
        anomaly_score = 0.0
        anomalies = []
        
        # Detectar velocidad anormal de acciones
        if len(recent_actions) > 100:  # Más de 100 acciones en 1 hora
            anomaly_score += 0.5
            anomalies.append("high_action_frequency")
        
        # Detectar acción inusual
        if action.action_type not in profile.common_actions[:5]:
            anomaly_score += 0.3
            anomalies.append("unusual_action_type")
        
        # Detectar horario inusual
        hour = action.timestamp.hour
        if profile.preferred_hours and hour not in profile.preferred_hours:
            anomaly_score += 0.2
            anomalies.append("unusual_hour")
        
        # Detectar cambio de IP frecuente
        recent_ips = set(a.ip_address for a in recent_actions if a.ip_address)
        if len(recent_ips) > 5:
            anomaly_score += 0.4
            anomalies.append("frequent_ip_change")
        
        if anomaly_score > 0.5:
            profile.risk_score = min(profile.risk_score + anomaly_score, 1.0)
            
            if anomaly_score > 0.7:
                profile.behavior_type = BehaviorType.SUSPICIOUS
            if anomaly_score > 0.9:
                profile.behavior_type = BehaviorType.FRAUDULENT
            
            anomaly_record = {
                "user_id": user_id,
                "anomaly_score": anomaly_score,
                "anomalies": anomalies,
                "timestamp": action.timestamp.isoformat(),
                "action_type": action.action_type,
            }
            
            async with self._lock:
                self.anomalies.append(anomaly_record)
                if len(self.anomalies) > 10000:
                    self.anomalies.pop(0)
    
    def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Obtener perfil de usuario."""
        profile = self.user_profiles.get(user_id)
        if not profile:
            return None
        
        return {
            "user_id": profile.user_id,
            "total_actions": profile.total_actions,
            "action_frequency": dict(profile.action_frequency),
            "common_actions": profile.common_actions,
            "preferred_hours": profile.preferred_hours,
            "risk_score": profile.risk_score,
            "behavior_type": profile.behavior_type.value,
            "last_seen": profile.last_seen.isoformat() if profile.last_seen else None,
        }
    
    def get_high_risk_users(
        self,
        threshold: float = 0.7,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Obtener usuarios de alto riesgo."""
        high_risk = [
            self.get_user_profile(uid)
            for uid, profile in self.user_profiles.items()
            if profile.risk_score >= threshold
        ]
        
        high_risk.sort(key=lambda p: p["risk_score"] if p else 0.0, reverse=True)
        return [p for p in high_risk if p][:limit]
    
    def get_anomalies(
        self,
        user_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Obtener anomalías."""
        anomalies = self.anomalies
        
        if user_id:
            anomalies = [a for a in anomalies if a.get("user_id") == user_id]
        
        anomalies.sort(key=lambda a: a.get("timestamp", ""), reverse=True)
        return anomalies[:limit]
    
    def get_behavior_analyzer_summary(self) -> Dict[str, Any]:
        """Obtener resumen del analizador."""
        by_behavior: Dict[str, int] = defaultdict(int)
        total_risk = 0.0
        
        for profile in self.user_profiles.values():
            by_behavior[profile.behavior_type.value] += 1
            total_risk += profile.risk_score
        
        avg_risk = total_risk / len(self.user_profiles) if self.user_profiles else 0.0
        
        return {
            "total_users": len(self.user_profiles),
            "users_by_behavior": dict(by_behavior),
            "total_actions": len(self.actions),
            "total_anomalies": len(self.anomalies),
            "average_risk_score": avg_risk,
        }














