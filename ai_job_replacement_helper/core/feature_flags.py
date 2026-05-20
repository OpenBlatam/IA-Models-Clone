"""
Feature Flags Service - Sistema de feature flags
================================================

Sistema para gestionar feature flags y rollouts graduales.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class FeatureFlagStatus(str, Enum):
    """Estados de feature flag"""
    DISABLED = "disabled"
    ENABLED = "enabled"
    ROLLING_OUT = "rolling_out"
    TESTING = "testing"


@dataclass
class FeatureFlag:
    """Feature flag"""
    id: str
    name: str
    description: str
    status: FeatureFlagStatus
    rollout_percentage: int = 0  # 0-100
    target_users: List[str] = field(default_factory=list)
    target_segments: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class FeatureFlagEvaluation:
    """Evaluación de feature flag"""
    flag_id: str
    user_id: str
    enabled: bool
    reason: str
    variant: Optional[str] = None


class FeatureFlagsService:
    """Servicio de feature flags"""
    
    def __init__(self):
        """Inicializar servicio"""
        self.flags: Dict[str, FeatureFlag] = {}
        logger.info("FeatureFlagsService initialized")
    
    def create_feature_flag(
        self,
        name: str,
        description: str,
        status: FeatureFlagStatus = FeatureFlagStatus.DISABLED
    ) -> FeatureFlag:
        """Crear feature flag"""
        flag_id = f"flag_{name.lower().replace(' ', '_')}"
        
        flag = FeatureFlag(
            id=flag_id,
            name=name,
            description=description,
            status=status,
        )
        
        self.flags[flag_id] = flag
        
        logger.info(f"Feature flag created: {flag_id}")
        return flag
    
    def evaluate_flag(
        self,
        flag_id: str,
        user_id: str,
        user_segments: Optional[List[str]] = None
    ) -> FeatureFlagEvaluation:
        """Evaluar feature flag para usuario"""
        flag = self.flags.get(flag_id)
        if not flag:
            return FeatureFlagEvaluation(
                flag_id=flag_id,
                user_id=user_id,
                enabled=False,
                reason="Flag not found",
            )
        
        # Verificar estado
        if flag.status == FeatureFlagStatus.DISABLED:
            return FeatureFlagEvaluation(
                flag_id=flag_id,
                user_id=user_id,
                enabled=False,
                reason="Flag is disabled",
            )
        
        if flag.status == FeatureFlagStatus.ENABLED:
            return FeatureFlagEvaluation(
                flag_id=flag_id,
                user_id=user_id,
                enabled=True,
                reason="Flag is enabled for all users",
            )
        
        # Verificar usuarios objetivo
        if flag.target_users and user_id in flag.target_users:
            return FeatureFlagEvaluation(
                flag_id=flag_id,
                user_id=user_id,
                enabled=True,
                reason="User in target list",
            )
        
        # Verificar segmentos
        if flag.target_segments and user_segments:
            if any(seg in flag.target_segments for seg in user_segments):
                return FeatureFlagEvaluation(
                    flag_id=flag_id,
                    user_id=user_id,
                    enabled=True,
                    reason="User in target segment",
                )
        
        # Verificar rollout percentage
        if flag.status == FeatureFlagStatus.ROLLING_OUT:
            user_hash = hash(f"{flag_id}:{user_id}") % 100
            if user_hash < flag.rollout_percentage:
                return FeatureFlagEvaluation(
                    flag_id=flag_id,
                    user_id=user_id,
                    enabled=True,
                    reason=f"User in rollout percentage ({flag.rollout_percentage}%)",
                )
            else:
                return FeatureFlagEvaluation(
                    flag_id=flag_id,
                    user_id=user_id,
                    enabled=False,
                    reason=f"User not in rollout percentage ({flag.rollout_percentage}%)",
                )
        
        return FeatureFlagEvaluation(
            flag_id=flag_id,
            user_id=user_id,
            enabled=False,
            reason="No matching conditions",
        )
    
    def update_flag(
        self,
        flag_id: str,
        status: Optional[FeatureFlagStatus] = None,
        rollout_percentage: Optional[int] = None,
        target_users: Optional[List[str]] = None,
        target_segments: Optional[List[str]] = None
    ) -> FeatureFlag:
        """Actualizar feature flag"""
        flag = self.flags.get(flag_id)
        if not flag:
            raise ValueError(f"Feature flag {flag_id} not found")
        
        if status is not None:
            flag.status = status
        if rollout_percentage is not None:
            flag.rollout_percentage = max(0, min(100, rollout_percentage))
        if target_users is not None:
            flag.target_users = target_users
        if target_segments is not None:
            flag.target_segments = target_segments
        
        flag.updated_at = datetime.now()
        
        return flag
    
    def get_all_flags(self) -> List[Dict[str, Any]]:
        """Obtener todos los feature flags"""
        return [
            {
                "id": f.id,
                "name": f.name,
                "description": f.description,
                "status": f.status.value,
                "rollout_percentage": f.rollout_percentage,
                "target_users_count": len(f.target_users),
                "target_segments": f.target_segments,
            }
            for f in self.flags.values()
        ]




