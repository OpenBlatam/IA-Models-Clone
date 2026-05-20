"""
Feature Toggle - Sistema de Feature Flags Avanzado
===================================================

Sistema avanzado de feature flags con rollouts graduales, A/B testing y targeting.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import hashlib
import json

logger = logging.getLogger(__name__)


class ToggleType(Enum):
    """Tipo de feature toggle."""
    BOOLEAN = "boolean"
    PERCENTAGE = "percentage"
    TARGETING = "targeting"
    A_B_TEST = "a_b_test"
    EXPERIMENTAL = "experimental"


class ToggleStatus(Enum):
    """Estado de toggle."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ARCHIVED = "archived"


@dataclass
class FeatureToggle:
    """Feature toggle."""
    toggle_id: str
    name: str
    toggle_type: ToggleType
    status: ToggleStatus = ToggleStatus.INACTIVE
    enabled: bool = False
    percentage: float = 0.0
    target_users: List[str] = field(default_factory=list)
    target_attributes: Dict[str, Any] = field(default_factory=dict)
    variants: Dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class FeatureToggleManager:
    """Gestor de feature toggles."""
    
    def __init__(self):
        self.toggles: Dict[str, FeatureToggle] = {}
        self.evaluation_history: List[Dict[str, Any]] = []
        self._lock = asyncio.Lock()
    
    def create_toggle(
        self,
        toggle_id: str,
        name: str,
        toggle_type: ToggleType,
        enabled: bool = False,
        percentage: float = 0.0,
        target_users: Optional[List[str]] = None,
        target_attributes: Optional[Dict[str, Any]] = None,
        variants: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Crear feature toggle."""
        toggle = FeatureToggle(
            toggle_id=toggle_id,
            name=name,
            toggle_type=toggle_type,
            enabled=enabled,
            percentage=percentage,
            target_users=target_users or [],
            target_attributes=target_attributes or {},
            variants=variants or {},
            metadata=metadata or {},
        )
        
        async def save_toggle():
            async with self._lock:
                self.toggles[toggle_id] = toggle
        
        asyncio.create_task(save_toggle())
        
        logger.info(f"Created feature toggle: {toggle_id} - {name}")
        return toggle_id
    
    def is_enabled(
        self,
        toggle_id: str,
        user_id: Optional[str] = None,
        user_attributes: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Verificar si toggle está habilitado."""
        toggle = self.toggles.get(toggle_id)
        if not toggle or toggle.status != ToggleStatus.ACTIVE:
            return False
        
        if not toggle.enabled:
            return False
        
        # Evaluar según tipo
        if toggle.toggle_type == ToggleType.BOOLEAN:
            return toggle.enabled
        
        elif toggle.toggle_type == ToggleType.PERCENTAGE:
            if user_id:
                # Hash user_id para distribución consistente
                hash_value = int(hashlib.md5(f"{toggle_id}_{user_id}".encode()).hexdigest(), 16)
                user_percentage = (hash_value % 100) / 100.0
                return user_percentage < toggle.percentage
            return toggle.percentage > 0.0
        
        elif toggle.toggle_type == ToggleType.TARGETING:
            # Verificar targeting
            if user_id and user_id in toggle.target_users:
                return True
            
            if user_attributes:
                for key, value in toggle.target_attributes.items():
                    if user_attributes.get(key) != value:
                        return False
                return True
            
            return False
        
        elif toggle.toggle_type == ToggleType.A_B_TEST:
            # Seleccionar variante
            if user_id:
                variant = self._select_variant(toggle_id, user_id, toggle.variants)
                return variant is not None
        
        return False
    
    def get_variant(
        self,
        toggle_id: str,
        user_id: Optional[str] = None,
    ) -> Optional[str]:
        """Obtener variante para A/B testing."""
        toggle = self.toggles.get(toggle_id)
        if not toggle or toggle.toggle_type != ToggleType.A_B_TEST:
            return None
        
        if not self.is_enabled(toggle_id, user_id):
            return None
        
        if user_id:
            return self._select_variant(toggle_id, user_id, toggle.variants)
        
        return None
    
    def _select_variant(self, toggle_id: str, user_id: str, variants: Dict[str, float]) -> Optional[str]:
        """Seleccionar variante basado en hash de usuario."""
        if not variants:
            return None
        
        hash_value = int(hashlib.md5(f"{toggle_id}_{user_id}".encode()).hexdigest(), 16)
        user_hash = (hash_value % 10000) / 10000.0
        
        cumulative = 0.0
        for variant, percentage in variants.items():
            cumulative += percentage
            if user_hash < cumulative:
                return variant
        
        return None
    
    def update_toggle(
        self,
        toggle_id: str,
        enabled: Optional[bool] = None,
        percentage: Optional[float] = None,
        target_users: Optional[List[str]] = None,
        target_attributes: Optional[Dict[str, Any]] = None,
        variants: Optional[Dict[str, float]] = None,
        status: Optional[ToggleStatus] = None,
    ) -> bool:
        """Actualizar feature toggle."""
        toggle = self.toggles.get(toggle_id)
        if not toggle:
            return False
        
        async def update():
            async with self._lock:
                if enabled is not None:
                    toggle.enabled = enabled
                if percentage is not None:
                    toggle.percentage = percentage
                if target_users is not None:
                    toggle.target_users = target_users
                if target_attributes is not None:
                    toggle.target_attributes = target_attributes
                if variants is not None:
                    toggle.variants = variants
                if status is not None:
                    toggle.status = status
        
        asyncio.create_task(update())
        
        logger.info(f"Updated feature toggle: {toggle_id}")
        return True
    
    def record_evaluation(
        self,
        toggle_id: str,
        user_id: Optional[str],
        enabled: bool,
        variant: Optional[str] = None,
    ):
        """Registrar evaluación de toggle."""
        async def save_evaluation():
            async with self._lock:
                self.evaluation_history.append({
                    "toggle_id": toggle_id,
                    "user_id": user_id,
                    "enabled": enabled,
                    "variant": variant,
                    "timestamp": datetime.now().isoformat(),
                })
                # Limitar historial
                if len(self.evaluation_history) > 100000:
                    self.evaluation_history.pop(0)
        
        asyncio.create_task(save_evaluation())
    
    def get_toggle(self, toggle_id: str) -> Optional[Dict[str, Any]]:
        """Obtener información de toggle."""
        toggle = self.toggles.get(toggle_id)
        if not toggle:
            return None
        
        return {
            "toggle_id": toggle.toggle_id,
            "name": toggle.name,
            "toggle_type": toggle.toggle_type.value,
            "status": toggle.status.value,
            "enabled": toggle.enabled,
            "percentage": toggle.percentage,
            "target_users": toggle.target_users,
            "target_attributes": toggle.target_attributes,
            "variants": toggle.variants,
            "created_at": toggle.created_at.isoformat(),
        }
    
    def get_evaluation_statistics(self, toggle_id: Optional[str] = None) -> Dict[str, Any]:
        """Obtener estadísticas de evaluación."""
        evaluations = self.evaluation_history
        
        if toggle_id:
            evaluations = [e for e in evaluations if e["toggle_id"] == toggle_id]
        
        total = len(evaluations)
        enabled_count = sum(1 for e in evaluations if e["enabled"])
        
        variant_counts: Dict[str, int] = defaultdict(int)
        for e in evaluations:
            if e.get("variant"):
                variant_counts[e["variant"]] += 1
        
        return {
            "total_evaluations": total,
            "enabled_count": enabled_count,
            "disabled_count": total - enabled_count,
            "enabled_percentage": (enabled_count / total * 100) if total > 0 else 0.0,
            "variant_counts": dict(variant_counts),
        }
    
    def get_feature_toggle_manager_summary(self) -> Dict[str, Any]:
        """Obtener resumen del gestor."""
        by_type: Dict[str, int] = defaultdict(int)
        by_status: Dict[str, int] = defaultdict(int)
        
        for toggle in self.toggles.values():
            by_type[toggle.toggle_type.value] += 1
            by_status[toggle.status.value] += 1
        
        return {
            "total_toggles": len(self.toggles),
            "toggles_by_type": dict(by_type),
            "toggles_by_status": dict(by_status),
            "active_toggles": len([t for t in self.toggles.values() if t.status == ToggleStatus.ACTIVE]),
            "total_evaluations": len(self.evaluation_history),
        }


