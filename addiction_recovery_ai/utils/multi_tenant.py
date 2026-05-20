"""
Multi-Tenant Support for Recovery AI
"""

import torch
import time
from typing import Dict, List, Optional, Any
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class TenantManager:
    """Manage multiple tenants"""
    
    def __init__(self):
        """Initialize tenant manager"""
        self.tenants = {}
        self.tenant_models = {}
        self.tenant_metrics = defaultdict(dict)
        
        logger.info("TenantManager initialized")
    
    def register_tenant(
        self,
        tenant_id: str,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Register new tenant
        
        Args:
            tenant_id: Tenant identifier
            config: Optional tenant configuration
        """
        self.tenants[tenant_id] = {
            "id": tenant_id,
            "config": config or {},
            "created_at": time.time(),
            "active": True
        }
        
        logger.info(f"Tenant registered: {tenant_id}")
    
    def get_tenant_model(
        self,
        tenant_id: str,
        model_type: str = "progress"
    ) -> Optional[torch.nn.Module]:
        """
        Get tenant-specific model
        
        Args:
            tenant_id: Tenant identifier
            model_type: Model type
        
        Returns:
            Model for tenant
        """
        key = f"{tenant_id}_{model_type}"
        
        if key not in self.tenant_models:
            # Create default model for tenant
            from addiction_recovery_ai import create_progress_predictor
            model = create_progress_predictor()
            self.tenant_models[key] = model
        
        return self.tenant_models[key]
    
    def set_tenant_model(
        self,
        tenant_id: str,
        model: torch.nn.Module,
        model_type: str = "progress"
    ):
        """
        Set tenant-specific model
        
        Args:
            tenant_id: Tenant identifier
            model: Model to set
            model_type: Model type
        """
        key = f"{tenant_id}_{model_type}"
        self.tenant_models[key] = model
        logger.info(f"Model set for tenant {tenant_id}: {model_type}")
    
    def record_tenant_metric(
        self,
        tenant_id: str,
        metric_name: str,
        value: float
    ):
        """Record metric for tenant"""
        if tenant_id not in self.tenant_metrics:
            self.tenant_metrics[tenant_id] = {}
        
        if metric_name not in self.tenant_metrics[tenant_id]:
            self.tenant_metrics[tenant_id][metric_name] = []
        
        self.tenant_metrics[tenant_id][metric_name].append(value)
    
    def get_tenant_stats(self, tenant_id: str) -> Dict[str, Any]:
        """Get statistics for tenant"""
        if tenant_id not in self.tenants:
            return {}
        
        metrics = self.tenant_metrics.get(tenant_id, {})
        
        stats = {
            "tenant_id": tenant_id,
            "config": self.tenants[tenant_id]["config"],
            "metrics": {}
        }
        
        for metric_name, values in metrics.items():
            if values:
                stats["metrics"][metric_name] = {
                    "count": len(values),
                    "avg": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values)
                }
        
        return stats


class TenantIsolation:
    """Isolate tenants with separate models and data"""
    
    def __init__(self, tenant_manager: TenantManager):
        """
        Initialize tenant isolation
        
        Args:
            tenant_manager: Tenant manager instance
        """
        self.tenant_manager = tenant_manager
    
    def predict_for_tenant(
        self,
        tenant_id: str,
        features: Dict[str, float],
        model_type: str = "progress"
    ) -> float:
        """
        Predict for specific tenant
        
        Args:
            tenant_id: Tenant identifier
            features: Feature dictionary
            model_type: Model type
        
        Returns:
            Prediction result
        """
        model = self.tenant_manager.get_tenant_model(tenant_id, model_type)
        
        # Convert features to tensor
        feature_list = [
            features.get("days_sober", 0) / 365.0,
            features.get("cravings_level", 5) / 10.0,
            features.get("stress_level", 5) / 10.0,
            features.get("support_level", 5) / 10.0,
            features.get("mood_score", 5) / 10.0,
            features.get("sleep_quality", 5) / 10.0,
            features.get("exercise_frequency", 2) / 7.0,
            features.get("therapy_sessions", 0) / 10.0,
            features.get("medication_compliance", 1.0),
            features.get("social_activity", 3) / 7.0
        ]
        
        feature_tensor = torch.tensor([feature_list], dtype=torch.float32)
        
        model.eval()
        with torch.no_grad():
            output = model(feature_tensor)
            result = output.item()
        
        # Record metric
        self.tenant_manager.record_tenant_metric(tenant_id, "predictions", result)
        
        return result

