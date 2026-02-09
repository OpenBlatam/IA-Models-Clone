"""
Billing System - Sistema de billing y pricing
==============================================
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class BillingSystem:
    """
    Gestiona billing, pricing y suscripciones.
    """
    
    def __init__(self, billing_dir: str = "data/billing"):
        """
        Inicializar sistema de billing.
        
        Args:
            billing_dir: Directorio para datos de billing
        """
        self.billing_dir = Path(billing_dir)
        self.billing_dir.mkdir(parents=True, exist_ok=True)
        
        self.plans = {
            "free": {
                "name": "Free",
                "price": 0,
                "currency": "USD",
                "features": ["10 papers", "50 improvements/month"]
            },
            "pro": {
                "name": "Pro",
                "price": 29.99,
                "currency": "USD",
                "billing_period": "monthly",
                "features": ["100 papers", "1000 improvements/month", "Priority support"]
            },
            "enterprise": {
                "name": "Enterprise",
                "price": 299.99,
                "currency": "USD",
                "billing_period": "monthly",
                "features": ["Unlimited papers", "Unlimited improvements", "Dedicated support"]
            }
        }
        
        self.subscriptions: Dict[str, Dict[str, Any]] = {}
        self._load_subscriptions()
    
    def create_subscription(
        self,
        user_id: str,
        plan: str,
        payment_method: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Crea una suscripción.
        
        Args:
            user_id: ID del usuario
            plan: Plan a suscribir
            payment_method: Método de pago (opcional)
            
        Returns:
            Información de la suscripción
        """
        if plan not in self.plans:
            raise ValueError(f"Plan no válido: {plan}")
        
        subscription = {
            "subscription_id": f"sub_{user_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "user_id": user_id,
            "plan": plan,
            "status": "active",
            "created_at": datetime.now().isoformat(),
            "current_period_start": datetime.now().isoformat(),
            "current_period_end": (datetime.now() + timedelta(days=30)).isoformat(),
            "payment_method": payment_method,
            "price": self.plans[plan]["price"]
        }
        
        self.subscriptions[subscription["subscription_id"]] = subscription
        self._save_subscription(subscription)
        
        logger.info(f"Suscripción creada: {subscription['subscription_id']} ({plan})")
        
        return subscription
    
    def get_usage_cost(
        self,
        user_id: str,
        usage: Dict[str, int]
    ) -> Dict[str, Any]:
        """
        Calcula costo basado en uso.
        
        Args:
            user_id: ID del usuario
            usage: Uso por recurso
            
        Returns:
            Cálculo de costo
        """
        subscription = self._get_user_subscription(user_id)
        
        if not subscription:
            return {
                "cost": 0,
                "plan": "free",
                "message": "No subscription found"
            }
        
        plan = subscription["plan"]
        
        # Calcular costo según plan y uso
        base_cost = self.plans[plan]["price"]
        
        # Costos adicionales por uso excedente (si aplica)
        overage_costs = {}
        total_overage = 0
        
        # En producción, esto calcularía costos reales
        cost_breakdown = {
            "base_cost": base_cost,
            "overage_costs": overage_costs,
            "total_overage": total_overage,
            "total_cost": base_cost + total_overage,
            "currency": "USD"
        }
        
        return cost_breakdown
    
    def _get_user_subscription(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene suscripción de un usuario"""
        for subscription in self.subscriptions.values():
            if subscription.get("user_id") == user_id and subscription.get("status") == "active":
                return subscription
        return None
    
    def _load_subscriptions(self):
        """Carga suscripciones desde disco"""
        subscriptions_file = self.billing_dir / "subscriptions.json"
        
        if subscriptions_file.exists():
            try:
                with open(subscriptions_file, "r", encoding="utf-8") as f:
                    self.subscriptions = json.load(f)
            except Exception as e:
                logger.error(f"Error cargando suscripciones: {e}")
    
    def _save_subscription(self, subscription: Dict[str, Any]):
        """Guarda suscripción en disco"""
        try:
            subscriptions_file = self.billing_dir / "subscriptions.json"
            
            # Cargar existentes
            if subscriptions_file.exists():
                with open(subscriptions_file, "r", encoding="utf-8") as f:
                    all_subscriptions = json.load(f)
            else:
                all_subscriptions = {}
            
            all_subscriptions[subscription["subscription_id"]] = subscription
            
            with open(subscriptions_file, "w", encoding="utf-8") as f:
                json.dump(all_subscriptions, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error guardando suscripción: {e}")




