"""
Monetization - Sistema de monetización
========================================
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class SubscriptionTier(str, Enum):
    """Niveles de suscripción"""
    FREE = "free"
    BASIC = "basic"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class PaymentStatus(str, Enum):
    """Estados de pago"""
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    REFUNDED = "refunded"


class Monetization:
    """Sistema de monetización"""
    
    def __init__(self):
        self.subscriptions: Dict[str, Dict[str, Any]] = {}
        self.payments: Dict[str, Dict[str, Any]] = {}
        self.pricing_plans: Dict[str, Dict[str, Any]] = {
            SubscriptionTier.FREE.value: {
                "name": "Free",
                "price": 0,
                "features": ["basic_prototypes", "limited_exports"]
            },
            SubscriptionTier.BASIC.value: {
                "name": "Basic",
                "price": 9.99,
                "features": ["unlimited_prototypes", "standard_exports", "priority_support"]
            },
            SubscriptionTier.PRO.value: {
                "name": "Pro",
                "price": 29.99,
                "features": ["unlimited_prototypes", "advanced_exports", "ml_recommendations", "priority_support"]
            },
            SubscriptionTier.ENTERPRISE.value: {
                "name": "Enterprise",
                "price": 99.99,
                "features": ["everything", "custom_integrations", "dedicated_support", "sla"]
            }
        }
    
    def create_subscription(self, user_id: str, tier: SubscriptionTier,
                          payment_method: str = "credit_card") -> Dict[str, Any]:
        """Crea una suscripción"""
        plan = self.pricing_plans.get(tier.value)
        if not plan:
            raise ValueError(f"Plan no encontrado: {tier.value}")
        
        subscription = {
            "user_id": user_id,
            "tier": tier.value,
            "plan": plan,
            "status": "active",
            "started_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(days=30)).isoformat(),
            "payment_method": payment_method,
            "auto_renew": True
        }
        
        self.subscriptions[user_id] = subscription
        
        # Crear pago inicial
        payment = self.create_payment(
            user_id=user_id,
            amount=plan["price"],
            description=f"Subscription: {plan['name']}",
            payment_method=payment_method
        )
        
        subscription["payment_id"] = payment["id"]
        
        logger.info(f"Suscripción creada: {user_id} - {tier.value}")
        return subscription
    
    def create_payment(self, user_id: str, amount: float, description: str,
                      payment_method: str = "credit_card") -> Dict[str, Any]:
        """Crea un pago"""
        from uuid import uuid4
        
        payment = {
            "id": str(uuid4()),
            "user_id": user_id,
            "amount": amount,
            "description": description,
            "payment_method": payment_method,
            "status": PaymentStatus.PENDING.value,
            "created_at": datetime.now().isoformat()
        }
        
        # Procesar pago (simulado)
        # En producción, integraría con Stripe, PayPal, etc.
        payment["status"] = PaymentStatus.COMPLETED.value
        payment["completed_at"] = datetime.now().isoformat()
        
        self.payments[payment["id"]] = payment
        
        logger.info(f"Pago procesado: {payment['id']} - ${amount}")
        return payment
    
    def get_user_subscription(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene suscripción de usuario"""
        subscription = self.subscriptions.get(user_id)
        
        if subscription:
            # Verificar si expiró
            expires_at = datetime.fromisoformat(subscription["expires_at"])
            if datetime.now() > expires_at and subscription["auto_renew"]:
                # Renovar automáticamente
                subscription = self.renew_subscription(user_id)
        
        return subscription
    
    def renew_subscription(self, user_id: str) -> Dict[str, Any]:
        """Renueva una suscripción"""
        subscription = self.subscriptions.get(user_id)
        if not subscription:
            raise ValueError(f"Suscripción no encontrada para usuario: {user_id}")
        
        plan = subscription["plan"]
        
        # Crear nuevo pago
        payment = self.create_payment(
            user_id=user_id,
            amount=plan["price"],
            description=f"Renewal: {plan['name']}",
            payment_method=subscription["payment_method"]
        )
        
        # Actualizar suscripción
        subscription["expires_at"] = (datetime.now() + timedelta(days=30)).isoformat()
        subscription["renewed_at"] = datetime.now().isoformat()
        subscription["payment_id"] = payment["id"]
        
        logger.info(f"Suscripción renovada: {user_id}")
        return subscription
    
    def cancel_subscription(self, user_id: str):
        """Cancela una suscripción"""
        subscription = self.subscriptions.get(user_id)
        if subscription:
            subscription["status"] = "cancelled"
            subscription["cancelled_at"] = datetime.now().isoformat()
            subscription["auto_renew"] = False
            
            logger.info(f"Suscripción cancelada: {user_id}")
    
    def get_pricing_plans(self) -> Dict[str, Any]:
        """Obtiene planes de precios"""
        return {
            "plans": list(self.pricing_plans.values()),
            "currency": "USD"
        }
    
    def check_feature_access(self, user_id: str, feature: str) -> bool:
        """Verifica acceso a una característica"""
        subscription = self.get_user_subscription(user_id)
        
        if not subscription:
            # Usuario free
            subscription = {"tier": SubscriptionTier.FREE.value, "plan": self.pricing_plans[SubscriptionTier.FREE.value]}
        
        features = subscription["plan"].get("features", [])
        return feature in features or "everything" in features
    
    def get_revenue_stats(self, start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """Obtiene estadísticas de ingresos"""
        if not start_date:
            start_date = datetime.now() - timedelta(days=30)
        if not end_date:
            end_date = datetime.now()
        
        relevant_payments = [
            p for p in self.payments.values()
            if start_date <= datetime.fromisoformat(p["created_at"]) <= end_date
            and p["status"] == PaymentStatus.COMPLETED.value
        ]
        
        total_revenue = sum(p["amount"] for p in relevant_payments)
        subscription_count = len(self.subscriptions)
        
        return {
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "total_revenue": total_revenue,
            "payment_count": len(relevant_payments),
            "active_subscriptions": subscription_count,
            "average_revenue_per_user": total_revenue / subscription_count if subscription_count > 0 else 0
        }




