"""
Subscriptions Service - Sistema de suscripciones y pagos
========================================================

Sistema para gestionar suscripciones premium.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class SubscriptionPlan(str, Enum):
    """Planes de suscripción"""
    FREE = "free"
    BASIC = "basic"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"


class SubscriptionStatus(str, Enum):
    """Estado de suscripción"""
    ACTIVE = "active"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    TRIAL = "trial"


@dataclass
class Subscription:
    """Suscripción"""
    user_id: str
    plan: SubscriptionPlan
    status: SubscriptionStatus
    start_date: datetime
    end_date: Optional[datetime] = None
    trial_end_date: Optional[datetime] = None
    payment_method: Optional[str] = None
    auto_renew: bool = True
    created_at: datetime = field(default_factory=datetime.now)


class SubscriptionsService:
    """Servicio de suscripciones"""
    
    def __init__(self):
        """Inicializar servicio"""
        self.subscriptions: Dict[str, Subscription] = {}
        self.plan_features = self._initialize_plan_features()
        logger.info("SubscriptionsService initialized")
    
    def _initialize_plan_features(self) -> Dict[str, Dict[str, Any]]:
        """Inicializar características de planes"""
        return {
            SubscriptionPlan.FREE: {
                "name": "Free",
                "price": 0,
                "features": [
                    "Búsqueda básica de trabajos",
                    "5 aplicaciones por mes",
                    "Roadmap básico",
                    "Comunidad",
                ],
                "limits": {
                    "applications_per_month": 5,
                    "cv_analyses": 1,
                    "interview_simulations": 2,
                }
            },
            SubscriptionPlan.BASIC: {
                "name": "Basic",
                "price": 9.99,
                "features": [
                    "Todo lo de Free",
                    "Aplicaciones ilimitadas",
                    "Análisis de CV ilimitado",
                    "Simulador de entrevistas",
                    "Recomendaciones avanzadas",
                ],
                "limits": {
                    "applications_per_month": -1,  # Ilimitado
                    "cv_analyses": -1,
                    "interview_simulations": 10,
                }
            },
            SubscriptionPlan.PREMIUM: {
                "name": "Premium",
                "price": 19.99,
                "features": [
                    "Todo lo de Basic",
                    "Mentoría con IA ilimitada",
                    "Eventos exclusivos",
                    "Prioridad en soporte",
                    "Reportes avanzados",
                ],
                "limits": {
                    "applications_per_month": -1,
                    "cv_analyses": -1,
                    "interview_simulations": -1,
                    "mentoring_sessions": -1,
                }
            },
            SubscriptionPlan.ENTERPRISE: {
                "name": "Enterprise",
                "price": 49.99,
                "features": [
                    "Todo lo de Premium",
                    "Soporte dedicado",
                    "Custom integrations",
                    "Analytics avanzado",
                ],
                "limits": {}
            }
        }
    
    def create_subscription(
        self,
        user_id: str,
        plan: SubscriptionPlan,
        trial_days: int = 7
    ) -> Subscription:
        """Crear suscripción"""
        subscription = Subscription(
            user_id=user_id,
            plan=plan,
            status=SubscriptionStatus.TRIAL if trial_days > 0 else SubscriptionStatus.ACTIVE,
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(days=30) if plan != SubscriptionPlan.FREE else None,
            trial_end_date=datetime.now() + timedelta(days=trial_days) if trial_days > 0 else None,
        )
        
        self.subscriptions[user_id] = subscription
        logger.info(f"Subscription created for user {user_id}: {plan.value}")
        return subscription
    
    def get_subscription(self, user_id: str) -> Optional[Subscription]:
        """Obtener suscripción del usuario"""
        subscription = self.subscriptions.get(user_id)
        
        if subscription:
            # Verificar si expiró
            if subscription.end_date and datetime.now() > subscription.end_date:
                subscription.status = SubscriptionStatus.EXPIRED
            
            # Verificar si el trial expiró
            if subscription.status == SubscriptionStatus.TRIAL:
                if subscription.trial_end_date and datetime.now() > subscription.trial_end_date:
                    subscription.status = SubscriptionStatus.EXPIRED
        
        return subscription
    
    def cancel_subscription(self, user_id: str) -> bool:
        """Cancelar suscripción"""
        subscription = self.subscriptions.get(user_id)
        if not subscription:
            return False
        
        subscription.status = SubscriptionStatus.CANCELLED
        subscription.auto_renew = False
        logger.info(f"Subscription cancelled for user {user_id}")
        return True
    
    def upgrade_plan(self, user_id: str, new_plan: SubscriptionPlan) -> Subscription:
        """Actualizar plan"""
        subscription = self.subscriptions.get(user_id)
        if not subscription:
            subscription = self.create_subscription(user_id, new_plan)
        else:
            subscription.plan = new_plan
            subscription.status = SubscriptionStatus.ACTIVE
            subscription.end_date = datetime.now() + timedelta(days=30)
        
        logger.info(f"Plan upgraded for user {user_id} to {new_plan.value}")
        return subscription
    
    def get_plan_features(self, plan: SubscriptionPlan) -> Dict[str, Any]:
        """Obtener características del plan"""
        return self.plan_features.get(plan, {})




