"""
Sistema de Monetización

Proporciona:
- Sistema de suscripciones
- Pagos y facturación
- Créditos y tokens
- Comisiones
- Análisis de ingresos
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class SubscriptionTier(Enum):
    """Niveles de suscripción"""
    FREE = "free"
    BASIC = "basic"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class PaymentStatus(Enum):
    """Estado de pago"""
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    REFUNDED = "refunded"


@dataclass
class Subscription:
    """Suscripción de usuario"""
    user_id: str
    tier: SubscriptionTier
    start_date: datetime
    end_date: Optional[datetime] = None
    auto_renew: bool = True
    payment_method: Optional[str] = None
    status: str = "active"


@dataclass
class Payment:
    """Pago"""
    payment_id: str
    user_id: str
    amount: float
    currency: str = "USD"
    description: str = ""
    status: PaymentStatus = PaymentStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None


@dataclass
class CreditTransaction:
    """Transacción de créditos"""
    transaction_id: str
    user_id: str
    amount: int  # Créditos
    type: str  # "earned", "spent", "purchased", "refunded"
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)


class MonetizationService:
    """Servicio de monetización"""
    
    def __init__(self):
        self.subscriptions: Dict[str, Subscription] = {}
        self.payments: Dict[str, Payment] = {}
        self.credit_transactions: Dict[str, List[CreditTransaction]] = {}
        self.user_credits: Dict[str, int] = defaultdict(int)
        self.commission_rate: float = 0.15  # 15% de comisión
        
        # Precios de suscripción (en USD)
        self.tier_prices = {
            SubscriptionTier.FREE: 0.0,
            SubscriptionTier.BASIC: 9.99,
            SubscriptionTier.PRO: 29.99,
            SubscriptionTier.ENTERPRISE: 99.99
        }
        
        logger.info("MonetizationService initialized")
    
    def create_subscription(
        self,
        user_id: str,
        tier: SubscriptionTier,
        duration_days: int = 30,
        auto_renew: bool = True
    ) -> Subscription:
        """
        Crea una suscripción
        
        Args:
            user_id: ID del usuario
            tier: Nivel de suscripción
            duration_days: Duración en días
            auto_renew: Auto-renovación
        
        Returns:
            Subscription
        """
        start_date = datetime.now()
        end_date = start_date + timedelta(days=duration_days)
        
        subscription = Subscription(
            user_id=user_id,
            tier=tier,
            start_date=start_date,
            end_date=end_date,
            auto_renew=auto_renew
        )
        
        self.subscriptions[user_id] = subscription
        
        # Procesar pago
        price = self.tier_prices.get(tier, 0.0)
        if price > 0:
            self.create_payment(
                payment_id=f"sub_{user_id}_{int(datetime.now().timestamp())}",
                user_id=user_id,
                amount=price,
                description=f"Subscription: {tier.value}"
            )
        
        logger.info(f"Subscription created for user {user_id}: {tier.value}")
        return subscription
    
    def create_payment(
        self,
        payment_id: str,
        user_id: str,
        amount: float,
        currency: str = "USD",
        description: str = ""
    ) -> Payment:
        """
        Crea un pago
        
        Args:
            payment_id: ID único del pago
            user_id: ID del usuario
            amount: Monto
            currency: Moneda
            description: Descripción
        
        Returns:
            Payment
        """
        payment = Payment(
            payment_id=payment_id,
            user_id=user_id,
            amount=amount,
            currency=currency,
            description=description
        )
        
        self.payments[payment_id] = payment
        
        # En producción, esto procesaría el pago real
        # Por ahora, marcamos como completado
        payment.status = PaymentStatus.COMPLETED
        payment.completed_at = datetime.now()
        
        logger.info(f"Payment created: {payment_id}")
        return payment
    
    def add_credits(
        self,
        user_id: str,
        amount: int,
        transaction_type: str = "purchased",
        description: str = ""
    ) -> CreditTransaction:
        """
        Agrega créditos a un usuario
        
        Args:
            user_id: ID del usuario
            amount: Cantidad de créditos
            transaction_type: Tipo de transacción
            description: Descripción
        
        Returns:
            CreditTransaction
        """
        transaction_id = f"credit_{user_id}_{int(datetime.now().timestamp())}"
        
        transaction = CreditTransaction(
            transaction_id=transaction_id,
            user_id=user_id,
            amount=amount,
            type=transaction_type,
            description=description
        )
        
        if user_id not in self.credit_transactions:
            self.credit_transactions[user_id] = []
        
        self.credit_transactions[user_id].append(transaction)
        self.user_credits[user_id] += amount
        
        logger.info(f"Credits added to user {user_id}: {amount}")
        return transaction
    
    def spend_credits(
        self,
        user_id: str,
        amount: int,
        description: str = ""
    ) -> bool:
        """
        Gasta créditos de un usuario
        
        Args:
            user_id: ID del usuario
            amount: Cantidad a gastar
            description: Descripción
        
        Returns:
            True si se gastaron exitosamente
        """
        if self.user_credits.get(user_id, 0) < amount:
            return False
        
        transaction = CreditTransaction(
            transaction_id=f"spend_{user_id}_{int(datetime.now().timestamp())}",
            user_id=user_id,
            amount=-amount,
            type="spent",
            description=description
        )
        
        if user_id not in self.credit_transactions:
            self.credit_transactions[user_id] = []
        
        self.credit_transactions[user_id].append(transaction)
        self.user_credits[user_id] -= amount
        
        logger.info(f"Credits spent by user {user_id}: {amount}")
        return True
    
    def get_user_credits(self, user_id: str) -> int:
        """Obtiene créditos de un usuario"""
        return self.user_credits.get(user_id, 0)
    
    def calculate_commission(self, amount: float) -> float:
        """
        Calcula comisión
        
        Args:
            amount: Monto
        
        Returns:
            Comisión
        """
        return amount * self.commission_rate
    
    def get_revenue_stats(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Obtiene estadísticas de ingresos
        
        Args:
            start_date: Fecha de inicio
            end_date: Fecha de fin
        
        Returns:
            Estadísticas
        """
        payments = list(self.payments.values())
        
        if start_date:
            payments = [p for p in payments if p.created_at >= start_date]
        if end_date:
            payments = [p for p in payments if p.created_at <= end_date]
        
        completed_payments = [p for p in payments if p.status == PaymentStatus.COMPLETED]
        
        total_revenue = sum(p.amount for p in completed_payments)
        total_commission = sum(self.calculate_commission(p.amount) for p in completed_payments)
        net_revenue = total_revenue - total_commission
        
        return {
            "total_revenue": total_revenue,
            "total_commission": total_commission,
            "net_revenue": net_revenue,
            "payment_count": len(completed_payments),
            "subscription_count": len([s for s in self.subscriptions.values() if s.status == "active"]),
            "total_credits_sold": sum(
                t.amount for transactions in self.credit_transactions.values()
                for t in transactions
                if t.type == "purchased"
            )
        }
    
    def get_user_subscription(self, user_id: str) -> Optional[Subscription]:
        """Obtiene suscripción de un usuario"""
        subscription = self.subscriptions.get(user_id)
        
        if subscription and subscription.end_date:
            if datetime.now() > subscription.end_date:
                subscription.status = "expired"
        
        return subscription
    
    def is_subscription_active(self, user_id: str, tier: Optional[SubscriptionTier] = None) -> bool:
        """
        Verifica si un usuario tiene suscripción activa
        
        Args:
            user_id: ID del usuario
            tier: Nivel requerido (opcional)
        
        Returns:
            True si tiene suscripción activa
        """
        subscription = self.get_user_subscription(user_id)
        
        if not subscription or subscription.status != "active":
            return False
        
        if tier:
            tier_order = {
                SubscriptionTier.FREE: 0,
                SubscriptionTier.BASIC: 1,
                SubscriptionTier.PRO: 2,
                SubscriptionTier.ENTERPRISE: 3
            }
            user_tier_level = tier_order.get(subscription.tier, 0)
            required_tier_level = tier_order.get(tier, 0)
            return user_tier_level >= required_tier_level
        
        return True


# Instancia global
_monetization_service: Optional[MonetizationService] = None


def get_monetization_service() -> MonetizationService:
    """Obtiene la instancia global del servicio de monetización"""
    global _monetization_service
    if _monetization_service is None:
        _monetization_service = MonetizationService()
    return _monetization_service

