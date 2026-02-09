"""
Billing Service - Sistema de facturación y pagos
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class PaymentStatus(str, Enum):
    """Estados de pago"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    REFUNDED = "refunded"


class SubscriptionPlan(str, Enum):
    """Planes de suscripción"""
    FREE = "free"
    BASIC = "basic"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


class BillingService:
    """Servicio para facturación y pagos"""
    
    def __init__(self):
        self.subscriptions: Dict[str, Dict[str, Any]] = {}
        self.invoices: Dict[str, List[Dict[str, Any]]] = {}
        self.payments: Dict[str, List[Dict[str, Any]]] = {}
    
    def create_subscription(
        self,
        user_id: str,
        plan: SubscriptionPlan,
        payment_method: Optional[str] = None
    ) -> Dict[str, Any]:
        """Crear suscripción"""
        
        plan_prices = {
            SubscriptionPlan.FREE: 0,
            SubscriptionPlan.BASIC: 29.99,
            SubscriptionPlan.PROFESSIONAL: 79.99,
            SubscriptionPlan.ENTERPRISE: 199.99
        }
        
        plan_features = {
            SubscriptionPlan.FREE: {
                "designs_limit": 3,
                "features": ["basic_design", "basic_analysis"]
            },
            SubscriptionPlan.BASIC: {
                "designs_limit": 10,
                "features": ["all_designs", "basic_analysis", "export_pdf"]
            },
            SubscriptionPlan.PROFESSIONAL: {
                "designs_limit": -1,  # Ilimitado
                "features": ["all_designs", "all_analysis", "all_exports", "collaboration"]
            },
            SubscriptionPlan.ENTERPRISE: {
                "designs_limit": -1,
                "features": ["all_features", "api_access", "priority_support", "custom_integrations"]
            }
        }
        
        subscription = {
            "subscription_id": f"sub_{user_id}_{datetime.now().strftime('%Y%m%d')}",
            "user_id": user_id,
            "plan": plan.value,
            "price": plan_prices[plan],
            "features": plan_features[plan]["features"],
            "designs_limit": plan_features[plan]["designs_limit"],
            "status": "active",
            "started_at": datetime.now().isoformat(),
            "next_billing_date": (datetime.now() + timedelta(days=30)).isoformat(),
            "payment_method": payment_method
        }
        
        self.subscriptions[user_id] = subscription
        
        # Crear factura inicial
        if plan != SubscriptionPlan.FREE:
            self.create_invoice(user_id, plan_prices[plan], "Monthly subscription")
        
        return subscription
    
    def create_invoice(
        self,
        user_id: str,
        amount: float,
        description: str,
        items: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Crear factura"""
        
        invoice_id = f"inv_{user_id}_{len(self.invoices.get(user_id, [])) + 1}"
        
        invoice = {
            "invoice_id": invoice_id,
            "user_id": user_id,
            "amount": amount,
            "description": description,
            "items": items or [{"description": description, "amount": amount}],
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "due_date": (datetime.now() + timedelta(days=30)).isoformat(),
            "paid_at": None
        }
        
        if user_id not in self.invoices:
            self.invoices[user_id] = []
        
        self.invoices[user_id].append(invoice)
        
        return invoice
    
    def process_payment(
        self,
        invoice_id: str,
        payment_method: str,
        amount: Optional[float] = None
    ) -> Dict[str, Any]:
        """Procesar pago"""
        
        # Buscar factura
        invoice = None
        user_id = None
        
        for uid, invoices in self.invoices.items():
            for inv in invoices:
                if inv["invoice_id"] == invoice_id:
                    invoice = inv
                    user_id = uid
                    break
            if invoice:
                break
        
        if not invoice:
            raise ValueError(f"Factura {invoice_id} no encontrada")
        
        payment_id = f"pay_{invoice_id}_{len(self.payments.get(user_id, [])) + 1}"
        payment_amount = amount or invoice["amount"]
        
        payment = {
            "payment_id": payment_id,
            "invoice_id": invoice_id,
            "user_id": user_id,
            "amount": payment_amount,
            "payment_method": payment_method,
            "status": PaymentStatus.PROCESSING.value,
            "created_at": datetime.now().isoformat()
        }
        
        # Simular procesamiento (en producción, integrar con gateway de pagos)
        if payment_amount == invoice["amount"]:
            payment["status"] = PaymentStatus.COMPLETED.value
            payment["completed_at"] = datetime.now().isoformat()
            
            invoice["status"] = "paid"
            invoice["paid_at"] = datetime.now().isoformat()
        else:
            payment["status"] = PaymentStatus.FAILED.value
            payment["error"] = "Amount mismatch"
        
        if user_id not in self.payments:
            self.payments[user_id] = []
        
        self.payments[user_id].append(payment)
        
        return payment
    
    def get_subscription(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Obtener suscripción de usuario"""
        return self.subscriptions.get(user_id)
    
    def get_invoices(self, user_id: str) -> List[Dict[str, Any]]:
        """Obtener facturas de usuario"""
        return self.invoices.get(user_id, [])
    
    def get_payment_history(self, user_id: str) -> List[Dict[str, Any]]:
        """Obtener historial de pagos"""
        return self.payments.get(user_id, [])
    
    def check_feature_access(
        self,
        user_id: str,
        feature: str
    ) -> bool:
        """Verificar acceso a funcionalidad"""
        subscription = self.subscriptions.get(user_id)
        
        if not subscription:
            return False
        
        features = subscription.get("features", [])
        return feature in features or "all_features" in features




