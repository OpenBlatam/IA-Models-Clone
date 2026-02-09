"""
Sistema de recordatorios inteligentes de productos
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import uuid


@dataclass
class ProductReminder:
    """Recordatorio de producto"""
    id: str
    user_id: str
    product_id: str
    product_name: str
    reminder_type: str  # "low_stock", "expiring", "reorder", "usage"
    message: str
    priority: int  # 1-5
    due_date: str
    completed: bool = False
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "product_id": self.product_id,
            "product_name": self.product_name,
            "reminder_type": self.reminder_type,
            "message": self.message,
            "priority": self.priority,
            "due_date": self.due_date,
            "completed": self.completed,
            "created_at": self.created_at
        }


class ProductReminderSystem:
    """Sistema de recordatorios de productos"""
    
    def __init__(self):
        """Inicializa el sistema"""
        self.reminders: Dict[str, List[ProductReminder]] = {}  # user_id -> [reminders]
        self.product_inventory: Dict[str, Dict] = {}  # user_id -> {product_id: inventory_data}
    
    def add_product_to_inventory(self, user_id: str, product_id: str, product_name: str,
                                 purchase_date: str, expiry_date: Optional[str] = None,
                                 quantity: int = 1, usage_frequency: str = "daily"):
        """Agrega producto al inventario"""
        if user_id not in self.product_inventory:
            self.product_inventory[user_id] = {}
        
        self.product_inventory[user_id][product_id] = {
            "product_name": product_name,
            "purchase_date": purchase_date,
            "expiry_date": expiry_date,
            "quantity": quantity,
            "usage_frequency": usage_frequency
        }
        
        # Crear recordatorios automáticos
        self._create_automatic_reminders(user_id, product_id, product_name, expiry_date, quantity)
    
    def _create_automatic_reminders(self, user_id: str, product_id: str, product_name: str,
                                   expiry_date: Optional[str], quantity: int):
        """Crea recordatorios automáticos"""
        reminders = []
        
        # Recordatorio de bajo stock
        if quantity <= 1:
            reminder = ProductReminder(
                id=str(uuid.uuid4()),
                user_id=user_id,
                product_id=product_id,
                product_name=product_name,
                reminder_type="low_stock",
                message=f"{product_name} está por agotarse. Considera reordenar.",
                priority=2,
                due_date=datetime.now().isoformat()
            )
            reminders.append(reminder)
        
        # Recordatorio de expiración
        if expiry_date:
            expiry = datetime.fromisoformat(expiry_date)
            days_until_expiry = (expiry - datetime.now()).days
            
            if 0 < days_until_expiry <= 30:
                reminder = ProductReminder(
                    id=str(uuid.uuid4()),
                    user_id=user_id,
                    product_id=product_id,
                    product_name=product_name,
                    reminder_type="expiring",
                    message=f"{product_name} expira en {days_until_expiry} días.",
                    priority=1 if days_until_expiry <= 7 else 3,
                    due_date=expiry_date
                )
                reminders.append(reminder)
        
        # Guardar recordatorios
        if user_id not in self.reminders:
            self.reminders[user_id] = []
        
        self.reminders[user_id].extend(reminders)
    
    def get_user_reminders(self, user_id: str, reminder_type: Optional[str] = None,
                          include_completed: bool = False) -> List[ProductReminder]:
        """Obtiene recordatorios del usuario"""
        user_reminders = self.reminders.get(user_id, [])
        
        if reminder_type:
            user_reminders = [r for r in user_reminders if r.reminder_type == reminder_type]
        
        if not include_completed:
            user_reminders = [r for r in user_reminders if not r.completed]
        
        user_reminders.sort(key=lambda x: (x.priority, x.due_date))
        return user_reminders
    
    def mark_reminder_completed(self, user_id: str, reminder_id: str) -> bool:
        """Marca recordatorio como completado"""
        user_reminders = self.reminders.get(user_id, [])
        
        for reminder in user_reminders:
            if reminder.id == reminder_id:
                reminder.completed = True
                return True
        
        return False
    
    def get_inventory_summary(self, user_id: str) -> Dict:
        """Obtiene resumen de inventario"""
        inventory = self.product_inventory.get(user_id, {})
        
        total_products = len(inventory)
        low_stock = sum(1 for p in inventory.values() if p.get("quantity", 0) <= 1)
        expiring_soon = 0
        
        for product in inventory.values():
            expiry = product.get("expiry_date")
            if expiry:
                days = (datetime.fromisoformat(expiry) - datetime.now()).days
                if 0 < days <= 30:
                    expiring_soon += 1
        
        return {
            "total_products": total_products,
            "low_stock_count": low_stock,
            "expiring_soon_count": expiring_soon
        }






