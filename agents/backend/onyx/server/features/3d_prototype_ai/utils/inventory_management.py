"""
Inventory Management - Sistema de gestión de inventario
========================================================
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class InventoryStatus(str, Enum):
    """Estados de inventario"""
    IN_STOCK = "in_stock"
    LOW_STOCK = "low_stock"
    OUT_OF_STOCK = "out_of_stock"
    RESERVED = "reserved"
    DISCONTINUED = "discontinued"


class InventoryManagement:
    """Sistema de gestión de inventario"""
    
    def __init__(self):
        self.inventory: Dict[str, Dict[str, Any]] = {}
        self.movements: List[Dict[str, Any]] = []
        self.reservations: Dict[str, Dict[str, Any]] = {}
        self.low_stock_threshold = 10
    
    def add_item(self, item_id: str, name: str, category: str,
                initial_quantity: int = 0, unit: str = "unit",
                reorder_point: int = 10, reorder_quantity: int = 50):
        """Agrega un item al inventario"""
        item = {
            "id": item_id,
            "name": name,
            "category": category,
            "quantity": initial_quantity,
            "unit": unit,
            "reorder_point": reorder_point,
            "reorder_quantity": reorder_quantity,
            "status": self._calculate_status(initial_quantity, reorder_point),
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        self.inventory[item_id] = item
        
        logger.info(f"Item agregado al inventario: {item_id} - {name}")
        return item
    
    def update_quantity(self, item_id: str, quantity_change: int,
                      reason: str = "manual", reference: Optional[str] = None):
        """Actualiza cantidad de un item"""
        item = self.inventory.get(item_id)
        if not item:
            raise ValueError(f"Item no encontrado: {item_id}")
        
        old_quantity = item["quantity"]
        new_quantity = old_quantity + quantity_change
        
        if new_quantity < 0:
            raise ValueError(f"Cantidad insuficiente: {item_id}")
        
        item["quantity"] = new_quantity
        item["status"] = self._calculate_status(new_quantity, item["reorder_point"])
        item["updated_at"] = datetime.now().isoformat()
        
        # Registrar movimiento
        movement = {
            "item_id": item_id,
            "item_name": item["name"],
            "old_quantity": old_quantity,
            "new_quantity": new_quantity,
            "change": quantity_change,
            "reason": reason,
            "reference": reference,
            "timestamp": datetime.now().isoformat()
        }
        
        self.movements.append(movement)
        
        # Mantener solo últimos 10000 movimientos
        if len(self.movements) > 10000:
            self.movements = self.movements[-10000:]
        
        logger.info(f"Inventario actualizado: {item_id} - {old_quantity} -> {new_quantity}")
        return item
    
    def reserve_item(self, item_id: str, quantity: int, reservation_id: str,
                    expires_at: Optional[datetime] = None):
        """Reserva un item"""
        item = self.inventory.get(item_id)
        if not item:
            raise ValueError(f"Item no encontrado: {item_id}")
        
        available = item["quantity"] - sum(
            r["quantity"] for r in self.reservations.values()
            if r["item_id"] == item_id and not r.get("expired", False)
        )
        
        if available < quantity:
            raise ValueError(f"Cantidad insuficiente disponible: {item_id}")
        
        if not expires_at:
            expires_at = datetime.now() + timedelta(hours=24)
        
        reservation = {
            "id": reservation_id,
            "item_id": item_id,
            "quantity": quantity,
            "created_at": datetime.now().isoformat(),
            "expires_at": expires_at.isoformat(),
            "expired": False
        }
        
        self.reservations[reservation_id] = reservation
        
        logger.info(f"Item reservado: {item_id} - {quantity} unidades")
        return reservation
    
    def release_reservation(self, reservation_id: str):
        """Libera una reserva"""
        reservation = self.reservations.get(reservation_id)
        if reservation:
            reservation["expired"] = True
            reservation["released_at"] = datetime.now().isoformat()
            logger.info(f"Reserva liberada: {reservation_id}")
    
    def _calculate_status(self, quantity: int, reorder_point: int) -> str:
        """Calcula estado del inventario"""
        if quantity <= 0:
            return InventoryStatus.OUT_OF_STOCK.value
        elif quantity <= reorder_point:
            return InventoryStatus.LOW_STOCK.value
        else:
            return InventoryStatus.IN_STOCK.value
    
    def check_low_stock(self) -> List[Dict[str, Any]]:
        """Verifica items con stock bajo"""
        low_stock_items = [
            item for item in self.inventory.values()
            if item["status"] == InventoryStatus.LOW_STOCK.value
            or item["status"] == InventoryStatus.OUT_OF_STOCK.value
        ]
        
        return low_stock_items
    
    def get_inventory_report(self) -> Dict[str, Any]:
        """Obtiene reporte de inventario"""
        total_items = len(self.inventory)
        total_value = sum(item.get("value", 0) * item["quantity"] for item in self.inventory.values())
        
        status_counts = {}
        for status in InventoryStatus:
            status_counts[status.value] = sum(
                1 for item in self.inventory.values()
                if item["status"] == status.value
            )
        
        return {
            "total_items": total_items,
            "total_value": total_value,
            "status_distribution": status_counts,
            "low_stock_items": len(self.check_low_stock()),
            "total_movements": len(self.movements),
            "active_reservations": sum(1 for r in self.reservations.values() if not r.get("expired", False))
        }
    
    def get_item_history(self, item_id: str, days: int = 30) -> List[Dict[str, Any]]:
        """Obtiene historial de un item"""
        cutoff = datetime.now() - timedelta(days=days)
        
        item_movements = [
            m for m in self.movements
            if m["item_id"] == item_id
            and datetime.fromisoformat(m["timestamp"]) > cutoff
        ]
        
        return sorted(item_movements, key=lambda x: x["timestamp"], reverse=True)




