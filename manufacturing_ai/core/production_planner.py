"""
Production Planner
==================

Sistema de planificación de producción inteligente.
"""

import logging
import uuid
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    """Estado de orden de producción."""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    DELAYED = "delayed"


class Priority(Enum):
    """Prioridad de orden."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4


@dataclass
class ProductionOrder:
    """Orden de producción."""
    order_id: str
    product_id: str
    quantity: int
    priority: Priority
    due_date: str
    status: OrderStatus = OrderStatus.PENDING
    estimated_duration: float = 0.0  # horas
    actual_duration: Optional[float] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    assigned_resources: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class Resource:
    """Recurso de manufactura."""
    resource_id: str
    name: str
    resource_type: str  # machine, robot, worker, etc.
    capacity: float = 1.0
    current_load: float = 0.0
    availability: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class ProductionPlanner:
    """
    Planificador de producción.
    
    Optimiza la planificación de órdenes de producción.
    """
    
    def __init__(self):
        """Inicializar planificador."""
        self.orders: Dict[str, ProductionOrder] = {}
        self.resources: Dict[str, Resource] = {}
        self.schedule: List[Dict[str, Any]] = []
    
    def create_order(
        self,
        product_id: str,
        quantity: int,
        due_date: str,
        priority: Priority = Priority.MEDIUM,
        estimated_duration: Optional[float] = None
    ) -> str:
        """
        Crear orden de producción.
        
        Args:
            product_id: ID del producto
            quantity: Cantidad a producir
            due_date: Fecha límite
            priority: Prioridad
            estimated_duration: Duración estimada (opcional)
            
        Returns:
            ID de la orden
        """
        order_id = str(uuid.uuid4())
        
        order = ProductionOrder(
            order_id=order_id,
            product_id=product_id,
            quantity=quantity,
            priority=priority,
            due_date=due_date,
            estimated_duration=estimated_duration or (quantity * 0.5)  # 0.5 horas por unidad
        )
        
        self.orders[order_id] = order
        logger.info(f"Created production order: {order_id}")
        
        return order_id
    
    def register_resource(
        self,
        resource_id: str,
        name: str,
        resource_type: str,
        capacity: float = 1.0
    ):
        """
        Registrar recurso.
        
        Args:
            resource_id: ID del recurso
            name: Nombre
            resource_type: Tipo de recurso
            capacity: Capacidad
        """
        resource = Resource(
            resource_id=resource_id,
            name=name,
            resource_type=resource_type,
            capacity=capacity
        )
        
        self.resources[resource_id] = resource
        logger.info(f"Registered resource: {resource_id}")
    
    def schedule_order(self, order_id: str) -> bool:
        """
        Programar orden.
        
        Args:
            order_id: ID de la orden
            
        Returns:
            True si se programó exitosamente
        """
        if order_id not in self.orders:
            logger.error(f"Order not found: {order_id}")
            return False
        
        order = self.orders[order_id]
        
        # Encontrar recursos disponibles
        available_resources = [
            r for r in self.resources.values()
            if r.availability and r.current_load < r.capacity
        ]
        
        if not available_resources:
            logger.warning("No available resources")
            return False
        
        # Asignar recursos (simplificado: asignar el primero disponible)
        assigned = available_resources[0]
        order.assigned_resources = [assigned.resource_id]
        order.status = OrderStatus.SCHEDULED
        
        # Calcular tiempo de inicio
        start_time = datetime.now()
        order.start_time = start_time.isoformat()
        
        # Actualizar carga del recurso
        assigned.current_load += order.estimated_duration
        
        # Agregar a schedule
        self.schedule.append({
            "order_id": order_id,
            "resource_id": assigned.resource_id,
            "start_time": order.start_time,
            "duration": order.estimated_duration
        })
        
        logger.info(f"Scheduled order {order_id} on resource {assigned.resource_id}")
        return True
    
    def start_order(self, order_id: str) -> bool:
        """Iniciar orden."""
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        if order.status != OrderStatus.SCHEDULED:
            logger.warning(f"Order {order_id} not scheduled")
            return False
        
        order.status = OrderStatus.IN_PROGRESS
        order.start_time = datetime.now().isoformat()
        logger.info(f"Started order: {order_id}")
        return True
    
    def complete_order(self, order_id: str, actual_duration: Optional[float] = None) -> bool:
        """Completar orden."""
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        order.status = OrderStatus.COMPLETED
        order.end_time = datetime.now().isoformat()
        
        if actual_duration:
            order.actual_duration = actual_duration
        
        # Liberar recursos
        for resource_id in order.assigned_resources:
            if resource_id in self.resources:
                resource = self.resources[resource_id]
                resource.current_load -= order.estimated_duration
                resource.current_load = max(0, resource.current_load)
        
        logger.info(f"Completed order: {order_id}")
        return True
    
    def optimize_schedule(self) -> Dict[str, Any]:
        """
        Optimizar schedule.
        
        Returns:
            Resultado de optimización
        """
        # Ordenar órdenes por prioridad y fecha límite
        pending_orders = [
            o for o in self.orders.values()
            if o.status == OrderStatus.PENDING
        ]
        
        pending_orders.sort(
            key=lambda x: (
                x.priority.value,
                datetime.fromisoformat(x.due_date).timestamp()
            ),
            reverse=True
        )
        
        # Programar órdenes
        scheduled_count = 0
        for order in pending_orders:
            if self.schedule_order(order.order_id):
                scheduled_count += 1
        
        return {
            "scheduled_orders": scheduled_count,
            "total_pending": len(pending_orders),
            "optimization_time": datetime.now().isoformat()
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas."""
        status_counts = {}
        for order in self.orders.values():
            status_counts[order.status.value] = status_counts.get(order.status.value, 0) + 1
        
        return {
            "total_orders": len(self.orders),
            "status_counts": status_counts,
            "total_resources": len(self.resources),
            "available_resources": sum(1 for r in self.resources.values() if r.availability)
        }

