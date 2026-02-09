"""
Booking Service - Sistema de reservas y citas
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class BookingStatus(str, Enum):
    """Estados de reserva"""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    NO_SHOW = "no_show"


class BookingService:
    """Servicio para reservas y citas"""
    
    def __init__(self):
        self.bookings: Dict[str, Dict[str, Any]] = {}
        self.availability: Dict[str, List[Dict[str, Any]]] = {}
        self.services: Dict[str, Dict[str, Any]] = {}
    
    def create_service(
        self,
        store_id: str,
        name: str,
        duration_minutes: int,
        price: float,
        description: Optional[str] = None
    ) -> Dict[str, Any]:
        """Crear servicio reservable"""
        
        service_id = f"svc_{store_id}_{len(self.services.get(store_id, [])) + 1}"
        
        service = {
            "service_id": service_id,
            "store_id": store_id,
            "name": name,
            "duration_minutes": duration_minutes,
            "price": price,
            "description": description,
            "is_active": True,
            "created_at": datetime.now().isoformat()
        }
        
        if store_id not in self.services:
            self.services[store_id] = {}
        
        self.services[store_id][service_id] = service
        
        return service
    
    def set_availability(
        self,
        store_id: str,
        day_of_week: int,  # 0=Monday, 6=Sunday
        start_time: str,  # "09:00"
        end_time: str,  # "18:00"
        is_available: bool = True
    ) -> Dict[str, Any]:
        """Configurar disponibilidad"""
        
        if store_id not in self.availability:
            self.availability[store_id] = []
        
        availability = {
            "store_id": store_id,
            "day_of_week": day_of_week,
            "start_time": start_time,
            "end_time": end_time,
            "is_available": is_available
        }
        
        self.availability[store_id].append(availability)
        
        return availability
    
    def create_booking(
        self,
        store_id: str,
        service_id: str,
        customer_name: str,
        customer_email: str,
        customer_phone: Optional[str],
        booking_date: str,
        booking_time: str,
        notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """Crear reserva"""
        
        # Verificar disponibilidad
        if not self._check_availability(store_id, booking_date, booking_time):
            raise ValueError("Horario no disponible")
        
        booking_id = f"bk_{store_id}_{len(self.bookings.get(store_id, [])) + 1}"
        
        service = self.services.get(store_id, {}).get(service_id)
        if not service:
            raise ValueError(f"Servicio {service_id} no encontrado")
        
        booking_datetime = datetime.fromisoformat(f"{booking_date}T{booking_time}")
        end_datetime = booking_datetime + timedelta(minutes=service["duration_minutes"])
        
        booking = {
            "booking_id": booking_id,
            "store_id": store_id,
            "service_id": service_id,
            "service_name": service["name"],
            "customer_name": customer_name,
            "customer_email": customer_email,
            "customer_phone": customer_phone,
            "booking_date": booking_date,
            "booking_time": booking_time,
            "booking_datetime": booking_datetime.isoformat(),
            "end_datetime": end_datetime.isoformat(),
            "duration_minutes": service["duration_minutes"],
            "price": service["price"],
            "status": BookingStatus.PENDING.value,
            "notes": notes,
            "created_at": datetime.now().isoformat()
        }
        
        if store_id not in self.bookings:
            self.bookings[store_id] = []
        
        self.bookings[store_id].append(booking)
        
        return booking
    
    def _check_availability(
        self,
        store_id: str,
        date: str,
        time: str
    ) -> bool:
        """Verificar disponibilidad"""
        booking_date = datetime.fromisoformat(date)
        day_of_week = booking_date.weekday()
        
        # Verificar disponibilidad configurada
        store_availability = self.availability.get(store_id, [])
        day_availability = next(
            (a for a in store_availability if a["day_of_week"] == day_of_week),
            None
        )
        
        if day_availability and not day_availability["is_available"]:
            return False
        
        # Verificar conflictos con otras reservas
        existing_bookings = self.bookings.get(store_id, [])
        booking_time = datetime.fromisoformat(f"{date}T{time}")
        
        for existing in existing_bookings:
            if existing["status"] in [BookingStatus.CONFIRMED.value, BookingStatus.PENDING.value]:
                existing_start = datetime.fromisoformat(existing["booking_datetime"])
                existing_end = datetime.fromisoformat(existing["end_datetime"])
                
                if booking_time < existing_end and booking_time >= existing_start:
                    return False
        
        return True
    
    def confirm_booking(self, booking_id: str) -> bool:
        """Confirmar reserva"""
        booking = self._find_booking(booking_id)
        
        if not booking:
            return False
        
        booking["status"] = BookingStatus.CONFIRMED.value
        booking["confirmed_at"] = datetime.now().isoformat()
        
        return True
    
    def cancel_booking(self, booking_id: str, reason: Optional[str] = None) -> bool:
        """Cancelar reserva"""
        booking = self._find_booking(booking_id)
        
        if not booking:
            return False
        
        booking["status"] = BookingStatus.CANCELLED.value
        booking["cancelled_at"] = datetime.now().isoformat()
        booking["cancellation_reason"] = reason
        
        return True
    
    def get_bookings(
        self,
        store_id: str,
        date: Optional[str] = None,
        status: Optional[BookingStatus] = None
    ) -> List[Dict[str, Any]]:
        """Obtener reservas"""
        bookings = self.bookings.get(store_id, [])
        
        if date:
            bookings = [b for b in bookings if b["booking_date"] == date]
        
        if status:
            bookings = [b for b in bookings if b["status"] == status.value]
        
        return bookings
    
    def get_available_slots(
        self,
        store_id: str,
        date: str,
        service_id: str
    ) -> List[str]:
        """Obtener slots disponibles"""
        service = self.services.get(store_id, {}).get(service_id)
        if not service:
            return []
        
        booking_date = datetime.fromisoformat(date)
        day_of_week = booking_date.weekday()
        
        # Obtener horario del día
        store_availability = self.availability.get(store_id, [])
        day_availability = next(
            (a for a in store_availability if a["day_of_week"] == day_of_week),
            None
        )
        
        if not day_availability or not day_availability["is_available"]:
            return []
        
        # Generar slots
        start_time = datetime.strptime(day_availability["start_time"], "%H:%M")
        end_time = datetime.strptime(day_availability["end_time"], "%H:%M")
        duration = timedelta(minutes=service["duration_minutes"])
        
        slots = []
        current = start_time
        
        while current + duration <= end_time:
            slot_time = current.strftime("%H:%M")
            if self._check_availability(store_id, date, slot_time):
                slots.append(slot_time)
            current += duration
        
        return slots
    
    def _find_booking(self, booking_id: str) -> Optional[Dict[str, Any]]:
        """Encontrar reserva"""
        for store_bookings in self.bookings.values():
            for booking in store_bookings:
                if booking["booking_id"] == booking_id:
                    return booking
        return None




