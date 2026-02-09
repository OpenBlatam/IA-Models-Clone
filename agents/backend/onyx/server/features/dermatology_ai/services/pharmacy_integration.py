"""
Sistema de integración con farmacias
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import uuid


@dataclass
class Pharmacy:
    """Farmacia"""
    id: str
    name: str
    address: str
    phone: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    rating: Optional[float] = None
    hours: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "id": self.id,
            "name": self.name,
            "address": self.address,
            "phone": self.phone,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "rating": self.rating,
            "hours": self.hours
        }


@dataclass
class ProductAvailability:
    """Disponibilidad de producto"""
    product_id: str
    product_name: str
    pharmacy_id: str
    pharmacy_name: str
    available: bool
    price: Optional[float] = None
    stock_quantity: Optional[int] = None
    distance_km: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "product_id": self.product_id,
            "product_name": self.product_name,
            "pharmacy_id": self.pharmacy_id,
            "pharmacy_name": self.pharmacy_name,
            "available": self.available,
            "price": self.price,
            "stock_quantity": self.stock_quantity,
            "distance_km": self.distance_km
        }


class PharmacyIntegration:
    """Sistema de integración con farmacias"""
    
    def __init__(self):
        """Inicializa el sistema"""
        self.pharmacies: Dict[str, Pharmacy] = {}  # pharmacy_id -> pharmacy
        self.availability: Dict[str, List[ProductAvailability]] = {}  # product_id -> [availability]
    
    def register_pharmacy(self, name: str, address: str, phone: Optional[str] = None,
                          latitude: Optional[float] = None, longitude: Optional[float] = None,
                          rating: Optional[float] = None) -> Pharmacy:
        """Registra una farmacia"""
        pharmacy = Pharmacy(
            id=str(uuid.uuid4()),
            name=name,
            address=address,
            phone=phone,
            latitude=latitude,
            longitude=longitude,
            rating=rating
        )
        
        self.pharmacies[pharmacy.id] = pharmacy
        return pharmacy
    
    def find_nearby_pharmacies(self, latitude: float, longitude: float,
                               radius_km: float = 5.0) -> List[Pharmacy]:
        """Encuentra farmacias cercanas"""
        # Simulación - en producción usaría cálculo real de distancia
        nearby = []
        
        for pharmacy in self.pharmacies.values():
            if pharmacy.latitude and pharmacy.longitude:
                # Cálculo simplificado de distancia
                distance = abs(latitude - pharmacy.latitude) + abs(longitude - pharmacy.longitude)
                if distance * 111 <= radius_km:  # Aproximación: 1 grado ≈ 111 km
                    nearby.append(pharmacy)
        
        # Ordenar por distancia
        nearby.sort(key=lambda p: abs(latitude - (p.latitude or 0)) + abs(longitude - (p.longitude or 0)))
        
        return nearby
    
    def check_product_availability(self, product_id: str, product_name: str,
                                   pharmacy_id: Optional[str] = None,
                                   latitude: Optional[float] = None,
                                   longitude: Optional[float] = None) -> List[ProductAvailability]:
        """Verifica disponibilidad de producto"""
        results = []
        
        pharmacies_to_check = []
        
        if pharmacy_id:
            if pharmacy_id in self.pharmacies:
                pharmacies_to_check = [self.pharmacies[pharmacy_id]]
        elif latitude and longitude:
            pharmacies_to_check = self.find_nearby_pharmacies(latitude, longitude)
        else:
            pharmacies_to_check = list(self.pharmacies.values())
        
        for pharmacy in pharmacies_to_check[:5]:  # Limitar a 5 farmacias
            # Simulación de disponibilidad
            available = True  # En producción consultaría API de farmacia
            price = 25.0  # Precio simulado
            
            availability = ProductAvailability(
                product_id=product_id,
                product_name=product_name,
                pharmacy_id=pharmacy.id,
                pharmacy_name=pharmacy.name,
                available=available,
                price=price,
                distance_km=None  # Calcular si hay coordenadas
            )
            
            results.append(availability)
        
        return results
    
    def get_pharmacy(self, pharmacy_id: str) -> Optional[Pharmacy]:
        """Obtiene información de farmacia"""
        return self.pharmacies.get(pharmacy_id)

