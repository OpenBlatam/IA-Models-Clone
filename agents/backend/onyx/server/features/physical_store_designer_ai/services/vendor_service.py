"""
Vendor Service - Integración con proveedores
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class VendorService:
    """Servicio para gestión de proveedores"""
    
    def __init__(self):
        self.vendors: Dict[str, Dict[str, Any]] = {}
        self.quotations: Dict[str, List[Dict[str, Any]]] = {}
    
    def register_vendor(
        self,
        name: str,
        category: str,  # "furniture", "decoration", "equipment", "materials"
        contact_info: Dict[str, str],
        specialties: List[str],
        rating: Optional[float] = None
    ) -> Dict[str, Any]:
        """Registrar nuevo proveedor"""
        
        vendor_id = f"vendor_{len(self.vendors) + 1}"
        
        vendor = {
            "vendor_id": vendor_id,
            "name": name,
            "category": category,
            "contact_info": contact_info,
            "specialties": specialties,
            "rating": rating or 5.0,
            "created_at": datetime.now().isoformat(),
            "is_active": True,
            "orders_count": 0
        }
        
        self.vendors[vendor_id] = vendor
        
        logger.info(f"Proveedor registrado: {name}")
        return vendor
    
    def get_vendors(
        self,
        category: Optional[str] = None,
        min_rating: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Obtener proveedores filtrados"""
        vendors = list(self.vendors.values())
        
        if category:
            vendors = [v for v in vendors if v["category"] == category]
        
        if min_rating:
            vendors = [v for v in vendors if v.get("rating", 0) >= min_rating]
        
        return vendors
    
    def request_quotation(
        self,
        store_id: str,
        vendor_id: str,
        items: List[Dict[str, Any]],
        requirements: Optional[str] = None
    ) -> Dict[str, Any]:
        """Solicitar cotización a proveedor"""
        
        if vendor_id not in self.vendors:
            raise ValueError(f"Proveedor {vendor_id} no encontrado")
        
        quotation_id = f"quote_{store_id}_{vendor_id}_{len(self.quotations.get(store_id, [])) + 1}"
        
        quotation = {
            "quotation_id": quotation_id,
            "store_id": store_id,
            "vendor_id": vendor_id,
            "vendor_name": self.vendors[vendor_id]["name"],
            "items": items,
            "requirements": requirements,
            "status": "pending",  # "pending", "received", "accepted", "rejected"
            "created_at": datetime.now().isoformat(),
            "total_amount": None,
            "valid_until": None
        }
        
        if store_id not in self.quotations:
            self.quotations[store_id] = []
        
        self.quotations[store_id].append(quotation)
        
        return quotation
    
    def get_quotations(self, store_id: str) -> List[Dict[str, Any]]:
        """Obtener cotizaciones de un diseño"""
        return self.quotations.get(store_id, [])
    
    def recommend_vendors(
        self,
        store_type: str,
        needs: List[str]
    ) -> List[Dict[str, Any]]:
        """Recomendar proveedores según necesidades"""
        
        recommendations = []
        
        for vendor in self.vendors.values():
            if not vendor.get("is_active", True):
                continue
            
            score = 0
            
            # Score por categoría
            if vendor["category"] in needs:
                score += 3
            
            # Score por especialidades
            for specialty in vendor.get("specialties", []):
                if specialty.lower() in [n.lower() for n in needs]:
                    score += 2
            
            # Score por rating
            score += vendor.get("rating", 0) * 0.5
            
            if score > 0:
                recommendations.append({
                    "vendor": vendor,
                    "score": score,
                    "match_reasons": self._get_match_reasons(vendor, needs)
                })
        
        # Ordenar por score
        recommendations.sort(key=lambda x: x["score"], reverse=True)
        
        return recommendations[:10]  # Top 10
    
    def _get_match_reasons(
        self,
        vendor: Dict[str, Any],
        needs: List[str]
    ) -> List[str]:
        """Obtener razones de match"""
        reasons = []
        
        if vendor["category"] in needs:
            reasons.append(f"Categoría: {vendor['category']}")
        
        for specialty in vendor.get("specialties", []):
            if specialty.lower() in [n.lower() for n in needs]:
                reasons.append(f"Especialidad: {specialty}")
        
        if vendor.get("rating", 0) >= 4.5:
            reasons.append("Alta calificación")
        
        return reasons
    
    def compare_quotations(
        self,
        store_id: str
    ) -> Dict[str, Any]:
        """Comparar cotizaciones recibidas"""
        quotations = self.quotations.get(store_id, [])
        
        received_quotes = [q for q in quotations if q["status"] == "received"]
        
        if not received_quotes:
            return {"message": "No hay cotizaciones recibidas para comparar"}
        
        comparison = {
            "total_quotations": len(received_quotes),
            "quotations": received_quotes,
            "cheapest": min(received_quotes, key=lambda x: x.get("total_amount", float('inf'))),
            "most_expensive": max(received_quotes, key=lambda x: x.get("total_amount", 0)),
            "average_price": sum(q.get("total_amount", 0) for q in received_quotes) / len(received_quotes) if received_quotes else 0
        }
        
        return comparison




