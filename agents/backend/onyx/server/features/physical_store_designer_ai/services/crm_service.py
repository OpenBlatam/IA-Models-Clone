"""
CRM Service - Integración con CRM
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class ContactStatus(str, Enum):
    """Estados de contacto"""
    LEAD = "lead"
    QUALIFIED = "qualified"
    PROPOSAL = "proposal"
    NEGOTIATION = "negotiation"
    CLOSED_WON = "closed_won"
    CLOSED_LOST = "closed_lost"


class CRMService:
    """Servicio para CRM"""
    
    def __init__(self):
        self.contacts: Dict[str, Dict[str, Any]] = {}
        self.deals: Dict[str, Dict[str, Any]] = {}
        self.activities: Dict[str, List[Dict[str, Any]]] = {}
    
    def create_contact(
        self,
        name: str,
        email: str,
        phone: Optional[str] = None,
        company: Optional[str] = None,
        source: Optional[str] = None
    ) -> Dict[str, Any]:
        """Crear contacto en CRM"""
        
        contact_id = f"contact_{len(self.contacts) + 1}"
        
        contact = {
            "contact_id": contact_id,
            "name": name,
            "email": email,
            "phone": phone,
            "company": company,
            "source": source or "website",
            "status": ContactStatus.LEAD.value,
            "created_at": datetime.now().isoformat(),
            "last_contact": None,
            "notes": []
        }
        
        self.contacts[contact_id] = contact
        
        return contact
    
    def create_deal(
        self,
        contact_id: str,
        title: str,
        value: float,
        expected_close_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """Crear deal/oportunidad"""
        
        deal_id = f"deal_{len(self.deals) + 1}"
        
        deal = {
            "deal_id": deal_id,
            "contact_id": contact_id,
            "title": title,
            "value": value,
            "stage": "prospecting",
            "probability": 10,
            "expected_close_date": expected_close_date or (datetime.now() + timedelta(days=30)).isoformat(),
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        self.deals[deal_id] = deal
        
        return deal
    
    def update_deal_stage(
        self,
        deal_id: str,
        stage: str,
        probability: Optional[int] = None
    ) -> bool:
        """Actualizar etapa de deal"""
        deal = self.deals.get(deal_id)
        
        if not deal:
            return False
        
        deal["stage"] = stage
        if probability is not None:
            deal["probability"] = probability
        deal["updated_at"] = datetime.now().isoformat()
        
        return True
    
    def add_activity(
        self,
        contact_id: Optional[str] = None,
        deal_id: Optional[str] = None,
        activity_type: str = "call",
        description: str = "",
        scheduled_at: Optional[str] = None
    ) -> Dict[str, Any]:
        """Agregar actividad"""
        
        activity = {
            "activity_id": f"act_{len(self.activities.get(contact_id or deal_id or 'general', [])) + 1}",
            "contact_id": contact_id,
            "deal_id": deal_id,
            "activity_type": activity_type,  # "call", "email", "meeting", "note"
            "description": description,
            "scheduled_at": scheduled_at or datetime.now().isoformat(),
            "completed": scheduled_at is None,
            "created_at": datetime.now().isoformat()
        }
        
        key = contact_id or deal_id or "general"
        if key not in self.activities:
            self.activities[key] = []
        
        self.activities[key].append(activity)
        
        return activity
    
    def get_contacts(
        self,
        status: Optional[ContactStatus] = None
    ) -> List[Dict[str, Any]]:
        """Obtener contactos"""
        contacts = list(self.contacts.values())
        
        if status:
            contacts = [c for c in contacts if c["status"] == status.value]
        
        return contacts
    
    def get_deals(
        self,
        stage: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Obtener deals"""
        deals = list(self.deals.values())
        
        if stage:
            deals = [d for d in deals if d["stage"] == stage]
        
        return deals
    
    def get_pipeline(self) -> Dict[str, Any]:
        """Obtener pipeline de ventas"""
        stages = {
            "prospecting": [],
            "qualification": [],
            "proposal": [],
            "negotiation": [],
            "closed_won": [],
            "closed_lost": []
        }
        
        for deal in self.deals.values():
            stage = deal["stage"]
            if stage in stages:
                stages[stage].append(deal)
        
        total_value = sum(d["value"] for d in self.deals.values())
        weighted_value = sum(d["value"] * (d["probability"] / 100) for d in self.deals.values())
        
        return {
            "stages": stages,
            "total_deals": len(self.deals),
            "total_value": total_value,
            "weighted_value": weighted_value,
            "conversion_rate": self._calculate_conversion_rate()
        }
    
    def _calculate_conversion_rate(self) -> float:
        """Calcular tasa de conversión"""
        won = len([d for d in self.deals.values() if d["stage"] == "closed_won"])
        total = len(self.deals)
        
        return (won / total * 100) if total > 0 else 0.0
    
    def sync_with_design(
        self,
        store_id: str,
        contact_id: str
    ) -> Dict[str, Any]:
        """Sincronizar diseño con contacto CRM"""
        contact = self.contacts.get(contact_id)
        
        if not contact:
            raise ValueError(f"Contacto {contact_id} no encontrado")
        
        # Crear deal automático
        deal = self.create_deal(
            contact_id=contact_id,
            title=f"Diseño para {contact.get('company', contact['name'])}",
            value=5000.0  # Valor estimado
        )
        
        # Agregar nota
        self.add_activity(
            contact_id=contact_id,
            deal_id=deal["deal_id"],
            activity_type="note",
            description=f"Diseño {store_id} creado y vinculado"
        )
        
        return {
            "contact_id": contact_id,
            "deal_id": deal["deal_id"],
            "store_id": store_id,
            "synced_at": datetime.now().isoformat()
        }




