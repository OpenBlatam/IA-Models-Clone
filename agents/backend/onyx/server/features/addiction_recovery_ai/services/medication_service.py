"""
Servicio de Seguimiento de Medicamentos - Gestión de medicamentos y tratamientos
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from enum import Enum


class MedicationStatus(str, Enum):
    """Estados de medicamento"""
    ACTIVE = "active"
    COMPLETED = "completed"
    DISCONTINUED = "discontinued"
    MISSED = "missed"


class MedicationService:
    """Servicio de seguimiento de medicamentos"""
    
    def __init__(self):
        """Inicializa el servicio de medicamentos"""
        pass
    
    def add_medication(
        self,
        user_id: str,
        medication_name: str,
        dosage: str,
        frequency: str,
        start_date: str,
        end_date: Optional[str] = None,
        doctor_name: Optional[str] = None,
        notes: Optional[str] = None
    ) -> Dict:
        """
        Agrega un medicamento al seguimiento
        
        Args:
            user_id: ID del usuario
            medication_name: Nombre del medicamento
            dosage: Dosis
            frequency: Frecuencia (diaria, dos veces al día, etc.)
            start_date: Fecha de inicio
            end_date: Fecha de fin (opcional)
            doctor_name: Nombre del doctor (opcional)
            notes: Notas adicionales (opcional)
        
        Returns:
            Medicamento agregado
        """
        medication = {
            "id": f"med_{datetime.now().timestamp()}",
            "user_id": user_id,
            "medication_name": medication_name,
            "dosage": dosage,
            "frequency": frequency,
            "start_date": start_date,
            "end_date": end_date,
            "doctor_name": doctor_name,
            "notes": notes,
            "status": MedicationStatus.ACTIVE,
            "created_at": datetime.now().isoformat()
        }
        
        return medication
    
    def record_medication_taken(
        self,
        medication_id: str,
        taken_at: datetime,
        notes: Optional[str] = None
    ) -> Dict:
        """
        Registra que se tomó un medicamento
        
        Args:
            medication_id: ID del medicamento
            taken_at: Fecha y hora en que se tomó
            notes: Notas adicionales
        
        Returns:
            Registro de toma
        """
        record = {
            "medication_id": medication_id,
            "taken_at": taken_at.isoformat(),
            "notes": notes,
            "recorded_at": datetime.now().isoformat()
        }
        
        return record
    
    def get_medication_schedule(
        self,
        user_id: str,
        date: Optional[datetime] = None
    ) -> List[Dict]:
        """
        Obtiene horario de medicamentos para una fecha
        
        Args:
            user_id: ID del usuario
            date: Fecha (opcional, por defecto hoy)
        
        Returns:
            Lista de medicamentos programados
        """
        if date is None:
            date = datetime.now()
        
        # En implementación real, esto vendría de la base de datos
        schedule = [
            {
                "medication_id": "med_1",
                "medication_name": "Medicamento A",
                "dosage": "10mg",
                "scheduled_time": "09:00",
                "taken": False,
                "status": "pending"
            },
            {
                "medication_id": "med_2",
                "medication_name": "Medicamento B",
                "dosage": "5mg",
                "scheduled_time": "21:00",
                "taken": False,
                "status": "pending"
            }
        ]
        
        return schedule
    
    def get_adherence_rate(
        self,
        user_id: str,
        medication_id: Optional[str] = None,
        days: int = 30
    ) -> Dict:
        """
        Calcula tasa de adherencia a medicamentos
        
        Args:
            user_id: ID del usuario
            medication_id: ID del medicamento (opcional, todos si no se especifica)
            days: Días a analizar
        
        Returns:
            Tasa de adherencia
        """
        # En implementación real, esto calcularía desde la base de datos
        return {
            "user_id": user_id,
            "medication_id": medication_id,
            "period_days": days,
            "total_doses": 60,
            "taken_doses": 55,
            "missed_doses": 5,
            "adherence_rate": round(55 / 60 * 100, 2),
            "status": "good" if (55 / 60) >= 0.8 else "needs_improvement"
        }
    
    def get_medication_reminders(
        self,
        user_id: str
    ) -> List[Dict]:
        """
        Obtiene recordatorios de medicamentos
        
        Args:
            user_id: ID del usuario
        
        Returns:
            Lista de recordatorios
        """
        schedule = self.get_medication_schedule(user_id)
        
        reminders = []
        for med in schedule:
            if not med.get("taken", False):
                reminders.append({
                    "medication_id": med.get("medication_id"),
                    "medication_name": med.get("medication_name"),
                    "dosage": med.get("dosage"),
                    "scheduled_time": med.get("scheduled_time"),
                    "reminder_type": "medication",
                    "priority": "high"
                })
        
        return reminders
    
    def track_side_effects(
        self,
        medication_id: str,
        side_effect: str,
        severity: str,
        notes: Optional[str] = None
    ) -> Dict:
        """
        Registra efectos secundarios
        
        Args:
            medication_id: ID del medicamento
            side_effect: Efecto secundario
            severity: Severidad (leve, moderado, severo)
            notes: Notas adicionales
        
        Returns:
            Registro de efecto secundario
        """
        record = {
            "medication_id": medication_id,
            "side_effect": side_effect,
            "severity": severity,
            "notes": notes,
            "reported_at": datetime.now().isoformat()
        }
        
        return record
    
    def get_medication_summary(
        self,
        user_id: str
    ) -> Dict:
        """
        Obtiene resumen de medicamentos del usuario
        
        Args:
            user_id: ID del usuario
        
        Returns:
            Resumen de medicamentos
        """
        # En implementación real, esto vendría de la base de datos
        return {
            "user_id": user_id,
            "active_medications": 2,
            "total_medications": 2,
            "adherence_rate": 91.67,
            "upcoming_doses": 2,
            "missed_doses_today": 0,
            "side_effects_reported": 0
        }

