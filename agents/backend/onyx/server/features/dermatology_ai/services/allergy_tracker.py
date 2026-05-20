"""
Sistema de seguimiento de alergias
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import uuid


@dataclass
class AllergyRecord:
    """Registro de alergia"""
    id: str
    user_id: str
    allergen: str  # Ingrediente o producto que causa alergia
    reaction_type: str  # "rash", "itching", "redness", "swelling", "other"
    severity: str  # "mild", "moderate", "severe"
    occurred_date: str
    product_name: Optional[str] = None
    notes: Optional[str] = None
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "allergen": self.allergen,
            "reaction_type": self.reaction_type,
            "severity": self.severity,
            "occurred_date": self.occurred_date,
            "product_name": self.product_name,
            "notes": self.notes,
            "created_at": self.created_at
        }


@dataclass
class AllergyProfile:
    """Perfil de alergias del usuario"""
    user_id: str
    known_allergens: List[str]
    reaction_history: List[AllergyRecord]
    risk_level: str  # "low", "medium", "high"
    recommendations: List[str]
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "user_id": self.user_id,
            "known_allergens": self.known_allergens,
            "reaction_history": [r.to_dict() for r in self.reaction_history],
            "risk_level": self.risk_level,
            "recommendations": self.recommendations
        }


class AllergyTracker:
    """Sistema de seguimiento de alergias"""
    
    def __init__(self):
        """Inicializa el tracker"""
        self.records: Dict[str, List[AllergyRecord]] = {}  # user_id -> [records]
    
    def record_allergy(self, user_id: str, allergen: str, reaction_type: str,
                      severity: str, occurred_date: str,
                      product_name: Optional[str] = None,
                      notes: Optional[str] = None) -> AllergyRecord:
        """Registra una reacción alérgica"""
        record = AllergyRecord(
            id=str(uuid.uuid4()),
            user_id=user_id,
            allergen=allergen,
            reaction_type=reaction_type,
            severity=severity,
            occurred_date=occurred_date,
            product_name=product_name,
            notes=notes
        )
        
        if user_id not in self.records:
            self.records[user_id] = []
        
        self.records[user_id].append(record)
        return record
    
    def get_allergy_profile(self, user_id: str) -> AllergyProfile:
        """Obtiene perfil de alergias del usuario"""
        user_records = self.records.get(user_id, [])
        
        # Extraer alérgenos conocidos
        known_allergens = list(set(r.allergen for r in user_records))
        
        # Determinar nivel de riesgo
        severe_reactions = [r for r in user_records if r.severity == "severe"]
        risk_level = "high" if len(severe_reactions) > 0 else "medium" if len(user_records) > 2 else "low"
        
        # Generar recomendaciones
        recommendations = self._generate_recommendations(user_records, known_allergens, risk_level)
        
        return AllergyProfile(
            user_id=user_id,
            known_allergens=known_allergens,
            reaction_history=user_records,
            risk_level=risk_level,
            recommendations=recommendations
        )
    
    def _generate_recommendations(self, records: List[AllergyRecord],
                                 allergens: List[str], risk_level: str) -> List[str]:
        """Genera recomendaciones basadas en alergias"""
        recommendations = []
        
        if allergens:
            recommendations.append(f"Evita estos ingredientes: {', '.join(allergens)}")
            recommendations.append("Siempre revisa la lista de ingredientes antes de comprar")
        
        if risk_level == "high":
            recommendations.append("Nivel de riesgo alto. Consulta con un dermatólogo.")
            recommendations.append("Considera hacer pruebas de parche antes de usar nuevos productos")
        elif risk_level == "medium":
            recommendations.append("Ten cuidado con nuevos productos")
            recommendations.append("Haz pruebas de parche en área pequeña primero")
        
        return recommendations
    
    def check_product_safety(self, user_id: str, product_ingredients: List[str]) -> Dict:
        """Verifica seguridad de producto para el usuario"""
        profile = self.get_allergy_profile(user_id)
        
        # Verificar si algún ingrediente es alérgeno conocido
        unsafe_ingredients = [
            ing for ing in product_ingredients
            if any(allergen.lower() in ing.lower() for allergen in profile.known_allergens)
        ]
        
        is_safe = len(unsafe_ingredients) == 0
        
        return {
            "is_safe": is_safe,
            "unsafe_ingredients": unsafe_ingredients,
            "risk_level": profile.risk_level,
            "recommendation": "Seguro para usar" if is_safe else "NO RECOMENDADO - Contiene alérgenos conocidos"
        }
    
    def get_user_records(self, user_id: str) -> List[AllergyRecord]:
        """Obtiene registros del usuario"""
        return self.records.get(user_id, [])






