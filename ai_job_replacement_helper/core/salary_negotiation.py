"""
Salary Negotiation Service - Negociación salarial
==================================================

Sistema de guía y simulación de negociación salarial.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class NegotiationStage(str, Enum):
    """Etapas de negociación"""
    INITIAL_OFFER = "initial_offer"
    COUNTER_OFFER = "counter_offer"
    FINAL_NEGOTIATION = "final_negotiation"
    ACCEPTED = "accepted"
    DECLINED = "declined"


@dataclass
class SalaryOffer:
    """Oferta salarial"""
    base_salary: float
    bonus: Optional[float] = None
    equity: Optional[float] = None
    benefits: List[str] = field(default_factory=list)
    total_compensation: Optional[float] = None


@dataclass
class NegotiationSession:
    """Sesión de negociación"""
    id: str
    user_id: str
    job_title: str
    company: str
    initial_offer: SalaryOffer
    target_salary: float
    current_stage: NegotiationStage
    offers_history: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


class SalaryNegotiationService:
    """Servicio de negociación salarial"""
    
    def __init__(self):
        """Inicializar servicio"""
        self.sessions: Dict[str, NegotiationSession] = {}
        self.market_data: Dict[str, Dict[str, float]] = {}  # job_title -> market_salary
        logger.info("SalaryNegotiationService initialized")
    
    def start_negotiation(
        self,
        user_id: str,
        job_title: str,
        company: str,
        initial_offer: SalaryOffer,
        target_salary: float
    ) -> NegotiationSession:
        """Iniciar sesión de negociación"""
        session_id = f"negotiation_{user_id}_{int(datetime.now().timestamp())}"
        
        session = NegotiationSession(
            id=session_id,
            user_id=user_id,
            job_title=job_title,
            company=company,
            initial_offer=initial_offer,
            target_salary=target_salary,
            current_stage=NegotiationStage.INITIAL_OFFER,
        )
        
        # Calcular compensación total
        if not initial_offer.total_compensation:
            initial_offer.total_compensation = self._calculate_total_compensation(initial_offer)
        
        session.offers_history.append({
            "offer": initial_offer,
            "stage": NegotiationStage.INITIAL_OFFER.value,
            "timestamp": datetime.now().isoformat(),
        })
        
        self.sessions[session_id] = session
        
        logger.info(f"Negotiation session started: {session_id}")
        return session
    
    def _calculate_total_compensation(self, offer: SalaryOffer) -> float:
        """Calcular compensación total"""
        total = offer.base_salary
        if offer.bonus:
            total += offer.bonus
        if offer.equity:
            total += offer.equity
        return total
    
    def get_negotiation_strategy(
        self,
        session_id: str
    ) -> Dict[str, Any]:
        """Obtener estrategia de negociación"""
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        market_salary = self._get_market_salary(session.job_title)
        offer_gap = session.target_salary - session.initial_offer.base_salary
        
        strategy = {
            "market_salary": market_salary,
            "offer_gap": offer_gap,
            "gap_percentage": (offer_gap / session.initial_offer.base_salary) * 100,
            "recommendations": [],
            "talking_points": [],
        }
        
        # Generar recomendaciones
        if offer_gap > 0:
            if offer_gap / session.initial_offer.base_salary > 0.2:
                strategy["recommendations"].append(
                    "La brecha es significativa (>20%). Considera negociar beneficios adicionales además del salario."
                )
            else:
                strategy["recommendations"].append(
                    "La brecha es manejable. Puedes negociar directamente el salario base."
                )
        
        # Talking points
        strategy["talking_points"] = [
            f"Basado en mi investigación, el rango de mercado para {session.job_title} es ${market_salary:,.0f}",
            "Tengo experiencia relevante que aporta valor inmediato",
            "Estoy muy interesado en esta oportunidad y en contribuir al éxito del equipo",
        ]
        
        return strategy
    
    def _get_market_salary(self, job_title: str) -> float:
        """Obtener salario de mercado"""
        # En producción, esto consultaría datos reales de mercado
        # Por ahora, simulamos
        return self.market_data.get(job_title, {}).get("median", 100000.0)
    
    def simulate_counter_offer(
        self,
        session_id: str,
        counter_amount: float
    ) -> Dict[str, Any]:
        """Simular contraoferta"""
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        # Simular respuesta de la empresa
        acceptance_probability = self._calculate_acceptance_probability(
            session.initial_offer.base_salary,
            counter_amount,
            session.target_salary
        )
        
        if acceptance_probability > 0.7:
            response = "accepted"
            final_offer = counter_amount
        elif acceptance_probability > 0.4:
            response = "counter"
            final_offer = (session.initial_offer.base_salary + counter_amount) / 2
        else:
            response = "declined"
            final_offer = session.initial_offer.base_salary
        
        result = {
            "response": response,
            "final_offer": final_offer,
            "acceptance_probability": acceptance_probability,
            "recommendation": self._get_recommendation(response, final_offer, session.target_salary),
        }
        
        return result
    
    def _calculate_acceptance_probability(
        self,
        initial: float,
        counter: float,
        target: float
    ) -> float:
        """Calcular probabilidad de aceptación"""
        gap = counter - initial
        target_gap = target - initial
        
        if gap <= 0:
            return 0.0
        
        if gap <= target_gap * 0.5:
            return 0.8  # Alta probabilidad si es razonable
        elif gap <= target_gap:
            return 0.5  # Media probabilidad
        else:
            return 0.2  # Baja probabilidad si es muy alto
    
    def _get_recommendation(
        self,
        response: str,
        final_offer: float,
        target: float
    ) -> str:
        """Obtener recomendación basada en respuesta"""
        if response == "accepted":
            return "¡Excelente! La oferta fue aceptada. Asegúrate de obtener todo por escrito."
        elif response == "counter":
            gap = target - final_offer
            if gap < target * 0.1:
                return "La contraoferta está cerca de tu objetivo. Considera aceptar o negociar beneficios adicionales."
            else:
                return "La contraoferta está por debajo de tu objetivo. Puedes intentar una última negociación."
        else:
            return "La oferta fue rechazada. Considera si puedes ajustar tus expectativas o busca otras oportunidades."
    
    def get_negotiation_tips(self, job_title: str, location: str) -> List[str]:
        """Obtener tips de negociación"""
        return [
            "Investiga el rango salarial de mercado antes de negociar",
            "Siempre negocia, incluso si la oferta inicial parece buena",
            "Considera el paquete completo: salario, bonus, equity, beneficios",
            "Sé profesional y respetuoso durante toda la negociación",
            "Ten un número objetivo pero sé flexible",
            "No aceptes la primera oferta inmediatamente",
            "Practica tu pitch de negociación antes de la conversación",
            "Considera negociar otros aspectos si el salario no es negociable",
        ]




