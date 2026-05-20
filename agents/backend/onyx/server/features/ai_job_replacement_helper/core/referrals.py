"""
Referrals Service - Sistema de referidos
=========================================

Sistema de referidos y recompensas.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Referral:
    """Referido"""
    referrer_id: str
    referred_id: str
    code: str
    status: str = "pending"  # pending, completed, rewarded
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    reward_given: bool = False


class ReferralsService:
    """Servicio de referidos"""
    
    def __init__(self):
        """Inicializar servicio"""
        self.referrals: Dict[str, List[Referral]] = {}  # referrer_id -> [referrals]
        self.referral_codes: Dict[str, str] = {}  # user_id -> code
        self.reward_points = 100  # Puntos por referido
        logger.info("ReferralsService initialized")
    
    def generate_referral_code(self, user_id: str) -> str:
        """Generar código de referido"""
        import secrets
        code = f"REF{secrets.token_hex(4).upper()}"
        self.referral_codes[user_id] = code
        logger.info(f"Referral code generated for user {user_id}: {code}")
        return code
    
    def get_referral_code(self, user_id: str) -> Optional[str]:
        """Obtener código de referido del usuario"""
        if user_id not in self.referral_codes:
            return self.generate_referral_code(user_id)
        return self.referral_codes[user_id]
    
    def register_referral(self, referrer_code: str, referred_user_id: str) -> Optional[Referral]:
        """Registrar referido"""
        # Buscar referrer por código
        referrer_id = None
        for user_id, code in self.referral_codes.items():
            if code == referrer_code:
                referrer_id = user_id
                break
        
        if not referrer_id:
            return None
        
        # Crear referral
        referral = Referral(
            referrer_id=referrer_id,
            referred_id=referred_user_id,
            code=referrer_code,
        )
        
        if referrer_id not in self.referrals:
            self.referrals[referrer_id] = []
        
        self.referrals[referrer_id].append(referral)
        
        logger.info(f"Referral registered: {referrer_id} -> {referred_user_id}")
        return referral
    
    def complete_referral(self, referred_user_id: str) -> bool:
        """Completar referral (cuando el referido completa registro)"""
        # Buscar referral
        for referrals_list in self.referrals.values():
            for referral in referrals_list:
                if referral.referred_id == referred_user_id and referral.status == "pending":
                    referral.status = "completed"
                    referral.completed_at = datetime.now()
                    return True
        return False
    
    def get_user_referrals(self, user_id: str) -> List[Referral]:
        """Obtener referidos del usuario"""
        return self.referrals.get(user_id, [])
    
    def get_referral_stats(self, user_id: str) -> Dict[str, int]:
        """Obtener estadísticas de referidos"""
        referrals = self.get_user_referrals(user_id)
        
        return {
            "total": len(referrals),
            "pending": sum(1 for r in referrals if r.status == "pending"),
            "completed": sum(1 for r in referrals if r.status == "completed"),
            "rewarded": sum(1 for r in referrals if r.reward_given),
            "total_rewards": sum(1 for r in referrals if r.reward_given) * self.reward_points,
        }




