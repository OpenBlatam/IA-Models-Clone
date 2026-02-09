"""
Loyalty Service - Sistema de lealtad y recompensas
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class RewardType(str, Enum):
    """Tipos de recompensa"""
    DISCOUNT = "discount"
    FREE_FEATURE = "free_feature"
    PREMIUM_UPGRADE = "premium_upgrade"
    CASHBACK = "cashback"
    BADGE = "badge"


class LoyaltyService:
    """Servicio para sistema de lealtad"""
    
    def __init__(self):
        self.members: Dict[str, Dict[str, Any]] = {}
        self.rewards: Dict[str, List[Dict[str, Any]]] = {}
        self.transactions: Dict[str, List[Dict[str, Any]]] = {}
    
    def enroll_member(
        self,
        user_id: str,
        tier: str = "bronze"  # "bronze", "silver", "gold", "platinum"
    ) -> Dict[str, Any]:
        """Inscribir miembro en programa de lealtad"""
        
        member = {
            "user_id": user_id,
            "tier": tier,
            "points": 0,
            "lifetime_points": 0,
            "enrolled_at": datetime.now().isoformat(),
            "last_activity": datetime.now().isoformat(),
            "rewards_claimed": 0,
            "referrals": 0
        }
        
        self.members[user_id] = member
        
        return member
    
    def get_member(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Obtener miembro"""
        return self.members.get(user_id)
    
    def add_points(
        self,
        user_id: str,
        points: int,
        reason: str
    ) -> Dict[str, Any]:
        """Agregar puntos"""
        member = self.get_member(user_id)
        
        if not member:
            member = self.enroll_member(user_id)
        
        old_points = member["points"]
        member["points"] += points
        member["lifetime_points"] += points
        member["last_activity"] = datetime.now().isoformat()
        
        # Verificar upgrade de tier
        old_tier = member["tier"]
        new_tier = self._calculate_tier(member["lifetime_points"])
        member["tier"] = new_tier
        
        # Registrar transacción
        transaction = {
            "transaction_id": f"txn_{len(self.transactions.get(user_id, [])) + 1}",
            "user_id": user_id,
            "points": points,
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        }
        
        if user_id not in self.transactions:
            self.transactions[user_id] = []
        
        self.transactions[user_id].append(transaction)
        
        return {
            "user_id": user_id,
            "points_added": points,
            "old_points": old_points,
            "new_points": member["points"],
            "old_tier": old_tier,
            "new_tier": new_tier,
            "tier_upgraded": new_tier != old_tier
        }
    
    def _calculate_tier(self, lifetime_points: int) -> str:
        """Calcular tier basado en puntos"""
        if lifetime_points >= 10000:
            return "platinum"
        elif lifetime_points >= 5000:
            return "gold"
        elif lifetime_points >= 1000:
            return "silver"
        else:
            return "bronze"
    
    def get_available_rewards(
        self,
        user_id: str
    ) -> List[Dict[str, Any]]:
        """Obtener recompensas disponibles"""
        member = self.get_member(user_id)
        
        if not member:
            return []
        
        points = member["points"]
        tier = member["tier"]
        
        rewards = []
        
        # Recompensas por puntos
        if points >= 1000:
            rewards.append({
                "reward_id": "discount_10",
                "type": RewardType.DISCOUNT.value,
                "name": "Descuento 10%",
                "description": "10% de descuento en tu próxima suscripción",
                "points_cost": 1000,
                "available": True
            })
        
        if points >= 2500:
            rewards.append({
                "reward_id": "free_month",
                "type": RewardType.FREE_FEATURE.value,
                "name": "Mes Gratis",
                "description": "1 mes gratis de suscripción Professional",
                "points_cost": 2500,
                "available": True
            })
        
        # Recompensas por tier
        if tier in ["gold", "platinum"]:
            rewards.append({
                "reward_id": "priority_support",
                "type": RewardType.FREE_FEATURE.value,
                "name": "Soporte Prioritario",
                "description": "Soporte prioritario por 30 días",
                "points_cost": 0,
                "tier_benefit": True,
                "available": True
            })
        
        return rewards
    
    def claim_reward(
        self,
        user_id: str,
        reward_id: str
    ) -> Dict[str, Any]:
        """Reclamar recompensa"""
        member = self.get_member(user_id)
        
        if not member:
            raise ValueError("Usuario no está en el programa de lealtad")
        
        available_rewards = self.get_available_rewards(user_id)
        reward = next((r for r in available_rewards if r["reward_id"] == reward_id), None)
        
        if not reward:
            raise ValueError(f"Recompensa {reward_id} no disponible")
        
        if reward.get("points_cost", 0) > member["points"]:
            raise ValueError("Puntos insuficientes")
        
        # Descontar puntos
        if reward.get("points_cost", 0) > 0:
            member["points"] -= reward["points_cost"]
        
        # Registrar recompensa
        claimed_reward = {
            "reward_id": reward_id,
            "user_id": user_id,
            "reward": reward,
            "claimed_at": datetime.now().isoformat(),
            "status": "active"
        }
        
        if user_id not in self.rewards:
            self.rewards[user_id] = []
        
        self.rewards[user_id].append(claimed_reward)
        member["rewards_claimed"] += 1
        
        return claimed_reward
    
    def get_referral_code(self, user_id: str) -> str:
        """Obtener código de referido"""
        member = self.get_member(user_id)
        
        if not member:
            member = self.enroll_member(user_id)
        
        return f"REF_{user_id[:8].upper()}"
    
    def process_referral(
        self,
        referrer_id: str,
        referred_id: str
    ) -> Dict[str, Any]:
        """Procesar referido"""
        referrer = self.get_member(referrer_id)
        referred = self.get_member(referred_id)
        
        if not referrer:
            referrer = self.enroll_member(referrer_id)
        
        if not referred:
            referred = self.enroll_member(referred_id)
        
        # Otorgar puntos al referrer
        referrer_points = self.add_points(referrer_id, 500, "Referral bonus")
        
        # Otorgar puntos al referido
        referred_points = self.add_points(referred_id, 200, "Welcome bonus")
        
        referrer["referrals"] += 1
        
        return {
            "referrer_id": referrer_id,
            "referred_id": referred_id,
            "referrer_points": referrer_points,
            "referred_points": referred_points
        }
    
    def get_loyalty_stats(self, user_id: str) -> Dict[str, Any]:
        """Obtener estadísticas de lealtad"""
        member = self.get_member(user_id)
        
        if not member:
            return {"message": "Usuario no está en el programa"}
        
        return {
            "user_id": user_id,
            "tier": member["tier"],
            "points": member["points"],
            "lifetime_points": member["lifetime_points"],
            "rewards_claimed": member["rewards_claimed"],
            "referrals": member["referrals"],
            "next_tier": self._get_next_tier(member["lifetime_points"]),
            "points_to_next_tier": self._points_to_next_tier(member["lifetime_points"])
        }
    
    def _get_next_tier(self, points: int) -> str:
        """Obtener siguiente tier"""
        if points < 1000:
            return "silver"
        elif points < 5000:
            return "gold"
        elif points < 10000:
            return "platinum"
        else:
            return "platinum"  # Ya está en el máximo
    
    def _points_to_next_tier(self, points: int) -> int:
        """Puntos necesarios para siguiente tier"""
        if points < 1000:
            return 1000 - points
        elif points < 5000:
            return 5000 - points
        elif points < 10000:
            return 10000 - points
        else:
            return 0




