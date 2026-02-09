"""
Servicio de Economía Virtual - Sistema de puntos, recompensas y economía
"""

from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum


class TransactionType(str, Enum):
    """Tipos de transacciones"""
    EARNED = "earned"
    SPENT = "spent"
    REWARD = "reward"
    BONUS = "bonus"
    PENALTY = "penalty"


class VirtualEconomyService:
    """Servicio de economía virtual y recompensas"""
    
    def __init__(self):
        """Inicializa el servicio de economía virtual"""
        self.point_values = self._setup_point_values()
        self.rewards_catalog = self._load_rewards_catalog()
    
    def earn_points(
        self,
        user_id: str,
        action_type: str,
        amount: int,
        description: str
    ) -> Dict:
        """
        Otorga puntos al usuario
        
        Args:
            user_id: ID del usuario
            action_type: Tipo de acción
            amount: Cantidad de puntos
            description: Descripción
        
        Returns:
            Transacción de puntos
        """
        transaction = {
            "id": f"tx_{datetime.now().timestamp()}",
            "user_id": user_id,
            "type": TransactionType.EARNED,
            "action_type": action_type,
            "amount": amount,
            "description": description,
            "timestamp": datetime.now().isoformat()
        }
        
        return transaction
    
    def spend_points(
        self,
        user_id: str,
        reward_id: str,
        points_cost: int
    ) -> Dict:
        """
        Gasta puntos en una recompensa
        
        Args:
            user_id: ID del usuario
            reward_id: ID de la recompensa
            points_cost: Costo en puntos
        
        Returns:
            Transacción de gasto
        """
        transaction = {
            "id": f"tx_{datetime.now().timestamp()}",
            "user_id": user_id,
            "type": TransactionType.SPENT,
            "reward_id": reward_id,
            "amount": -points_cost,
            "description": f"Canjeado: {reward_id}",
            "timestamp": datetime.now().isoformat()
        }
        
        return transaction
    
    def get_user_balance(self, user_id: str) -> Dict:
        """
        Obtiene balance de puntos del usuario
        
        Args:
            user_id: ID del usuario
        
        Returns:
            Balance y estadísticas
        """
        # En implementación real, esto calcularía desde transacciones
        return {
            "user_id": user_id,
            "total_points": 0,
            "available_points": 0,
            "spent_points": 0,
            "lifetime_earned": 0,
            "level": 1,
            "next_level_points": 100
        }
    
    def get_rewards_catalog(
        self,
        category: Optional[str] = None,
        max_points: Optional[int] = None
    ) -> List[Dict]:
        """
        Obtiene catálogo de recompensas
        
        Args:
            category: Filtrar por categoría (opcional)
            max_points: Filtrar por puntos máximos (opcional)
        
        Returns:
            Lista de recompensas disponibles
        """
        rewards = self.rewards_catalog.copy()
        
        if category:
            rewards = [r for r in rewards if r.get("category") == category]
        
        if max_points:
            rewards = [r for r in rewards if r.get("points_cost", 0) <= max_points]
        
        return rewards
    
    def purchase_reward(
        self,
        user_id: str,
        reward_id: str
    ) -> Dict:
        """
        Compra una recompensa
        
        Args:
            user_id: ID del usuario
            reward_id: ID de la recompensa
        
        Returns:
            Compra realizada
        """
        reward = next((r for r in self.rewards_catalog if r.get("id") == reward_id), None)
        
        if not reward:
            return {
                "success": False,
                "message": "Recompensa no encontrada"
            }
        
        purchase = {
            "user_id": user_id,
            "reward_id": reward_id,
            "reward_name": reward.get("name"),
            "points_cost": reward.get("points_cost", 0),
            "purchased_at": datetime.now().isoformat(),
            "status": "completed"
        }
        
        return purchase
    
    def _setup_point_values(self) -> Dict:
        """Configura valores de puntos por acción"""
        return {
            "sober_day": 10,
            "check_in": 5,
            "journal_entry": 5,
            "exercise": 15,
            "therapy_session": 25,
            "milestone": 50,
            "help_others": 20,
            "complete_goal": 30
        }
    
    def _load_rewards_catalog(self) -> List[Dict]:
        """Carga catálogo de recompensas"""
        return [
            {
                "id": "reward_1",
                "name": "Badge de Logro",
                "category": "badge",
                "points_cost": 50,
                "description": "Badge personalizado de logro",
                "icon": "🏆"
            },
            {
                "id": "reward_2",
                "name": "Perfil Personalizado",
                "category": "customization",
                "points_cost": 100,
                "description": "Personaliza tu perfil con temas exclusivos",
                "icon": "🎨"
            },
            {
                "id": "reward_3",
                "name": "Sesión de Coaching Premium",
                "category": "service",
                "points_cost": 200,
                "description": "Sesión extendida de coaching",
                "icon": "💎"
            },
            {
                "id": "reward_4",
                "name": "Certificado de Logro",
                "category": "certificate",
                "points_cost": 500,
                "description": "Certificado oficial de logro",
                "icon": "📜"
            }
        ]

