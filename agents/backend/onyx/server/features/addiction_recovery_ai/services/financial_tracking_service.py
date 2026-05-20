"""
Servicio de Seguimiento Financiero - Sistema de seguimiento de gastos y ahorros
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta


class FinancialTrackingService:
    """Servicio de seguimiento financiero"""
    
    def __init__(self):
        """Inicializa el servicio de seguimiento financiero"""
        pass
    
    def calculate_savings(
        self,
        user_id: str,
        addiction_type: str,
        days_sober: int,
        daily_cost: float
    ) -> Dict:
        """
        Calcula ahorros por días de sobriedad
        
        Args:
            user_id: ID del usuario
            addiction_type: Tipo de adicción
            days_sober: Días de sobriedad
            daily_cost: Costo diario estimado
        
        Returns:
            Cálculo de ahorros
        """
        total_saved = days_sober * daily_cost
        
        # Proyecciones
        weekly_saved = daily_cost * 7
        monthly_saved = daily_cost * 30
        yearly_saved = daily_cost * 365
        
        return {
            "user_id": user_id,
            "addiction_type": addiction_type,
            "days_sober": days_sober,
            "daily_cost": daily_cost,
            "total_saved": round(total_saved, 2),
            "weekly_projection": round(weekly_saved, 2),
            "monthly_projection": round(monthly_saved, 2),
            "yearly_projection": round(yearly_saved, 2),
            "milestones": self._calculate_milestones(total_saved),
            "calculated_at": datetime.now().isoformat()
        }
    
    def track_expense(
        self,
        user_id: str,
        category: str,
        amount: float,
        description: str,
        date: Optional[str] = None
    ) -> Dict:
        """
        Registra un gasto
        
        Args:
            user_id: ID del usuario
            category: Categoría del gasto
            amount: Monto
            description: Descripción
            date: Fecha (opcional)
        
        Returns:
            Gasto registrado
        """
        expense = {
            "id": f"expense_{datetime.now().timestamp()}",
            "user_id": user_id,
            "category": category,
            "amount": amount,
            "description": description,
            "date": date or datetime.now().date().isoformat(),
            "created_at": datetime.now().isoformat()
        }
        
        return expense
    
    def get_financial_summary(
        self,
        user_id: str,
        days: int = 30
    ) -> Dict:
        """
        Obtiene resumen financiero
        
        Args:
            user_id: ID del usuario
            days: Número de días a analizar
        
        Returns:
            Resumen financiero
        """
        return {
            "user_id": user_id,
            "period_days": days,
            "total_saved": 0.0,
            "total_spent": 0.0,
            "net_savings": 0.0,
            "savings_rate": 0.0,
            "top_categories": [],
            "trend": "improving",
            "generated_at": datetime.now().isoformat()
        }
    
    def suggest_financial_goals(
        self,
        user_id: str,
        current_savings: float
    ) -> List[Dict]:
        """
        Sugiere metas financieras
        
        Args:
            user_id: ID del usuario
            current_savings: Ahorros actuales
        
        Returns:
            Lista de metas sugeridas
        """
        goals = [
            {
                "id": "goal_1",
                "name": "Ahorro de Emergencia",
                "target_amount": 1000.0,
                "current_amount": current_savings,
                "progress": min(100, (current_savings / 1000.0) * 100),
                "description": "Construir fondo de emergencia de $1000"
            },
            {
                "id": "goal_2",
                "name": "Tratamiento Premium",
                "target_amount": 500.0,
                "current_amount": current_savings,
                "progress": min(100, (current_savings / 500.0) * 100),
                "description": "Ahorrar para tratamiento premium"
            }
        ]
        
        return goals
    
    def _calculate_milestones(self, total_saved: float) -> List[Dict]:
        """Calcula hitos financieros alcanzados"""
        milestones = [
            {"amount": 100, "name": "Primeros $100"},
            {"amount": 500, "name": "Medio Milenio"},
            {"amount": 1000, "name": "Mil Dólares"},
            {"amount": 5000, "name": "Cinco Mil"},
            {"amount": 10000, "name": "Diez Mil"}
        ]
        
        achieved = [m for m in milestones if total_saved >= m["amount"]]
        next_milestone = next((m for m in milestones if total_saved < m["amount"]), None)
        
        return {
            "achieved": achieved,
            "next": next_milestone,
            "progress_to_next": ((total_saved / next_milestone["amount"]) * 100) if next_milestone else 100
        }

