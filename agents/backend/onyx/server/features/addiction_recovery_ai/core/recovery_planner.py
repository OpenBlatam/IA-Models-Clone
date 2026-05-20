"""
Planificador de recuperación - Crea planes personalizados de recuperación
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel


class RecoveryPlan(BaseModel):
    """Modelo para plan de recuperación"""
    user_id: str
    addiction_type: str
    start_date: str
    approach: str  # "abstinencia_total" o "reduccion_gradual"
    goals: List[Dict]
    milestones: List[Dict]
    strategies: List[Dict]
    daily_tasks: List[Dict]
    weekly_tasks: List[Dict]
    support_resources: List[Dict]
    created_at: str
    updated_at: str


class RecoveryPlanner:
    """Planificador de planes de recuperación personalizados"""
    
    def __init__(self):
        """Inicializa el planificador"""
        self.strategies_by_type = self._load_strategies()
    
    def create_plan(
        self,
        user_id: str,
        addiction_type: str,
        assessment_data: Dict,
        approach: Optional[str] = None
    ) -> Dict:
        """
        Crea un plan de recuperación personalizado
        
        Args:
            user_id: ID del usuario
            addiction_type: Tipo de adicción
            assessment_data: Datos de evaluación
            approach: Enfoque preferido (opcional)
        
        Returns:
            Plan de recuperación completo
        """
        if not approach:
            approach = self._determine_approach(assessment_data)
        
        start_date = datetime.now().isoformat()
        
        plan = {
            "user_id": user_id,
            "addiction_type": addiction_type,
            "start_date": start_date,
            "approach": approach,
            "goals": self._create_goals(addiction_type, assessment_data),
            "milestones": self._create_milestones(),
            "strategies": self._get_strategies(addiction_type, assessment_data),
            "daily_tasks": self._create_daily_tasks(addiction_type, approach),
            "weekly_tasks": self._create_weekly_tasks(addiction_type),
            "support_resources": self._get_support_resources(addiction_type),
            "created_at": start_date,
            "updated_at": start_date
        }
        
        return plan
    
    def _determine_approach(self, assessment_data: Dict) -> str:
        """Determina el enfoque recomendado"""
        severity = assessment_data.get("severity", "moderada")
        
        if severity in ["severa", "crítica"]:
            return "abstinencia_total"
        else:
            # Dar opción, pero recomendar abstinencia total
            return "abstinencia_total"
    
    def _create_goals(self, addiction_type: str, assessment_data: Dict) -> List[Dict]:
        """Crea objetivos del plan"""
        goals = [
            {
                "id": "goal_1",
                "title": "Primera semana sin consumo",
                "description": "Completar 7 días consecutivos sin consumo",
                "target_date": (datetime.now() + timedelta(days=7)).isoformat(),
                "status": "pending"
            },
            {
                "id": "goal_2",
                "title": "Primer mes de sobriedad",
                "description": "Alcanzar 30 días de sobriedad continua",
                "target_date": (datetime.now() + timedelta(days=30)).isoformat(),
                "status": "pending"
            },
            {
                "id": "goal_3",
                "title": "Tres meses de recuperación",
                "description": "Completar 90 días de sobriedad",
                "target_date": (datetime.now() + timedelta(days=90)).isoformat(),
                "status": "pending"
            }
        ]
        
        daily_cost = assessment_data.get("daily_cost")
        if daily_cost:
            goals.append({
                "id": "goal_money",
                "title": "Ahorro económico",
                "description": f"Ahorrar ${daily_cost * 30:.2f} en el primer mes",
                "target_date": (datetime.now() + timedelta(days=30)).isoformat(),
                "status": "pending"
            })
        
        return goals
    
    def _create_milestones(self) -> List[Dict]:
        """Crea hitos de recuperación"""
        return [
            {"days": 1, "title": "Primer día", "reward": "Reconocimiento personal"},
            {"days": 7, "title": "Primera semana", "reward": "Actividad especial"},
            {"days": 30, "title": "Primer mes", "reward": "Celebración significativa"},
            {"days": 90, "title": "Tres meses", "reward": "Recompensa importante"},
            {"days": 180, "title": "Seis meses", "reward": "Celebración grande"},
            {"days": 365, "title": "Un año", "reward": "Logro mayor"}
        ]
    
    def _get_strategies(self, addiction_type: str, assessment_data: Dict) -> List[Dict]:
        """Obtiene estrategias específicas para el tipo de adicción"""
        base_strategies = self.strategies_by_type.get(addiction_type.lower(), [])
        
        # Personalizar según triggers
        triggers = assessment_data.get("triggers", [])
        personalized = []
        
        for strategy in base_strategies:
            personalized.append({
                "name": strategy["name"],
                "description": strategy["description"],
                "applicable": True,
                "priority": strategy.get("priority", "medium")
            })
        
        # Agregar estrategias específicas para triggers
        if "estrés" in triggers or "stress" in triggers:
            personalized.append({
                "name": "Manejo de estrés",
                "description": "Técnicas de relajación y manejo de estrés",
                "applicable": True,
                "priority": "high"
            })
        
        if "social" in triggers or "amigos" in triggers:
            personalized.append({
                "name": "Estrategias sociales",
                "description": "Cómo manejar situaciones sociales sin consumo",
                "applicable": True,
                "priority": "high"
            })
        
        return personalized
    
    def _create_daily_tasks(self, addiction_type: str, approach: str) -> List[Dict]:
        """Crea tareas diarias"""
        tasks = [
            {
                "task": "Registrar estado emocional",
                "time": "mañana",
                "description": "Anotar cómo te sientes al despertar"
            },
            {
                "task": "Practicar técnica de relajación",
                "time": "tarde",
                "description": "5-10 minutos de respiración o meditación"
            },
            {
                "task": "Revisar motivaciones",
                "time": "noche",
                "description": "Leer lista de razones para dejar la adicción"
            }
        ]
        
        if addiction_type.lower() in ["cigarrillos", "tabaco"]:
            tasks.append({
                "task": "Ejercicio físico",
                "time": "mañana",
                "description": "Caminar o hacer ejercicio para reducir ansiedad"
            })
        
        return tasks
    
    def _create_weekly_tasks(self, addiction_type: str) -> List[Dict]:
        """Crea tareas semanales"""
        return [
            {
                "task": "Revisar progreso semanal",
                "day": "domingo",
                "description": "Evaluar logros y desafíos de la semana"
            },
            {
                "task": "Actualizar plan si es necesario",
                "day": "domingo",
                "description": "Ajustar estrategias según lo aprendido"
            }
        ]
    
    def _get_support_resources(self, addiction_type: str) -> List[Dict]:
        """Obtiene recursos de apoyo"""
        resources = [
            {
                "type": "grupo_soporte",
                "name": "Grupos de apoyo local",
                "description": "Buscar grupos de 12 pasos o similares"
            },
            {
                "type": "profesional",
                "name": "Consejería profesional",
                "description": "Considerar terapia o consejería especializada"
            },
            {
                "type": "app",
                "name": "Aplicación móvil",
                "description": "Usar esta app para seguimiento diario"
            }
        ]
        
        return resources
    
    def _load_strategies(self) -> Dict[str, List[Dict]]:
        """Carga estrategias por tipo de adicción"""
        return {
            "cigarrillos": [
                {
                    "name": "Terapia de reemplazo de nicotina",
                    "description": "Considerar parches, chicles o pastillas de nicotina",
                    "priority": "medium"
                },
                {
                    "name": "Evitar situaciones de fumar",
                    "description": "Identificar y evitar lugares/actividades asociadas",
                    "priority": "high"
                }
            ],
            "tabaco": [
                {
                    "name": "Terapia de reemplazo de nicotina",
                    "description": "Considerar parches, chicles o pastillas de nicotina",
                    "priority": "medium"
                }
            ],
            "alcohol": [
                {
                    "name": "Desintoxicación supervisada",
                    "description": "Para dependencia severa, considerar supervisión médica",
                    "priority": "high"
                },
                {
                    "name": "Evitar ambientes con alcohol",
                    "description": "Temporalmente evitar bares, fiestas, etc.",
                    "priority": "high"
                }
            ],
            "drogas": [
                {
                    "name": "Tratamiento profesional",
                    "description": "Buscar programa de tratamiento especializado",
                    "priority": "high"
                },
                {
                    "name": "Grupos de apoyo",
                    "description": "NA, AA, o grupos similares",
                    "priority": "high"
                }
            ]
        }

