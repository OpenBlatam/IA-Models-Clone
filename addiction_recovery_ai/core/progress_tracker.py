"""
Rastreador de progreso - Monitorea y analiza el progreso de recuperación
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from utils.helpers import (
    calculate_days_sober,
    calculate_money_saved,
    calculate_health_improvements,
    get_milestone_message,
    format_time_sober
)


class ProgressTracker:
    """Rastreador de progreso de recuperación"""
    
    def __init__(self):
        """Inicializa el rastreador"""
        pass
    
    def log_entry(
        self,
        user_id: str,
        date: str,
        mood: str,
        cravings_level: int,
        triggers_encountered: List[str],
        consumed: bool,
        notes: Optional[str] = None
    ) -> Dict:
        """
        Registra una entrada diaria
        
        Args:
            user_id: ID del usuario
            date: Fecha de la entrada
            mood: Estado de ánimo
            cravings_level: Nivel de cravings (1-10)
            triggers_encountered: Lista de triggers encontrados
            consumed: Si hubo consumo o no
            notes: Notas adicionales
        
        Returns:
            Entrada registrada
        """
        entry = {
            "user_id": user_id,
            "date": date,
            "mood": mood,
            "cravings_level": cravings_level,
            "triggers_encountered": triggers_encountered,
            "consumed": consumed,
            "notes": notes,
            "logged_at": datetime.now().isoformat()
        }
        
        return entry
    
    def get_progress(
        self,
        user_id: str,
        start_date: Optional[datetime] = None,
        entries: List[Dict] = None
    ) -> Dict:
        """
        Obtiene progreso del usuario
        
        Args:
            user_id: ID del usuario
            start_date: Fecha de inicio (opcional)
            entries: Lista de entradas (opcional, normalmente vendría de BD)
        
        Returns:
            Diccionario con progreso completo
        """
        if entries is None:
            entries = []
        
        # Calcular días de sobriedad
        last_consumption = self._get_last_consumption_date(entries)
        days_sober = calculate_days_sober(last_consumption)
        
        # Calcular estadísticas
        total_entries = len(entries)
        days_without_consumption = sum(1 for e in entries if not e.get("consumed", False))
        average_cravings = self._calculate_average_cravings(entries)
        most_common_triggers = self._get_most_common_triggers(entries)
        
        # Calcular mejoras de salud (necesitaría tipo de adicción del perfil)
        health_improvements = calculate_health_improvements(days_sober, "general")
        
        progress = {
            "user_id": user_id,
            "days_sober": days_sober,
            "time_sober_formatted": format_time_sober(days_sober),
            "total_entries": total_entries,
            "days_without_consumption": days_without_consumption,
            "success_rate": (days_without_consumption / total_entries * 100) if total_entries > 0 else 0,
            "average_cravings_level": average_cravings,
            "most_common_triggers": most_common_triggers,
            "health_improvements": health_improvements,
            "current_milestone": self._get_current_milestone(days_sober),
            "next_milestone": self._get_next_milestone(days_sober),
            "milestone_message": get_milestone_message(days_sober),
            "last_consumption_date": last_consumption.isoformat() if last_consumption else None,
            "streak_days": self._calculate_streak(entries)
        }
        
        return progress
    
    def get_stats(self, user_id: str, entries: List[Dict] = None) -> Dict:
        """
        Obtiene estadísticas detalladas
        
        Args:
            user_id: ID del usuario
            entries: Lista de entradas
        
        Returns:
            Estadísticas completas
        """
        if entries is None:
            entries = []
        
        # Análisis por día de la semana
        day_of_week_stats = self._analyze_by_day_of_week(entries)
        
        # Análisis de tendencias
        trends = self._analyze_trends(entries)
        
        # Análisis de triggers
        trigger_analysis = self._analyze_triggers(entries)
        
        # Análisis de estado de ánimo
        mood_analysis = self._analyze_mood(entries)
        
        stats = {
            "user_id": user_id,
            "total_days_tracked": len(entries),
            "day_of_week_analysis": day_of_week_stats,
            "trends": trends,
            "trigger_analysis": trigger_analysis,
            "mood_analysis": mood_analysis,
            "generated_at": datetime.now().isoformat()
        }
        
        return stats
    
    def get_timeline(self, user_id: str, entries: List[Dict] = None) -> List[Dict]:
        """
        Obtiene línea de tiempo de progreso
        
        Args:
            user_id: ID del usuario
            entries: Lista de entradas
        
        Returns:
            Lista de eventos en la línea de tiempo
        """
        if entries is None:
            entries = []
        
        timeline = []
        
        # Ordenar entradas por fecha
        sorted_entries = sorted(entries, key=lambda x: x.get("date", ""))
        
        for entry in sorted_entries:
            event = {
                "date": entry.get("date"),
                "type": "consumption" if entry.get("consumed") else "sober_day",
                "mood": entry.get("mood"),
                "cravings_level": entry.get("cravings_level"),
                "notes": entry.get("notes")
            }
            timeline.append(event)
        
        # Agregar hitos
        milestones = [1, 7, 30, 90, 180, 365]
        for milestone in milestones:
            # Verificar si se alcanzó el hito
            days_sober_at_milestone = self._get_days_sober_at_date(sorted_entries, milestone)
            if days_sober_at_milestone >= milestone:
                timeline.append({
                    "date": None,  # Se calcularía basado en la fecha
                    "type": "milestone",
                    "milestone_days": milestone,
                    "title": f"{milestone} días de sobriedad"
                })
        
        return sorted(timeline, key=lambda x: x.get("date", ""))
    
    def _get_last_consumption_date(self, entries: List[Dict]) -> Optional[datetime]:
        """Obtiene fecha de último consumo"""
        consumption_entries = [e for e in entries if e.get("consumed", False)]
        if not consumption_entries:
            return None
        
        last_entry = max(consumption_entries, key=lambda x: x.get("date", ""))
        try:
            return datetime.fromisoformat(last_entry["date"].replace('Z', '+00:00'))
        except:
            return None
    
    def _calculate_average_cravings(self, entries: List[Dict]) -> float:
        """Calcula promedio de cravings"""
        if not entries:
            return 0.0
        
        cravings = [e.get("cravings_level", 0) for e in entries if e.get("cravings_level")]
        return sum(cravings) / len(cravings) if cravings else 0.0
    
    def _get_most_common_triggers(self, entries: List[Dict]) -> List[Dict]:
        """Obtiene triggers más comunes"""
        trigger_count = {}
        
        for entry in entries:
            triggers = entry.get("triggers_encountered", [])
            for trigger in triggers:
                trigger_count[trigger] = trigger_count.get(trigger, 0) + 1
        
        sorted_triggers = sorted(trigger_count.items(), key=lambda x: x[1], reverse=True)
        
        return [{"trigger": t[0], "count": t[1]} for t in sorted_triggers[:5]]
    
    def _get_current_milestone(self, days_sober: int) -> Optional[Dict]:
        """Obtiene hito actual"""
        milestones = [1, 7, 30, 90, 180, 365]
        
        for milestone in reversed(milestones):
            if days_sober >= milestone:
                return {
                    "days": milestone,
                    "title": f"{milestone} días",
                    "achieved": True
                }
        
        return None
    
    def _get_next_milestone(self, days_sober: int) -> Optional[Dict]:
        """Obtiene próximo hito"""
        milestones = [1, 7, 30, 90, 180, 365]
        
        for milestone in milestones:
            if days_sober < milestone:
                return {
                    "days": milestone,
                    "title": f"{milestone} días",
                    "days_remaining": milestone - days_sober,
                    "achieved": False
                }
        
        return None
    
    def _calculate_streak(self, entries: List[Dict]) -> int:
        """Calcula racha actual de días sin consumo"""
        if not entries:
            return 0
        
        sorted_entries = sorted(entries, key=lambda x: x.get("date", ""), reverse=True)
        streak = 0
        
        for entry in sorted_entries:
            if entry.get("consumed", False):
                break
            streak += 1
        
        return streak
    
    def _analyze_by_day_of_week(self, entries: List[Dict]) -> Dict:
        """Analiza por día de la semana"""
        day_stats = {}
        days = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
        
        for day in days:
            day_stats[day] = {
                "total_entries": 0,
                "consumption_count": 0,
                "avg_cravings": 0.0
            }
        
        for entry in entries:
            try:
                date = datetime.fromisoformat(entry.get("date", "").replace('Z', '+00:00'))
                day_name = days[date.weekday()]
                
                day_stats[day_name]["total_entries"] += 1
                if entry.get("consumed", False):
                    day_stats[day_name]["consumption_count"] += 1
            except:
                continue
        
        return day_stats
    
    def _analyze_trends(self, entries: List[Dict]) -> Dict:
        """Analiza tendencias"""
        if len(entries) < 7:
            return {"insufficient_data": True}
        
        # Últimos 7 días vs anteriores 7 días
        sorted_entries = sorted(entries, key=lambda x: x.get("date", ""))
        recent = sorted_entries[-7:]
        previous = sorted_entries[-14:-7] if len(sorted_entries) >= 14 else []
        
        recent_consumption_rate = sum(1 for e in recent if e.get("consumed", False)) / len(recent)
        previous_consumption_rate = sum(1 for e in previous if e.get("consumed", False)) / len(previous) if previous else 0
        
        return {
            "recent_consumption_rate": recent_consumption_rate,
            "previous_consumption_rate": previous_consumption_rate,
            "trend": "mejorando" if recent_consumption_rate < previous_consumption_rate else "empeorando" if recent_consumption_rate > previous_consumption_rate else "estable"
        }
    
    def _analyze_triggers(self, entries: List[Dict]) -> Dict:
        """Analiza triggers en detalle"""
        trigger_analysis = {}
        
        for entry in entries:
            triggers = entry.get("triggers_encountered", [])
            consumed = entry.get("consumed", False)
            
            for trigger in triggers:
                if trigger not in trigger_analysis:
                    trigger_analysis[trigger] = {
                        "count": 0,
                        "consumption_count": 0
                    }
                
                trigger_analysis[trigger]["count"] += 1
                if consumed:
                    trigger_analysis[trigger]["consumption_count"] += 1
        
        # Calcular tasa de consumo por trigger
        for trigger, data in trigger_analysis.items():
            data["consumption_rate"] = data["consumption_count"] / data["count"] if data["count"] > 0 else 0
        
        return trigger_analysis
    
    def _analyze_mood(self, entries: List[Dict]) -> Dict:
        """Analiza estado de ánimo"""
        mood_count = {}
        
        for entry in entries:
            mood = entry.get("mood", "neutral")
            mood_count[mood] = mood_count.get(mood, 0) + 1
        
        return mood_count
    
    def _get_days_sober_at_date(self, entries: List[Dict], target_days: int) -> int:
        """Calcula días de sobriedad en una fecha específica"""
        # Implementación simplificada
        sober_days = 0
        for entry in entries:
            if not entry.get("consumed", False):
                sober_days += 1
            else:
                sober_days = 0
            if sober_days >= target_days:
                return sober_days
        return sober_days

