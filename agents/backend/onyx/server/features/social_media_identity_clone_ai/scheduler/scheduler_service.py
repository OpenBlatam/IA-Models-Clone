"""
Sistema de scheduling de contenido
"""

import logging
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from sqlalchemy import Column, String, Text, DateTime, JSON, Boolean, Integer, Enum as SQLEnum
from sqlalchemy.orm import Session
import asyncio

from ..db.base import Base, get_db_session

logger = logging.getLogger(__name__)


class ScheduleType(str, Enum):
    """Tipos de scheduling"""
    ONCE = "once"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    CUSTOM = "custom"


class ScheduledTaskModel(Base):
    """Modelo de tarea programada en BD"""
    __tablename__ = "scheduled_tasks"
    
    id = Column(String(64), primary_key=True, index=True)
    identity_profile_id = Column(String(64), nullable=False, index=True)
    task_type = Column(String(50), nullable=False)  # generate_content, extract_profile, etc.
    schedule_type = Column(String(20), nullable=False)
    schedule_config = Column(JSON, nullable=False)  # Configuración del schedule
    payload = Column(JSON, nullable=False)  # Datos para la tarea
    enabled = Column(Boolean, default=True, nullable=False, index=True)
    next_run_at = Column(DateTime, nullable=False, index=True)
    last_run_at = Column(DateTime, nullable=True)
    run_count = Column(Integer, default=0, nullable=False)
    max_runs = Column(Integer, nullable=True)  # None = ilimitado
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


@dataclass
class ScheduledTask:
    """Tarea programada"""
    task_id: str
    identity_profile_id: str
    task_type: str
    schedule_type: ScheduleType
    schedule_config: Dict[str, Any]
    payload: Dict[str, Any]
    enabled: bool = True
    next_run_at: datetime = field(default_factory=datetime.utcnow)
    last_run_at: Optional[datetime] = None
    run_count: int = 0
    max_runs: Optional[int] = None


class SchedulerService:
    """Servicio de scheduling"""
    
    def __init__(self):
        self._init_table()
        self.running = False
        self.scheduler_task: Optional[asyncio.Task] = None
    
    def _init_table(self):
        """Inicializa tabla de scheduled tasks"""
        from ..db.base import init_db
        init_db()
    
    def create_schedule(
        self,
        identity_profile_id: str,
        task_type: str,
        schedule_type: ScheduleType,
        schedule_config: Dict[str, Any],
        payload: Dict[str, Any],
        enabled: bool = True,
        max_runs: Optional[int] = None
    ) -> str:
        """
        Crea una tarea programada
        
        Args:
            identity_profile_id: ID de la identidad
            task_type: Tipo de tarea
            schedule_type: Tipo de schedule
            schedule_config: Configuración del schedule
            payload: Datos para la tarea
            enabled: Si está habilitada
            max_runs: Máximo de ejecuciones (None = ilimitado)
            
        Returns:
            ID de la tarea programada
        """
        task_id = str(uuid.uuid4())
        next_run_at = self._calculate_next_run(schedule_type, schedule_config)
        
        with get_db_session() as db:
            scheduled_task = ScheduledTaskModel(
                id=task_id,
                identity_profile_id=identity_profile_id,
                task_type=task_type,
                schedule_type=schedule_type.value,
                schedule_config=schedule_config,
                payload=payload,
                enabled=enabled,
                next_run_at=next_run_at,
                run_count=0,
                max_runs=max_runs
            )
            db.add(scheduled_task)
            db.commit()
        
        logger.info(f"Tarea programada creada: {task_id} ({schedule_type.value})")
        return task_id
    
    def _calculate_next_run(self, schedule_type: ScheduleType, config: Dict[str, Any]) -> datetime:
        """Calcula próxima ejecución"""
        now = datetime.utcnow()
        
        if schedule_type == ScheduleType.ONCE:
            return datetime.fromisoformat(config["run_at"])
        elif schedule_type == ScheduleType.DAILY:
            hour = config.get("hour", 9)
            minute = config.get("minute", 0)
            next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if next_run <= now:
                next_run += timedelta(days=1)
            return next_run
        elif schedule_type == ScheduleType.WEEKLY:
            weekday = config.get("weekday", 0)  # 0 = Monday
            hour = config.get("hour", 9)
            minute = config.get("minute", 0)
            days_ahead = weekday - now.weekday()
            if days_ahead <= 0:
                days_ahead += 7
            next_run = now + timedelta(days=days_ahead)
            next_run = next_run.replace(hour=hour, minute=minute, second=0, microsecond=0)
            return next_run
        elif schedule_type == ScheduleType.MONTHLY:
            day = config.get("day", 1)
            hour = config.get("hour", 9)
            minute = config.get("minute", 0)
            next_run = now.replace(day=day, hour=hour, minute=minute, second=0, microsecond=0)
            if next_run <= now:
                # Próximo mes
                if next_run.month == 12:
                    next_run = next_run.replace(year=next_run.year + 1, month=1)
                else:
                    next_run = next_run.replace(month=next_run.month + 1)
            return next_run
        else:
            return now + timedelta(hours=1)
    
    def get_schedules(
        self,
        identity_profile_id: Optional[str] = None,
        enabled_only: bool = False,
        limit: int = 50
    ) -> List[ScheduledTask]:
        """Obtiene tareas programadas"""
        with get_db_session() as db:
            query = db.query(ScheduledTaskModel)
            
            if identity_profile_id:
                query = query.filter_by(identity_profile_id=identity_profile_id)
            
            if enabled_only:
                query = query.filter_by(enabled=True)
            
            results = query.order_by(
                ScheduledTaskModel.next_run_at.asc()
            ).limit(limit).all()
            
            return [
                ScheduledTask(
                    task_id=r.id,
                    identity_profile_id=r.identity_profile_id,
                    task_type=r.task_type,
                    schedule_type=ScheduleType(r.schedule_type),
                    schedule_config=r.schedule_config,
                    payload=r.payload,
                    enabled=r.enabled,
                    next_run_at=r.next_run_at,
                    last_run_at=r.last_run_at,
                    run_count=r.run_count,
                    max_runs=r.max_runs
                )
                for r in results
            ]
    
    async def start_scheduler(self):
        """Inicia el scheduler"""
        self.running = True
        logger.info("Scheduler iniciado")
        
        while self.running:
            try:
                await self._process_due_tasks()
                await asyncio.sleep(60)  # Verificar cada minuto
            except Exception as e:
                logger.error(f"Error en scheduler: {e}", exc_info=True)
                await asyncio.sleep(60)
    
    async def _process_due_tasks(self):
        """Procesa tareas que deben ejecutarse"""
        now = datetime.utcnow()
        
        with get_db_session() as db:
            due_tasks = db.query(ScheduledTaskModel).filter(
                ScheduledTaskModel.enabled == True,
                ScheduledTaskModel.next_run_at <= now
            ).all()
            
            for task in due_tasks:
                try:
                    await self._execute_scheduled_task(task)
                except Exception as e:
                    logger.error(f"Error ejecutando tarea programada {task.id}: {e}", exc_info=True)
    
    async def _execute_scheduled_task(self, task: ScheduledTaskModel):
        """Ejecuta una tarea programada"""
        logger.info(f"Ejecutando tarea programada: {task.id} ({task.task_type})")
        
        from ..queue.task_queue import get_task_queue
        
        task_queue = get_task_queue()
        
        # Crear tarea en la cola
        queue_task_id = await task_queue.add_task(
            task_type=task.task_type,
            payload=task.payload
        )
        
        # Actualizar scheduled task
        with get_db_session() as db:
            db_task = db.query(ScheduledTaskModel).filter_by(id=task.id).first()
            if db_task:
                db_task.last_run_at = datetime.utcnow()
                db_task.run_count += 1
                
                # Verificar si debe continuar
                if db_task.max_runs and db_task.run_count >= db_task.max_runs:
                    db_task.enabled = False
                else:
                    # Calcular próxima ejecución
                    db_task.next_run_at = self._calculate_next_run(
                        ScheduleType(db_task.schedule_type),
                        db_task.schedule_config
                    )
                
                db.commit()
    
    def stop_scheduler(self):
        """Detiene el scheduler"""
        self.running = False
        logger.info("Scheduler detenido")


# Singleton global
_scheduler_service: Optional[SchedulerService] = None


def get_scheduler_service() -> SchedulerService:
    """Obtiene instancia singleton del scheduler"""
    global _scheduler_service
    if _scheduler_service is None:
        _scheduler_service = SchedulerService()
    return _scheduler_service




