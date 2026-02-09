"""
Scheduler - Sistema de scheduler/cron jobs
===========================================
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import re

logger = logging.getLogger(__name__)


class ScheduleType(str, Enum):
    """Tipos de schedule"""
    ONCE = "once"
    INTERVAL = "interval"
    CRON = "cron"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


@dataclass
class ScheduledJob:
    """Job programado"""
    id: str
    name: str
    schedule_type: ScheduleType
    schedule_value: str  # cron expression, interval, etc.
    task: Callable
    enabled: bool = True
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    run_count: int = 0
    max_runs: Optional[int] = None


class Scheduler:
    """Sistema de scheduler"""
    
    def __init__(self):
        self.jobs: Dict[str, ScheduledJob] = {}
        self.running = False
        self.scheduler_task: Optional[asyncio.Task] = None
    
    def schedule_job(self, job_id: str, name: str, schedule_type: ScheduleType,
                    schedule_value: str, task: Callable, 
                    max_runs: Optional[int] = None) -> ScheduledJob:
        """Programa un job"""
        job = ScheduledJob(
            id=job_id,
            name=name,
            schedule_type=schedule_type,
            schedule_value=schedule_value,
            task=task,
            max_runs=max_runs
        )
        
        job.next_run = self._calculate_next_run(job)
        self.jobs[job_id] = job
        
        logger.info(f"Job programado: {job_id} - Próxima ejecución: {job.next_run}")
        return job
    
    def _calculate_next_run(self, job: ScheduledJob) -> datetime:
        """Calcula próxima ejecución"""
        now = datetime.now()
        
        if job.schedule_type == ScheduleType.ONCE:
            return datetime.fromisoformat(job.schedule_value)
        
        elif job.schedule_type == ScheduleType.INTERVAL:
            seconds = int(job.schedule_value)
            return now + timedelta(seconds=seconds)
        
        elif job.schedule_type == ScheduleType.DAILY:
            time_str = job.schedule_value  # "HH:MM"
            hour, minute = map(int, time_str.split(":"))
            next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if next_run <= now:
                next_run += timedelta(days=1)
            return next_run
        
        elif job.schedule_type == ScheduleType.CRON:
            # Parsear expresión cron simple (minuto hora día mes día-semana)
            parts = job.schedule_value.split()
            if len(parts) >= 2:
                minute, hour = int(parts[0]), int(parts[1])
                next_run = now.replace(minute=minute, hour=hour, second=0, microsecond=0)
                if next_run <= now:
                    next_run += timedelta(days=1)
                return next_run
        
        return now + timedelta(hours=1)  # Default
    
    async def start(self):
        """Inicia el scheduler"""
        if self.running:
            return
        
        self.running = True
        self.scheduler_task = asyncio.create_task(self._scheduler_loop())
        logger.info("Scheduler iniciado")
    
    async def stop(self):
        """Detiene el scheduler"""
        self.running = False
        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass
        logger.info("Scheduler detenido")
    
    async def _scheduler_loop(self):
        """Loop principal del scheduler"""
        while self.running:
            try:
                now = datetime.now()
                
                for job in self.jobs.values():
                    if not job.enabled:
                        continue
                    
                    if job.max_runs and job.run_count >= job.max_runs:
                        continue
                    
                    if job.next_run and now >= job.next_run:
                        await self._execute_job(job)
                        job.next_run = self._calculate_next_run(job)
                
                await asyncio.sleep(1)  # Verificar cada segundo
            
            except Exception as e:
                logger.error(f"Error en scheduler loop: {e}")
                await asyncio.sleep(5)
    
    async def _execute_job(self, job: ScheduledJob):
        """Ejecuta un job"""
        try:
            logger.info(f"Ejecutando job: {job.name}")
            job.last_run = datetime.now()
            
            if asyncio.iscoroutinefunction(job.task):
                await job.task()
            else:
                job.task()
            
            job.run_count += 1
            logger.info(f"Job {job.name} completado (ejecución #{job.run_count})")
        
        except Exception as e:
            logger.error(f"Error ejecutando job {job.name}: {e}")
    
    def get_jobs(self) -> List[Dict[str, Any]]:
        """Obtiene lista de jobs"""
        return [
            {
                "id": job.id,
                "name": job.name,
                "schedule_type": job.schedule_type.value,
                "enabled": job.enabled,
                "last_run": job.last_run.isoformat() if job.last_run else None,
                "next_run": job.next_run.isoformat() if job.next_run else None,
                "run_count": job.run_count,
                "max_runs": job.max_runs
            }
            for job in self.jobs.values()
        ]
    
    def enable_job(self, job_id: str):
        """Habilita un job"""
        job = self.jobs.get(job_id)
        if job:
            job.enabled = True
            logger.info(f"Job habilitado: {job_id}")
    
    def disable_job(self, job_id: str):
        """Deshabilita un job"""
        job = self.jobs.get(job_id)
        if job:
            job.enabled = False
            logger.info(f"Job deshabilitado: {job_id}")




