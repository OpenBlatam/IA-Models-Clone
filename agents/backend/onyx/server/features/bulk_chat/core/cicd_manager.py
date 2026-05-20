"""
CI/CD Manager - Gestor de CI/CD
================================

Sistema de integración y despliegue continuo con pipelines, stages y automatización.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


class PipelineStatus(Enum):
    """Estado de pipeline."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


class StageStatus(Enum):
    """Estado de stage."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Stage:
    """Stage de pipeline."""
    stage_id: str
    name: str
    commands: List[str]
    status: StageStatus = StageStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    logs: List[str] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Pipeline:
    """Pipeline de CI/CD."""
    pipeline_id: str
    name: str
    stages: List[Stage]
    status: PipelineStatus = PipelineStatus.PENDING
    trigger: str = "manual"  # manual, push, schedule, webhook
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class CICDManager:
    """Gestor de CI/CD."""
    
    def __init__(self):
        self.pipelines: Dict[str, Pipeline] = {}
        self.pipeline_templates: Dict[str, Dict[str, Any]] = {}
        self.pipeline_history: List[str] = []
        self._lock = asyncio.Lock()
    
    def register_pipeline_template(
        self,
        template_id: str,
        name: str,
        stages_config: List[Dict[str, Any]],
    ):
        """Registrar plantilla de pipeline."""
        self.pipeline_templates[template_id] = {
            "template_id": template_id,
            "name": name,
            "stages_config": stages_config,
        }
        
        logger.info(f"Registered pipeline template: {template_id} - {name}")
    
    async def create_pipeline(
        self,
        pipeline_id: str,
        name: str,
        stages_config: List[Dict[str, Any]],
        trigger: str = "manual",
    ) -> str:
        """Crear pipeline."""
        stages = []
        
        for stage_config in stages_config:
            stage = Stage(
                stage_id=f"{pipeline_id}_stage_{len(stages)}",
                name=stage_config.get("name", "unnamed"),
                commands=stage_config.get("commands", []),
            )
            stages.append(stage)
        
        pipeline = Pipeline(
            pipeline_id=pipeline_id,
            name=name,
            stages=stages,
            trigger=trigger,
        )
        
        async with self._lock:
            self.pipelines[pipeline_id] = pipeline
        
        logger.info(f"Created pipeline: {pipeline_id} - {name}")
        return pipeline_id
    
    async def create_pipeline_from_template(
        self,
        template_id: str,
        pipeline_id: Optional[str] = None,
        override_config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Crear pipeline desde plantilla."""
        template = self.pipeline_templates.get(template_id)
        if not template:
            raise ValueError(f"Template not found: {template_id}")
        
        if pipeline_id is None:
            pipeline_id = f"{template_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        stages_config = template["stages_config"]
        if override_config:
            stages_config = override_config.get("stages", stages_config)
        
        return await self.create_pipeline(
            pipeline_id=pipeline_id,
            name=template["name"],
            stages_config=stages_config,
            trigger=override_config.get("trigger", "manual") if override_config else "manual",
        )
    
    async def run_pipeline(self, pipeline_id: str) -> bool:
        """Ejecutar pipeline."""
        pipeline = self.pipelines.get(pipeline_id)
        if not pipeline:
            return False
        
        async with self._lock:
            pipeline.status = PipelineStatus.RUNNING
            pipeline.started_at = datetime.now()
        
        logger.info(f"Running pipeline: {pipeline_id}")
        
        # Ejecutar stages secuencialmente
        for stage in pipeline.stages:
            stage.status = StageStatus.RUNNING
            stage.start_time = datetime.now()
            
            # Simular ejecución de comandos
            for command in stage.commands:
                stage.logs.append(f"[{datetime.now()}] Executing: {command}")
                await asyncio.sleep(0.1)  # Simular tiempo de ejecución
            
            stage.end_time = datetime.now()
            stage.status = StageStatus.SUCCESS
            
            # Si algún stage falla, detener pipeline
            if stage.status == StageStatus.FAILED:
                async with self._lock:
                    pipeline.status = PipelineStatus.FAILED
                    pipeline.completed_at = datetime.now()
                return False
        
        async with self._lock:
            pipeline.status = PipelineStatus.SUCCESS
            pipeline.completed_at = datetime.now()
            self.pipeline_history.append(pipeline_id)
        
        logger.info(f"Pipeline completed: {pipeline_id} - {pipeline.status.value}")
        return True
    
    async def cancel_pipeline(self, pipeline_id: str) -> bool:
        """Cancelar pipeline."""
        pipeline = self.pipelines.get(pipeline_id)
        if not pipeline:
            return False
        
        async with self._lock:
            pipeline.status = PipelineStatus.CANCELLED
            pipeline.completed_at = datetime.now()
            
            # Cancelar stages en ejecución
            for stage in pipeline.stages:
                if stage.status == StageStatus.RUNNING:
                    stage.status = StageStatus.SKIPPED
        
        logger.info(f"Cancelled pipeline: {pipeline_id}")
        return True
    
    def get_pipeline(self, pipeline_id: str) -> Optional[Dict[str, Any]]:
        """Obtener pipeline."""
        pipeline = self.pipelines.get(pipeline_id)
        if not pipeline:
            return None
        
        return {
            "pipeline_id": pipeline.pipeline_id,
            "name": pipeline.name,
            "status": pipeline.status.value,
            "trigger": pipeline.trigger,
            "stages": [
                {
                    "stage_id": s.stage_id,
                    "name": s.name,
                    "status": s.status.value,
                    "start_time": s.start_time.isoformat() if s.start_time else None,
                    "end_time": s.end_time.isoformat() if s.end_time else None,
                    "commands_count": len(s.commands),
                    "logs_count": len(s.logs),
                }
                for s in pipeline.stages
            ],
            "created_at": pipeline.created_at.isoformat(),
            "started_at": pipeline.started_at.isoformat() if pipeline.started_at else None,
            "completed_at": pipeline.completed_at.isoformat() if pipeline.completed_at else None,
        }
    
    def get_pipeline_logs(self, pipeline_id: str, stage_id: Optional[str] = None) -> List[str]:
        """Obtener logs de pipeline."""
        pipeline = self.pipelines.get(pipeline_id)
        if not pipeline:
            return []
        
        if stage_id:
            stage = next((s for s in pipeline.stages if s.stage_id == stage_id), None)
            if stage:
                return stage.logs
            return []
        
        # Concatenar logs de todos los stages
        all_logs = []
        for stage in pipeline.stages:
            all_logs.extend([f"[{stage.name}] {log}" for log in stage.logs])
        
        return all_logs
    
    def get_cicd_summary(self) -> Dict[str, Any]:
        """Obtener resumen de CI/CD."""
        by_status: Dict[str, int] = defaultdict(int)
        
        for pipeline in self.pipelines.values():
            by_status[pipeline.status.value] += 1
        
        return {
            "total_pipelines": len(self.pipelines),
            "pipelines_by_status": dict(by_status),
            "total_templates": len(self.pipeline_templates),
            "total_executions": len(self.pipeline_history),
        }
















