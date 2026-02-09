"""
Workflow Automation System
==========================
Sistema de automatización de workflows con triggers y acciones
"""

import time
import json
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict


class TriggerType(Enum):
    """Tipos de triggers"""
    TIME_BASED = "time_based"
    EVENT_BASED = "event_based"
    CONDITION_BASED = "condition_based"
    MANUAL = "manual"


class ActionType(Enum):
    """Tipos de acciones"""
    PROCESS_IMAGE = "process_image"
    SEND_NOTIFICATION = "send_notification"
    EXPORT_DATA = "export_data"
    SYNC_CLOUD = "sync_cloud"
    RUN_ANALYSIS = "run_analysis"
    CUSTOM = "custom"


@dataclass
class Trigger:
    """Trigger de workflow"""
    id: str
    trigger_type: TriggerType
    config: Dict[str, Any]
    enabled: bool = True


@dataclass
class Action:
    """Acción de workflow"""
    id: str
    action_type: ActionType
    config: Dict[str, Any]
    retry_count: int = 3
    timeout: float = 30.0


@dataclass
class Workflow:
    """Workflow completo"""
    id: str
    name: str
    description: str
    triggers: List[Trigger]
    actions: List[Action]
    enabled: bool = True
    created_at: float = 0.0
    last_run: Optional[float] = None
    run_count: int = 0
    success_count: int = 0
    failure_count: int = 0


@dataclass
class WorkflowExecution:
    """Ejecución de workflow"""
    id: str
    workflow_id: str
    started_at: float
    completed_at: Optional[float] = None
    status: str = "running"  # running, completed, failed
    results: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.results is None:
            self.results = {}


class WorkflowAutomation:
    """
    Sistema de automatización de workflows
    """
    
    def __init__(self):
        self.workflows: Dict[str, Workflow] = {}
        self.executions: Dict[str, List[WorkflowExecution]] = defaultdict(list)
        self.action_handlers: Dict[ActionType, Callable] = {}
        self.trigger_checkers: Dict[TriggerType, Callable] = {}
        self._init_default_handlers()
    
    def _init_default_handlers(self):
        """Inicializar handlers por defecto"""
        # Time-based trigger checker
        self.trigger_checkers[TriggerType.TIME_BASED] = self._check_time_trigger
        # Event-based trigger checker
        self.trigger_checkers[TriggerType.EVENT_BASED] = self._check_event_trigger
        # Condition-based trigger checker
        self.trigger_checkers[TriggerType.CONDITION_BASED] = self._check_condition_trigger
    
    def create_workflow(
        self,
        name: str,
        description: str,
        triggers: List[Dict[str, Any]],
        actions: List[Dict[str, Any]],
        enabled: bool = True
    ) -> Workflow:
        """
        Crear workflow
        
        Args:
            name: Nombre del workflow
            description: Descripción
            triggers: Lista de triggers
            actions: Lista de acciones
            enabled: Si está habilitado
        """
        workflow_id = f"workflow_{int(time.time())}"
        
        workflow_triggers = [
            Trigger(
                id=f"trigger_{i}",
                trigger_type=TriggerType(t['type']),
                config=t.get('config', {}),
                enabled=t.get('enabled', True)
            )
            for i, t in enumerate(triggers)
        ]
        
        workflow_actions = [
            Action(
                id=f"action_{i}",
                action_type=ActionType(a['type']),
                config=a.get('config', {}),
                retry_count=a.get('retry_count', 3),
                timeout=a.get('timeout', 30.0)
            )
            for i, a in enumerate(actions)
        ]
        
        workflow = Workflow(
            id=workflow_id,
            name=name,
            description=description,
            triggers=workflow_triggers,
            actions=workflow_actions,
            enabled=enabled,
            created_at=time.time()
        )
        
        self.workflows[workflow_id] = workflow
        return workflow
    
    def register_action_handler(
        self,
        action_type: ActionType,
        handler: Callable
    ):
        """Registrar handler para acción"""
        self.action_handlers[action_type] = handler
    
    def register_trigger_checker(
        self,
        trigger_type: TriggerType,
        checker: Callable
    ):
        """Registrar checker para trigger"""
        self.trigger_checkers[trigger_type] = checker
    
    def execute_workflow(
        self,
        workflow_id: str,
        manual: bool = False
    ) -> WorkflowExecution:
        """
        Ejecutar workflow
        
        Args:
            workflow_id: ID del workflow
            manual: Si es ejecución manual
        """
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self.workflows[workflow_id]
        
        if not workflow.enabled and not manual:
            raise ValueError(f"Workflow {workflow_id} is disabled")
        
        execution_id = f"exec_{int(time.time())}"
        execution = WorkflowExecution(
            id=execution_id,
            workflow_id=workflow_id,
            started_at=time.time()
        )
        
        self.executions[workflow_id].append(execution)
        
        try:
            # Ejecutar acciones en orden
            for action in workflow.actions:
                result = self._execute_action(action, execution)
                execution.results[action.id] = result
            
            execution.status = "completed"
            execution.completed_at = time.time()
            workflow.success_count += 1
            
        except Exception as e:
            execution.status = "failed"
            execution.completed_at = time.time()
            execution.results['error'] = str(e)
            workflow.failure_count += 1
        
        finally:
            workflow.last_run = time.time()
            workflow.run_count += 1
        
        return execution
    
    def _execute_action(
        self,
        action: Action,
        execution: WorkflowExecution
    ) -> Any:
        """Ejecutar acción individual"""
        handler = self.action_handlers.get(action.action_type)
        
        if not handler:
            raise ValueError(f"No handler registered for action type {action.action_type}")
        
        # Retry logic
        last_error = None
        for attempt in range(action.retry_count):
            try:
                result = handler(action.config, execution)
                return result
            except Exception as e:
                last_error = e
                if attempt < action.retry_count - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        raise Exception(f"Action {action.id} failed after {action.retry_count} attempts: {last_error}")
    
    def check_triggers(self) -> List[str]:
        """
        Verificar triggers y ejecutar workflows
        
        Returns:
            Lista de workflow IDs ejecutados
        """
        executed = []
        
        for workflow_id, workflow in self.workflows.items():
            if not workflow.enabled:
                continue
            
            # Verificar si algún trigger se activa
            should_run = False
            for trigger in workflow.triggers:
                if not trigger.enabled:
                    continue
                
                checker = self.trigger_checkers.get(trigger.trigger_type)
                if checker and checker(trigger):
                    should_run = True
                    break
            
            if should_run:
                try:
                    self.execute_workflow(workflow_id)
                    executed.append(workflow_id)
                except Exception as e:
                    print(f"Error executing workflow {workflow_id}: {e}")
        
        return executed
    
    def _check_time_trigger(self, trigger: Trigger) -> bool:
        """Verificar trigger basado en tiempo"""
        config = trigger.config
        schedule = config.get('schedule', {})
        
        # Verificar intervalo
        if 'interval_seconds' in schedule:
            last_run = config.get('last_triggered', 0)
            if time.time() - last_run >= schedule['interval_seconds']:
                config['last_triggered'] = time.time()
                return True
        
        # Verificar hora específica
        if 'time' in schedule:
            current_time = time.strftime('%H:%M')
            if current_time == schedule['time']:
                last_triggered_date = config.get('last_triggered_date', '')
                today = time.strftime('%Y-%m-%d')
                if last_triggered_date != today:
                    config['last_triggered_date'] = today
                    return True
        
        return False
    
    def _check_event_trigger(self, trigger: Trigger) -> bool:
        """Verificar trigger basado en eventos"""
        # En implementación real, verificar eventos del sistema
        # Por ahora, retornar False (debe ser activado manualmente)
        return False
    
    def _check_condition_trigger(self, trigger: Trigger) -> bool:
        """Verificar trigger basado en condiciones"""
        config = trigger.config
        condition = config.get('condition')
        
        if not condition:
            return False
        
        # Evaluar condición (simplificado)
        # En implementación real, usar un evaluador de expresiones
        return config.get('condition_met', False)
    
    def get_workflow_statistics(self, workflow_id: str) -> Dict[str, Any]:
        """Obtener estadísticas de workflow"""
        if workflow_id not in self.workflows:
            return {}
        
        workflow = self.workflows[workflow_id]
        executions = self.executions[workflow_id]
        
        successful = len([e for e in executions if e.status == 'completed'])
        failed = len([e for e in executions if e.status == 'failed'])
        
        avg_duration = 0
        if executions:
            durations = [
                (e.completed_at or time.time()) - e.started_at
                for e in executions if e.completed_at
            ]
            if durations:
                avg_duration = sum(durations) / len(durations)
        
        return {
            'workflow_id': workflow_id,
            'name': workflow.name,
            'enabled': workflow.enabled,
            'total_runs': workflow.run_count,
            'successful_runs': successful,
            'failed_runs': failed,
            'success_rate': successful / len(executions) if executions else 0,
            'average_duration': avg_duration,
            'last_run': workflow.last_run,
            'created_at': workflow.created_at
        }
    
    def enable_workflow(self, workflow_id: str):
        """Habilitar workflow"""
        if workflow_id in self.workflows:
            self.workflows[workflow_id].enabled = True
    
    def disable_workflow(self, workflow_id: str):
        """Deshabilitar workflow"""
        if workflow_id in self.workflows:
            self.workflows[workflow_id].enabled = False
    
    def delete_workflow(self, workflow_id: str) -> bool:
        """Eliminar workflow"""
        if workflow_id in self.workflows:
            del self.workflows[workflow_id]
            if workflow_id in self.executions:
                del self.executions[workflow_id]
            return True
        return False


# Instancia global
workflow_automation = WorkflowAutomation()

