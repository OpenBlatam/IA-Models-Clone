"""
Motor de Automatización AI
=========================

Motor para automatización inteligente de procesos, RPA (Robotic Process Automation) y orquestación de tareas.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
from pathlib import Path
import hashlib
import subprocess
import os

logger = logging.getLogger(__name__)

class AutomationType(str, Enum):
    """Tipos de automatización"""
    DOCUMENT_PROCESSING = "document_processing"
    DATA_EXTRACTION = "data_extraction"
    WORKFLOW_AUTOMATION = "workflow_automation"
    SYSTEM_INTEGRATION = "system_integration"
    REPORT_GENERATION = "report_generation"
    EMAIL_AUTOMATION = "email_automation"
    FILE_MANAGEMENT = "file_management"
    API_AUTOMATION = "api_automation"

class TriggerType(str, Enum):
    """Tipos de trigger"""
    SCHEDULED = "scheduled"
    FILE_WATCH = "file_watch"
    EMAIL_RECEIVED = "email_received"
    API_CALL = "api_call"
    MANUAL = "manual"
    CONDITION_BASED = "condition_based"
    EVENT_DRIVEN = "event_driven"

class AutomationStatus(str, Enum):
    """Estados de automatización"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    COMPLETED = "completed"

@dataclass
class AutomationRule:
    """Regla de automatización"""
    id: str
    name: str
    description: str
    automation_type: AutomationType
    trigger_type: TriggerType
    trigger_config: Dict[str, Any] = field(default_factory=dict)
    actions: List[Dict[str, Any]] = field(default_factory=list)
    conditions: List[Dict[str, Any]] = field(default_factory=list)
    status: AutomationStatus = AutomationStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    last_executed: Optional[datetime] = None
    execution_count: int = 0
    success_count: int = 0
    error_count: int = 0

@dataclass
class AutomationExecution:
    """Ejecución de automatización"""
    id: str
    rule_id: str
    status: AutomationStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    trigger_data: Dict[str, Any] = field(default_factory=dict)
    execution_log: List[Dict[str, Any]] = field(default_factory=list)
    result_data: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None

@dataclass
class AutomationTemplate:
    """Plantilla de automatización"""
    id: str
    name: str
    description: str
    category: str
    template_config: Dict[str, Any] = field(default_factory=dict)
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    examples: List[Dict[str, Any]] = field(default_factory=list)

class AIAutomationEngine:
    """Motor de automatización AI"""
    
    def __init__(self):
        self.automation_rules: Dict[str, AutomationRule] = {}
        self.automation_executions: Dict[str, AutomationExecution] = {}
        self.automation_templates: Dict[str, AutomationTemplate] = {}
        self.running_automations: Dict[str, asyncio.Task] = {}
        self.file_watchers: Dict[str, Any] = {}
        self.scheduled_tasks: Dict[str, asyncio.Task] = {}
        
        # Configuración
        self.max_concurrent_automations = 10
        self.automation_timeout = 300  # 5 minutos
        
    async def initialize(self):
        """Inicializa el motor de automatización"""
        logger.info("Inicializando motor de automatización AI...")
        
        # Cargar plantillas predefinidas
        await self._load_automation_templates()
        
        # Cargar reglas existentes
        await self._load_automation_rules()
        
        # Iniciar scheduler
        asyncio.create_task(self._scheduler_worker())
        
        # Iniciar file watchers
        asyncio.create_task(self._file_watcher_worker())
        
        logger.info("Motor de automatización AI inicializado")
    
    async def _load_automation_templates(self):
        """Carga plantillas de automatización predefinidas"""
        try:
            # Plantilla de procesamiento automático de documentos
            doc_processing_template = AutomationTemplate(
                id="auto_document_processing",
                name="Procesamiento Automático de Documentos",
                description="Procesa automáticamente documentos que llegan a una carpeta",
                category="document_processing",
                template_config={
                    "trigger_type": "file_watch",
                    "automation_type": "document_processing",
                    "watch_path": "/input/documents",
                    "file_patterns": ["*.pdf", "*.docx", "*.md"],
                    "actions": [
                        {
                            "type": "process_document",
                            "config": {
                                "target_format": "consultancy",
                                "include_analysis": True
                            }
                        },
                        {
                            "type": "move_file",
                            "config": {
                                "destination": "/processed/documents"
                            }
                        },
                        {
                            "type": "send_notification",
                            "config": {
                                "message": "Document processed successfully"
                            }
                        }
                    ]
                },
                parameters=[
                    {
                        "name": "watch_path",
                        "type": "string",
                        "description": "Ruta a monitorear",
                        "required": True
                    },
                    {
                        "name": "target_format",
                        "type": "select",
                        "options": ["consultancy", "technical", "academic", "commercial", "legal"],
                        "description": "Formato objetivo",
                        "required": True
                    }
                ],
                examples=[
                    {
                        "name": "Procesar facturas",
                        "description": "Procesa automáticamente facturas PDF",
                        "config": {
                            "watch_path": "/invoices",
                            "file_patterns": ["*.pdf"],
                            "target_format": "commercial"
                        }
                    }
                ]
            )
            
            # Plantilla de extracción de datos
            data_extraction_template = AutomationTemplate(
                id="auto_data_extraction",
                name="Extracción Automática de Datos",
                description="Extrae datos específicos de documentos automáticamente",
                category="data_extraction",
                template_config={
                    "trigger_type": "file_watch",
                    "automation_type": "data_extraction",
                    "watch_path": "/input/data",
                    "extraction_rules": [
                        {
                            "field": "invoice_number",
                            "pattern": r"Invoice\s+#(\d+)",
                            "type": "regex"
                        },
                        {
                            "field": "amount",
                            "pattern": r"Total:\s+\$?([\d,]+\.?\d*)",
                            "type": "regex"
                        }
                    ],
                    "actions": [
                        {
                            "type": "extract_data",
                            "config": {
                                "output_format": "json"
                            }
                        },
                        {
                            "type": "save_to_database",
                            "config": {
                                "table": "extracted_data"
                            }
                        }
                    ]
                },
                parameters=[
                    {
                        "name": "extraction_rules",
                        "type": "array",
                        "description": "Reglas de extracción",
                        "required": True
                    },
                    {
                        "name": "output_format",
                        "type": "select",
                        "options": ["json", "csv", "xml"],
                        "description": "Formato de salida",
                        "required": True
                    }
                ]
            )
            
            # Plantilla de generación de reportes
            report_generation_template = AutomationTemplate(
                id="auto_report_generation",
                name="Generación Automática de Reportes",
                description="Genera reportes automáticamente basados en datos",
                category="report_generation",
                template_config={
                    "trigger_type": "scheduled",
                    "automation_type": "report_generation",
                    "schedule": "0 9 * * 1",  # Lunes a las 9 AM
                    "data_source": "analytics_service",
                    "report_template": "weekly_summary",
                    "actions": [
                        {
                            "type": "collect_data",
                            "config": {
                                "time_range": "7d",
                                "metrics": ["processing_count", "success_rate", "error_rate"]
                            }
                        },
                        {
                            "type": "generate_report",
                            "config": {
                                "template": "weekly_summary",
                                "format": "pdf"
                            }
                        },
                        {
                            "type": "send_email",
                            "config": {
                                "recipients": ["admin@company.com"],
                                "subject": "Weekly Processing Report"
                            }
                        }
                    ]
                },
                parameters=[
                    {
                        "name": "schedule",
                        "type": "string",
                        "description": "Expresión cron para programación",
                        "required": True
                    },
                    {
                        "name": "report_template",
                        "type": "string",
                        "description": "Plantilla de reporte",
                        "required": True
                    }
                ]
            )
            
            # Plantilla de integración de sistemas
            system_integration_template = AutomationTemplate(
                id="auto_system_integration",
                name="Integración Automática de Sistemas",
                description="Integra automáticamente datos entre sistemas",
                category="system_integration",
                template_config={
                    "trigger_type": "api_call",
                    "automation_type": "system_integration",
                    "source_system": "crm",
                    "target_system": "analytics",
                    "mapping_rules": [
                        {
                            "source_field": "customer_id",
                            "target_field": "client_id"
                        },
                        {
                            "source_field": "created_date",
                            "target_field": "registration_date"
                        }
                    ],
                    "actions": [
                        {
                            "type": "fetch_data",
                            "config": {
                                "endpoint": "/api/customers",
                                "method": "GET"
                            }
                        },
                        {
                            "type": "transform_data",
                            "config": {
                                "mapping_rules": "mapping_rules"
                            }
                        },
                        {
                            "type": "send_data",
                            "config": {
                                "endpoint": "/api/analytics/import",
                                "method": "POST"
                            }
                        }
                    ]
                },
                parameters=[
                    {
                        "name": "source_system",
                        "type": "string",
                        "description": "Sistema fuente",
                        "required": True
                    },
                    {
                        "name": "target_system",
                        "type": "string",
                        "description": "Sistema destino",
                        "required": True
                    }
                ]
            )
            
            # Guardar plantillas
            self.automation_templates["auto_document_processing"] = doc_processing_template
            self.automation_templates["auto_data_extraction"] = data_extraction_template
            self.automation_templates["auto_report_generation"] = report_generation_template
            self.automation_templates["auto_system_integration"] = system_integration_template
            
            logger.info(f"Cargadas {len(self.automation_templates)} plantillas de automatización")
            
        except Exception as e:
            logger.error(f"Error cargando plantillas de automatización: {e}")
    
    async def _load_automation_rules(self):
        """Carga reglas de automatización existentes"""
        try:
            rules_file = Path("data/automation_rules.json")
            if rules_file.exists():
                with open(rules_file, 'r', encoding='utf-8') as f:
                    rules_data = json.load(f)
                
                for rule_data in rules_data:
                    rule = AutomationRule(
                        id=rule_data["id"],
                        name=rule_data["name"],
                        description=rule_data["description"],
                        automation_type=AutomationType(rule_data["automation_type"]),
                        trigger_type=TriggerType(rule_data["trigger_type"]),
                        trigger_config=rule_data["trigger_config"],
                        actions=rule_data["actions"],
                        conditions=rule_data["conditions"],
                        status=AutomationStatus(rule_data["status"]),
                        created_at=datetime.fromisoformat(rule_data["created_at"]),
                        updated_at=datetime.fromisoformat(rule_data["updated_at"]),
                        last_executed=datetime.fromisoformat(rule_data["last_executed"]) if rule_data.get("last_executed") else None,
                        execution_count=rule_data.get("execution_count", 0),
                        success_count=rule_data.get("success_count", 0),
                        error_count=rule_data.get("error_count", 0)
                    )
                    self.automation_rules[rule.id] = rule
                
                logger.info(f"Cargadas {len(self.automation_rules)} reglas de automatización")
            
        except Exception as e:
            logger.error(f"Error cargando reglas de automatización: {e}")
    
    async def create_automation_rule(
        self,
        name: str,
        description: str,
        automation_type: AutomationType,
        trigger_type: TriggerType,
        trigger_config: Dict[str, Any],
        actions: List[Dict[str, Any]],
        conditions: List[Dict[str, Any]] = None
    ) -> str:
        """Crea una nueva regla de automatización"""
        try:
            rule_id = f"rule_{uuid.uuid4().hex[:8]}"
            
            rule = AutomationRule(
                id=rule_id,
                name=name,
                description=description,
                automation_type=automation_type,
                trigger_type=trigger_type,
                trigger_config=trigger_config,
                actions=actions,
                conditions=conditions or []
            )
            
            self.automation_rules[rule_id] = rule
            
            # Configurar trigger según tipo
            await self._setup_trigger(rule)
            
            # Guardar regla
            await self._save_automation_rules()
            
            logger.info(f"Regla de automatización creada: {rule_id}")
            return rule_id
            
        except Exception as e:
            logger.error(f"Error creando regla de automatización: {e}")
            raise
    
    async def create_automation_from_template(
        self,
        template_id: str,
        name: str,
        parameters: Dict[str, Any]
    ) -> str:
        """Crea automatización desde plantilla"""
        try:
            if template_id not in self.automation_templates:
                raise ValueError(f"Plantilla no encontrada: {template_id}")
            
            template = self.automation_templates[template_id]
            template_config = template.template_config.copy()
            
            # Aplicar parámetros
            for param_name, param_value in parameters.items():
                template_config = self._apply_parameter(template_config, param_name, param_value)
            
            # Crear regla
            rule_id = await self.create_automation_rule(
                name=name,
                description=template.description,
                automation_type=AutomationType(template_config["automation_type"]),
                trigger_type=TriggerType(template_config["trigger_type"]),
                trigger_config=template_config.get("trigger_config", {}),
                actions=template_config.get("actions", []),
                conditions=template_config.get("conditions", [])
            )
            
            logger.info(f"Automatización creada desde plantilla: {rule_id}")
            return rule_id
            
        except Exception as e:
            logger.error(f"Error creando automatización desde plantilla: {e}")
            raise
    
    def _apply_parameter(self, config: Dict[str, Any], param_name: str, param_value: Any) -> Dict[str, Any]:
        """Aplica parámetro a configuración"""
        try:
            # Búsqueda recursiva y reemplazo
            if isinstance(config, dict):
                for key, value in config.items():
                    if key == param_name:
                        config[key] = param_value
                    elif isinstance(value, (dict, list)):
                        config[key] = self._apply_parameter(value, param_name, param_value)
            elif isinstance(config, list):
                for i, item in enumerate(config):
                    config[i] = self._apply_parameter(item, param_name, param_value)
            
            return config
            
        except Exception as e:
            logger.error(f"Error aplicando parámetro: {e}")
            return config
    
    async def _setup_trigger(self, rule: AutomationRule):
        """Configura trigger para la regla"""
        try:
            if rule.trigger_type == TriggerType.SCHEDULED:
                await self._setup_scheduled_trigger(rule)
            elif rule.trigger_type == TriggerType.FILE_WATCH:
                await self._setup_file_watch_trigger(rule)
            elif rule.trigger_type == TriggerType.EMAIL_RECEIVED:
                await self._setup_email_trigger(rule)
            elif rule.trigger_type == TriggerType.API_CALL:
                await self._setup_api_trigger(rule)
            
        except Exception as e:
            logger.error(f"Error configurando trigger: {e}")
    
    async def _setup_scheduled_trigger(self, rule: AutomationRule):
        """Configura trigger programado"""
        try:
            schedule = rule.trigger_config.get("schedule")
            if not schedule:
                logger.warning(f"No se especificó horario para regla {rule.id}")
                return
            
            # En implementación real, usar croniter
            # Por ahora, simular configuración
            task = asyncio.create_task(self._scheduled_automation_worker(rule))
            self.scheduled_tasks[rule.id] = task
            
            logger.info(f"Trigger programado configurado para regla {rule.id}")
            
        except Exception as e:
            logger.error(f"Error configurando trigger programado: {e}")
    
    async def _setup_file_watch_trigger(self, rule: AutomationRule):
        """Configura trigger de monitoreo de archivos"""
        try:
            watch_path = rule.trigger_config.get("watch_path")
            if not watch_path:
                logger.warning(f"No se especificó ruta para monitorear en regla {rule.id}")
                return
            
            # En implementación real, usar watchdog
            # Por ahora, simular configuración
            self.file_watchers[rule.id] = {
                "path": watch_path,
                "patterns": rule.trigger_config.get("file_patterns", ["*"]),
                "rule": rule
            }
            
            logger.info(f"Monitoreo de archivos configurado para regla {rule.id}")
            
        except Exception as e:
            logger.error(f"Error configurando monitoreo de archivos: {e}")
    
    async def _setup_email_trigger(self, rule: AutomationRule):
        """Configura trigger de email"""
        try:
            # En implementación real, configurar IMAP/POP3
            logger.info(f"Trigger de email configurado para regla {rule.id}")
            
        except Exception as e:
            logger.error(f"Error configurando trigger de email: {e}")
    
    async def _setup_api_trigger(self, rule: AutomationRule):
        """Configura trigger de API"""
        try:
            # En implementación real, configurar webhook
            logger.info(f"Trigger de API configurado para regla {rule.id}")
            
        except Exception as e:
            logger.error(f"Error configurando trigger de API: {e}")
    
    async def execute_automation(
        self,
        rule_id: str,
        trigger_data: Dict[str, Any] = None
    ) -> str:
        """Ejecuta una automatización"""
        try:
            if rule_id not in self.automation_rules:
                raise ValueError(f"Regla no encontrada: {rule_id}")
            
            rule = self.automation_rules[rule_id]
            
            if rule.status != AutomationStatus.ACTIVE:
                raise ValueError(f"Regla no está activa: {rule_id}")
            
            # Verificar condiciones
            if not await self._check_conditions(rule, trigger_data or {}):
                logger.info(f"Condiciones no cumplidas para regla {rule_id}")
                return None
            
            # Crear ejecución
            execution_id = f"exec_{uuid.uuid4().hex[:8]}"
            execution = AutomationExecution(
                id=execution_id,
                rule_id=rule_id,
                status=AutomationStatus.RUNNING,
                started_at=datetime.now(),
                trigger_data=trigger_data or {}
            )
            
            self.automation_executions[execution_id] = execution
            
            # Ejecutar automatización
            task = asyncio.create_task(self._execute_automation_actions(rule, execution))
            self.running_automations[execution_id] = task
            
            # Actualizar estadísticas de la regla
            rule.execution_count += 1
            rule.last_executed = datetime.now()
            
            logger.info(f"Automatización ejecutada: {execution_id}")
            return execution_id
            
        except Exception as e:
            logger.error(f"Error ejecutando automatización: {e}")
            raise
    
    async def _check_conditions(self, rule: AutomationRule, trigger_data: Dict[str, Any]) -> bool:
        """Verifica condiciones de la regla"""
        try:
            if not rule.conditions:
                return True
            
            for condition in rule.conditions:
                condition_type = condition.get("type")
                condition_config = condition.get("config", {})
                
                if condition_type == "file_size":
                    file_path = trigger_data.get("file_path")
                    if file_path:
                        file_size = os.path.getsize(file_path)
                        max_size = condition_config.get("max_size", 0)
                        if file_size > max_size:
                            return False
                
                elif condition_type == "file_type":
                    file_path = trigger_data.get("file_path")
                    if file_path:
                        file_ext = Path(file_path).suffix.lower()
                        allowed_types = condition_config.get("allowed_types", [])
                        if file_ext not in allowed_types:
                            return False
                
                elif condition_type == "time_range":
                    current_hour = datetime.now().hour
                    start_hour = condition_config.get("start_hour", 0)
                    end_hour = condition_config.get("end_hour", 23)
                    if not (start_hour <= current_hour <= end_hour):
                        return False
                
                elif condition_type == "custom":
                    # Evaluar condición personalizada
                    expression = condition_config.get("expression")
                    if expression and not self._evaluate_expression(expression, trigger_data):
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error verificando condiciones: {e}")
            return False
    
    def _evaluate_expression(self, expression: str, data: Dict[str, Any]) -> bool:
        """Evalúa expresión personalizada"""
        try:
            # Implementación simplificada
            # En implementación real, usar un evaluador de expresiones seguro
            return True
            
        except Exception as e:
            logger.error(f"Error evaluando expresión: {e}")
            return False
    
    async def _execute_automation_actions(self, rule: AutomationRule, execution: AutomationExecution):
        """Ejecuta acciones de automatización"""
        try:
            for action in rule.actions:
                action_type = action.get("type")
                action_config = action.get("config", {})
                
                # Log de acción
                execution.execution_log.append({
                    "timestamp": datetime.now().isoformat(),
                    "action": action_type,
                    "status": "started"
                })
                
                try:
                    # Ejecutar acción
                    result = await self._execute_action(action_type, action_config, execution.trigger_data)
                    
                    execution.execution_log.append({
                        "timestamp": datetime.now().isoformat(),
                        "action": action_type,
                        "status": "completed",
                        "result": result
                    })
                    
                except Exception as e:
                    execution.execution_log.append({
                        "timestamp": datetime.now().isoformat(),
                        "action": action_type,
                        "status": "failed",
                        "error": str(e)
                    })
                    
                    # Continuar con siguiente acción o fallar según configuración
                    if action_config.get("fail_on_error", True):
                        raise e
            
            # Completar ejecución
            execution.status = AutomationStatus.COMPLETED
            execution.completed_at = datetime.now()
            
            # Actualizar estadísticas
            rule.success_count += 1
            
            # Limpiar ejecución en memoria después de un tiempo
            asyncio.create_task(self._cleanup_execution(execution.id))
            
        except Exception as e:
            logger.error(f"Error ejecutando acciones de automatización: {e}")
            execution.status = AutomationStatus.ERROR
            execution.error_message = str(e)
            execution.completed_at = datetime.now()
            
            # Actualizar estadísticas
            rule.error_count += 1
    
    async def _execute_action(self, action_type: str, config: Dict[str, Any], trigger_data: Dict[str, Any]) -> Any:
        """Ejecuta una acción específica"""
        try:
            if action_type == "process_document":
                return await self._action_process_document(config, trigger_data)
            elif action_type == "move_file":
                return await self._action_move_file(config, trigger_data)
            elif action_type == "copy_file":
                return await self._action_copy_file(config, trigger_data)
            elif action_type == "delete_file":
                return await self._action_delete_file(config, trigger_data)
            elif action_type == "extract_data":
                return await self._action_extract_data(config, trigger_data)
            elif action_type == "save_to_database":
                return await self._action_save_to_database(config, trigger_data)
            elif action_type == "send_notification":
                return await self._action_send_notification(config, trigger_data)
            elif action_type == "send_email":
                return await self._action_send_email(config, trigger_data)
            elif action_type == "call_api":
                return await self._action_call_api(config, trigger_data)
            elif action_type == "generate_report":
                return await self._action_generate_report(config, trigger_data)
            elif action_type == "collect_data":
                return await self._action_collect_data(config, trigger_data)
            elif action_type == "transform_data":
                return await self._action_transform_data(config, trigger_data)
            elif action_type == "send_data":
                return await self._action_send_data(config, trigger_data)
            else:
                raise ValueError(f"Tipo de acción no soportado: {action_type}")
                
        except Exception as e:
            logger.error(f"Error ejecutando acción {action_type}: {e}")
            raise
    
    # Implementaciones de acciones
    async def _action_process_document(self, config: Dict[str, Any], trigger_data: Dict[str, Any]) -> Dict[str, Any]:
        """Acción: procesar documento"""
        try:
            file_path = trigger_data.get("file_path")
            if not file_path:
                raise ValueError("file_path no encontrado en trigger_data")
            
            # Importar servicios necesarios
            from services.document_processor import DocumentProcessor
            from models.document_models import DocumentProcessingRequest, ProfessionalFormat
            
            processor = DocumentProcessor()
            await processor.initialize()
            
            request = DocumentProcessingRequest(
                filename=Path(file_path).name,
                target_format=ProfessionalFormat(config.get("target_format", "consultancy")),
                language=config.get("language", "es"),
                include_analysis=config.get("include_analysis", True)
            )
            
            result = await processor.process_document(file_path, request)
            
            return {
                "action": "process_document",
                "file_path": file_path,
                "result": result,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error procesando documento: {e}")
            raise
    
    async def _action_move_file(self, config: Dict[str, Any], trigger_data: Dict[str, Any]) -> Dict[str, Any]:
        """Acción: mover archivo"""
        try:
            source_path = trigger_data.get("file_path")
            destination = config.get("destination")
            
            if not source_path or not destination:
                raise ValueError("source_path y destination son requeridos")
            
            # Crear directorio destino si no existe
            dest_dir = Path(destination)
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            # Mover archivo
            dest_path = dest_dir / Path(source_path).name
            os.rename(source_path, dest_path)
            
            return {
                "action": "move_file",
                "source": source_path,
                "destination": str(dest_path),
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error moviendo archivo: {e}")
            raise
    
    async def _action_copy_file(self, config: Dict[str, Any], trigger_data: Dict[str, Any]) -> Dict[str, Any]:
        """Acción: copiar archivo"""
        try:
            source_path = trigger_data.get("file_path")
            destination = config.get("destination")
            
            if not source_path or not destination:
                raise ValueError("source_path y destination son requeridos")
            
            # Crear directorio destino si no existe
            dest_dir = Path(destination)
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            # Copiar archivo
            dest_path = dest_dir / Path(source_path).name
            import shutil
            shutil.copy2(source_path, dest_path)
            
            return {
                "action": "copy_file",
                "source": source_path,
                "destination": str(dest_path),
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error copiando archivo: {e}")
            raise
    
    async def _action_delete_file(self, config: Dict[str, Any], trigger_data: Dict[str, Any]) -> Dict[str, Any]:
        """Acción: eliminar archivo"""
        try:
            file_path = trigger_data.get("file_path")
            
            if not file_path:
                raise ValueError("file_path no encontrado en trigger_data")
            
            # Eliminar archivo
            os.remove(file_path)
            
            return {
                "action": "delete_file",
                "file_path": file_path,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error eliminando archivo: {e}")
            raise
    
    async def _action_extract_data(self, config: Dict[str, Any], trigger_data: Dict[str, Any]) -> Dict[str, Any]:
        """Acción: extraer datos"""
        try:
            file_path = trigger_data.get("file_path")
            extraction_rules = config.get("extraction_rules", [])
            
            if not file_path:
                raise ValueError("file_path no encontrado en trigger_data")
            
            # Leer archivo
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extraer datos según reglas
            extracted_data = {}
            for rule in extraction_rules:
                field = rule.get("field")
                pattern = rule.get("pattern")
                rule_type = rule.get("type", "regex")
                
                if rule_type == "regex":
                    import re
                    match = re.search(pattern, content)
                    if match:
                        extracted_data[field] = match.group(1) if match.groups() else match.group(0)
            
            return {
                "action": "extract_data",
                "file_path": file_path,
                "extracted_data": extracted_data,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error extrayendo datos: {e}")
            raise
    
    async def _action_save_to_database(self, config: Dict[str, Any], trigger_data: Dict[str, Any]) -> Dict[str, Any]:
        """Acción: guardar en base de datos"""
        try:
            table = config.get("table")
            data = trigger_data.get("extracted_data", {})
            
            if not table:
                raise ValueError("table es requerido")
            
            # En implementación real, usar ORM o conexión a BD
            # Por ahora, simular guardado
            logger.info(f"Guardando datos en tabla {table}: {data}")
            
            return {
                "action": "save_to_database",
                "table": table,
                "data": data,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error guardando en base de datos: {e}")
            raise
    
    async def _action_send_notification(self, config: Dict[str, Any], trigger_data: Dict[str, Any]) -> Dict[str, Any]:
        """Acción: enviar notificación"""
        try:
            message = config.get("message", "Automatización completada")
            
            # Importar servicio de notificaciones
            from services.notification_service import NotificationService, NotificationType, NotificationPriority
            
            notification_service = NotificationService()
            await notification_service.initialize()
            
            await notification_service.send_notification(
                title="Automatización AI",
                content=message,
                priority=NotificationPriority.MEDIUM,
                notification_type=NotificationType.CONSOLE
            )
            
            return {
                "action": "send_notification",
                "message": message,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error enviando notificación: {e}")
            raise
    
    async def _action_send_email(self, config: Dict[str, Any], trigger_data: Dict[str, Any]) -> Dict[str, Any]:
        """Acción: enviar email"""
        try:
            recipients = config.get("recipients", [])
            subject = config.get("subject", "Automatización AI")
            body = config.get("body", "Proceso automatizado completado")
            
            if not recipients:
                raise ValueError("recipients es requerido")
            
            # En implementación real, usar servicio de email
            logger.info(f"Enviando email a {recipients}: {subject}")
            
            return {
                "action": "send_email",
                "recipients": recipients,
                "subject": subject,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error enviando email: {e}")
            raise
    
    async def _action_call_api(self, config: Dict[str, Any], trigger_data: Dict[str, Any]) -> Dict[str, Any]:
        """Acción: llamar API"""
        try:
            url = config.get("url")
            method = config.get("method", "GET")
            headers = config.get("headers", {})
            data = config.get("data")
            
            if not url:
                raise ValueError("url es requerido")
            
            # En implementación real, usar aiohttp
            logger.info(f"Llamando API {method} {url}")
            
            return {
                "action": "call_api",
                "url": url,
                "method": method,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error llamando API: {e}")
            raise
    
    async def _action_generate_report(self, config: Dict[str, Any], trigger_data: Dict[str, Any]) -> Dict[str, Any]:
        """Acción: generar reporte"""
        try:
            template = config.get("template")
            format_type = config.get("format", "pdf")
            data = trigger_data.get("collected_data", {})
            
            if not template:
                raise ValueError("template es requerido")
            
            # En implementación real, usar generador de reportes
            logger.info(f"Generando reporte {template} en formato {format_type}")
            
            return {
                "action": "generate_report",
                "template": template,
                "format": format_type,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error generando reporte: {e}")
            raise
    
    async def _action_collect_data(self, config: Dict[str, Any], trigger_data: Dict[str, Any]) -> Dict[str, Any]:
        """Acción: recopilar datos"""
        try:
            time_range = config.get("time_range", "7d")
            metrics = config.get("metrics", [])
            
            # En implementación real, recopilar datos de servicios
            collected_data = {
                "time_range": time_range,
                "metrics": metrics,
                "data": {}  # Datos recopilados
            }
            
            return {
                "action": "collect_data",
                "collected_data": collected_data,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error recopilando datos: {e}")
            raise
    
    async def _action_transform_data(self, config: Dict[str, Any], trigger_data: Dict[str, Any]) -> Dict[str, Any]:
        """Acción: transformar datos"""
        try:
            mapping_rules = config.get("mapping_rules", [])
            source_data = trigger_data.get("fetched_data", {})
            
            # Aplicar reglas de mapeo
            transformed_data = {}
            for rule in mapping_rules:
                source_field = rule.get("source_field")
                target_field = rule.get("target_field")
                
                if source_field in source_data:
                    transformed_data[target_field] = source_data[source_field]
            
            return {
                "action": "transform_data",
                "transformed_data": transformed_data,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error transformando datos: {e}")
            raise
    
    async def _action_send_data(self, config: Dict[str, Any], trigger_data: Dict[str, Any]) -> Dict[str, Any]:
        """Acción: enviar datos"""
        try:
            endpoint = config.get("endpoint")
            method = config.get("method", "POST")
            data = trigger_data.get("transformed_data", {})
            
            if not endpoint:
                raise ValueError("endpoint es requerido")
            
            # En implementación real, enviar datos
            logger.info(f"Enviando datos a {endpoint}")
            
            return {
                "action": "send_data",
                "endpoint": endpoint,
                "method": method,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error enviando datos: {e}")
            raise
    
    async def _scheduler_worker(self):
        """Worker para automatizaciones programadas"""
        while True:
            try:
                await asyncio.sleep(60)  # Verificar cada minuto
                
                # Verificar reglas programadas
                for rule in self.automation_rules.values():
                    if (rule.trigger_type == TriggerType.SCHEDULED and 
                        rule.status == AutomationStatus.ACTIVE):
                        
                        # En implementación real, usar croniter para verificar horario
                        # Por ahora, simular ejecución cada 5 minutos
                        if (rule.last_executed is None or 
                            datetime.now() - rule.last_executed > timedelta(minutes=5)):
                            
                            await self.execute_automation(rule.id)
                
            except Exception as e:
                logger.error(f"Error en scheduler worker: {e}")
    
    async def _file_watcher_worker(self):
        """Worker para monitoreo de archivos"""
        while True:
            try:
                await asyncio.sleep(10)  # Verificar cada 10 segundos
                
                # Verificar archivos en rutas monitoreadas
                for rule_id, watcher in self.file_watchers.items():
                    rule = watcher["rule"]
                    watch_path = watcher["path"]
                    patterns = watcher["patterns"]
                    
                    if rule.status == AutomationStatus.ACTIVE:
                        # En implementación real, usar watchdog
                        # Por ahora, simular detección de archivos
                        pass
                
            except Exception as e:
                logger.error(f"Error en file watcher worker: {e}")
    
    async def _scheduled_automation_worker(self, rule: AutomationRule):
        """Worker para automatización programada específica"""
        while rule.status == AutomationStatus.ACTIVE:
            try:
                # En implementación real, usar croniter
                await asyncio.sleep(300)  # Esperar 5 minutos
                
                if rule.status == AutomationStatus.ACTIVE:
                    await self.execute_automation(rule.id)
                
            except Exception as e:
                logger.error(f"Error en worker de automatización {rule.id}: {e}")
                await asyncio.sleep(60)  # Esperar antes de reintentar
    
    async def _cleanup_execution(self, execution_id: str):
        """Limpia ejecución después de un tiempo"""
        try:
            await asyncio.sleep(3600)  # Esperar 1 hora
            
            if execution_id in self.running_automations:
                del self.running_automations[execution_id]
            
            if execution_id in self.automation_executions:
                del self.automation_executions[execution_id]
            
            logger.info(f"Ejecución limpiada: {execution_id}")
            
        except Exception as e:
            logger.error(f"Error limpiando ejecución: {e}")
    
    async def _save_automation_rules(self):
        """Guarda reglas de automatización"""
        try:
            # Crear directorio de datos
            data_dir = Path("data")
            data_dir.mkdir(exist_ok=True)
            
            # Convertir a formato serializable
            rules_data = []
            for rule in self.automation_rules.values():
                rules_data.append({
                    "id": rule.id,
                    "name": rule.name,
                    "description": rule.description,
                    "automation_type": rule.automation_type.value,
                    "trigger_type": rule.trigger_type.value,
                    "trigger_config": rule.trigger_config,
                    "actions": rule.actions,
                    "conditions": rule.conditions,
                    "status": rule.status.value,
                    "created_at": rule.created_at.isoformat(),
                    "updated_at": rule.updated_at.isoformat(),
                    "last_executed": rule.last_executed.isoformat() if rule.last_executed else None,
                    "execution_count": rule.execution_count,
                    "success_count": rule.success_count,
                    "error_count": rule.error_count
                })
            
            # Guardar archivo
            rules_file = data_dir / "automation_rules.json"
            with open(rules_file, 'w', encoding='utf-8') as f:
                json.dump(rules_data, f, indent=2, ensure_ascii=False, default=str)
            
        except Exception as e:
            logger.error(f"Error guardando reglas de automatización: {e}")
    
    async def get_automation_status(self, rule_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene estado de automatización"""
        try:
            if rule_id not in self.automation_rules:
                return None
            
            rule = self.automation_rules[rule_id]
            
            return {
                "rule_id": rule.id,
                "name": rule.name,
                "status": rule.status.value,
                "automation_type": rule.automation_type.value,
                "trigger_type": rule.trigger_type.value,
                "execution_count": rule.execution_count,
                "success_count": rule.success_count,
                "error_count": rule.error_count,
                "success_rate": (rule.success_count / rule.execution_count * 100) if rule.execution_count > 0 else 0,
                "last_executed": rule.last_executed.isoformat() if rule.last_executed else None,
                "created_at": rule.created_at.isoformat(),
                "updated_at": rule.updated_at.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo estado de automatización: {e}")
            return None
    
    async def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene estado de ejecución"""
        try:
            if execution_id not in self.automation_executions:
                return None
            
            execution = self.automation_executions[execution_id]
            
            return {
                "execution_id": execution.id,
                "rule_id": execution.rule_id,
                "status": execution.status.value,
                "started_at": execution.started_at.isoformat(),
                "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
                "duration_seconds": (
                    (execution.completed_at - execution.started_at).total_seconds()
                    if execution.completed_at else None
                ),
                "execution_log": execution.execution_log,
                "error_message": execution.error_message
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo estado de ejecución: {e}")
            return None
    
    async def get_automation_templates(self) -> List[Dict[str, Any]]:
        """Obtiene plantillas de automatización"""
        try:
            return [
                {
                    "id": template.id,
                    "name": template.name,
                    "description": template.description,
                    "category": template.category,
                    "parameters": template.parameters,
                    "examples": template.examples
                }
                for template in self.automation_templates.values()
            ]
        except Exception as e:
            logger.error(f"Error obteniendo plantillas de automatización: {e}")
            return []
    
    async def get_automation_dashboard_data(self) -> Dict[str, Any]:
        """Obtiene datos para dashboard de automatización"""
        try:
            # Estadísticas generales
            total_rules = len(self.automation_rules)
            active_rules = len([r for r in self.automation_rules.values() if r.status == AutomationStatus.ACTIVE])
            
            # Estadísticas de ejecución
            total_executions = sum(r.execution_count for r in self.automation_rules.values())
            total_successes = sum(r.success_count for r in self.automation_rules.values())
            total_errors = sum(r.error_count for r in self.automation_rules.values())
            
            # Distribución por tipo
            type_distribution = {}
            for rule in self.automation_rules.values():
                rule_type = rule.automation_type.value
                type_distribution[rule_type] = type_distribution.get(rule_type, 0) + 1
            
            # Distribución por trigger
            trigger_distribution = {}
            for rule in self.automation_rules.values():
                trigger_type = rule.trigger_type.value
                trigger_distribution[trigger_type] = trigger_distribution.get(trigger_type, 0) + 1
            
            # Ejecuciones recientes
            recent_executions = [
                {
                    "execution_id": exec_id,
                    "rule_id": execution.rule_id,
                    "status": execution.status.value,
                    "started_at": execution.started_at.isoformat(),
                    "duration_seconds": (
                        (execution.completed_at - execution.started_at).total_seconds()
                        if execution.completed_at else None
                    )
                }
                for exec_id, execution in list(self.automation_executions.items())[-10:]
            ]
            
            return {
                "total_rules": total_rules,
                "active_rules": active_rules,
                "inactive_rules": total_rules - active_rules,
                "total_executions": total_executions,
                "total_successes": total_successes,
                "total_errors": total_errors,
                "success_rate": (total_successes / total_executions * 100) if total_executions > 0 else 0,
                "type_distribution": type_distribution,
                "trigger_distribution": trigger_distribution,
                "recent_executions": recent_executions,
                "running_automations": len(self.running_automations),
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo datos del dashboard de automatización: {e}")
            return {"error": str(e)}
    
    async def pause_automation(self, rule_id: str) -> bool:
        """Pausa automatización"""
        try:
            if rule_id not in self.automation_rules:
                return False
            
            rule = self.automation_rules[rule_id]
            rule.status = AutomationStatus.PAUSED
            rule.updated_at = datetime.now()
            
            # Cancelar tareas programadas
            if rule_id in self.scheduled_tasks:
                self.scheduled_tasks[rule_id].cancel()
                del self.scheduled_tasks[rule_id]
            
            await self._save_automation_rules()
            
            logger.info(f"Automatización pausada: {rule_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error pausando automatización: {e}")
            return False
    
    async def resume_automation(self, rule_id: str) -> bool:
        """Reanuda automatización"""
        try:
            if rule_id not in self.automation_rules:
                return False
            
            rule = self.automation_rules[rule_id]
            rule.status = AutomationStatus.ACTIVE
            rule.updated_at = datetime.now()
            
            # Reconfigurar trigger
            await self._setup_trigger(rule)
            
            await self._save_automation_rules()
            
            logger.info(f"Automatización reanudada: {rule_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error reanudando automatización: {e}")
            return False
    
    async def delete_automation(self, rule_id: str) -> bool:
        """Elimina automatización"""
        try:
            if rule_id not in self.automation_rules:
                return False
            
            # Cancelar tareas
            if rule_id in self.scheduled_tasks:
                self.scheduled_tasks[rule_id].cancel()
                del self.scheduled_tasks[rule_id]
            
            # Eliminar file watcher
            if rule_id in self.file_watchers:
                del self.file_watchers[rule_id]
            
            # Eliminar regla
            del self.automation_rules[rule_id]
            
            await self._save_automation_rules()
            
            logger.info(f"Automatización eliminada: {rule_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error eliminando automatización: {e}")
            return False

