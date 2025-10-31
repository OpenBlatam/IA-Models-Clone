"""
Application Commands - Comandos de Aplicación
===========================================

Comandos CQRS que representan intenciones de cambio de estado
en el sistema.
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime
import uuid

from ..domain.value_objects import ContentId, ModelType


@dataclass
class Command:
    """
    Comando base.
    
    Todos los comandos deben heredar de esta clase.
    """
    
    command_id: str
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if not self.command_id:
            self.command_id = str(uuid.uuid4())


@dataclass
class CreateHistoryCommand(Command):
    """
    Comando para crear una nueva entrada de historial.
    """
    
    model_type: ModelType
    content: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    assess_quality: bool = False
    analyze_content: bool = True


@dataclass
class UpdateHistoryCommand(Command):
    """
    Comando para actualizar una entrada de historial existente.
    """
    
    entry_id: ContentId
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    assess_quality: bool = False
    analyze_content: bool = False


@dataclass
class DeleteHistoryCommand(Command):
    """
    Comando para eliminar una entrada de historial.
    """
    
    entry_id: ContentId
    reason: Optional[str] = None


@dataclass
class CompareEntriesCommand(Command):
    """
    Comando para comparar dos entradas de historial.
    """
    
    entry_1_id: ContentId
    entry_2_id: ContentId
    include_differences: bool = True
    analysis_options: Optional[Dict[str, Any]] = None


@dataclass
class AssessQualityCommand(Command):
    """
    Comando para evaluar la calidad de una entrada.
    """
    
    entry_id: ContentId
    include_recommendations: bool = True
    detailed_analysis: bool = False
    assessor_version: Optional[str] = None


@dataclass
class StartAnalysisCommand(Command):
    """
    Comando para iniciar un análisis en lote.
    """
    
    analysis_type: str
    user_id: Optional[str] = None
    model_type: Optional[ModelType] = None
    filters: Optional[Dict[str, Any]] = None
    options: Optional[Dict[str, Any]] = None


@dataclass
class BulkAnalyzeCommand(Command):
    """
    Comando para análisis masivo de entradas.
    """
    
    entry_ids: List[ContentId]
    analysis_types: List[str]
    options: Optional[Dict[str, Any]] = None


@dataclass
class ExportDataCommand(Command):
    """
    Comando para exportar datos del sistema.
    """
    
    export_type: str
    format: str  # json, csv, xlsx
    filters: Optional[Dict[str, Any]] = None
    include_metadata: bool = True


@dataclass
class ImportDataCommand(Command):
    """
    Comando para importar datos al sistema.
    """
    
    import_type: str
    data: List[Dict[str, Any]]
    validate_data: bool = True
    overwrite_existing: bool = False


@dataclass
class SystemMaintenanceCommand(Command):
    """
    Comando para operaciones de mantenimiento del sistema.
    """
    
    operation: str  # cleanup, backup, optimize, migrate
    parameters: Optional[Dict[str, Any]] = None
    dry_run: bool = False


@dataclass
class PluginManagementCommand(Command):
    """
    Comando para gestión de plugins.
    """
    
    action: str  # install, uninstall, enable, disable, update
    plugin_name: str
    plugin_version: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


@dataclass
class ConfigurationUpdateCommand(Command):
    """
    Comando para actualizar configuración del sistema.
    """
    
    configuration_key: str
    configuration_value: Any
    scope: str = "global"  # global, service, user
    target_service: Optional[str] = None


@dataclass
class HealthCheckCommand(Command):
    """
    Comando para verificar la salud del sistema.
    """
    
    service_name: Optional[str] = None
    include_dependencies: bool = True
    detailed_report: bool = False


@dataclass
class MetricsCollectionCommand(Command):
    """
    Comando para recolección de métricas.
    """
    
    metric_types: List[str]
    time_range: Optional[str] = None
    aggregation_level: str = "service"  # service, system, user


@dataclass
class NotificationCommand(Command):
    """
    Comando para enviar notificaciones.
    """
    
    notification_type: str
    recipients: List[str]
    message: str
    priority: str = "normal"  # low, normal, high, urgent
    channels: List[str] = None  # email, sms, push, webhook


@dataclass
class AuditLogCommand(Command):
    """
    Comando para registrar eventos de auditoría.
    """
    
    event_type: str
    user_id: Optional[str] = None
    resource_id: Optional[str] = None
    action: str
    details: Optional[Dict[str, Any]] = None
    severity: str = "info"  # info, warning, error, critical


# Factory para crear comandos
class CommandFactory:
    """
    Factory para crear comandos de aplicación.
    """
    
    @staticmethod
    def create_history_command(
        model_type: ModelType,
        content: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        assess_quality: bool = False,
        analyze_content: bool = True,
        correlation_id: Optional[str] = None,
        causation_id: Optional[str] = None
    ) -> CreateHistoryCommand:
        """Crear comando de creación de historial."""
        return CreateHistoryCommand(
            command_id=str(uuid.uuid4()),
            model_type=model_type,
            content=content,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata,
            assess_quality=assess_quality,
            analyze_content=analyze_content,
            correlation_id=correlation_id,
            causation_id=causation_id
        )
    
    @staticmethod
    def compare_entries_command(
        entry_1_id: ContentId,
        entry_2_id: ContentId,
        include_differences: bool = True,
        analysis_options: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        causation_id: Optional[str] = None
    ) -> CompareEntriesCommand:
        """Crear comando de comparación de entradas."""
        return CompareEntriesCommand(
            command_id=str(uuid.uuid4()),
            entry_1_id=entry_1_id,
            entry_2_id=entry_2_id,
            include_differences=include_differences,
            analysis_options=analysis_options,
            correlation_id=correlation_id,
            causation_id=causation_id
        )
    
    @staticmethod
    def assess_quality_command(
        entry_id: ContentId,
        include_recommendations: bool = True,
        detailed_analysis: bool = False,
        assessor_version: Optional[str] = None,
        correlation_id: Optional[str] = None,
        causation_id: Optional[str] = None
    ) -> AssessQualityCommand:
        """Crear comando de evaluación de calidad."""
        return AssessQualityCommand(
            command_id=str(uuid.uuid4()),
            entry_id=entry_id,
            include_recommendations=include_recommendations,
            detailed_analysis=detailed_analysis,
            assessor_version=assessor_version,
            correlation_id=correlation_id,
            causation_id=causation_id
        )
    
    @staticmethod
    def start_analysis_command(
        analysis_type: str,
        user_id: Optional[str] = None,
        model_type: Optional[ModelType] = None,
        filters: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        causation_id: Optional[str] = None
    ) -> StartAnalysisCommand:
        """Crear comando de inicio de análisis."""
        return StartAnalysisCommand(
            command_id=str(uuid.uuid4()),
            analysis_type=analysis_type,
            user_id=user_id,
            model_type=model_type,
            filters=filters,
            options=options,
            correlation_id=correlation_id,
            causation_id=causation_id
        )




