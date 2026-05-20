"""
Service Factory
===============

Factory para crear servicios con dependency injection.
"""

import logging
from typing import Dict, Any, Optional
from ..config.config_loader import get_config
from ..services import (
    DatabaseService,
    NotificationService,
    AnalyticsService,
    BackupService,
    TemplateService,
    ReportingService,
    WebhookService,
    SyncService,
    SearchService,
    AlertService,
    ExportService
)

logger = logging.getLogger(__name__)


class ServiceFactory:
    """Factory de servicios."""
    
    def __init__(self, config: Optional[Any] = None):
        """
        Inicializar factory.
        
        Args:
            config: Configuración (opcional)
        """
        self.config = config or get_config()
        self._services: Dict[str, Any] = {}
        self._logger = logger
    
    def get_database_service(self) -> DatabaseService:
        """Obtener servicio de base de datos."""
        if "database" not in self._services:
            self._services["database"] = DatabaseService(
                db_path=self.config.database.path
            )
        return self._services["database"]
    
    def get_notification_service(self) -> NotificationService:
        """Obtener servicio de notificaciones."""
        if "notification" not in self._services:
            self._services["notification"] = NotificationService()
        return self._services["notification"]
    
    def get_analytics_service(self) -> AnalyticsService:
        """Obtener servicio de analytics."""
        if "analytics" not in self._services:
            self._services["analytics"] = AnalyticsService()
        return self._services["analytics"]
    
    def get_backup_service(self) -> BackupService:
        """Obtener servicio de backup."""
        if "backup" not in self._services:
            self._services["backup"] = BackupService()
        return self._services["backup"]
    
    def get_template_service(self) -> TemplateService:
        """Obtener servicio de plantillas."""
        if "template" not in self._services:
            self._services["template"] = TemplateService()
        return self._services["template"]
    
    def get_reporting_service(self) -> ReportingService:
        """Obtener servicio de reportes."""
        if "reporting" not in self._services:
            self._services["reporting"] = ReportingService()
        return self._services["reporting"]
    
    def get_webhook_service(self) -> WebhookService:
        """Obtener servicio de webhooks."""
        if "webhook" not in self._services:
            self._services["webhook"] = WebhookService()
        return self._services["webhook"]
    
    def get_sync_service(self) -> SyncService:
        """Obtener servicio de sincronización."""
        if "sync" not in self._services:
            self._services["sync"] = SyncService()
        return self._services["sync"]
    
    def get_search_service(self) -> SearchService:
        """Obtener servicio de búsqueda."""
        if "search" not in self._services:
            self._services["search"] = SearchService()
        return self._services["search"]
    
    def get_alert_service(self) -> AlertService:
        """Obtener servicio de alertas."""
        if "alert" not in self._services:
            self._services["alert"] = AlertService()
        return self._services["alert"]
    
    def get_export_service(self) -> ExportService:
        """Obtener servicio de exportación."""
        if "export" not in self._services:
            self._services["export"] = ExportService()
        return self._services["export"]
    
    def get_all_services(self) -> Dict[str, Any]:
        """Obtener todos los servicios."""
        return self._services.copy()
    
    def reset(self):
        """Resetear factory."""
        self._services.clear()




