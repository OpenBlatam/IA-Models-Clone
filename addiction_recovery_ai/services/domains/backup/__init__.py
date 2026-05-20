"""
Backup domain services
"""

from services.domains import register_service

try:
    from services.backup_service import BackupService
    
    def register_services():
        register_service("backup", "backup", BackupService)
except ImportError:
    pass



