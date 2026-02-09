"""
Dashboard domain services
"""

from services.domains import register_service

try:
    from services.dashboard_service import DashboardService
    
    def register_services():
        register_service("dashboard", "dashboard", DashboardService)
except ImportError:
    pass



