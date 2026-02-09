"""
Financial domain services
"""

from services.domains import register_service

try:
    from services.financial_tracking_service import FinancialTrackingService
    
    def register_services():
        register_service("financial", "tracking", FinancialTrackingService)
except ImportError:
    pass



