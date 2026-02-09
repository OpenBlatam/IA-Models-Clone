"""
Reporting domain services
"""

from services.domains import register_service

try:
    from services.report_service import ReportService
    from services.advanced_reporting_service import AdvancedReportingService
    from services.certificate_service import CertificateService
    
    def register_services():
        register_service("reporting", "report", ReportService)
        register_service("reporting", "advanced", AdvancedReportingService)
        register_service("reporting", "certificate", CertificateService)
except ImportError:
    pass



