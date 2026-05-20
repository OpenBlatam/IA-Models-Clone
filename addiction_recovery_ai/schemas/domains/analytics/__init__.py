"""
Analytics domain schemas
"""

from schemas.domains import register_schema

try:
    from schemas.analytics import (
        AnalyticsResponse,
        GenerateReportRequest,
        ReportResponse,
        InsightsResponse
    )
    
    def register_schemas():
        register_schema("analytics", "AnalyticsResponse", AnalyticsResponse)
        register_schema("analytics", "GenerateReportRequest", GenerateReportRequest)
        register_schema("analytics", "ReportResponse", ReportResponse)
        register_schema("analytics", "InsightsResponse", InsightsResponse)
except ImportError:
    pass



