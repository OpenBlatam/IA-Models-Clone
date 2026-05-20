"""
Pydantic schemas for analytics and reporting endpoints
"""

from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from datetime import datetime


class AnalyticsResponse(BaseModel):
    """Response schema for user analytics"""
    user_id: str = Field(..., description="User ID")
    progress_summary: Dict = Field(default_factory=dict, description="Progress summary")
    statistics: Dict = Field(default_factory=dict, description="Detailed statistics")
    trends: Dict = Field(default_factory=dict, description="Progress trends")
    generated_at: datetime = Field(default_factory=datetime.now, description="Generation timestamp")


class GenerateReportRequest(BaseModel):
    """Request schema for generating reports"""
    user_id: str = Field(..., description="User ID")
    report_type: str = Field(default="summary", description="Type of report")
    include_charts: bool = Field(default=True, description="Include charts in report")
    date_range: Optional[Dict] = Field(default=None, description="Date range for report")


class ReportResponse(BaseModel):
    """Response schema for generated report"""
    user_id: str = Field(..., description="User ID")
    report_id: str = Field(..., description="Report ID")
    report_type: str = Field(..., description="Type of report")
    generated_at: datetime = Field(default_factory=datetime.now, description="Generation timestamp")
    progress_summary: Dict = Field(default_factory=dict, description="Progress summary")
    detailed_statistics: Dict = Field(default_factory=dict, description="Detailed statistics")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations")
    charts: Optional[List[Dict]] = Field(default=None, description="Chart data")


class InsightsResponse(BaseModel):
    """Response schema for user insights"""
    user_id: str = Field(..., description="User ID")
    key_insights: List[str] = Field(default_factory=list, description="Key insights")
    trends: Dict = Field(default_factory=dict, description="Progress trends")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations")
    generated_at: datetime = Field(default_factory=datetime.now, description="Generation timestamp")

