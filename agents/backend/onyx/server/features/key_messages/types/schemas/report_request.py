from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from enum import Enum
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Report request API schemas for cybersecurity tools.
"""

class ReportType(str, Enum):
    """Types of security reports."""
    VULNERABILITY_REPORT = "vulnerability_report"
    PENETRATION_TEST_REPORT = "penetration_test_report"
    INCIDENT_REPORT = "incident_report"
    THREAT_INTELLIGENCE_REPORT = "threat_intelligence_report"
    COMPLIANCE_REPORT = "compliance_report"
    AUDIT_REPORT = "audit_report"
    FORENSIC_REPORT = "forensic_report"

class ReportStatus(str, Enum):
    """Report status."""
    DRAFT = "draft"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    PUBLISHED = "published"
    ARCHIVED = "archived"

class ReportSeverity(str, Enum):
    """Report severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

# Request Schemas
class CreateReportRequest(BaseModel):
    """Request schema for creating a report."""
    title: str = Field(..., min_length=1, max_length=200, description="Report title")
    report_type: ReportType = Field(..., description="Type of report")
    executive_summary: str = Field(..., min_length=1, description="Executive summary")
    scope: str = Field(..., description="Assessment scope")
    methodology: str = Field(..., description="Assessment methodology")
    objectives: List[str] = Field(default_factory=list, description="Assessment objectives")
    limitations: List[str] = Field(default_factory=list, description="Assessment limitations")
    severity: ReportSeverity = Field(default=ReportSeverity.MEDIUM, description="Overall report severity")
    assessment_start_date: Optional[datetime] = Field(None, description="Assessment start date")
    assessment_end_date: Optional[datetime] = Field(None, description="Assessment end date")
    report_due_date: Optional[datetime] = Field(None, description="Report due date")
    tools_used: List[str] = Field(default_factory=list, description="Tools used in assessment")
    techniques_employed: List[str] = Field(default_factory=list, description="Techniques employed")
    standards_followed: List[str] = Field(default_factory=list, description="Standards followed")
    compliance_frameworks: List[str] = Field(default_factory=list, description="Compliance frameworks")
    assets_assessed: List[str] = Field(default_factory=list, description="Assets assessed")
    systems_tested: List[str] = Field(default_factory=list, description="Systems tested")
    networks_scanned: List[str] = Field(default_factory=list, description="Networks scanned")
    applications_tested: List[str] = Field(default_factory=list, description="Applications tested")
    scan_range: Optional[str] = Field(None, description="Port range scanned")
    distribution_list: List[str] = Field(default_factory=list, description="Distribution list")
    access_level: str = Field(default="confidential", description="Report access level")
    classification: Optional[str] = Field(None, description="Report classification")
    tags: List[str] = Field(default_factory=list, description="Report tags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @field_validator('title')
    def validate_title(cls, v) -> bool:
        if not v.strip():
            raise ValueError("Title cannot be empty or whitespace only")
        return v.strip()
    
    @field_validator('executive_summary')
    def validate_executive_summary(cls, v) -> bool:
        if not v.strip():
            raise ValueError("Executive summary cannot be empty or whitespace only")
        return v.strip()
    
    @field_validator('scope')
    def validate_scope(cls, v) -> bool:
        if not v.strip():
            raise ValueError("Scope cannot be empty")
        return v.strip()
    
    @field_validator('methodology')
    def validate_methodology(cls, v) -> bool:
        if not v.strip():
            raise ValueError("Methodology cannot be empty")
        return v.strip()

class UpdateReportRequest(BaseModel):
    """Request schema for updating a report."""
    title: Optional[str] = Field(None, min_length=1, max_length=200, description="Report title")
    executive_summary: Optional[str] = Field(None, min_length=1, description="Executive summary")
    scope: Optional[str] = Field(None, description="Assessment scope")
    methodology: Optional[str] = Field(None, description="Assessment methodology")
    objectives: Optional[List[str]] = Field(None, description="Assessment objectives")
    limitations: Optional[List[str]] = Field(None, description="Assessment limitations")
    severity: Optional[ReportSeverity] = Field(None, description="Overall report severity")
    status: Optional[ReportStatus] = Field(None, description="Report status")
    assessment_end_date: Optional[datetime] = Field(None, description="Assessment end date")
    report_due_date: Optional[datetime] = Field(None, description="Report due date")
    tools_used: Optional[List[str]] = Field(None, description="Tools used in assessment")
    techniques_employed: Optional[List[str]] = Field(None, description="Techniques employed")
    standards_followed: Optional[List[str]] = Field(None, description="Standards followed")
    compliance_frameworks: Optional[List[str]] = Field(None, description="Compliance frameworks")
    assets_assessed: Optional[List[str]] = Field(None, description="Assets assessed")
    systems_tested: Optional[List[str]] = Field(None, description="Systems tested")
    networks_scanned: Optional[List[str]] = Field(None, description="Networks scanned")
    applications_tested: Optional[List[str]] = Field(None, description="Applications tested")
    distribution_list: Optional[List[str]] = Field(None, description="Distribution list")
    access_level: Optional[str] = Field(None, description="Report access level")
    classification: Optional[str] = Field(None, description="Report classification")
    tags: Optional[List[str]] = Field(None, description="Report tags")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class ReportFilterRequest(BaseModel):
    """Request schema for filtering reports."""
    report_type: Optional[ReportType] = Field(None, description="Filter by report type")
    status: Optional[ReportStatus] = Field(None, description="Filter by status")
    severity: Optional[ReportSeverity] = Field(None, description="Filter by severity")
    author: Optional[str] = Field(None, description="Filter by author")
    created_after: Optional[datetime] = Field(None, description="Filter by creation date (after)")
    created_before: Optional[datetime] = Field(None, description="Filter by creation date (before)")
    published_after: Optional[datetime] = Field(None, description="Filter by publication date (after)")
    published_before: Optional[datetime] = Field(None, description="Filter by publication date (before)")
    assigned_to: Optional[str] = Field(None, description="Filter by assignee")
    compliance_frameworks: Optional[List[str]] = Field(None, description="Filter by compliance frameworks")
    search: Optional[str] = Field(None, description="Search in title and content")
    limit: Optional[int] = Field(None, ge=1, le=1000, description="Maximum number of results")
    offset: Optional[int] = Field(None, ge=0, description="Number of results to skip")
    sort_by: Optional[str] = Field(None, description="Sort field")
    sort_order: Optional[str] = Field(None, description="Sort order (asc/desc)")

# Response Schemas
class FindingInfoResponse(BaseModel):
    """Response schema for finding information."""
    id: str = Field(..., description="Finding identifier")
    title: str = Field(..., description="Finding title")
    description: str = Field(..., description="Finding description")
    severity: ReportSeverity = Field(..., description="Finding severity")
    cvss_score: Optional[float] = Field(None, description="CVSS score")
    affected_assets: List[str] = Field(..., description="Affected assets")
    remediation: str = Field(..., description="Remediation steps")
    references: List[str] = Field(..., description="Reference links")
    evidence: List[str] = Field(..., description="Evidence files")
    status: str = Field(..., description="Finding status")

class RecommendationInfoResponse(BaseModel):
    """Response schema for recommendation information."""
    id: str = Field(..., description="Recommendation identifier")
    title: str = Field(..., description="Recommendation title")
    description: str = Field(..., description="Recommendation description")
    priority: str = Field(..., description="Recommendation priority")
    effort: str = Field(..., description="Implementation effort")
    cost: Optional[str] = Field(None, description="Estimated cost")
    timeline: Optional[str] = Field(None, description="Implementation timeline")
    responsible_party: Optional[str] = Field(None, description="Responsible party")

class ReportResponse(BaseModel):
    """Response schema for report data."""
    id: str = Field(..., description="Unique report identifier")
    title: str = Field(..., description="Report title")
    report_type: ReportType = Field(..., description="Type of report")
    executive_summary: str = Field(..., description="Executive summary")
    status: ReportStatus = Field(..., description="Report status")
    severity: ReportSeverity = Field(..., description="Overall report severity")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    published_at: Optional[datetime] = Field(None, description="Publication timestamp")
    author: str = Field(..., description="Report author")
    reviewers: List[str] = Field(..., description="Report reviewers")
    approver: Optional[str] = Field(None, description="Report approver")
    scope: str = Field(..., description="Assessment scope")
    methodology: str = Field(..., description="Assessment methodology")
    objectives: List[str] = Field(..., description="Assessment objectives")
    limitations: List[str] = Field(..., description="Assessment limitations")
    findings: List[FindingInfoResponse] = Field(..., description="Security findings")
    recommendations: List[RecommendationInfoResponse] = Field(..., description="Security recommendations")
    total_findings: int = Field(..., description="Total number of findings")
    critical_findings: int = Field(..., description="Number of critical findings")
    high_findings: int = Field(..., description="Number of high findings")
    medium_findings: int = Field(..., description="Number of medium findings")
    low_findings: int = Field(..., description="Number of low findings")
    info_findings: int = Field(..., description="Number of info findings")
    overall_risk_score: Optional[float] = Field(None, description="Overall risk score")
    risk_level: Optional[str] = Field(None, description="Overall risk level")
    risk_factors: List[str] = Field(..., description="Key risk factors")
    compliance_frameworks: List[str] = Field(..., description="Compliance frameworks")
    compliance_status: Dict[str, str] = Field(..., description="Compliance status by framework")
    gaps_identified: List[str] = Field(..., description="Compliance gaps identified")
    assets_assessed: List[str] = Field(..., description="Assets assessed")
    systems_tested: List[str] = Field(..., description="Systems tested")
    networks_scanned: List[str] = Field(..., description="Networks scanned")
    applications_tested: List[str] = Field(..., description="Applications tested")
    assessment_start_date: Optional[datetime] = Field(None, description="Assessment start date")
    assessment_end_date: Optional[datetime] = Field(None, description="Assessment end date")
    report_due_date: Optional[datetime] = Field(None, description="Report due date")
    tools_used: List[str] = Field(..., description="Tools used in assessment")
    techniques_employed: List[str] = Field(..., description="Techniques employed")
    standards_followed: List[str] = Field(..., description="Standards followed")
    evidence_files: List[str] = Field(..., description="Evidence files")
    screenshots: List[str] = Field(..., description="Screenshot files")
    logs: List[str] = Field(..., description="Log files")
    raw_data: List[str] = Field(..., description="Raw data files")
    distribution_list: List[str] = Field(..., description="Distribution list")
    access_level: str = Field(..., description="Report access level")
    classification: Optional[str] = Field(None, description="Report classification")
    version: str = Field(..., description="Report version")
    revision_history: List[Dict[str, Any]] = Field(..., description="Revision history")
    metadata: Dict[str, Any] = Field(..., description="Additional metadata")
    tags: List[str] = Field(..., description="Report tags")
    notes: Optional[str] = Field(None, description="Additional notes")
    is_published: bool = Field(..., description="Whether report is published")
    is_approved: bool = Field(..., description="Whether report is approved")
    risk_level_calculated: str = Field(..., description="Calculated risk level")
    summary_stats: Dict[str, Any] = Field(..., description="Summary statistics")

class ReportListResponse(BaseModel):
    """Response schema for report list."""
    reports: List[ReportResponse] = Field(..., description="List of reports")
    total_count: int = Field(..., description="Total number of reports")
    filtered_count: int = Field(..., description="Number of reports after filtering")
    page: Optional[int] = Field(None, description="Current page number")
    page_size: Optional[int] = Field(None, description="Page size")
    has_next: bool = Field(..., description="Whether there are more results")
    has_previous: bool = Field(..., description="Whether there are previous results")

class ReportStatsResponse(BaseModel):
    """Response schema for report statistics."""
    total_reports: int = Field(..., description="Total number of reports")
    draft_reports: int = Field(..., description="Number of draft reports")
    in_review_reports: int = Field(..., description="Number of reports in review")
    approved_reports: int = Field(..., description="Number of approved reports")
    published_reports: int = Field(..., description="Number of published reports")
    archived_reports: int = Field(..., description="Number of archived reports")
    critical_reports: int = Field(..., description="Number of critical reports")
    high_reports: int = Field(..., description="Number of high severity reports")
    medium_reports: int = Field(..., description="Number of medium severity reports")
    low_reports: int = Field(..., description="Number of low severity reports")
    info_reports: int = Field(..., description="Number of info reports")
    reports_by_type: Dict[str, int] = Field(..., description="Reports count by type")
    reports_by_status: Dict[str, int] = Field(..., description="Reports count by status")
    reports_by_severity: Dict[str, int] = Field(..., description="Reports count by severity")
    total_findings: int = Field(..., description="Total findings across all reports")
    total_recommendations: int = Field(..., description="Total recommendations across all reports")
    recent_reports: List[ReportResponse] = Field(..., description="Recent reports")

class ReportCreateResponse(BaseModel):
    """Response schema for report creation."""
    id: str = Field(..., description="Created report identifier")
    message: str = Field(..., description="Success message")
    status: ReportStatus = Field(..., description="Initial report status")
    version: str = Field(..., description="Report version")
    created_at: datetime = Field(..., description="Creation timestamp")

class ReportUpdateResponse(BaseModel):
    """Response schema for report update."""
    id: str = Field(..., description="Updated report identifier")
    message: str = Field(..., description="Success message")
    updated_at: datetime = Field(..., description="Update timestamp")
    changes: Dict[str, Any] = Field(..., description="Changes made")
    new_version: str = Field(..., description="New report version")

class ReportPublishResponse(BaseModel):
    """Response schema for report publication."""
    id: str = Field(..., description="Published report identifier")
    message: str = Field(..., description="Success message")
    published_at: datetime = Field(..., description="Publication timestamp")
    download_url: Optional[str] = Field(None, description="Download URL for published report")

# Error Schemas
class ReportErrorResponse(BaseModel):
    """Error response schema for report operations."""
    error: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Error code")
    details: Optional[Dict[str, Any]] = Field(None, description="Error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")

# Bulk Operation Schemas
class BulkReportRequest(BaseModel):
    """Request schema for bulk report operations."""
    operation: str = Field(..., description="Bulk operation type")
    report_ids: List[str] = Field(..., description="List of report IDs")
    updates: Optional[Dict[str, Any]] = Field(None, description="Updates to apply")

class BulkReportResponse(BaseModel):
    """Response schema for bulk report operations."""
    operation: str = Field(..., description="Bulk operation type")
    total_reports: int = Field(..., description="Total number of reports")
    successful_operations: int = Field(..., description="Number of successful operations")
    failed_operations: int = Field(..., description="Number of failed operations")
    errors: List[Dict[str, Any]] = Field(default_factory=list, description="Operation errors")
    completed_at: datetime = Field(..., description="Completion timestamp")

# Export Schemas
class ReportExportRequest(BaseModel):
    """Request schema for report export."""
    format: str = Field(..., description="Export format (pdf, docx, html, json)")
    filters: Optional[ReportFilterRequest] = Field(None, description="Export filters")
    include_findings: bool = Field(default=True, description="Include findings in export")
    include_recommendations: bool = Field(default=True, description="Include recommendations in export")
    include_evidence: bool = Field(default=False, description="Include evidence files in export")
    template: Optional[str] = Field(None, description="Export template to use")

class ReportExportResponse(BaseModel):
    """Response schema for report export."""
    download_url: str = Field(..., description="Download URL for exported file")
    file_size: int = Field(..., description="File size in bytes")
    format: str = Field(..., description="Export format")
    expires_at: datetime = Field(..., description="Download link expiration")
    report_count: int = Field(..., description="Number of reports exported")
    template_used: Optional[str] = Field(None, description="Template used for export") 