from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from enum import Enum
        import re
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Report data model for cybersecurity tools.
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

class FindingInfo(BaseModel):
    """Security finding information."""
    id: str = Field(..., description="Finding identifier")
    title: str = Field(..., description="Finding title")
    description: str = Field(..., description="Finding description")
    severity: ReportSeverity = Field(..., description="Finding severity")
    cvss_score: Optional[float] = Field(None, ge=0.0, le=10.0, description="CVSS score")
    affected_assets: List[str] = Field(default_factory=list, description="Affected assets")
    remediation: str = Field(..., description="Remediation steps")
    references: List[str] = Field(default_factory=list, description="Reference links")
    evidence: List[str] = Field(default_factory=list, description="Evidence files")
    status: str = Field(default="open", description="Finding status")

class RecommendationInfo(BaseModel):
    """Security recommendation information."""
    id: str = Field(..., description="Recommendation identifier")
    title: str = Field(..., description="Recommendation title")
    description: str = Field(..., description="Recommendation description")
    priority: str = Field(..., description="Recommendation priority")
    effort: str = Field(..., description="Implementation effort")
    cost: Optional[str] = Field(None, description="Estimated cost")
    timeline: Optional[str] = Field(None, description="Implementation timeline")
    responsible_party: Optional[str] = Field(None, description="Responsible party")

class ReportModel(BaseModel):
    """Report data model."""
    
    # Core fields
    id: str = Field(..., description="Unique report identifier")
    title: str = Field(..., description="Report title")
    report_type: ReportType = Field(..., description="Type of report")
    executive_summary: str = Field(..., description="Executive summary")
    
    # Status and metadata
    status: ReportStatus = Field(default=ReportStatus.DRAFT, description="Report status")
    severity: ReportSeverity = Field(default=ReportSeverity.MEDIUM, description="Overall report severity")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    published_at: Optional[datetime] = Field(None, description="Publication timestamp")
    
    # Authorship
    author: str = Field(..., description="Report author")
    reviewers: List[str] = Field(default_factory=list, description="Report reviewers")
    approver: Optional[str] = Field(None, description="Report approver")
    
    # Scope and context
    scope: str = Field(..., description="Assessment scope")
    methodology: str = Field(..., description="Assessment methodology")
    objectives: List[str] = Field(default_factory=list, description="Assessment objectives")
    limitations: List[str] = Field(default_factory=list, description="Assessment limitations")
    
    # Findings and recommendations
    findings: List[FindingInfo] = Field(default_factory=list, description="Security findings")
    recommendations: List[RecommendationInfo] = Field(default_factory=list, description="Security recommendations")
    
    # Statistics
    total_findings: int = Field(default=0, description="Total number of findings")
    critical_findings: int = Field(default=0, description="Number of critical findings")
    high_findings: int = Field(default=0, description="Number of high findings")
    medium_findings: int = Field(default=0, description="Number of medium findings")
    low_findings: int = Field(default=0, description="Number of low findings")
    info_findings: int = Field(default=0, description="Number of info findings")
    
    # Risk assessment
    overall_risk_score: Optional[float] = Field(None, ge=0.0, le=10.0, description="Overall risk score")
    risk_level: Optional[str] = Field(None, description="Overall risk level")
    risk_factors: List[str] = Field(default_factory=list, description="Key risk factors")
    
    # Compliance
    compliance_frameworks: List[str] = Field(default_factory=list, description="Compliance frameworks")
    compliance_status: Dict[str, str] = Field(default_factory=dict, description="Compliance status by framework")
    gaps_identified: List[str] = Field(default_factory=list, description="Compliance gaps identified")
    
    # Assets and targets
    assets_assessed: List[str] = Field(default_factory=list, description="Assets assessed")
    systems_tested: List[str] = Field(default_factory=list, description="Systems tested")
    networks_scanned: List[str] = Field(default_factory=list, description="Networks scanned")
    applications_tested: List[str] = Field(default_factory=list, description="Applications tested")
    
    # Timeline
    assessment_start_date: Optional[datetime] = Field(None, description="Assessment start date")
    assessment_end_date: Optional[datetime] = Field(None, description="Assessment end date")
    report_due_date: Optional[datetime] = Field(None, description="Report due date")
    
    # Tools and techniques
    tools_used: List[str] = Field(default_factory=list, description="Tools used in assessment")
    techniques_employed: List[str] = Field(default_factory=list, description="Techniques employed")
    standards_followed: List[str] = Field(default_factory=list, description="Standards followed")
    
    # Evidence and artifacts
    evidence_files: List[str] = Field(default_factory=list, description="Evidence files")
    screenshots: List[str] = Field(default_factory=list, description="Screenshot files")
    logs: List[str] = Field(default_factory=list, description="Log files")
    raw_data: List[str] = Field(default_factory=list, description="Raw data files")
    
    # Distribution and access
    distribution_list: List[str] = Field(default_factory=list, description="Distribution list")
    access_level: str = Field(default="confidential", description="Report access level")
    classification: Optional[str] = Field(None, description="Report classification")
    
    # Version control
    version: str = Field(default="1.0", description="Report version")
    revision_history: List[Dict[str, Any]] = Field(default_factory=list, description="Revision history")
    
    # Custom fields
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    tags: List[str] = Field(default_factory=list, description="Report tags")
    
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
    
    @field_validator('author')
    def validate_author(cls, v) -> bool:
        if not v.strip():
            raise ValueError("Author cannot be empty")
        return v.strip()
    
    @field_validator('version')
    def validate_version(cls, v) -> bool:
        version_pattern = r'^\d+\.\d+(\.\d+)?$'
        if not re.match(version_pattern, v):
            raise ValueError("Invalid version format. Use format: X.Y or X.Y.Z")
        return v
    
    def is_published(self) -> bool:
        """Check if report is published."""
        return self.status == ReportStatus.PUBLISHED
    
    def is_approved(self) -> bool:
        """Check if report is approved."""
        return self.status in [ReportStatus.APPROVED, ReportStatus.PUBLISHED]
    
    def get_risk_level(self) -> str:
        """Get risk level based on overall risk score."""
        if self.overall_risk_score is None:
            return "unknown"
        elif self.overall_risk_score >= 8.0:
            return "critical"
        elif self.overall_risk_score >= 6.0:
            return "high"
        elif self.overall_risk_score >= 4.0:
            return "medium"
        elif self.overall_risk_score >= 2.0:
            return "low"
        else:
            return "minimal"
    
    def calculate_risk_score(self) -> float:
        """Calculate overall risk score based on findings."""
        score = 0.0
        
        # Weight findings by severity
        score += self.critical_findings * 10.0
        score += self.high_findings * 7.0
        score += self.medium_findings * 4.0
        score += self.low_findings * 1.0
        score += self.info_findings * 0.5
        
        # Normalize to 0-10 scale
        return min(score / 10.0, 10.0)
    
    def add_finding(self, finding: FindingInfo) -> None:
        """Add a finding to the report."""
        self.findings.append(finding)
        self.total_findings += 1
        
        # Update severity counts
        if finding.severity == ReportSeverity.CRITICAL:
            self.critical_findings += 1
        elif finding.severity == ReportSeverity.HIGH:
            self.high_findings += 1
        elif finding.severity == ReportSeverity.MEDIUM:
            self.medium_findings += 1
        elif finding.severity == ReportSeverity.LOW:
            self.low_findings += 1
        elif finding.severity == ReportSeverity.INFO:
            self.info_findings += 1
    
    def add_recommendation(self, recommendation: RecommendationInfo) -> None:
        """Add a recommendation to the report."""
        self.recommendations.append(recommendation)
    
    def update_status(self, status: ReportStatus, approver: Optional[str] = None) -> None:
        """Update report status."""
        self.status = status
        self.updated_at = datetime.utcnow()
        
        if status == ReportStatus.PUBLISHED:
            self.published_at = datetime.utcnow()
        
        if approver:
            self.approver = approver
    
    def add_reviewer(self, reviewer: str) -> None:
        """Add a reviewer to the report."""
        if reviewer not in self.reviewers:
            self.reviewers.append(reviewer)
    
    def add_revision(self, version: str, changes: str, author: str) -> None:
        """Add a revision to the report."""
        revision = {
            "version": version,
            "changes": changes,
            "author": author,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.revision_history.append(revision)
        self.version = version
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for the report."""
        return {
            "total_findings": self.total_findings,
            "critical_findings": self.critical_findings,
            "high_findings": self.high_findings,
            "medium_findings": self.medium_findings,
            "low_findings": self.low_findings,
            "info_findings": self.info_findings,
            "total_recommendations": len(self.recommendations),
            "risk_score": self.overall_risk_score,
            "risk_level": self.get_risk_level(),
            "compliance_frameworks": len(self.compliance_frameworks),
            "assets_assessed": len(self.assets_assessed)
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReportModel':
        """Create model from dictionary."""
        return cls(**data)
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        validate_assignment = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        } 