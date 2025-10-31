"""
Enterprise Compliance Engine
============================

Advanced compliance checking for legal, regulatory, and industry standards.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from uuid import UUID, uuid4
import re
import json

from ..schemas import CopywritingRequest, CopywritingVariant
from ..exceptions import ComplianceError, ValidationError

logger = logging.getLogger(__name__)


class ComplianceType(str, Enum):
    """Types of compliance checks"""
    LEGAL = "legal"
    REGULATORY = "regulatory"
    INDUSTRY = "industry"
    BRAND = "brand"
    ACCESSIBILITY = "accessibility"
    PRIVACY = "privacy"
    SECURITY = "security"


class ComplianceLevel(str, Enum):
    """Compliance severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class IndustryType(str, Enum):
    """Industry types for compliance"""
    HEALTHCARE = "healthcare"
    FINANCE = "finance"
    EDUCATION = "education"
    TECHNOLOGY = "technology"
    RETAIL = "retail"
    MANUFACTURING = "manufacturing"
    REAL_ESTATE = "real_estate"
    LEGAL = "legal"
    INSURANCE = "insurance"
    AUTOMOTIVE = "automotive"


@dataclass
class ComplianceRule:
    """Individual compliance rule"""
    id: UUID
    name: str
    description: str
    compliance_type: ComplianceType
    industry: Optional[IndustryType] = None
    level: ComplianceLevel = ComplianceLevel.WARNING
    pattern: Optional[str] = None  # Regex pattern
    keywords: List[str] = None
    required_elements: List[str] = None
    forbidden_elements: List[str] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    is_active: bool = True
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.keywords is None:
            self.keywords = []
        if self.required_elements is None:
            self.required_elements = []
        if self.forbidden_elements is None:
            self.forbidden_elements = []


@dataclass
class ComplianceViolation:
    """Compliance violation record"""
    id: UUID
    rule_id: UUID
    content_id: UUID
    violation_type: ComplianceType
    level: ComplianceLevel
    description: str
    suggested_fix: str
    line_number: Optional[int] = None
    word_position: Optional[int] = None
    detected_at: datetime = None
    
    def __post_init__(self):
        if self.detected_at is None:
            self.detected_at = datetime.utcnow()


@dataclass
class ComplianceReport:
    """Comprehensive compliance report"""
    content_id: UUID
    industry: IndustryType
    overall_compliance_score: float  # 0-1
    violations: List[ComplianceViolation]
    recommendations: List[str]
    compliance_summary: Dict[str, Any]
    generated_at: datetime = None
    
    def __post_init__(self):
        if self.generated_at is None:
            self.generated_at = datetime.utcnow()


class ComplianceEngine:
    """Enterprise compliance checking engine"""
    
    def __init__(self):
        self.compliance_rules: Dict[UUID, ComplianceRule] = {}
        self.violations: Dict[UUID, List[ComplianceViolation]] = {}
        self.reports: Dict[UUID, ComplianceReport] = {}
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """Initialize default compliance rules"""
        
        # Healthcare Compliance Rules
        healthcare_rules = [
            ComplianceRule(
                id=uuid4(),
                name="HIPAA Compliance",
                description="Ensure HIPAA compliance in healthcare content",
                compliance_type=ComplianceType.LEGAL,
                industry=IndustryType.HEALTHCARE,
                level=ComplianceLevel.CRITICAL,
                required_elements=["privacy policy", "HIPAA compliance"],
                forbidden_elements=["patient names", "medical records", "specific diagnoses"]
            ),
            ComplianceRule(
                id=uuid4(),
                name="Medical Claims",
                description="Avoid unsubstantiated medical claims",
                compliance_type=ComplianceType.REGULATORY,
                industry=IndustryType.HEALTHCARE,
                level=ComplianceLevel.ERROR,
                forbidden_elements=["cure", "guarantee", "miracle", "100% effective"],
                keywords=["cure", "guarantee", "miracle", "instant", "permanent"]
            ),
            ComplianceRule(
                id=uuid4(),
                name="FDA Disclaimer",
                description="Include FDA disclaimer for health products",
                compliance_type=ComplianceType.REGULATORY,
                industry=IndustryType.HEALTHCARE,
                level=ComplianceLevel.ERROR,
                required_elements=["FDA disclaimer", "not evaluated by FDA"]
            )
        ]
        
        for rule in healthcare_rules:
            self.compliance_rules[rule.id] = rule
        
        # Financial Services Compliance Rules
        finance_rules = [
            ComplianceRule(
                id=uuid4(),
                name="SEC Compliance",
                description="Ensure SEC compliance for financial content",
                compliance_type=ComplianceType.LEGAL,
                industry=IndustryType.FINANCE,
                level=ComplianceLevel.CRITICAL,
                required_elements=["SEC compliance", "investment disclaimer"],
                forbidden_elements=["guaranteed returns", "risk-free investment"]
            ),
            ComplianceRule(
                id=uuid4(),
                name="Investment Disclaimers",
                description="Include proper investment disclaimers",
                compliance_type=ComplianceType.REGULATORY,
                industry=IndustryType.FINANCE,
                level=ComplianceLevel.ERROR,
                required_elements=["past performance", "investment risks", "no guarantee"],
                keywords=["investment", "returns", "portfolio", "trading"]
            ),
            ComplianceRule(
                id=uuid4(),
                name="Anti-Money Laundering",
                description="Comply with AML regulations",
                compliance_type=ComplianceType.REGULATORY,
                industry=IndustryType.FINANCE,
                level=ComplianceLevel.ERROR,
                required_elements=["AML compliance", "KYC procedures"]
            )
        ]
        
        for rule in finance_rules:
            self.compliance_rules[rule.id] = rule
        
        # General Legal Compliance Rules
        general_rules = [
            ComplianceRule(
                id=uuid4(),
                name="Copyright Compliance",
                description="Ensure copyright compliance",
                compliance_type=ComplianceType.LEGAL,
                level=ComplianceLevel.ERROR,
                forbidden_elements=["copyrighted material", "trademark infringement"]
            ),
            ComplianceRule(
                id=uuid4(),
                name="Privacy Policy",
                description="Include privacy policy reference",
                compliance_type=ComplianceType.PRIVACY,
                level=ComplianceLevel.WARNING,
                required_elements=["privacy policy", "data protection"]
            ),
            ComplianceRule(
                id=uuid4(),
                name="Accessibility Compliance",
                description="Ensure accessibility compliance",
                compliance_type=ComplianceType.ACCESSIBILITY,
                level=ComplianceLevel.WARNING,
                required_elements=["alt text", "accessible content"]
            ),
            ComplianceRule(
                id=uuid4(),
                name="GDPR Compliance",
                description="Ensure GDPR compliance for EU content",
                compliance_type=ComplianceType.PRIVACY,
                level=ComplianceLevel.ERROR,
                required_elements=["GDPR compliance", "data processing", "consent"],
                keywords=["personal data", "EU", "European", "GDPR"]
            ),
            ComplianceRule(
                id=uuid4(),
                name="CCPA Compliance",
                description="Ensure CCPA compliance for California content",
                compliance_type=ComplianceType.PRIVACY,
                level=ComplianceLevel.ERROR,
                required_elements=["CCPA compliance", "California privacy rights"],
                keywords=["California", "personal information", "CCPA"]
            )
        ]
        
        for rule in general_rules:
            self.compliance_rules[rule.id] = rule
        
        # Technology Industry Rules
        tech_rules = [
            ComplianceRule(
                id=uuid4(),
                name="Software Licensing",
                description="Comply with software licensing requirements",
                compliance_type=ComplianceType.LEGAL,
                industry=IndustryType.TECHNOLOGY,
                level=ComplianceLevel.WARNING,
                required_elements=["software license", "terms of service"]
            ),
            ComplianceRule(
                id=uuid4(),
                name="Data Security",
                description="Ensure data security compliance",
                compliance_type=ComplianceType.SECURITY,
                industry=IndustryType.TECHNOLOGY,
                level=ComplianceLevel.ERROR,
                required_elements=["data encryption", "security measures"]
            )
        ]
        
        for rule in tech_rules:
            self.compliance_rules[rule.id] = rule
    
    async def check_compliance(
        self,
        content: str,
        industry: IndustryType,
        content_id: Optional[UUID] = None,
        additional_rules: Optional[List[UUID]] = None
    ) -> ComplianceReport:
        """Check content compliance against all applicable rules"""
        
        violations = []
        applicable_rules = []
        
        # Get rules applicable to the industry
        for rule in self.compliance_rules.values():
            if rule.is_active and (rule.industry is None or rule.industry == industry):
                applicable_rules.append(rule)
        
        # Add additional rules if specified
        if additional_rules:
            for rule_id in additional_rules:
                if rule_id in self.compliance_rules:
                    applicable_rules.append(self.compliance_rules[rule_id])
        
        # Check each rule
        for rule in applicable_rules:
            rule_violations = await self._check_rule_compliance(content, rule, content_id)
            violations.extend(rule_violations)
        
        # Generate recommendations
        recommendations = await self._generate_compliance_recommendations(violations, industry)
        
        # Calculate compliance score
        total_rules = len(applicable_rules)
        critical_violations = len([v for v in violations if v.level == ComplianceLevel.CRITICAL])
        error_violations = len([v for v in violations if v.level == ComplianceLevel.ERROR])
        warning_violations = len([v for v in violations if v.level == ComplianceLevel.WARNING])
        
        # Calculate score (penalize critical and error violations heavily)
        penalty = (critical_violations * 0.5) + (error_violations * 0.3) + (warning_violations * 0.1)
        compliance_score = max(0, 1 - (penalty / total_rules)) if total_rules > 0 else 1
        
        # Create compliance summary
        compliance_summary = {
            "total_rules_checked": total_rules,
            "violations_by_level": {
                "critical": critical_violations,
                "error": error_violations,
                "warning": warning_violations,
                "info": len([v for v in violations if v.level == ComplianceLevel.INFO])
            },
            "violations_by_type": self._group_violations_by_type(violations),
            "compliance_percentage": compliance_score * 100
        }
        
        report = ComplianceReport(
            content_id=content_id or uuid4(),
            industry=industry,
            overall_compliance_score=compliance_score,
            violations=violations,
            recommendations=recommendations,
            compliance_summary=compliance_summary
        )
        
        # Store violations and report
        if content_id:
            self.violations[content_id] = violations
            self.reports[content_id] = report
        
        logger.info(f"Compliance check completed for {industry.value}: {compliance_score:.2f} score")
        
        return report
    
    async def _check_rule_compliance(
        self,
        content: str,
        rule: ComplianceRule,
        content_id: Optional[UUID]
    ) -> List[ComplianceViolation]:
        """Check compliance against a specific rule"""
        violations = []
        content_lower = content.lower()
        
        # Check required elements
        for element in rule.required_elements:
            if element.lower() not in content_lower:
                violations.append(ComplianceViolation(
                    id=uuid4(),
                    rule_id=rule.id,
                    content_id=content_id or uuid4(),
                    violation_type=rule.compliance_type,
                    level=rule.level,
                    description=f"Required element '{element}' not found",
                    suggested_fix=f"Include '{element}' in the content"
                ))
        
        # Check forbidden elements
        for element in rule.forbidden_elements:
            if element.lower() in content_lower:
                violations.append(ComplianceViolation(
                    id=uuid4(),
                    rule_id=rule.id,
                    content_id=content_id or uuid4(),
                    violation_type=rule.compliance_type,
                    level=rule.level,
                    description=f"Forbidden element '{element}' found",
                    suggested_fix=f"Remove '{element}' from the content"
                ))
        
        # Check keywords
        for keyword in rule.keywords:
            if keyword.lower() in content_lower:
                violations.append(ComplianceViolation(
                    id=uuid4(),
                    rule_id=rule.id,
                    content_id=content_id or uuid4(),
                    violation_type=rule.compliance_type,
                    level=rule.level,
                    description=f"Restricted keyword '{keyword}' found",
                    suggested_fix=f"Review usage of '{keyword}' for compliance"
                ))
        
        # Check regex pattern
        if rule.pattern:
            if re.search(rule.pattern, content, re.IGNORECASE):
                violations.append(ComplianceViolation(
                    id=uuid4(),
                    rule_id=rule.id,
                    content_id=content_id or uuid4(),
                    violation_type=rule.compliance_type,
                    level=rule.level,
                    description=f"Content matches restricted pattern: {rule.pattern}",
                    suggested_fix="Review content for compliance with pattern restrictions"
                ))
        
        # Check length constraints
        if rule.min_length and len(content) < rule.min_length:
            violations.append(ComplianceViolation(
                id=uuid4(),
                rule_id=rule.id,
                content_id=content_id or uuid4(),
                violation_type=rule.compliance_type,
                level=rule.level,
                description=f"Content too short (minimum {rule.min_length} characters)",
                suggested_fix=f"Expand content to at least {rule.min_length} characters"
            ))
        
        if rule.max_length and len(content) > rule.max_length:
            violations.append(ComplianceViolation(
                id=uuid4(),
                rule_id=rule.id,
                content_id=content_id or uuid4(),
                violation_type=rule.compliance_type,
                level=rule.level,
                description=f"Content too long (maximum {rule.max_length} characters)",
                suggested_fix=f"Reduce content to maximum {rule.max_length} characters"
            ))
        
        return violations
    
    async def _generate_compliance_recommendations(
        self,
        violations: List[ComplianceViolation],
        industry: IndustryType
    ) -> List[str]:
        """Generate compliance recommendations"""
        recommendations = []
        
        # Critical violations
        critical_violations = [v for v in violations if v.level == ComplianceLevel.CRITICAL]
        if critical_violations:
            recommendations.append("Address critical compliance violations immediately before publishing")
        
        # Error violations
        error_violations = [v for v in violations if v.level == ComplianceLevel.ERROR]
        if error_violations:
            recommendations.append("Fix error-level compliance issues before content goes live")
        
        # Industry-specific recommendations
        if industry == IndustryType.HEALTHCARE:
            recommendations.append("Ensure all health claims are substantiated and include proper disclaimers")
            recommendations.append("Include HIPAA compliance statements where applicable")
        elif industry == IndustryType.FINANCE:
            recommendations.append("Include investment disclaimers and risk warnings")
            recommendations.append("Ensure SEC compliance for all financial content")
        elif industry == IndustryType.TECHNOLOGY:
            recommendations.append("Include software licensing and terms of service information")
            recommendations.append("Ensure data security and privacy compliance")
        
        # General recommendations
        if not violations:
            recommendations.append("Content appears to be compliant with applicable regulations")
        else:
            recommendations.append("Review all flagged issues and implement suggested fixes")
        
        return recommendations
    
    def _group_violations_by_type(self, violations: List[ComplianceViolation]) -> Dict[str, int]:
        """Group violations by compliance type"""
        type_counts = {}
        for violation in violations:
            violation_type = violation.violation_type.value
            type_counts[violation_type] = type_counts.get(violation_type, 0) + 1
        return type_counts
    
    async def create_custom_rule(
        self,
        name: str,
        description: str,
        compliance_type: ComplianceType,
        level: ComplianceLevel,
        industry: Optional[IndustryType] = None,
        **kwargs
    ) -> ComplianceRule:
        """Create a custom compliance rule"""
        
        rule = ComplianceRule(
            id=uuid4(),
            name=name,
            description=description,
            compliance_type=compliance_type,
            industry=industry,
            level=level,
            pattern=kwargs.get('pattern'),
            keywords=kwargs.get('keywords', []),
            required_elements=kwargs.get('required_elements', []),
            forbidden_elements=kwargs.get('forbidden_elements', []),
            min_length=kwargs.get('min_length'),
            max_length=kwargs.get('max_length')
        )
        
        self.compliance_rules[rule.id] = rule
        
        logger.info(f"Created custom compliance rule: {name}")
        
        return rule
    
    async def get_compliance_analytics(self) -> Dict[str, Any]:
        """Get compliance analytics and metrics"""
        
        total_reports = len(self.reports)
        if total_reports == 0:
            return {"message": "No compliance reports available"}
        
        # Calculate average compliance scores
        avg_compliance_score = sum(report.overall_compliance_score for report in self.reports.values()) / total_reports
        
        # Count violations by level and type
        level_counts = {"critical": 0, "error": 0, "warning": 0, "info": 0}
        type_counts = {}
        industry_counts = {}
        
        for report in self.reports.values():
            industry_counts[report.industry.value] = industry_counts.get(report.industry.value, 0) + 1
            
            for violation in report.violations:
                level_counts[violation.level.value] += 1
                violation_type = violation.violation_type.value
                type_counts[violation_type] = type_counts.get(violation_type, 0) + 1
        
        # Most common violations
        most_common_violations = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Industry compliance scores
        industry_scores = {}
        for industry in IndustryType:
            industry_reports = [r for r in self.reports.values() if r.industry == industry]
            if industry_reports:
                industry_scores[industry.value] = sum(r.overall_compliance_score for r in industry_reports) / len(industry_reports)
        
        return {
            "summary": {
                "total_reports": total_reports,
                "average_compliance_score": avg_compliance_score,
                "total_violations": sum(len(v) for v in self.violations.values()),
                "active_rules": len([r for r in self.compliance_rules.values() if r.is_active])
            },
            "violation_analysis": {
                "by_level": level_counts,
                "by_type": type_counts,
                "most_common": most_common_violations
            },
            "industry_analysis": {
                "reports_by_industry": industry_counts,
                "compliance_scores_by_industry": industry_scores
            },
            "rules_summary": {
                "total_rules": len(self.compliance_rules),
                "rules_by_type": {
                    rule_type.value: len([r for r in self.compliance_rules.values() if r.compliance_type == rule_type])
                    for rule_type in ComplianceType
                },
                "rules_by_industry": {
                    industry.value: len([r for r in self.compliance_rules.values() if r.industry == industry])
                    for industry in IndustryType
                }
            }
        }
    
    async def get_industry_compliance_requirements(self, industry: IndustryType) -> List[ComplianceRule]:
        """Get compliance requirements for a specific industry"""
        
        industry_rules = []
        general_rules = []
        
        for rule in self.compliance_rules.values():
            if rule.is_active:
                if rule.industry == industry:
                    industry_rules.append(rule)
                elif rule.industry is None:  # General rules
                    general_rules.append(rule)
        
        return industry_rules + general_rules


# Global compliance engine instance
compliance_engine = ComplianceEngine()






























