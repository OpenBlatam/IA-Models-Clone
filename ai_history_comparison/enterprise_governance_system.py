"""
Enterprise Governance System
============================

Advanced enterprise governance system for AI model analysis with
comprehensive governance frameworks, policy management, and compliance monitoring.
"""

import asyncio
import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import hashlib
import yaml
import xml.etree.ElementTree as ET
from pathlib import Path
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class GovernanceLevel(str, Enum):
    """Governance levels"""
    ORGANIZATIONAL = "organizational"
    DEPARTMENTAL = "departmental"
    PROJECT = "project"
    INDIVIDUAL = "individual"


class PolicyType(str, Enum):
    """Policy types"""
    DATA_GOVERNANCE = "data_governance"
    MODEL_GOVERNANCE = "model_governance"
    ETHICS = "ethics"
    SECURITY = "security"
    COMPLIANCE = "compliance"
    QUALITY = "quality"
    ACCESS_CONTROL = "access_control"
    AUDIT = "audit"
    RETENTION = "retention"
    PRIVACY = "privacy"


class ApprovalStatus(str, Enum):
    """Approval status"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    UNDER_REVIEW = "under_review"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class RiskLevel(str, Enum):
    """Risk levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ComplianceFramework(str, Enum):
    """Compliance frameworks"""
    GDPR = "gdpr"
    CCPA = "ccpa"
    HIPAA = "hipaa"
    SOX = "sox"
    PCI_DSS = "pci_dss"
    ISO27001 = "iso27001"
    NIST = "nist"
    COBIT = "cobit"
    ITIL = "itil"
    AGILE = "agile"


@dataclass
class GovernancePolicy:
    """Governance policy definition"""
    policy_id: str
    name: str
    description: str
    policy_type: PolicyType
    governance_level: GovernanceLevel
    framework: ComplianceFramework
    version: str
    effective_date: datetime
    expiry_date: datetime
    owner: str
    approvers: List[str]
    stakeholders: List[str]
    rules: List[Dict[str, Any]]
    enforcement_actions: List[str]
    risk_assessment: Dict[str, Any]
    compliance_requirements: List[str]
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class ApprovalRequest:
    """Approval request"""
    request_id: str
    policy_id: str
    requester: str
    request_type: str
    description: str
    justification: str
    risk_level: RiskLevel
    status: ApprovalStatus
    approvers: List[str]
    approvals: List[Dict[str, Any]]
    comments: List[Dict[str, Any]]
    created_at: datetime
    due_date: datetime
    completed_at: datetime = None
    
    def __post_init__(self):
        if self.completed_at is None:
            self.completed_at = datetime.now()


@dataclass
class ComplianceAssessment:
    """Compliance assessment"""
    assessment_id: str
    policy_id: str
    framework: ComplianceFramework
    assessor: str
    assessment_date: datetime
    compliance_score: float
    findings: List[Dict[str, Any]]
    recommendations: List[str]
    remediation_plan: Dict[str, Any]
    next_assessment: datetime
    status: str = "completed"
    
    def __post_init__(self):
        if self.next_assessment is None:
            self.next_assessment = datetime.now() + timedelta(days=90)


@dataclass
class RiskAssessment:
    """Risk assessment"""
    risk_id: str
    policy_id: str
    risk_type: str
    description: str
    risk_level: RiskLevel
    probability: float
    impact: float
    risk_score: float
    mitigation_measures: List[str]
    residual_risk: RiskLevel
    owner: str
    assessor: str
    assessment_date: datetime
    review_date: datetime
    status: str = "active"


class EnterpriseGovernanceSystem:
    """Advanced enterprise governance system for AI model analysis"""
    
    def __init__(self, max_policies: int = 1000, max_assessments: int = 10000):
        self.max_policies = max_policies
        self.max_assessments = max_assessments
        
        self.governance_policies: Dict[str, GovernancePolicy] = {}
        self.approval_requests: List[ApprovalRequest] = []
        self.compliance_assessments: List[ComplianceAssessment] = []
        self.risk_assessments: List[RiskAssessment] = []
        
        # Governance frameworks
        self.frameworks: Dict[str, Dict[str, Any]] = {}
        
        # Approval workflow
        self.approval_workflows: Dict[str, Dict[str, Any]] = {}
        
        # Risk management
        self.risk_matrix: Dict[str, Dict[str, Any]] = {}
        
        # Initialize governance frameworks
        self._initialize_frameworks()
        self._initialize_approval_workflows()
        self._initialize_risk_matrix()
        
        # Start background tasks
        self._start_background_tasks()
    
    async def create_governance_policy(self, 
                                     name: str,
                                     description: str,
                                     policy_type: PolicyType,
                                     governance_level: GovernanceLevel,
                                     framework: ComplianceFramework,
                                     version: str,
                                     effective_date: datetime,
                                     expiry_date: datetime,
                                     owner: str,
                                     approvers: List[str],
                                     stakeholders: List[str],
                                     rules: List[Dict[str, Any]],
                                     enforcement_actions: List[str],
                                     risk_assessment: Dict[str, Any] = None,
                                     compliance_requirements: List[str] = None) -> GovernancePolicy:
        """Create governance policy"""
        try:
            policy_id = hashlib.md5(f"{name}_{version}_{datetime.now()}".encode()).hexdigest()
            
            if risk_assessment is None:
                risk_assessment = {"level": "medium", "factors": []}
            if compliance_requirements is None:
                compliance_requirements = []
            
            policy = GovernancePolicy(
                policy_id=policy_id,
                name=name,
                description=description,
                policy_type=policy_type,
                governance_level=governance_level,
                framework=framework,
                version=version,
                effective_date=effective_date,
                expiry_date=expiry_date,
                owner=owner,
                approvers=approvers,
                stakeholders=stakeholders,
                rules=rules,
                enforcement_actions=enforcement_actions,
                risk_assessment=risk_assessment,
                compliance_requirements=compliance_requirements
            )
            
            self.governance_policies[policy_id] = policy
            
            logger.info(f"Created governance policy: {name} v{version}")
            
            return policy
            
        except Exception as e:
            logger.error(f"Error creating governance policy: {str(e)}")
            raise e
    
    async def create_approval_request(self, 
                                    policy_id: str,
                                    requester: str,
                                    request_type: str,
                                    description: str,
                                    justification: str,
                                    risk_level: RiskLevel,
                                    approvers: List[str],
                                    due_date: datetime) -> ApprovalRequest:
        """Create approval request"""
        try:
            if policy_id not in self.governance_policies:
                raise ValueError(f"Policy {policy_id} not found")
            
            request_id = hashlib.md5(f"{policy_id}_{requester}_{datetime.now()}".encode()).hexdigest()
            
            request = ApprovalRequest(
                request_id=request_id,
                policy_id=policy_id,
                requester=requester,
                request_type=request_type,
                description=description,
                justification=justification,
                risk_level=risk_level,
                status=ApprovalStatus.PENDING,
                approvers=approvers,
                approvals=[],
                comments=[],
                created_at=datetime.now(),
                due_date=due_date
            )
            
            self.approval_requests.append(request)
            
            logger.info(f"Created approval request: {request_id}")
            
            return request
            
        except Exception as e:
            logger.error(f"Error creating approval request: {str(e)}")
            raise e
    
    async def approve_request(self, 
                            request_id: str,
                            approver: str,
                            decision: bool,
                            comments: str = "") -> bool:
        """Approve or reject request"""
        try:
            request = next((r for r in self.approval_requests if r.request_id == request_id), None)
            if not request:
                raise ValueError(f"Approval request {request_id} not found")
            
            if approver not in request.approvers:
                raise ValueError(f"User {approver} is not authorized to approve this request")
            
            # Add approval
            approval = {
                "approver": approver,
                "decision": decision,
                "comments": comments,
                "timestamp": datetime.now().isoformat()
            }
            request.approvals.append(approval)
            
            # Add comment
            if comments:
                comment = {
                    "user": approver,
                    "comment": comments,
                    "timestamp": datetime.now().isoformat()
                }
                request.comments.append(comment)
            
            # Check if all approvers have responded
            if len(request.approvals) >= len(request.approvers):
                # Determine overall decision
                approved_count = sum(1 for a in request.approvals if a["decision"])
                if approved_count > len(request.approvers) / 2:
                    request.status = ApprovalStatus.APPROVED
                else:
                    request.status = ApprovalStatus.REJECTED
                
                request.completed_at = datetime.now()
                
                logger.info(f"Approval request {request_id} completed: {request.status.value}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error approving request: {str(e)}")
            return False
    
    async def conduct_compliance_assessment(self, 
                                          policy_id: str,
                                          framework: ComplianceFramework,
                                          assessor: str,
                                          findings: List[Dict[str, Any]] = None,
                                          recommendations: List[str] = None) -> ComplianceAssessment:
        """Conduct compliance assessment"""
        try:
            if policy_id not in self.governance_policies:
                raise ValueError(f"Policy {policy_id} not found")
            
            assessment_id = hashlib.md5(f"{policy_id}_{framework}_{datetime.now()}".encode()).hexdigest()
            
            if findings is None:
                findings = []
            if recommendations is None:
                recommendations = []
            
            # Calculate compliance score
            compliance_score = await self._calculate_compliance_score(policy_id, framework, findings)
            
            # Generate remediation plan
            remediation_plan = await self._generate_remediation_plan(findings, recommendations)
            
            assessment = ComplianceAssessment(
                assessment_id=assessment_id,
                policy_id=policy_id,
                framework=framework,
                assessor=assessor,
                assessment_date=datetime.now(),
                compliance_score=compliance_score,
                findings=findings,
                recommendations=recommendations,
                remediation_plan=remediation_plan,
                next_assessment=datetime.now() + timedelta(days=90)
            )
            
            self.compliance_assessments.append(assessment)
            
            logger.info(f"Conducted compliance assessment: {assessment_id}")
            
            return assessment
            
        except Exception as e:
            logger.error(f"Error conducting compliance assessment: {str(e)}")
            raise e
    
    async def conduct_risk_assessment(self, 
                                    policy_id: str,
                                    risk_type: str,
                                    description: str,
                                    probability: float,
                                    impact: float,
                                    mitigation_measures: List[str],
                                    owner: str,
                                    assessor: str) -> RiskAssessment:
        """Conduct risk assessment"""
        try:
            if policy_id not in self.governance_policies:
                raise ValueError(f"Policy {policy_id} not found")
            
            risk_id = hashlib.md5(f"{policy_id}_{risk_type}_{datetime.now()}".encode()).hexdigest()
            
            # Calculate risk score
            risk_score = probability * impact
            
            # Determine risk level
            risk_level = await self._determine_risk_level(risk_score)
            
            # Calculate residual risk
            residual_risk = await self._calculate_residual_risk(risk_level, mitigation_measures)
            
            assessment = RiskAssessment(
                risk_id=risk_id,
                policy_id=policy_id,
                risk_type=risk_type,
                description=description,
                risk_level=risk_level,
                probability=probability,
                impact=impact,
                risk_score=risk_score,
                mitigation_measures=mitigation_measures,
                residual_risk=residual_risk,
                owner=owner,
                assessor=assessor,
                assessment_date=datetime.now(),
                review_date=datetime.now() + timedelta(days=180)
            )
            
            self.risk_assessments.append(assessment)
            
            logger.info(f"Conducted risk assessment: {risk_id}")
            
            return assessment
            
        except Exception as e:
            logger.error(f"Error conducting risk assessment: {str(e)}")
            raise e
    
    async def get_governance_dashboard(self) -> Dict[str, Any]:
        """Get governance dashboard data"""
        try:
            dashboard = {
                "total_policies": len(self.governance_policies),
                "active_policies": len([p for p in self.governance_policies.values() if p.effective_date <= datetime.now() <= p.expiry_date]),
                "expired_policies": len([p for p in self.governance_policies.values() if p.expiry_date < datetime.now()]),
                "pending_approvals": len([r for r in self.approval_requests if r.status == ApprovalStatus.PENDING]),
                "total_assessments": len(self.compliance_assessments),
                "total_risks": len(self.risk_assessments),
                "high_risks": len([r for r in self.risk_assessments if r.risk_level == RiskLevel.HIGH]),
                "critical_risks": len([r for r in self.risk_assessments if r.risk_level == RiskLevel.CRITICAL]),
                "policy_types": {},
                "governance_levels": {},
                "compliance_frameworks": {},
                "risk_distribution": {},
                "approval_trends": {},
                "compliance_trends": {},
                "recent_activities": []
            }
            
            # Analyze policy types
            for policy in self.governance_policies.values():
                policy_type = policy.policy_type.value
                if policy_type not in dashboard["policy_types"]:
                    dashboard["policy_types"][policy_type] = 0
                dashboard["policy_types"][policy_type] += 1
            
            # Analyze governance levels
            for policy in self.governance_policies.values():
                level = policy.governance_level.value
                if level not in dashboard["governance_levels"]:
                    dashboard["governance_levels"][level] = 0
                dashboard["governance_levels"][level] += 1
            
            # Analyze compliance frameworks
            for assessment in self.compliance_assessments:
                framework = assessment.framework.value
                if framework not in dashboard["compliance_frameworks"]:
                    dashboard["compliance_frameworks"][framework] = 0
                dashboard["compliance_frameworks"][framework] += 1
            
            # Analyze risk distribution
            for risk in self.risk_assessments:
                risk_level = risk.risk_level.value
                if risk_level not in dashboard["risk_distribution"]:
                    dashboard["risk_distribution"][risk_level] = 0
                dashboard["risk_distribution"][risk_level] += 1
            
            # Approval trends (daily)
            daily_approvals = defaultdict(int)
            for request in self.approval_requests:
                date_key = request.created_at.date()
                daily_approvals[date_key] += 1
            
            dashboard["approval_trends"] = {
                date.isoformat(): count for date, count in daily_approvals.items()
            }
            
            # Compliance trends (monthly)
            monthly_assessments = defaultdict(int)
            for assessment in self.compliance_assessments:
                month_key = assessment.assessment_date.replace(day=1)
                monthly_assessments[month_key] += 1
            
            dashboard["compliance_trends"] = {
                month.isoformat(): count for month, count in monthly_assessments.items()
            }
            
            # Recent activities
            recent_activities = []
            
            # Recent policies
            recent_policies = sorted(
                self.governance_policies.values(),
                key=lambda p: p.created_at,
                reverse=True
            )[:5]
            
            for policy in recent_policies:
                recent_activities.append({
                    "type": "policy_created",
                    "description": f"Policy '{policy.name}' created",
                    "timestamp": policy.created_at.isoformat(),
                    "user": policy.owner
                })
            
            # Recent approvals
            recent_approvals = sorted(
                self.approval_requests,
                key=lambda r: r.created_at,
                reverse=True
            )[:5]
            
            for request in recent_approvals:
                recent_activities.append({
                    "type": "approval_request",
                    "description": f"Approval request for {request.request_type}",
                    "timestamp": request.created_at.isoformat(),
                    "user": request.requester,
                    "status": request.status.value
                })
            
            # Sort activities by timestamp
            recent_activities.sort(key=lambda a: a["timestamp"], reverse=True)
            dashboard["recent_activities"] = recent_activities[:10]
            
            return dashboard
            
        except Exception as e:
            logger.error(f"Error getting governance dashboard: {str(e)}")
            return {"error": str(e)}
    
    async def get_policy_compliance_report(self, policy_id: str) -> Dict[str, Any]:
        """Get policy compliance report"""
        try:
            if policy_id not in self.governance_policies:
                raise ValueError(f"Policy {policy_id} not found")
            
            policy = self.governance_policies[policy_id]
            
            # Get related assessments
            related_assessments = [a for a in self.compliance_assessments if a.policy_id == policy_id]
            related_risks = [r for r in self.risk_assessments if r.policy_id == policy_id]
            related_approvals = [r for r in self.approval_requests if r.policy_id == policy_id]
            
            report = {
                "policy_id": policy_id,
                "policy_name": policy.name,
                "policy_version": policy.version,
                "policy_type": policy.policy_type.value,
                "governance_level": policy.governance_level.value,
                "framework": policy.framework.value,
                "effective_date": policy.effective_date.isoformat(),
                "expiry_date": policy.expiry_date.isoformat(),
                "owner": policy.owner,
                "status": "active" if policy.effective_date <= datetime.now() <= policy.expiry_date else "inactive",
                "compliance_score": 0.0,
                "risk_score": 0.0,
                "total_assessments": len(related_assessments),
                "total_risks": len(related_risks),
                "total_approvals": len(related_approvals),
                "assessments": [],
                "risks": [],
                "approvals": [],
                "recommendations": []
            }
            
            # Process assessments
            for assessment in related_assessments:
                assessment_data = {
                    "assessment_id": assessment.assessment_id,
                    "framework": assessment.framework.value,
                    "assessor": assessment.assessor,
                    "assessment_date": assessment.assessment_date.isoformat(),
                    "compliance_score": assessment.compliance_score,
                    "findings_count": len(assessment.findings),
                    "recommendations_count": len(assessment.recommendations),
                    "status": assessment.status
                }
                report["assessments"].append(assessment_data)
            
            # Process risks
            for risk in related_risks:
                risk_data = {
                    "risk_id": risk.risk_id,
                    "risk_type": risk.risk_type,
                    "risk_level": risk.risk_level.value,
                    "probability": risk.probability,
                    "impact": risk.impact,
                    "risk_score": risk.risk_score,
                    "residual_risk": risk.residual_risk.value,
                    "owner": risk.owner,
                    "assessment_date": risk.assessment_date.isoformat(),
                    "status": risk.status
                }
                report["risks"].append(risk_data)
            
            # Process approvals
            for approval in related_approvals:
                approval_data = {
                    "request_id": approval.request_id,
                    "requester": approval.requester,
                    "request_type": approval.request_type,
                    "status": approval.status.value,
                    "created_at": approval.created_at.isoformat(),
                    "due_date": approval.due_date.isoformat(),
                    "approvals_count": len(approval.approvals),
                    "comments_count": len(approval.comments)
                }
                report["approvals"].append(approval_data)
            
            # Calculate overall scores
            if related_assessments:
                report["compliance_score"] = sum(a.compliance_score for a in related_assessments) / len(related_assessments)
            
            if related_risks:
                report["risk_score"] = sum(r.risk_score for r in related_risks) / len(related_risks)
            
            # Generate recommendations
            report["recommendations"] = await self._generate_policy_recommendations(policy, related_assessments, related_risks)
            
            return report
            
        except Exception as e:
            logger.error(f"Error getting policy compliance report: {str(e)}")
            return {"error": str(e)}
    
    # Private helper methods
    def _initialize_frameworks(self) -> None:
        """Initialize governance frameworks"""
        try:
            # GDPR framework
            self.frameworks["gdpr"] = {
                "name": "General Data Protection Regulation",
                "description": "EU data protection and privacy regulation",
                "requirements": [
                    "Data minimization",
                    "Purpose limitation",
                    "Storage limitation",
                    "Accuracy",
                    "Security",
                    "Accountability"
                ],
                "assessment_criteria": {
                    "data_protection": 0.3,
                    "consent_management": 0.2,
                    "data_subject_rights": 0.2,
                    "security_measures": 0.2,
                    "documentation": 0.1
                }
            }
            
            # SOC2 framework
            self.frameworks["soc2"] = {
                "name": "SOC 2 Type II",
                "description": "Security, availability, processing integrity, confidentiality, and privacy",
                "requirements": [
                    "Security",
                    "Availability",
                    "Processing integrity",
                    "Confidentiality",
                    "Privacy"
                ],
                "assessment_criteria": {
                    "security": 0.3,
                    "availability": 0.2,
                    "processing_integrity": 0.2,
                    "confidentiality": 0.15,
                    "privacy": 0.15
                }
            }
            
            # ISO 27001 framework
            self.frameworks["iso27001"] = {
                "name": "ISO/IEC 27001",
                "description": "Information security management system",
                "requirements": [
                    "Information security policies",
                    "Organization of information security",
                    "Human resource security",
                    "Asset management",
                    "Access control",
                    "Cryptography",
                    "Physical and environmental security",
                    "Operations security",
                    "Communications security",
                    "System acquisition, development and maintenance",
                    "Supplier relationships",
                    "Information security incident management",
                    "Information security aspects of business continuity management",
                    "Compliance"
                ],
                "assessment_criteria": {
                    "policies": 0.1,
                    "organization": 0.1,
                    "human_resources": 0.1,
                    "asset_management": 0.1,
                    "access_control": 0.1,
                    "cryptography": 0.1,
                    "physical_security": 0.1,
                    "operations": 0.1,
                    "communications": 0.1,
                    "system_development": 0.1
                }
            }
            
            logger.info(f"Initialized {len(self.frameworks)} governance frameworks")
            
        except Exception as e:
            logger.error(f"Error initializing frameworks: {str(e)}")
    
    def _initialize_approval_workflows(self) -> None:
        """Initialize approval workflows"""
        try:
            # Standard approval workflow
            self.approval_workflows["standard"] = {
                "name": "Standard Approval Workflow",
                "description": "Standard approval process for policies",
                "steps": [
                    {
                        "step": 1,
                        "role": "policy_owner",
                        "action": "create_request",
                        "required": True
                    },
                    {
                        "step": 2,
                        "role": "department_head",
                        "action": "review",
                        "required": True
                    },
                    {
                        "step": 3,
                        "role": "compliance_officer",
                        "action": "approve",
                        "required": True
                    },
                    {
                        "step": 4,
                        "role": "legal_counsel",
                        "action": "final_approval",
                        "required": False
                    }
                ],
                "escalation_rules": {
                    "timeout_hours": 72,
                    "escalation_role": "compliance_director"
                }
            }
            
            # High-risk approval workflow
            self.approval_workflows["high_risk"] = {
                "name": "High-Risk Approval Workflow",
                "description": "Enhanced approval process for high-risk policies",
                "steps": [
                    {
                        "step": 1,
                        "role": "policy_owner",
                        "action": "create_request",
                        "required": True
                    },
                    {
                        "step": 2,
                        "role": "risk_manager",
                        "action": "risk_assessment",
                        "required": True
                    },
                    {
                        "step": 3,
                        "role": "department_head",
                        "action": "review",
                        "required": True
                    },
                    {
                        "step": 4,
                        "role": "compliance_officer",
                        "action": "approve",
                        "required": True
                    },
                    {
                        "step": 5,
                        "role": "legal_counsel",
                        "action": "legal_review",
                        "required": True
                    },
                    {
                        "step": 6,
                        "role": "executive_committee",
                        "action": "final_approval",
                        "required": True
                    }
                ],
                "escalation_rules": {
                    "timeout_hours": 48,
                    "escalation_role": "ceo"
                }
            }
            
            logger.info(f"Initialized {len(self.approval_workflows)} approval workflows")
            
        except Exception as e:
            logger.error(f"Error initializing approval workflows: {str(e)}")
    
    def _initialize_risk_matrix(self) -> None:
        """Initialize risk matrix"""
        try:
            self.risk_matrix = {
                "probability": {
                    "very_low": 0.1,
                    "low": 0.3,
                    "medium": 0.5,
                    "high": 0.7,
                    "very_high": 0.9
                },
                "impact": {
                    "very_low": 0.1,
                    "low": 0.3,
                    "medium": 0.5,
                    "high": 0.7,
                    "very_high": 0.9
                },
                "risk_levels": {
                    "low": {"min": 0.0, "max": 0.3},
                    "medium": {"min": 0.3, "max": 0.6},
                    "high": {"min": 0.6, "max": 0.8},
                    "critical": {"min": 0.8, "max": 1.0}
                }
            }
            
            logger.info("Initialized risk matrix")
            
        except Exception as e:
            logger.error(f"Error initializing risk matrix: {str(e)}")
    
    async def _calculate_compliance_score(self, policy_id: str, framework: ComplianceFramework, findings: List[Dict[str, Any]]) -> float:
        """Calculate compliance score"""
        try:
            framework_config = self.frameworks.get(framework.value, {})
            criteria = framework_config.get("assessment_criteria", {})
            
            if not criteria:
                return 0.0
            
            total_score = 0.0
            total_weight = 0.0
            
            for criterion, weight in criteria.items():
                # Find findings for this criterion
                criterion_findings = [f for f in findings if f.get("criterion") == criterion]
                
                if criterion_findings:
                    # Calculate score based on findings
                    severity_scores = {
                        "low": 0.8,
                        "medium": 0.6,
                        "high": 0.4,
                        "critical": 0.2
                    }
                    
                    criterion_score = 1.0
                    for finding in criterion_findings:
                        severity = finding.get("severity", "medium")
                        score_reduction = 1.0 - severity_scores.get(severity, 0.6)
                        criterion_score -= score_reduction
                    
                    criterion_score = max(0.0, criterion_score)
                else:
                    criterion_score = 1.0
                
                total_score += criterion_score * weight
                total_weight += weight
            
            return total_score / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating compliance score: {str(e)}")
            return 0.0
    
    async def _generate_remediation_plan(self, findings: List[Dict[str, Any]], recommendations: List[str]) -> Dict[str, Any]:
        """Generate remediation plan"""
        try:
            plan = {
                "priority_actions": [],
                "timeline": {},
                "resources_required": [],
                "success_metrics": []
            }
            
            # Categorize findings by severity
            critical_findings = [f for f in findings if f.get("severity") == "critical"]
            high_findings = [f for f in findings if f.get("severity") == "high"]
            medium_findings = [f for f in findings if f.get("severity") == "medium"]
            low_findings = [f for f in findings if f.get("severity") == "low"]
            
            # Generate priority actions
            if critical_findings:
                plan["priority_actions"].append({
                    "priority": "critical",
                    "description": "Address critical findings immediately",
                    "timeline": "1-7 days",
                    "findings_count": len(critical_findings)
                })
            
            if high_findings:
                plan["priority_actions"].append({
                    "priority": "high",
                    "description": "Address high-severity findings",
                    "timeline": "1-4 weeks",
                    "findings_count": len(high_findings)
                })
            
            if medium_findings:
                plan["priority_actions"].append({
                    "priority": "medium",
                    "description": "Address medium-severity findings",
                    "timeline": "1-3 months",
                    "findings_count": len(medium_findings)
                })
            
            if low_findings:
                plan["priority_actions"].append({
                    "priority": "low",
                    "description": "Address low-severity findings",
                    "timeline": "3-6 months",
                    "findings_count": len(low_findings)
                })
            
            # Generate timeline
            plan["timeline"] = {
                "immediate": "1-7 days",
                "short_term": "1-4 weeks",
                "medium_term": "1-3 months",
                "long_term": "3-6 months"
            }
            
            # Generate resource requirements
            plan["resources_required"] = [
                "Compliance team",
                "IT security team",
                "Legal counsel",
                "Training resources",
                "Technology updates"
            ]
            
            # Generate success metrics
            plan["success_metrics"] = [
                "100% of critical findings addressed",
                "90% of high findings addressed",
                "80% of medium findings addressed",
                "70% of low findings addressed",
                "Compliance score improvement of 20%"
            ]
            
            return plan
            
        except Exception as e:
            logger.error(f"Error generating remediation plan: {str(e)}")
            return {}
    
    async def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """Determine risk level based on score"""
        try:
            for level, range_config in self.risk_matrix["risk_levels"].items():
                if range_config["min"] <= risk_score <= range_config["max"]:
                    return RiskLevel(level)
            
            return RiskLevel.LOW
            
        except Exception as e:
            logger.error(f"Error determining risk level: {str(e)}")
            return RiskLevel.LOW
    
    async def _calculate_residual_risk(self, risk_level: RiskLevel, mitigation_measures: List[str]) -> RiskLevel:
        """Calculate residual risk after mitigation"""
        try:
            # Simple mitigation effectiveness
            mitigation_effectiveness = {
                "low": 0.1,
                "medium": 0.3,
                "high": 0.5,
                "critical": 0.7
            }
            
            effectiveness = mitigation_effectiveness.get(risk_level.value, 0.3)
            
            # Reduce risk level based on mitigation measures
            if len(mitigation_measures) >= 3 and effectiveness > 0.5:
                if risk_level == RiskLevel.CRITICAL:
                    return RiskLevel.HIGH
                elif risk_level == RiskLevel.HIGH:
                    return RiskLevel.MEDIUM
                elif risk_level == RiskLevel.MEDIUM:
                    return RiskLevel.LOW
            
            return risk_level
            
        except Exception as e:
            logger.error(f"Error calculating residual risk: {str(e)}")
            return risk_level
    
    async def _generate_policy_recommendations(self, 
                                            policy: GovernancePolicy,
                                            assessments: List[ComplianceAssessment],
                                            risks: List[RiskAssessment]) -> List[str]:
        """Generate policy recommendations"""
        try:
            recommendations = []
            
            # Check policy expiry
            if policy.expiry_date < datetime.now() + timedelta(days=30):
                recommendations.append("Policy is expiring soon - consider renewal or update")
            
            # Check compliance scores
            if assessments:
                avg_compliance = sum(a.compliance_score for a in assessments) / len(assessments)
                if avg_compliance < 0.7:
                    recommendations.append("Compliance score is below threshold - review and improve")
            
            # Check high risks
            high_risks = [r for r in risks if r.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]]
            if high_risks:
                recommendations.append(f"Address {len(high_risks)} high-risk items")
            
            # Check assessment frequency
            if assessments:
                last_assessment = max(a.assessment_date for a in assessments)
                if (datetime.now() - last_assessment).days > 180:
                    recommendations.append("Schedule new compliance assessment")
            
            # Check approval status
            pending_approvals = [r for r in self.approval_requests if r.policy_id == policy.policy_id and r.status == ApprovalStatus.PENDING]
            if pending_approvals:
                recommendations.append(f"Process {len(pending_approvals)} pending approval requests")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating policy recommendations: {str(e)}")
            return []
    
    def _start_background_tasks(self) -> None:
        """Start background tasks"""
        try:
            # Start policy expiry monitoring
            asyncio.create_task(self._monitor_policy_expiry())
            
            # Start approval deadline monitoring
            asyncio.create_task(self._monitor_approval_deadlines())
            
            logger.info("Started governance background tasks")
            
        except Exception as e:
            logger.error(f"Error starting background tasks: {str(e)}")
    
    async def _monitor_policy_expiry(self) -> None:
        """Monitor policy expiry"""
        try:
            while True:
                await asyncio.sleep(3600)  # Check every hour
                
                expiring_policies = [
                    p for p in self.governance_policies.values()
                    if p.expiry_date < datetime.now() + timedelta(days=30)
                    and p.expiry_date > datetime.now()
                ]
                
                for policy in expiring_policies:
                    logger.warning(f"Policy {policy.name} expires on {policy.expiry_date}")
                    
        except Exception as e:
            logger.error(f"Error monitoring policy expiry: {str(e)}")
    
    async def _monitor_approval_deadlines(self) -> None:
        """Monitor approval deadlines"""
        try:
            while True:
                await asyncio.sleep(1800)  # Check every 30 minutes
                
                overdue_approvals = [
                    r for r in self.approval_requests
                    if r.due_date < datetime.now()
                    and r.status == ApprovalStatus.PENDING
                ]
                
                for approval in overdue_approvals:
                    logger.warning(f"Approval request {approval.request_id} is overdue")
                    
        except Exception as e:
            logger.error(f"Error monitoring approval deadlines: {str(e)}")


# Global governance system instance
_governance_system: Optional[EnterpriseGovernanceSystem] = None


def get_enterprise_governance_system(max_policies: int = 1000, max_assessments: int = 10000) -> EnterpriseGovernanceSystem:
    """Get or create global enterprise governance system instance"""
    global _governance_system
    if _governance_system is None:
        _governance_system = EnterpriseGovernanceSystem(max_policies, max_assessments)
    return _governance_system


# Example usage
async def main():
    """Example usage of the enterprise governance system"""
    system = get_enterprise_governance_system()
    
    # Create governance policy
    policy = await system.create_governance_policy(
        name="AI Model Data Governance Policy",
        description="Governance policy for AI model data handling",
        policy_type=PolicyType.DATA_GOVERNANCE,
        governance_level=GovernanceLevel.ORGANIZATIONAL,
        framework=ComplianceFramework.GDPR,
        version="1.0",
        effective_date=datetime.now(),
        expiry_date=datetime.now() + timedelta(days=365),
        owner="data_governance_team",
        approvers=["compliance_officer", "legal_counsel"],
        stakeholders=["data_team", "ai_team", "security_team"],
        rules=[
            {"type": "data_minimization", "enabled": True},
            {"type": "purpose_limitation", "enabled": True},
            {"type": "storage_limitation", "max_days": 2555}  # 7 years
        ],
        enforcement_actions=["data_encryption", "access_control", "audit_logging"],
        risk_assessment={"level": "medium", "factors": ["data_volume", "sensitivity"]},
        compliance_requirements=["GDPR", "SOC2", "ISO27001"]
    )
    print(f"Created governance policy: {policy.policy_id}")
    
    # Create approval request
    approval_request = await system.create_approval_request(
        policy_id=policy.policy_id,
        requester="data_governance_team",
        request_type="policy_creation",
        description="Request approval for new AI model data governance policy",
        justification="Required for GDPR compliance and data protection",
        risk_level=RiskLevel.MEDIUM,
        approvers=["compliance_officer", "legal_counsel"],
        due_date=datetime.now() + timedelta(days=7)
    )
    print(f"Created approval request: {approval_request.request_id}")
    
    # Approve request
    approved = await system.approve_request(
        request_id=approval_request.request_id,
        approver="compliance_officer",
        decision=True,
        comments="Policy looks good, approved for implementation"
    )
    print(f"Approval processed: {approved}")
    
    # Conduct compliance assessment
    assessment = await system.conduct_compliance_assessment(
        policy_id=policy.policy_id,
        framework=ComplianceFramework.GDPR,
        assessor="compliance_auditor",
        findings=[
            {
                "criterion": "data_protection",
                "severity": "low",
                "description": "Minor documentation gap",
                "recommendation": "Update documentation"
            }
        ],
        recommendations=[
            "Implement automated data classification",
            "Enhance audit logging",
            "Conduct regular compliance training"
        ]
    )
    print(f"Conducted compliance assessment: {assessment.assessment_id}")
    print(f"Compliance score: {assessment.compliance_score:.2f}")
    
    # Conduct risk assessment
    risk_assessment = await system.conduct_risk_assessment(
        policy_id=policy.policy_id,
        risk_type="data_breach",
        description="Risk of unauthorized access to AI model data",
        probability=0.3,
        impact=0.7,
        mitigation_measures=[
            "Implement encryption at rest",
            "Enforce access controls",
            "Regular security audits"
        ],
        owner="security_team",
        assessor="risk_manager"
    )
    print(f"Conducted risk assessment: {risk_assessment.risk_id}")
    print(f"Risk level: {risk_assessment.risk_level.value}")
    print(f"Residual risk: {risk_assessment.residual_risk.value}")
    
    # Get governance dashboard
    dashboard = await system.get_governance_dashboard()
    print(f"Governance dashboard:")
    print(f"  Total policies: {dashboard.get('total_policies', 0)}")
    print(f"  Active policies: {dashboard.get('active_policies', 0)}")
    print(f"  Pending approvals: {dashboard.get('pending_approvals', 0)}")
    print(f"  High risks: {dashboard.get('high_risks', 0)}")
    
    # Get policy compliance report
    report = await system.get_policy_compliance_report(policy.policy_id)
    print(f"Policy compliance report:")
    print(f"  Policy: {report['policy_name']}")
    print(f"  Compliance score: {report['compliance_score']:.2f}")
    print(f"  Risk score: {report['risk_score']:.2f}")
    print(f"  Recommendations: {len(report['recommendations'])}")


if __name__ == "__main__":
    asyncio.run(main())

























