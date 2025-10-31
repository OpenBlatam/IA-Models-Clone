"""
Content Governance Engine - Advanced Content Governance and Compliance Management
============================================================================

This module provides comprehensive content governance capabilities including:
- Content policy management and enforcement
- Regulatory compliance monitoring
- Content approval workflows
- Risk assessment and mitigation
- Audit trails and compliance reporting
- Content lifecycle governance
- Data privacy and protection
- Multi-jurisdiction compliance
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
import hashlib
import re
from collections import defaultdict, deque
import redis
import sqlalchemy as sa
from sqlalchemy import create_engine, text
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from bs4 import BeautifulSoup
import yaml
import xml.etree.ElementTree as ET
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
import jwt
from passlib.context import CryptContext
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import boto3
from google.cloud import storage
import openai
import anthropic

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComplianceStandard(Enum):
    """Compliance standard enumeration"""
    GDPR = "gdpr"
    CCPA = "ccpa"
    HIPAA = "hipaa"
    SOX = "sox"
    PCI_DSS = "pci_dss"
    ISO_27001 = "iso_27001"
    SOC2 = "soc2"
    FERPA = "ferpa"
    COPPA = "coppa"
    ADA = "ada"

class PolicyType(Enum):
    """Policy type enumeration"""
    CONTENT_POLICY = "content_policy"
    PRIVACY_POLICY = "privacy_policy"
    SECURITY_POLICY = "security_policy"
    ACCESS_POLICY = "access_policy"
    RETENTION_POLICY = "retention_policy"
    SHARING_POLICY = "sharing_policy"
    QUALITY_POLICY = "quality_policy"
    BRAND_POLICY = "brand_policy"

class RiskLevel(Enum):
    """Risk level enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ApprovalStatus(Enum):
    """Approval status enumeration"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    REQUIRES_REVISION = "requires_revision"
    ESCALATED = "escalated"

class GovernanceAction(Enum):
    """Governance action enumeration"""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    SHARE = "share"
    PUBLISH = "publish"
    ARCHIVE = "archive"
    EXPORT = "export"
    IMPORT = "import"

@dataclass
class GovernancePolicy:
    """Governance policy data structure"""
    policy_id: str
    name: str
    policy_type: PolicyType
    description: str
    rules: List[Dict[str, Any]] = field(default_factory=list)
    compliance_standards: List[ComplianceStandard] = field(default_factory=list)
    risk_level: RiskLevel = RiskLevel.MEDIUM
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = ""
    approved_by: str = ""

@dataclass
class ComplianceCheck:
    """Compliance check data structure"""
    check_id: str
    content_id: str
    policy_id: str
    compliance_standard: ComplianceStandard
    status: str  # compliant, non_compliant, warning
    violations: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    risk_score: float = 0.0
    checked_at: datetime = field(default_factory=datetime.utcnow)
    checked_by: str = "system"

@dataclass
class ApprovalWorkflow:
    """Approval workflow data structure"""
    workflow_id: str
    name: str
    description: str
    steps: List[Dict[str, Any]] = field(default_factory=list)
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = ""

@dataclass
class ApprovalRequest:
    """Approval request data structure"""
    request_id: str
    content_id: str
    workflow_id: str
    requester_id: str
    current_step: int = 0
    status: ApprovalStatus = ApprovalStatus.PENDING
    approvers: List[str] = field(default_factory=list)
    comments: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

@dataclass
class RiskAssessment:
    """Risk assessment data structure"""
    assessment_id: str
    content_id: str
    risk_level: RiskLevel
    risk_factors: List[str] = field(default_factory=list)
    mitigation_strategies: List[str] = field(default_factory=list)
    residual_risk: RiskLevel = RiskLevel.LOW
    assessed_at: datetime = field(default_factory=datetime.utcnow)
    assessed_by: str = ""

@dataclass
class AuditLog:
    """Audit log data structure"""
    log_id: str
    user_id: str
    action: GovernanceAction
    resource_id: str
    resource_type: str
    details: Dict[str, Any] = field(default_factory=dict)
    ip_address: str = ""
    user_agent: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    compliance_impact: str = ""

class ContentGovernanceEngine:
    """
    Advanced Content Governance Engine
    
    Provides comprehensive content governance and compliance management capabilities
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the Content Governance Engine"""
        self.config = config
        self.governance_policies = {}
        self.compliance_checks = {}
        self.approval_workflows = {}
        self.approval_requests = {}
        self.risk_assessments = {}
        self.audit_logs = []
        self.compliance_rules = {}
        self.redis_client = None
        self.database_engine = None
        
        # Initialize components
        self._initialize_database()
        self._initialize_redis()
        self._initialize_compliance_rules()
        self._initialize_encryption()
        
        # Start background tasks
        self._start_background_tasks()
        
        logger.info("Content Governance Engine initialized successfully")
    
    def _initialize_database(self):
        """Initialize database connection"""
        try:
            if self.config.get("database_url"):
                self.database_engine = create_engine(self.config["database_url"])
                logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
    
    def _initialize_redis(self):
        """Initialize Redis connection"""
        try:
            if self.config.get("redis_url"):
                self.redis_client = redis.Redis.from_url(self.config["redis_url"])
                logger.info("Redis connection established")
        except Exception as e:
            logger.error(f"Error initializing Redis: {e}")
    
    def _initialize_compliance_rules(self):
        """Initialize compliance rules for different standards"""
        try:
            # GDPR compliance rules
            self.compliance_rules[ComplianceStandard.GDPR] = {
                "personal_data_patterns": [
                    r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
                    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
                    r'\b\d{3}-\d{3}-\d{4}\b',  # Phone
                    r'\b\d{4}\s\d{4}\s\d{4}\s\d{4}\b'  # Credit card
                ],
                "required_elements": [
                    "privacy_policy",
                    "consent_mechanism",
                    "data_retention_policy",
                    "right_to_deletion",
                    "data_portability"
                ],
                "data_retention_limit": 365,  # days
                "consent_required": True,
                "data_processing_basis": ["consent", "legitimate_interest", "contract"]
            }
            
            # HIPAA compliance rules
            self.compliance_rules[ComplianceStandard.HIPAA] = {
                "phi_patterns": [
                    r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
                    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
                    r'\b\d{3}-\d{3}-\d{4}\b',  # Phone
                    r'\b\d{2}/\d{2}/\d{4}\b'  # Date of birth
                ],
                "required_safeguards": [
                    "administrative_safeguards",
                    "physical_safeguards",
                    "technical_safeguards"
                ],
                "encryption_required": True,
                "access_controls_required": True,
                "audit_trail_required": True
            }
            
            # PCI DSS compliance rules
            self.compliance_rules[ComplianceStandard.PCI_DSS] = {
                "card_data_patterns": [
                    r'\b\d{4}\s\d{4}\s\d{4}\s\d{4}\b',  # Credit card
                    r'\b\d{3}\b'  # CVV
                ],
                "required_controls": [
                    "firewall_configuration",
                    "password_protection",
                    "card_data_encryption",
                    "secure_networks",
                    "regular_monitoring"
                ],
                "encryption_required": True,
                "network_security_required": True,
                "regular_testing_required": True
            }
            
            # CCPA compliance rules
            self.compliance_rules[ComplianceStandard.CCPA] = {
                "personal_information_patterns": [
                    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
                    r'\b\d{3}-\d{3}-\d{4}\b',  # Phone
                    r'\b\d{5}(-\d{4})?\b'  # ZIP code
                ],
                "consumer_rights": [
                    "right_to_know",
                    "right_to_delete",
                    "right_to_opt_out",
                    "right_to_non_discrimination"
                ],
                "privacy_policy_required": True,
                "opt_out_mechanism_required": True
            }
            
            logger.info("Compliance rules initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing compliance rules: {e}")
    
    def _initialize_encryption(self):
        """Initialize encryption components"""
        try:
            # Generate encryption key
            self.encryption_key = Fernet.generate_key()
            self.cipher_suite = Fernet(self.encryption_key)
            
            # Initialize password hashing
            self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
            
            logger.info("Encryption components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing encryption: {e}")
    
    def _start_background_tasks(self):
        """Start background tasks"""
        try:
            # Start compliance monitoring task
            asyncio.create_task(self._monitor_compliance_periodically())
            
            # Start audit log processing task
            asyncio.create_task(self._process_audit_logs_periodically())
            
            # Start risk assessment task
            asyncio.create_task(self._assess_risks_periodically())
            
            logger.info("Background tasks started")
            
        except Exception as e:
            logger.error(f"Error starting background tasks: {e}")
    
    async def create_governance_policy(self, policy_data: Dict[str, Any]) -> GovernancePolicy:
        """Create a new governance policy"""
        try:
            policy_id = str(uuid.uuid4())
            
            policy = GovernancePolicy(
                policy_id=policy_id,
                name=policy_data["name"],
                policy_type=PolicyType(policy_data["policy_type"]),
                description=policy_data["description"],
                rules=policy_data.get("rules", []),
                compliance_standards=[ComplianceStandard(s) for s in policy_data.get("compliance_standards", [])],
                risk_level=RiskLevel(policy_data.get("risk_level", "medium")),
                created_by=policy_data.get("created_by", "system")
            )
            
            # Store policy
            self.governance_policies[policy_id] = policy
            
            # Log policy creation
            await self._log_audit_event(
                user_id=policy.created_by,
                action=GovernanceAction.CREATE,
                resource_id=policy_id,
                resource_type="governance_policy",
                details={"policy_name": policy.name, "policy_type": policy.policy_type.value}
            )
            
            logger.info(f"Governance policy {policy_id} created successfully")
            
            return policy
            
        except Exception as e:
            logger.error(f"Error creating governance policy: {e}")
            raise
    
    async def check_content_compliance(self, content_id: str, content: str, 
                                     compliance_standards: List[ComplianceStandard]) -> List[ComplianceCheck]:
        """Check content compliance against standards"""
        try:
            compliance_checks = []
            
            for standard in compliance_standards:
                check_id = str(uuid.uuid4())
                
                # Get compliance rules for standard
                rules = self.compliance_rules.get(standard, {})
                
                # Perform compliance check
                violations = []
                recommendations = []
                risk_score = 0.0
                
                if standard == ComplianceStandard.GDPR:
                    violations, recommendations, risk_score = await self._check_gdpr_compliance(content, rules)
                elif standard == ComplianceStandard.HIPAA:
                    violations, recommendations, risk_score = await self._check_hipaa_compliance(content, rules)
                elif standard == ComplianceStandard.PCI_DSS:
                    violations, recommendations, risk_score = await self._check_pci_compliance(content, rules)
                elif standard == ComplianceStandard.CCPA:
                    violations, recommendations, risk_score = await self._check_ccpa_compliance(content, rules)
                
                # Determine compliance status
                if not violations:
                    status = "compliant"
                elif risk_score < 0.5:
                    status = "warning"
                else:
                    status = "non_compliant"
                
                # Create compliance check
                compliance_check = ComplianceCheck(
                    check_id=check_id,
                    content_id=content_id,
                    policy_id="",  # Would be linked to specific policy
                    compliance_standard=standard,
                    status=status,
                    violations=violations,
                    recommendations=recommendations,
                    risk_score=risk_score
                )
                
                compliance_checks.append(compliance_check)
                self.compliance_checks[check_id] = compliance_check
            
            # Log compliance check
            await self._log_audit_event(
                user_id="system",
                action=GovernanceAction.UPDATE,
                resource_id=content_id,
                resource_type="content",
                details={
                    "compliance_standards": [s.value for s in compliance_standards],
                    "checks_performed": len(compliance_checks)
                }
            )
            
            logger.info(f"Compliance check completed for content {content_id}")
            
            return compliance_checks
            
        except Exception as e:
            logger.error(f"Error checking content compliance: {e}")
            raise
    
    async def _check_gdpr_compliance(self, content: str, rules: Dict[str, Any]) -> Tuple[List[str], List[str], float]:
        """Check GDPR compliance"""
        try:
            violations = []
            recommendations = []
            risk_score = 0.0
            
            # Check for personal data
            personal_data_patterns = rules.get("personal_data_patterns", [])
            for pattern in personal_data_patterns:
                if re.search(pattern, content):
                    violations.append(f"Personal data detected: {pattern}")
                    risk_score += 0.3
            
            # Check for required elements
            required_elements = rules.get("required_elements", [])
            for element in required_elements:
                if element not in content.lower():
                    violations.append(f"Missing required element: {element}")
                    risk_score += 0.2
            
            # Generate recommendations
            if violations:
                recommendations.extend([
                    "Remove or encrypt personal data",
                    "Add privacy policy and consent mechanisms",
                    "Implement data retention policies",
                    "Ensure right to deletion is available"
                ])
            
            return violations, recommendations, min(risk_score, 1.0)
            
        except Exception as e:
            logger.error(f"Error checking GDPR compliance: {e}")
            return [], [], 0.0
    
    async def _check_hipaa_compliance(self, content: str, rules: Dict[str, Any]) -> Tuple[List[str], List[str], float]:
        """Check HIPAA compliance"""
        try:
            violations = []
            recommendations = []
            risk_score = 0.0
            
            # Check for PHI
            phi_patterns = rules.get("phi_patterns", [])
            for pattern in phi_patterns:
                if re.search(pattern, content):
                    violations.append(f"Protected health information detected: {pattern}")
                    risk_score += 0.4
            
            # Check for required safeguards
            required_safeguards = rules.get("required_safeguards", [])
            for safeguard in required_safeguards:
                if safeguard not in content.lower():
                    violations.append(f"Missing required safeguard: {safeguard}")
                    risk_score += 0.2
            
            # Generate recommendations
            if violations:
                recommendations.extend([
                    "Encrypt protected health information",
                    "Implement access controls",
                    "Add audit logging",
                    "Ensure administrative, physical, and technical safeguards"
                ])
            
            return violations, recommendations, min(risk_score, 1.0)
            
        except Exception as e:
            logger.error(f"Error checking HIPAA compliance: {e}")
            return [], [], 0.0
    
    async def _check_pci_compliance(self, content: str, rules: Dict[str, Any]) -> Tuple[List[str], List[str], float]:
        """Check PCI DSS compliance"""
        try:
            violations = []
            recommendations = []
            risk_score = 0.0
            
            # Check for card data
            card_data_patterns = rules.get("card_data_patterns", [])
            for pattern in card_data_patterns:
                if re.search(pattern, content):
                    violations.append(f"Card data detected: {pattern}")
                    risk_score += 0.5
            
            # Check for required controls
            required_controls = rules.get("required_controls", [])
            for control in required_controls:
                if control not in content.lower():
                    violations.append(f"Missing required control: {control}")
                    risk_score += 0.1
            
            # Generate recommendations
            if violations:
                recommendations.extend([
                    "Encrypt card data",
                    "Implement network security",
                    "Regular security testing",
                    "Firewall configuration"
                ])
            
            return violations, recommendations, min(risk_score, 1.0)
            
        except Exception as e:
            logger.error(f"Error checking PCI compliance: {e}")
            return [], [], 0.0
    
    async def _check_ccpa_compliance(self, content: str, rules: Dict[str, Any]) -> Tuple[List[str], List[str], float]:
        """Check CCPA compliance"""
        try:
            violations = []
            recommendations = []
            risk_score = 0.0
            
            # Check for personal information
            personal_info_patterns = rules.get("personal_information_patterns", [])
            for pattern in personal_info_patterns:
                if re.search(pattern, content):
                    violations.append(f"Personal information detected: {pattern}")
                    risk_score += 0.3
            
            # Check for consumer rights
            consumer_rights = rules.get("consumer_rights", [])
            for right in consumer_rights:
                if right not in content.lower():
                    violations.append(f"Missing consumer right: {right}")
                    risk_score += 0.2
            
            # Generate recommendations
            if violations:
                recommendations.extend([
                    "Add privacy policy",
                    "Implement opt-out mechanism",
                    "Ensure consumer rights are available",
                    "Provide data deletion options"
                ])
            
            return violations, recommendations, min(risk_score, 1.0)
            
        except Exception as e:
            logger.error(f"Error checking CCPA compliance: {e}")
            return [], [], 0.0
    
    async def create_approval_workflow(self, workflow_data: Dict[str, Any]) -> ApprovalWorkflow:
        """Create approval workflow"""
        try:
            workflow_id = str(uuid.uuid4())
            
            workflow = ApprovalWorkflow(
                workflow_id=workflow_id,
                name=workflow_data["name"],
                description=workflow_data["description"],
                steps=workflow_data.get("steps", []),
                created_by=workflow_data.get("created_by", "system")
            )
            
            # Store workflow
            self.approval_workflows[workflow_id] = workflow
            
            # Log workflow creation
            await self._log_audit_event(
                user_id=workflow.created_by,
                action=GovernanceAction.CREATE,
                resource_id=workflow_id,
                resource_type="approval_workflow",
                details={"workflow_name": workflow.name}
            )
            
            logger.info(f"Approval workflow {workflow_id} created successfully")
            
            return workflow
            
        except Exception as e:
            logger.error(f"Error creating approval workflow: {e}")
            raise
    
    async def submit_approval_request(self, content_id: str, workflow_id: str, 
                                    requester_id: str) -> ApprovalRequest:
        """Submit content for approval"""
        try:
            if workflow_id not in self.approval_workflows:
                raise ValueError(f"Workflow {workflow_id} not found")
            
            workflow = self.approval_workflows[workflow_id]
            
            request_id = str(uuid.uuid4())
            
            # Get first step approvers
            first_step = workflow.steps[0] if workflow.steps else {}
            approvers = first_step.get("approvers", [])
            
            approval_request = ApprovalRequest(
                request_id=request_id,
                content_id=content_id,
                workflow_id=workflow_id,
                requester_id=requester_id,
                approvers=approvers
            )
            
            # Store request
            self.approval_requests[request_id] = approval_request
            
            # Log approval request
            await self._log_audit_event(
                user_id=requester_id,
                action=GovernanceAction.CREATE,
                resource_id=content_id,
                resource_type="content",
                details={
                    "workflow_id": workflow_id,
                    "approval_request_id": request_id
                }
            )
            
            logger.info(f"Approval request {request_id} submitted for content {content_id}")
            
            return approval_request
            
        except Exception as e:
            logger.error(f"Error submitting approval request: {e}")
            raise
    
    async def process_approval(self, request_id: str, approver_id: str, 
                             decision: ApprovalStatus, comments: str = "") -> bool:
        """Process approval decision"""
        try:
            if request_id not in self.approval_requests:
                raise ValueError(f"Approval request {request_id} not found")
            
            request = self.approval_requests[request_id]
            
            if approver_id not in request.approvers:
                raise ValueError(f"User {approver_id} is not authorized to approve this request")
            
            # Add comment
            if comments:
                request.comments.append({
                    "approver_id": approver_id,
                    "comment": comments,
                    "timestamp": datetime.utcnow().isoformat()
                })
            
            # Process decision
            if decision == ApprovalStatus.APPROVED:
                # Move to next step or complete
                if request.current_step < len(self.approval_workflows[request.workflow_id].steps) - 1:
                    request.current_step += 1
                    next_step = self.approval_workflows[request.workflow_id].steps[request.current_step]
                    request.approvers = next_step.get("approvers", [])
                else:
                    request.status = ApprovalStatus.APPROVED
                    request.completed_at = datetime.utcnow()
            
            elif decision == ApprovalStatus.REJECTED:
                request.status = ApprovalStatus.REJECTED
                request.completed_at = datetime.utcnow()
            
            elif decision == ApprovalStatus.REQUIRES_REVISION:
                request.status = ApprovalStatus.REQUIRES_REVISION
                request.completed_at = datetime.utcnow()
            
            # Log approval decision
            await self._log_audit_event(
                user_id=approver_id,
                action=GovernanceAction.UPDATE,
                resource_id=request.content_id,
                resource_type="content",
                details={
                    "approval_request_id": request_id,
                    "decision": decision.value,
                    "comments": comments
                }
            )
            
            logger.info(f"Approval decision processed: {decision.value} for request {request_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing approval: {e}")
            return False
    
    async def assess_content_risk(self, content_id: str, content: str) -> RiskAssessment:
        """Assess content risk"""
        try:
            assessment_id = str(uuid.uuid4())
            
            risk_factors = []
            mitigation_strategies = []
            risk_score = 0.0
            
            # Check for sensitive information
            sensitive_patterns = [
                r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
                r'\b\d{3}-\d{3}-\d{4}\b',  # Phone
                r'\b\d{4}\s\d{4}\s\d{4}\s\d{4}\b'  # Credit card
            ]
            
            for pattern in sensitive_patterns:
                if re.search(pattern, content):
                    risk_factors.append(f"Sensitive information detected: {pattern}")
                    risk_score += 0.3
            
            # Check for controversial content
            controversial_keywords = [
                "hate", "violence", "discrimination", "harassment",
                "illegal", "fraud", "scam", "misleading"
            ]
            
            for keyword in controversial_keywords:
                if keyword.lower() in content.lower():
                    risk_factors.append(f"Controversial content detected: {keyword}")
                    risk_score += 0.2
            
            # Check content length and complexity
            if len(content) > 10000:
                risk_factors.append("Very long content may be difficult to review")
                risk_score += 0.1
            
            # Determine risk level
            if risk_score >= 0.8:
                risk_level = RiskLevel.CRITICAL
            elif risk_score >= 0.6:
                risk_level = RiskLevel.HIGH
            elif risk_score >= 0.3:
                risk_level = RiskLevel.MEDIUM
            else:
                risk_level = RiskLevel.LOW
            
            # Generate mitigation strategies
            if risk_factors:
                mitigation_strategies.extend([
                    "Remove or encrypt sensitive information",
                    "Review content for accuracy and compliance",
                    "Implement additional approval steps",
                    "Add content warnings if necessary"
                ])
            
            # Calculate residual risk
            residual_risk = RiskLevel.LOW if not risk_factors else RiskLevel.MEDIUM
            
            assessment = RiskAssessment(
                assessment_id=assessment_id,
                content_id=content_id,
                risk_level=risk_level,
                risk_factors=risk_factors,
                mitigation_strategies=mitigation_strategies,
                residual_risk=residual_risk
            )
            
            # Store assessment
            self.risk_assessments[assessment_id] = assessment
            
            # Log risk assessment
            await self._log_audit_event(
                user_id="system",
                action=GovernanceAction.UPDATE,
                resource_id=content_id,
                resource_type="content",
                details={
                    "risk_assessment_id": assessment_id,
                    "risk_level": risk_level.value,
                    "risk_score": risk_score
                }
            )
            
            logger.info(f"Risk assessment completed for content {content_id}: {risk_level.value}")
            
            return assessment
            
        except Exception as e:
            logger.error(f"Error assessing content risk: {e}")
            raise
    
    async def _log_audit_event(self, user_id: str, action: GovernanceAction, 
                             resource_id: str, resource_type: str, 
                             details: Dict[str, Any] = None, ip_address: str = "",
                             user_agent: str = ""):
        """Log audit event"""
        try:
            audit_log = AuditLog(
                log_id=str(uuid.uuid4()),
                user_id=user_id,
                action=action,
                resource_id=resource_id,
                resource_type=resource_type,
                details=details or {},
                ip_address=ip_address,
                user_agent=user_agent
            )
            
            self.audit_logs.append(audit_log)
            
            # Store in Redis for quick access
            if self.redis_client:
                log_data = {
                    "log_id": audit_log.log_id,
                    "user_id": user_id,
                    "action": action.value,
                    "resource_id": resource_id,
                    "resource_type": resource_type,
                    "timestamp": audit_log.timestamp.isoformat(),
                    "details": details or {}
                }
                self.redis_client.setex(f"audit_log:{audit_log.log_id}", 86400, json.dumps(log_data))
            
            logger.info(f"Audit event logged: {action.value} on {resource_type} {resource_id}")
            
        except Exception as e:
            logger.error(f"Error logging audit event: {e}")
    
    async def get_audit_trail(self, resource_id: str = None, user_id: str = None, 
                            start_date: datetime = None, end_date: datetime = None) -> List[AuditLog]:
        """Get audit trail"""
        try:
            filtered_logs = self.audit_logs.copy()
            
            # Filter by resource ID
            if resource_id:
                filtered_logs = [log for log in filtered_logs if log.resource_id == resource_id]
            
            # Filter by user ID
            if user_id:
                filtered_logs = [log for log in filtered_logs if log.user_id == user_id]
            
            # Filter by date range
            if start_date:
                filtered_logs = [log for log in filtered_logs if log.timestamp >= start_date]
            
            if end_date:
                filtered_logs = [log for log in filtered_logs if log.timestamp <= end_date]
            
            # Sort by timestamp (newest first)
            filtered_logs.sort(key=lambda x: x.timestamp, reverse=True)
            
            return filtered_logs
            
        except Exception as e:
            logger.error(f"Error getting audit trail: {e}")
            return []
    
    async def generate_compliance_report(self, time_period: str = "30d", 
                                       compliance_standards: List[ComplianceStandard] = None) -> Dict[str, Any]:
        """Generate compliance report"""
        try:
            # Calculate time range
            end_date = datetime.utcnow()
            if time_period == "7d":
                start_date = end_date - timedelta(days=7)
            elif time_period == "30d":
                start_date = end_date - timedelta(days=30)
            elif time_period == "90d":
                start_date = end_date - timedelta(days=90)
            else:
                start_date = end_date - timedelta(days=30)
            
            # Filter compliance checks by time period and standards
            filtered_checks = [
                check for check in self.compliance_checks.values()
                if start_date <= check.checked_at <= end_date
            ]
            
            if compliance_standards:
                filtered_checks = [
                    check for check in filtered_checks
                    if check.compliance_standard in compliance_standards
                ]
            
            # Generate report
            report = {
                "report_id": str(uuid.uuid4()),
                "generated_at": datetime.utcnow().isoformat(),
                "time_period": time_period,
                "compliance_standards": [s.value for s in compliance_standards] if compliance_standards else "all",
                "summary": {
                    "total_checks": len(filtered_checks),
                    "compliant": len([c for c in filtered_checks if c.status == "compliant"]),
                    "non_compliant": len([c for c in filtered_checks if c.status == "non_compliant"]),
                    "warnings": len([c for c in filtered_checks if c.status == "warning"]),
                    "compliance_rate": 0.0
                },
                "detailed_results": {},
                "risk_analysis": {},
                "recommendations": []
            }
            
            # Calculate compliance rate
            if filtered_checks:
                compliant_count = len([c for c in filtered_checks if c.status == "compliant"])
                report["summary"]["compliance_rate"] = (compliant_count / len(filtered_checks)) * 100
            
            # Detailed results by standard
            for standard in compliance_standards or list(ComplianceStandard):
                standard_checks = [c for c in filtered_checks if c.compliance_standard == standard]
                if standard_checks:
                    report["detailed_results"][standard.value] = {
                        "total_checks": len(standard_checks),
                        "compliant": len([c for c in standard_checks if c.status == "compliant"]),
                        "non_compliant": len([c for c in standard_checks if c.status == "non_compliant"]),
                        "warnings": len([c for c in standard_checks if c.status == "warning"]),
                        "average_risk_score": np.mean([c.risk_score for c in standard_checks]),
                        "common_violations": self._get_common_violations(standard_checks)
                    }
            
            # Risk analysis
            report["risk_analysis"] = await self._analyze_compliance_risks(filtered_checks)
            
            # Generate recommendations
            report["recommendations"] = await self._generate_compliance_recommendations(filtered_checks)
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating compliance report: {e}")
            return {"error": str(e)}
    
    def _get_common_violations(self, checks: List[ComplianceCheck]) -> List[str]:
        """Get common violations from compliance checks"""
        try:
            all_violations = []
            for check in checks:
                all_violations.extend(check.violations)
            
            # Count violations
            violation_counts = Counter(all_violations)
            
            # Return top 5 most common violations
            return [violation for violation, count in violation_counts.most_common(5)]
            
        except Exception as e:
            logger.error(f"Error getting common violations: {e}")
            return []
    
    async def _analyze_compliance_risks(self, checks: List[ComplianceCheck]) -> Dict[str, Any]:
        """Analyze compliance risks"""
        try:
            risk_analysis = {
                "high_risk_content": [],
                "risk_trends": {},
                "risk_distribution": {}
            }
            
            # Identify high-risk content
            high_risk_checks = [c for c in checks if c.risk_score > 0.7]
            risk_analysis["high_risk_content"] = [
                {
                    "content_id": check.content_id,
                    "risk_score": check.risk_score,
                    "violations": check.violations
                }
                for check in high_risk_checks
            ]
            
            # Risk distribution
            risk_scores = [c.risk_score for c in checks]
            if risk_scores:
                risk_analysis["risk_distribution"] = {
                    "average_risk": np.mean(risk_scores),
                    "max_risk": np.max(risk_scores),
                    "min_risk": np.min(risk_scores),
                    "risk_std": np.std(risk_scores)
                }
            
            return risk_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing compliance risks: {e}")
            return {}
    
    async def _generate_compliance_recommendations(self, checks: List[ComplianceCheck]) -> List[str]:
        """Generate compliance recommendations"""
        try:
            recommendations = []
            
            # Get all recommendations from checks
            all_recommendations = []
            for check in checks:
                all_recommendations.extend(check.recommendations)
            
            # Count recommendations
            recommendation_counts = Counter(all_recommendations)
            
            # Return top recommendations
            top_recommendations = [rec for rec, count in recommendation_counts.most_common(10)]
            
            # Add general recommendations
            if len([c for c in checks if c.status == "non_compliant"]) > len(checks) * 0.3:
                recommendations.append("Implement comprehensive compliance training program")
                recommendations.append("Establish regular compliance monitoring and auditing")
                recommendations.append("Create compliance checklists and templates")
            
            recommendations.extend(top_recommendations)
            
            return list(set(recommendations))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Error generating compliance recommendations: {e}")
            return []
    
    async def _monitor_compliance_periodically(self):
        """Monitor compliance periodically"""
        while True:
            try:
                await asyncio.sleep(3600)  # Monitor every hour
                
                # In production, this would check for new content and run compliance checks
                logger.info("Compliance monitoring completed")
                
            except Exception as e:
                logger.error(f"Error in compliance monitoring: {e}")
                await asyncio.sleep(3600)
    
    async def _process_audit_logs_periodically(self):
        """Process audit logs periodically"""
        while True:
            try:
                await asyncio.sleep(1800)  # Process every 30 minutes
                
                # In production, this would process and store audit logs
                logger.info("Audit logs processed")
                
            except Exception as e:
                logger.error(f"Error processing audit logs: {e}")
                await asyncio.sleep(1800)
    
    async def _assess_risks_periodically(self):
        """Assess risks periodically"""
        while True:
            try:
                await asyncio.sleep(7200)  # Assess every 2 hours
                
                # In production, this would assess risks for new content
                logger.info("Risk assessment completed")
                
            except Exception as e:
                logger.error(f"Error in risk assessment: {e}")
                await asyncio.sleep(7200)

# Example usage and testing
async def main():
    """Example usage of the Content Governance Engine"""
    try:
        # Initialize engine
        config = {
            "database_url": "postgresql://user:password@localhost/governancedb",
            "redis_url": "redis://localhost:6379"
        }
        
        engine = ContentGovernanceEngine(config)
        
        # Create governance policy
        print("Creating governance policy...")
        policy = await engine.create_governance_policy({
            "name": "Content Quality Policy",
            "policy_type": "content_policy",
            "description": "Policy for maintaining content quality and compliance",
            "rules": [
                {"type": "min_length", "value": 100},
                {"type": "max_length", "value": 5000},
                {"type": "required_elements", "value": ["title", "content"]}
            ],
            "compliance_standards": ["gdpr", "ccpa"],
            "risk_level": "medium",
            "created_by": "admin"
        })
        
        # Check content compliance
        print("Checking content compliance...")
        test_content = "This is a test content with email john.doe@example.com and phone 123-456-7890"
        compliance_checks = await engine.check_content_compliance(
            "content_001",
            test_content,
            [ComplianceStandard.GDPR, ComplianceStandard.CCPA]
        )
        
        for check in compliance_checks:
            print(f"Compliance check: {check.compliance_standard.value} - {check.status}")
            print(f"Risk score: {check.risk_score}")
            print(f"Violations: {check.violations}")
        
        # Create approval workflow
        print("Creating approval workflow...")
        workflow = await engine.create_approval_workflow({
            "name": "Content Approval Workflow",
            "description": "Standard content approval process",
            "steps": [
                {
                    "name": "Initial Review",
                    "approvers": ["reviewer1", "reviewer2"],
                    "required_approvals": 1
                },
                {
                    "name": "Final Approval",
                    "approvers": ["manager1"],
                    "required_approvals": 1
                }
            ],
            "created_by": "admin"
        })
        
        # Submit approval request
        print("Submitting approval request...")
        approval_request = await engine.submit_approval_request(
            "content_001",
            workflow.workflow_id,
            "author1"
        )
        print(f"Approval request submitted: {approval_request.request_id}")
        
        # Process approval
        print("Processing approval...")
        await engine.process_approval(
            approval_request.request_id,
            "reviewer1",
            ApprovalStatus.APPROVED,
            "Content looks good, approved for publication"
        )
        
        # Assess content risk
        print("Assessing content risk...")
        risk_assessment = await engine.assess_content_risk("content_001", test_content)
        print(f"Risk level: {risk_assessment.risk_level.value}")
        print(f"Risk factors: {risk_assessment.risk_factors}")
        print(f"Mitigation strategies: {risk_assessment.mitigation_strategies}")
        
        # Get audit trail
        print("Getting audit trail...")
        audit_trail = await engine.get_audit_trail(resource_id="content_001")
        print(f"Audit trail entries: {len(audit_trail)}")
        
        # Generate compliance report
        print("Generating compliance report...")
        compliance_report = await engine.generate_compliance_report("7d", [ComplianceStandard.GDPR])
        print(f"Compliance report generated: {compliance_report['report_id']}")
        print(f"Total checks: {compliance_report['summary']['total_checks']}")
        print(f"Compliance rate: {compliance_report['summary']['compliance_rate']:.1f}%")
        
        print("\nContent Governance Engine demo completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main demo: {e}")

if __name__ == "__main__":
    asyncio.run(main())
























