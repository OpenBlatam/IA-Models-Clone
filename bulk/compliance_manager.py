"""
BUL Compliance Manager
=====================

Advanced compliance management system for the BUL system.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import sqlite3
from dataclasses import dataclass
from enum import Enum
import yaml
import re
from collections import defaultdict

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from config_optimized import BULConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComplianceStandard(Enum):
    """Compliance standards."""
    GDPR = "gdpr"
    HIPAA = "hipaa"
    SOX = "sox"
    PCI_DSS = "pci_dss"
    ISO27001 = "iso27001"
    SOC2 = "soc2"
    CCPA = "ccpa"
    FERPA = "ferpa"

class ComplianceStatus(Enum):
    """Compliance status."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NOT_ASSESSED = "not_assessed"
    IN_PROGRESS = "in_progress"

class ComplianceLevel(Enum):
    """Compliance levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ComplianceRequirement:
    """Compliance requirement definition."""
    id: str
    standard: ComplianceStandard
    title: str
    description: str
    level: ComplianceLevel
    category: str
    controls: List[str]
    evidence_required: List[str]
    assessment_frequency: str
    last_assessed: Optional[datetime] = None
    status: ComplianceStatus = ComplianceStatus.NOT_ASSESSED
    notes: str = ""

@dataclass
class ComplianceAssessment:
    """Compliance assessment definition."""
    id: str
    requirement_id: str
    assessor: str
    assessment_date: datetime
    status: ComplianceStatus
    evidence: List[str]
    findings: List[str]
    recommendations: List[str]
    next_assessment: datetime
    score: Optional[float] = None

@dataclass
class ComplianceAudit:
    """Compliance audit definition."""
    id: str
    standard: ComplianceStandard
    audit_date: datetime
    auditor: str
    scope: str
    findings: List[Dict[str, Any]]
    recommendations: List[str]
    overall_status: ComplianceStatus
    next_audit: datetime

class ComplianceManager:
    """Advanced compliance management system for BUL system."""
    
    def __init__(self, config: Optional[BULConfig] = None):
        self.config = config or BULConfig()
        self.requirements = {}
        self.assessments = {}
        self.audits = {}
        self.init_compliance_environment()
        self.load_requirements()
        self.load_assessments()
        self.load_audits()
    
    def init_compliance_environment(self):
        """Initialize compliance environment."""
        print("📋 Initializing compliance management environment...")
        
        # Create compliance directories
        self.compliance_dir = Path("compliance")
        self.compliance_dir.mkdir(exist_ok=True)
        
        self.assessments_dir = Path("compliance_assessments")
        self.assessments_dir.mkdir(exist_ok=True)
        
        self.audits_dir = Path("compliance_audits")
        self.audits_dir.mkdir(exist_ok=True)
        
        self.evidence_dir = Path("compliance_evidence")
        self.evidence_dir.mkdir(exist_ok=True)
        
        # Initialize database
        self.init_compliance_database()
        
        print("✅ Compliance management environment initialized")
    
    def init_compliance_database(self):
        """Initialize compliance database."""
        conn = sqlite3.connect("compliance.db")
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS compliance_requirements (
                id TEXT PRIMARY KEY,
                standard TEXT,
                title TEXT,
                description TEXT,
                level TEXT,
                category TEXT,
                controls TEXT,
                evidence_required TEXT,
                assessment_frequency TEXT,
                last_assessed DATETIME,
                status TEXT,
                notes TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS compliance_assessments (
                id TEXT PRIMARY KEY,
                requirement_id TEXT,
                assessor TEXT,
                assessment_date DATETIME,
                status TEXT,
                evidence TEXT,
                findings TEXT,
                recommendations TEXT,
                next_assessment DATETIME,
                score REAL,
                FOREIGN KEY (requirement_id) REFERENCES compliance_requirements (id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS compliance_audits (
                id TEXT PRIMARY KEY,
                standard TEXT,
                audit_date DATETIME,
                auditor TEXT,
                scope TEXT,
                findings TEXT,
                recommendations TEXT,
                overall_status TEXT,
                next_audit DATETIME
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def load_requirements(self):
        """Load existing compliance requirements."""
        conn = sqlite3.connect("compliance.db")
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM compliance_requirements")
        rows = cursor.fetchall()
        
        for row in rows:
            requirement = ComplianceRequirement(
                id=row[0],
                standard=ComplianceStandard(row[1]),
                title=row[2],
                description=row[3],
                level=ComplianceLevel(row[4]),
                category=row[5],
                controls=json.loads(row[6]),
                evidence_required=json.loads(row[7]),
                assessment_frequency=row[8],
                last_assessed=datetime.fromisoformat(row[9]) if row[9] else None,
                status=ComplianceStatus(row[10]) if row[10] else ComplianceStatus.NOT_ASSESSED,
                notes=row[11] or ""
            )
            self.requirements[requirement.id] = requirement
        
        conn.close()
        
        # Create default requirements if none exist
        if not self.requirements:
            self.create_default_requirements()
        
        print(f"✅ Loaded {len(self.requirements)} compliance requirements")
    
    def load_assessments(self):
        """Load existing compliance assessments."""
        conn = sqlite3.connect("compliance.db")
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM compliance_assessments")
        rows = cursor.fetchall()
        
        for row in rows:
            assessment = ComplianceAssessment(
                id=row[0],
                requirement_id=row[1],
                assessor=row[2],
                assessment_date=datetime.fromisoformat(row[3]),
                status=ComplianceStatus(row[4]),
                evidence=json.loads(row[5]),
                findings=json.loads(row[6]),
                recommendations=json.loads(row[7]),
                next_assessment=datetime.fromisoformat(row[8]),
                score=row[9]
            )
            self.assessments[assessment.id] = assessment
        
        conn.close()
        
        print(f"✅ Loaded {len(self.assessments)} compliance assessments")
    
    def load_audits(self):
        """Load existing compliance audits."""
        conn = sqlite3.connect("compliance.db")
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM compliance_audits")
        rows = cursor.fetchall()
        
        for row in rows:
            audit = ComplianceAudit(
                id=row[0],
                standard=ComplianceStandard(row[1]),
                audit_date=datetime.fromisoformat(row[2]),
                auditor=row[3],
                scope=row[4],
                findings=json.loads(row[5]),
                recommendations=json.loads(row[6]),
                overall_status=ComplianceStatus(row[7]),
                next_audit=datetime.fromisoformat(row[8])
            )
            self.audits[audit.id] = audit
        
        conn.close()
        
        print(f"✅ Loaded {len(self.audits)} compliance audits")
    
    def create_default_requirements(self):
        """Create default compliance requirements."""
        default_requirements = [
            {
                'id': 'gdpr_data_protection',
                'standard': ComplianceStandard.GDPR,
                'title': 'Data Protection by Design and by Default',
                'description': 'Implement appropriate technical and organizational measures to ensure data protection',
                'level': ComplianceLevel.HIGH,
                'category': 'Data Protection',
                'controls': [
                    'Data minimization',
                    'Purpose limitation',
                    'Storage limitation',
                    'Accuracy',
                    'Security of processing'
                ],
                'evidence_required': [
                    'Data protection impact assessment',
                    'Privacy by design documentation',
                    'Data processing records',
                    'Security measures documentation'
                ],
                'assessment_frequency': 'Annual'
            },
            {
                'id': 'gdpr_consent_management',
                'standard': ComplianceStandard.GDPR,
                'title': 'Consent Management',
                'description': 'Implement mechanisms for obtaining and managing user consent',
                'level': ComplianceLevel.HIGH,
                'category': 'Consent Management',
                'controls': [
                    'Clear consent request',
                    'Granular consent options',
                    'Easy withdrawal mechanism',
                    'Consent records maintenance'
                ],
                'evidence_required': [
                    'Consent forms',
                    'Consent management system',
                    'Withdrawal mechanisms',
                    'Consent audit logs'
                ],
                'assessment_frequency': 'Quarterly'
            },
            {
                'id': 'hipaa_administrative_safeguards',
                'standard': ComplianceStandard.HIPAA,
                'title': 'Administrative Safeguards',
                'description': 'Implement administrative safeguards for protected health information',
                'level': ComplianceLevel.CRITICAL,
                'category': 'Administrative Safeguards',
                'controls': [
                    'Security officer designation',
                    'Workforce training',
                    'Access management',
                    'Information access management',
                    'Security awareness training'
                ],
                'evidence_required': [
                    'Security officer appointment',
                    'Training records',
                    'Access control policies',
                    'Incident response procedures'
                ],
                'assessment_frequency': 'Annual'
            },
            {
                'id': 'sox_financial_controls',
                'standard': ComplianceStandard.SOX,
                'title': 'Financial Controls',
                'description': 'Implement internal controls over financial reporting',
                'level': ComplianceLevel.CRITICAL,
                'category': 'Financial Controls',
                'controls': [
                    'Control environment',
                    'Risk assessment',
                    'Control activities',
                    'Information and communication',
                    'Monitoring'
                ],
                'evidence_required': [
                    'Control documentation',
                    'Testing results',
                    'Management certifications',
                    'Audit reports'
                ],
                'assessment_frequency': 'Quarterly'
            },
            {
                'id': 'pci_dss_data_security',
                'standard': ComplianceStandard.PCI_DSS,
                'title': 'Data Security Standards',
                'description': 'Implement security measures for cardholder data',
                'level': ComplianceLevel.CRITICAL,
                'category': 'Data Security',
                'controls': [
                    'Firewall configuration',
                    'Default password protection',
                    'Cardholder data protection',
                    'Encryption of data transmission',
                    'Antivirus software',
                    'Secure systems and applications'
                ],
                'evidence_required': [
                    'Network diagrams',
                    'Firewall rules',
                    'Encryption documentation',
                    'Vulnerability scans',
                    'Penetration tests'
                ],
                'assessment_frequency': 'Annual'
            }
        ]
        
        for req_data in default_requirements:
            self.create_requirement(
                requirement_id=req_data['id'],
                standard=req_data['standard'],
                title=req_data['title'],
                description=req_data['description'],
                level=req_data['level'],
                category=req_data['category'],
                controls=req_data['controls'],
                evidence_required=req_data['evidence_required'],
                assessment_frequency=req_data['assessment_frequency']
            )
    
    def create_requirement(self, requirement_id: str, standard: ComplianceStandard,
                          title: str, description: str, level: ComplianceLevel,
                          category: str, controls: List[str], evidence_required: List[str],
                          assessment_frequency: str) -> ComplianceRequirement:
        """Create a new compliance requirement."""
        requirement = ComplianceRequirement(
            id=requirement_id,
            standard=standard,
            title=title,
            description=description,
            level=level,
            category=category,
            controls=controls,
            evidence_required=evidence_required,
            assessment_frequency=assessment_frequency,
            status=ComplianceStatus.NOT_ASSESSED
        )
        
        self.requirements[requirement_id] = requirement
        
        # Save to database
        conn = sqlite3.connect("compliance.db")
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO compliance_requirements 
            (id, standard, title, description, level, category, controls, 
             evidence_required, assessment_frequency, status, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (requirement_id, standard.value, title, description, level.value,
              category, json.dumps(controls), json.dumps(evidence_required),
              assessment_frequency, ComplianceStatus.NOT_ASSESSED.value, ""))
        
        conn.commit()
        conn.close()
        
        print(f"✅ Created compliance requirement: {title}")
        return requirement
    
    def create_assessment(self, assessment_id: str, requirement_id: str, assessor: str,
                         status: ComplianceStatus, evidence: List[str],
                         findings: List[str], recommendations: List[str]) -> ComplianceAssessment:
        """Create a new compliance assessment."""
        if requirement_id not in self.requirements:
            raise ValueError(f"Requirement {requirement_id} not found")
        
        requirement = self.requirements[requirement_id]
        
        # Calculate next assessment date
        next_assessment = self._calculate_next_assessment(requirement.assessment_frequency)
        
        assessment = ComplianceAssessment(
            id=assessment_id,
            requirement_id=requirement_id,
            assessor=assessor,
            assessment_date=datetime.now(),
            status=status,
            evidence=evidence,
            findings=findings,
            recommendations=recommendations,
            next_assessment=next_assessment
        )
        
        self.assessments[assessment_id] = assessment
        
        # Update requirement status
        requirement.status = status
        requirement.last_assessed = assessment.assessment_date
        self._save_requirement(requirement)
        
        # Save to database
        conn = sqlite3.connect("compliance.db")
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO compliance_assessments 
            (id, requirement_id, assessor, assessment_date, status, evidence, 
             findings, recommendations, next_assessment, score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (assessment_id, requirement_id, assessor, assessment.assessment_date.isoformat(),
              status.value, json.dumps(evidence), json.dumps(findings),
              json.dumps(recommendations), next_assessment.isoformat(), assessment.score))
        
        conn.commit()
        conn.close()
        
        print(f"✅ Created compliance assessment: {requirement.title}")
        return assessment
    
    def create_audit(self, audit_id: str, standard: ComplianceStandard, auditor: str,
                    scope: str, findings: List[Dict[str, Any]], recommendations: List[str],
                    overall_status: ComplianceStatus) -> ComplianceAudit:
        """Create a new compliance audit."""
        # Calculate next audit date (typically annual)
        next_audit = datetime.now() + timedelta(days=365)
        
        audit = ComplianceAudit(
            id=audit_id,
            standard=standard,
            audit_date=datetime.now(),
            auditor=auditor,
            scope=scope,
            findings=findings,
            recommendations=recommendations,
            overall_status=overall_status,
            next_audit=next_audit
        )
        
        self.audits[audit_id] = audit
        
        # Save to database
        conn = sqlite3.connect("compliance.db")
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO compliance_audits 
            (id, standard, audit_date, auditor, scope, findings, 
             recommendations, overall_status, next_audit)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (audit_id, standard.value, audit.audit_date.isoformat(), auditor,
              scope, json.dumps(findings), json.dumps(recommendations),
              overall_status.value, next_audit.isoformat()))
        
        conn.commit()
        conn.close()
        
        print(f"✅ Created compliance audit: {standard.value.upper()}")
        return audit
    
    def assess_requirement(self, requirement_id: str, assessor: str,
                          evidence: List[str], findings: List[str],
                          recommendations: List[str]) -> ComplianceAssessment:
        """Assess a compliance requirement."""
        if requirement_id not in self.requirements:
            raise ValueError(f"Requirement {requirement_id} not found")
        
        requirement = self.requirements[requirement_id]
        
        # Determine compliance status based on findings
        status = self._determine_compliance_status(findings, requirement.level)
        
        # Calculate compliance score
        score = self._calculate_compliance_score(requirement, evidence, findings)
        
        # Create assessment
        assessment_id = f"assess_{requirement_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        assessment = self.create_assessment(
            assessment_id=assessment_id,
            requirement_id=requirement_id,
            assessor=assessor,
            status=status,
            evidence=evidence,
            findings=findings,
            recommendations=recommendations
        )
        
        assessment.score = score
        self._save_assessment(assessment)
        
        return assessment
    
    def generate_compliance_report(self, standard: Optional[ComplianceStandard] = None) -> str:
        """Generate compliance report."""
        report = f"""
BUL Compliance Management Report
===============================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        if standard:
            report += f"Standard: {standard.value.upper()}\n"
        
        # Filter requirements by standard if specified
        requirements = self.requirements.values()
        if standard:
            requirements = [r for r in requirements if r.standard == standard]
        
        # Group by standard
        by_standard = defaultdict(list)
        for req in requirements:
            by_standard[req.standard].append(req)
        
        for std, reqs in by_standard.items():
            report += f"""
{std.value.upper()} COMPLIANCE
{'=' * (len(std.value) + 12)}
Total Requirements: {len(reqs)}
"""
            
            # Status summary
            status_counts = defaultdict(int)
            for req in reqs:
                status_counts[req.status] += 1
            
            report += f"""
Status Summary:
  Compliant: {status_counts[ComplianceStatus.COMPLIANT]}
  Non-Compliant: {status_counts[ComplianceStatus.NON_COMPLIANT]}
  Partially Compliant: {status_counts[ComplianceStatus.PARTIALLY_COMPLIANT]}
  Not Assessed: {status_counts[ComplianceStatus.NOT_ASSESSED]}
  In Progress: {status_counts[ComplianceStatus.IN_PROGRESS]}
"""
            
            # Requirements details
            for req in reqs:
                report += f"""
{req.title} ({req.id}):
  Level: {req.level.value}
  Category: {req.category}
  Status: {req.status.value}
  Last Assessed: {req.last_assessed.strftime('%Y-%m-%d') if req.last_assessed else 'Never'}
  Controls: {len(req.controls)}
  Evidence Required: {len(req.evidence_required)}
"""
        
        # Recent assessments
        recent_assessments = sorted(
            self.assessments.values(),
            key=lambda x: x.assessment_date,
            reverse=True
        )[:10]
        
        if recent_assessments:
            report += f"""
RECENT ASSESSMENTS
------------------
"""
            for assessment in recent_assessments:
                req = self.requirements[assessment.requirement_id]
                report += f"""
{req.title} - {assessment.assessment_date.strftime('%Y-%m-%d')}
  Assessor: {assessment.assessor}
  Status: {assessment.status.value}
  Score: {assessment.score:.1f}% if assessment.score else 'N/A'
  Findings: {len(assessment.findings)}
  Recommendations: {len(assessment.recommendations)}
"""
        
        # Upcoming assessments
        upcoming_assessments = []
        for req in self.requirements.values():
            if req.last_assessed:
                next_assessment = self._calculate_next_assessment(req.assessment_frequency, req.last_assessed)
                if next_assessment <= datetime.now() + timedelta(days=30):
                    upcoming_assessments.append((req, next_assessment))
        
        if upcoming_assessments:
            report += f"""
UPCOMING ASSESSMENTS (Next 30 Days)
-----------------------------------
"""
            for req, next_date in sorted(upcoming_assessments, key=lambda x: x[1]):
                report += f"""
{req.title} - {next_date.strftime('%Y-%m-%d')}
  Standard: {req.standard.value.upper()}
  Level: {req.level.value}
  Frequency: {req.assessment_frequency}
"""
        
        return report
    
    def get_compliance_stats(self) -> Dict[str, Any]:
        """Get compliance statistics."""
        total_requirements = len(self.requirements)
        total_assessments = len(self.assessments)
        total_audits = len(self.audits)
        
        # Status counts
        status_counts = defaultdict(int)
        for req in self.requirements.values():
            status_counts[req.status] += 1
        
        # Standard counts
        standard_counts = defaultdict(int)
        for req in self.requirements.values():
            standard_counts[req.standard] += 1
        
        # Level counts
        level_counts = defaultdict(int)
        for req in self.requirements.values():
            level_counts[req.level] += 1
        
        # Overdue assessments
        overdue_count = 0
        for req in self.requirements.values():
            if req.last_assessed:
                next_assessment = self._calculate_next_assessment(req.assessment_frequency, req.last_assessed)
                if next_assessment < datetime.now():
                    overdue_count += 1
        
        return {
            'total_requirements': total_requirements,
            'total_assessments': total_assessments,
            'total_audits': total_audits,
            'status_counts': {k.value: v for k, v in status_counts.items()},
            'standard_counts': {k.value: v for k, v in standard_counts.items()},
            'level_counts': {k.value: v for k, v in level_counts.items()},
            'overdue_assessments': overdue_count,
            'compliance_rate': (status_counts[ComplianceStatus.COMPLIANT] / total_requirements * 100) if total_requirements > 0 else 0
        }
    
    def _calculate_next_assessment(self, frequency: str, last_assessed: Optional[datetime] = None) -> datetime:
        """Calculate next assessment date."""
        base_date = last_assessed or datetime.now()
        
        if frequency.lower() == 'annual':
            return base_date + timedelta(days=365)
        elif frequency.lower() == 'quarterly':
            return base_date + timedelta(days=90)
        elif frequency.lower() == 'monthly':
            return base_date + timedelta(days=30)
        elif frequency.lower() == 'weekly':
            return base_date + timedelta(days=7)
        else:
            return base_date + timedelta(days=365)  # Default to annual
    
    def _determine_compliance_status(self, findings: List[str], level: ComplianceLevel) -> ComplianceStatus:
        """Determine compliance status based on findings."""
        if not findings:
            return ComplianceStatus.COMPLIANT
        
        # Count critical and high findings
        critical_findings = len([f for f in findings if 'critical' in f.lower() or 'high' in f.lower()])
        medium_findings = len([f for f in findings if 'medium' in f.lower()])
        low_findings = len([f for f in findings if 'low' in f.lower()])
        
        if critical_findings > 0:
            return ComplianceStatus.NON_COMPLIANT
        elif medium_findings > 2 or (medium_findings > 0 and level == ComplianceLevel.CRITICAL):
            return ComplianceStatus.PARTIALLY_COMPLIANT
        elif low_findings > 5:
            return ComplianceStatus.PARTIALLY_COMPLIANT
        else:
            return ComplianceStatus.COMPLIANT
    
    def _calculate_compliance_score(self, requirement: ComplianceRequirement, 
                                   evidence: List[str], findings: List[str]) -> float:
        """Calculate compliance score for a requirement."""
        # Base score
        score = 100.0
        
        # Deduct for missing evidence
        evidence_ratio = len(evidence) / len(requirement.evidence_required) if requirement.evidence_required else 1.0
        score *= evidence_ratio
        
        # Deduct for findings
        for finding in findings:
            if 'critical' in finding.lower():
                score -= 25
            elif 'high' in finding.lower():
                score -= 15
            elif 'medium' in finding.lower():
                score -= 10
            elif 'low' in finding.lower():
                score -= 5
        
        # Ensure score is not negative
        return max(0.0, score)
    
    def _save_requirement(self, requirement: ComplianceRequirement):
        """Save requirement to database."""
        conn = sqlite3.connect("compliance.db")
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE compliance_requirements 
            SET last_assessed = ?, status = ?, notes = ?
            WHERE id = ?
        ''', (requirement.last_assessed.isoformat() if requirement.last_assessed else None,
              requirement.status.value, requirement.notes, requirement.id))
        
        conn.commit()
        conn.close()
    
    def _save_assessment(self, assessment: ComplianceAssessment):
        """Save assessment to database."""
        conn = sqlite3.connect("compliance.db")
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE compliance_assessments 
            SET score = ?
            WHERE id = ?
        ''', (assessment.score, assessment.id))
        
        conn.commit()
        conn.close()

def main():
    """Main compliance manager function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="BUL Compliance Manager")
    parser.add_argument("--create-requirement", help="Create a new compliance requirement")
    parser.add_argument("--create-assessment", help="Create a new compliance assessment")
    parser.add_argument("--create-audit", help="Create a new compliance audit")
    parser.add_argument("--assess-requirement", help="Assess a compliance requirement")
    parser.add_argument("--list-requirements", action="store_true", help="List all requirements")
    parser.add_argument("--list-assessments", action="store_true", help="List all assessments")
    parser.add_argument("--list-audits", action="store_true", help="List all audits")
    parser.add_argument("--stats", action="store_true", help="Show compliance statistics")
    parser.add_argument("--report", action="store_true", help="Generate compliance report")
    parser.add_argument("--title", help="Title for requirement")
    parser.add_argument("--description", help="Description for requirement")
    parser.add_argument("--standard", choices=['gdpr', 'hipaa', 'sox', 'pci_dss', 'iso27001', 'soc2', 'ccpa', 'ferpa'],
                       help="Compliance standard")
    parser.add_argument("--level", choices=['low', 'medium', 'high', 'critical'],
                       help="Compliance level")
    parser.add_argument("--category", help="Category for requirement")
    parser.add_argument("--controls", help="JSON string of controls")
    parser.add_argument("--evidence", help="JSON string of evidence")
    parser.add_argument("--frequency", help="Assessment frequency")
    parser.add_argument("--assessor", help="Assessor name")
    parser.add_argument("--auditor", help="Auditor name")
    parser.add_argument("--scope", help="Audit scope")
    parser.add_argument("--findings", help="JSON string of findings")
    parser.add_argument("--recommendations", help="JSON string of recommendations")
    parser.add_argument("--status", choices=['compliant', 'non_compliant', 'partially_compliant', 'not_assessed', 'in_progress'],
                       help="Compliance status")
    
    args = parser.parse_args()
    
    compliance_manager = ComplianceManager()
    
    print("📋 BUL Compliance Manager")
    print("=" * 30)
    
    if args.create_requirement:
        if not all([args.title, args.description, args.standard, args.level, args.category]):
            print("❌ Error: --title, --description, --standard, --level, and --category are required")
            return 1
        
        controls = []
        if args.controls:
            try:
                controls = json.loads(args.controls)
            except json.JSONDecodeError:
                print("❌ Error: Invalid JSON in --controls")
                return 1
        
        evidence = []
        if args.evidence:
            try:
                evidence = json.loads(args.evidence)
            except json.JSONDecodeError:
                print("❌ Error: Invalid JSON in --evidence")
                return 1
        
        requirement = compliance_manager.create_requirement(
            requirement_id=args.create_requirement,
            standard=ComplianceStandard(args.standard),
            title=args.title,
            description=args.description,
            level=ComplianceLevel(args.level),
            category=args.category,
            controls=controls,
            evidence_required=evidence,
            assessment_frequency=args.frequency or "Annual"
        )
        print(f"✅ Created requirement: {requirement.title}")
    
    elif args.create_assessment:
        if not all([args.assessor, args.status]):
            print("❌ Error: --assessor and --status are required")
            return 1
        
        evidence = []
        if args.evidence:
            try:
                evidence = json.loads(args.evidence)
            except json.JSONDecodeError:
                print("❌ Error: Invalid JSON in --evidence")
                return 1
        
        findings = []
        if args.findings:
            try:
                findings = json.loads(args.findings)
            except json.JSONDecodeError:
                print("❌ Error: Invalid JSON in --findings")
                return 1
        
        recommendations = []
        if args.recommendations:
            try:
                recommendations = json.loads(args.recommendations)
            except json.JSONDecodeError:
                print("❌ Error: Invalid JSON in --recommendations")
                return 1
        
        assessment = compliance_manager.create_assessment(
            assessment_id=args.create_assessment,
            requirement_id=args.create_requirement or "gdpr_data_protection",
            assessor=args.assessor,
            status=ComplianceStatus(args.status),
            evidence=evidence,
            findings=findings,
            recommendations=recommendations
        )
        print(f"✅ Created assessment: {assessment.id}")
    
    elif args.create_audit:
        if not all([args.standard, args.auditor, args.scope, args.status]):
            print("❌ Error: --standard, --auditor, --scope, and --status are required")
            return 1
        
        findings = []
        if args.findings:
            try:
                findings = json.loads(args.findings)
            except json.JSONDecodeError:
                print("❌ Error: Invalid JSON in --findings")
                return 1
        
        recommendations = []
        if args.recommendations:
            try:
                recommendations = json.loads(args.recommendations)
            except json.JSONDecodeError:
                print("❌ Error: Invalid JSON in --recommendations")
                return 1
        
        audit = compliance_manager.create_audit(
            audit_id=args.create_audit,
            standard=ComplianceStandard(args.standard),
            auditor=args.auditor,
            scope=args.scope,
            findings=findings,
            recommendations=recommendations,
            overall_status=ComplianceStatus(args.status)
        )
        print(f"✅ Created audit: {audit.id}")
    
    elif args.assess_requirement:
        if not all([args.assessor]):
            print("❌ Error: --assessor is required")
            return 1
        
        evidence = []
        if args.evidence:
            try:
                evidence = json.loads(args.evidence)
            except json.JSONDecodeError:
                print("❌ Error: Invalid JSON in --evidence")
                return 1
        
        findings = []
        if args.findings:
            try:
                findings = json.loads(args.findings)
            except json.JSONDecodeError:
                print("❌ Error: Invalid JSON in --findings")
                return 1
        
        recommendations = []
        if args.recommendations:
            try:
                recommendations = json.loads(args.recommendations)
            except json.JSONDecodeError:
                print("❌ Error: Invalid JSON in --recommendations")
                return 1
        
        assessment = compliance_manager.assess_requirement(
            requirement_id=args.assess_requirement,
            assessor=args.assessor,
            evidence=evidence,
            findings=findings,
            recommendations=recommendations
        )
        print(f"✅ Assessed requirement: {assessment.requirement_id}")
        print(f"   Status: {assessment.status.value}")
        print(f"   Score: {assessment.score:.1f}%")
    
    elif args.list_requirements:
        requirements = compliance_manager.requirements
        if requirements:
            print(f"\n📋 Compliance Requirements ({len(requirements)}):")
            print("-" * 60)
            for req_id, req in requirements.items():
                print(f"{req.title} ({req_id}):")
                print(f"  Standard: {req.standard.value.upper()}")
                print(f"  Level: {req.level.value}")
                print(f"  Status: {req.status.value}")
                print(f"  Category: {req.category}")
                print()
        else:
            print("No requirements found.")
    
    elif args.list_assessments:
        assessments = compliance_manager.assessments
        if assessments:
            print(f"\n📋 Compliance Assessments ({len(assessments)}):")
            print("-" * 60)
            for assess_id, assess in assessments.items():
                req = compliance_manager.requirements[assess.requirement_id]
                print(f"{req.title} - {assess.assessment_date.strftime('%Y-%m-%d')} ({assess_id}):")
                print(f"  Assessor: {assess.assessor}")
                print(f"  Status: {assess.status.value}")
                print(f"  Score: {assess.score:.1f}% if assess.score else 'N/A'")
                print()
        else:
            print("No assessments found.")
    
    elif args.list_audits:
        audits = compliance_manager.audits
        if audits:
            print(f"\n📋 Compliance Audits ({len(audits)}):")
            print("-" * 60)
            for audit_id, audit in audits.items():
                print(f"{audit.standard.value.upper()} - {audit.audit_date.strftime('%Y-%m-%d')} ({audit_id}):")
                print(f"  Auditor: {audit.auditor}")
                print(f"  Status: {audit.overall_status.value}")
                print(f"  Scope: {audit.scope}")
                print()
        else:
            print("No audits found.")
    
    elif args.stats:
        stats = compliance_manager.get_compliance_stats()
        print(f"\n📊 Compliance Statistics:")
        print(f"   Total Requirements: {stats['total_requirements']}")
        print(f"   Total Assessments: {stats['total_assessments']}")
        print(f"   Total Audits: {stats['total_audits']}")
        print(f"   Compliance Rate: {stats['compliance_rate']:.1f}%")
        print(f"   Overdue Assessments: {stats['overdue_assessments']}")
        print(f"   Status Distribution:")
        for status, count in stats['status_counts'].items():
            print(f"     {status.title()}: {count}")
        print(f"   Standard Distribution:")
        for standard, count in stats['standard_counts'].items():
            print(f"     {standard.upper()}: {count}")
    
    elif args.report:
        report = compliance_manager.generate_compliance_report()
        print(report)
        
        # Save report
        report_file = f"compliance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"📄 Report saved to: {report_file}")
    
    else:
        # Show quick overview
        stats = compliance_manager.get_compliance_stats()
        print(f"📋 Requirements: {stats['total_requirements']}")
        print(f"📋 Assessments: {stats['total_assessments']}")
        print(f"📋 Audits: {stats['total_audits']}")
        print(f"📊 Compliance Rate: {stats['compliance_rate']:.1f}%")
        print(f"⚠️ Overdue Assessments: {stats['overdue_assessments']}")
        print(f"\n💡 Use --list-requirements to see all requirements")
        print(f"💡 Use --create-assessment to create a new assessment")
        print(f"💡 Use --report to generate compliance report")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
