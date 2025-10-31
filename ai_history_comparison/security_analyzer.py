"""
Advanced Security and Privacy Analysis System for AI History Comparison
Sistema avanzado de análisis de seguridad y privacidad para análisis de historial de IA
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import re
import hashlib
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    """Niveles de seguridad"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    SECURE = "secure"

class PrivacyRisk(Enum):
    """Riesgos de privacidad"""
    PII_EXPOSURE = "pii_exposure"
    DATA_LEAKAGE = "data_leakage"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    DATA_BREACH = "data_breach"
    COMPLIANCE_VIOLATION = "compliance_violation"

@dataclass
class SecurityIssue:
    """Problema de seguridad"""
    id: str
    issue_type: str
    severity: SecurityLevel
    description: str
    location: str
    recommendation: str
    detected_at: datetime = field(default_factory=datetime.now)

@dataclass
class PrivacyAnalysis:
    """Análisis de privacidad"""
    id: str
    document_id: str
    pii_detected: List[str]
    privacy_risks: List[PrivacyRisk]
    compliance_status: Dict[str, bool]
    risk_score: float
    recommendations: List[str]
    created_at: datetime = field(default_factory=datetime.now)

class AdvancedSecurityAnalyzer:
    """
    Analizador avanzado de seguridad y privacidad
    """
    
    def __init__(self):
        self.security_issues: Dict[str, SecurityIssue] = {}
        self.privacy_analyses: Dict[str, PrivacyAnalysis] = {}
        
        # Patrones de PII
        self.pii_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            'ip_address': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
        }
        
        # Patrones de seguridad
        self.security_patterns = {
            'password': r'password\s*[:=]\s*["\']?[^"\'\s]+["\']?',
            'api_key': r'api[_-]?key\s*[:=]\s*["\']?[^"\'\s]+["\']?',
            'secret': r'secret\s*[:=]\s*["\']?[^"\'\s]+["\']?',
            'token': r'token\s*[:=]\s*["\']?[^"\'\s]+["\']?'
        }
    
    async def analyze_security(self, text: str, document_id: str) -> List[SecurityIssue]:
        """Analizar seguridad del texto"""
        issues = []
        
        try:
            # Detectar credenciales expuestas
            for pattern_name, pattern in self.security_patterns.items():
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    issue = SecurityIssue(
                        id=f"security_{pattern_name}_{document_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        issue_type=pattern_name,
                        severity=SecurityLevel.CRITICAL,
                        description=f"Potential {pattern_name} exposure detected",
                        location=f"Document {document_id}",
                        recommendation=f"Remove or mask {pattern_name} information"
                    )
                    issues.append(issue)
            
            # Detectar URLs sospechosas
            url_pattern = r'https?://[^\s]+'
            urls = re.findall(url_pattern, text)
            for url in urls:
                if any(suspicious in url.lower() for suspicious in ['localhost', '127.0.0.1', 'admin', 'test']):
                    issue = SecurityIssue(
                        id=f"security_url_{document_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        issue_type="suspicious_url",
                        severity=SecurityLevel.MEDIUM,
                        description=f"Suspicious URL detected: {url}",
                        location=f"Document {document_id}",
                        recommendation="Review URL for security implications"
                    )
                    issues.append(issue)
            
            # Almacenar issues
            for issue in issues:
                self.security_issues[issue.id] = issue
            
            return issues
            
        except Exception as e:
            logger.error(f"Error analyzing security: {e}")
            return []
    
    async def analyze_privacy(self, text: str, document_id: str) -> PrivacyAnalysis:
        """Analizar privacidad del texto"""
        try:
            # Detectar PII
            pii_detected = []
            for pii_type, pattern in self.pii_patterns.items():
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    pii_detected.extend([f"{pii_type}: {match}" for match in matches])
            
            # Determinar riesgos de privacidad
            privacy_risks = []
            if any('email' in pii for pii in pii_detected):
                privacy_risks.append(PrivacyRisk.PII_EXPOSURE)
            if any('ssn' in pii for pii in pii_detected):
                privacy_risks.append(PrivacyRisk.DATA_BREACH)
            
            # Calcular score de riesgo
            risk_score = len(pii_detected) * 0.2 + len(privacy_risks) * 0.3
            
            # Estado de cumplimiento
            compliance_status = {
                "gdpr": len(pii_detected) == 0,
                "ccpa": len(pii_detected) == 0,
                "hipaa": 'ssn' not in ' '.join(pii_detected).lower()
            }
            
            # Generar recomendaciones
            recommendations = []
            if pii_detected:
                recommendations.append("Remove or mask PII information")
            if privacy_risks:
                recommendations.append("Implement data protection measures")
            
            # Crear análisis
            analysis = PrivacyAnalysis(
                id=f"privacy_{document_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                document_id=document_id,
                pii_detected=pii_detected,
                privacy_risks=privacy_risks,
                compliance_status=compliance_status,
                risk_score=min(1.0, risk_score),
                recommendations=recommendations
            )
            
            # Almacenar análisis
            self.privacy_analyses[analysis.id] = analysis
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing privacy: {e}")
            raise
    
    async def get_security_summary(self) -> Dict[str, Any]:
        """Obtener resumen de seguridad"""
        return {
            "total_security_issues": len(self.security_issues),
            "total_privacy_analyses": len(self.privacy_analyses),
            "critical_issues": len([i for i in self.security_issues.values() if i.severity == SecurityLevel.CRITICAL]),
            "high_issues": len([i for i in self.security_issues.values() if i.severity == SecurityLevel.HIGH]),
            "medium_issues": len([i for i in self.security_issues.values() if i.severity == SecurityLevel.MEDIUM]),
            "low_issues": len([i for i in self.security_issues.values() if i.severity == SecurityLevel.LOW])
        }
    
    async def export_security_data(self, filepath: str = None) -> str:
        """Exportar datos de seguridad"""
        try:
            if filepath is None:
                filepath = f"exports/security_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            import os
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            export_data = {
                "security_issues": {
                    issue_id: {
                        "issue_type": issue.issue_type,
                        "severity": issue.severity.value,
                        "description": issue.description,
                        "location": issue.location,
                        "recommendation": issue.recommendation,
                        "detected_at": issue.detected_at.isoformat()
                    }
                    for issue_id, issue in self.security_issues.items()
                },
                "privacy_analyses": {
                    analysis_id: {
                        "document_id": analysis.document_id,
                        "pii_detected": analysis.pii_detected,
                        "privacy_risks": [risk.value for risk in analysis.privacy_risks],
                        "compliance_status": analysis.compliance_status,
                        "risk_score": analysis.risk_score,
                        "recommendations": analysis.recommendations,
                        "created_at": analysis.created_at.isoformat()
                    }
                    for analysis_id, analysis in self.privacy_analyses.items()
                },
                "summary": await self.get_security_summary(),
                "exported_at": datetime.now().isoformat()
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Security data exported to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error exporting security data: {e}")
            raise

























