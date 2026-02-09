"""
Security Audit System
=====================

Sistema de auditoría de seguridad.
"""

import logging
import hashlib
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SecurityIssue:
    """Problema de seguridad."""
    issue_id: str
    severity: str  # "critical", "high", "medium", "low"
    category: str
    description: str
    recommendation: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None


class SecurityAuditor:
    """
    Auditor de seguridad.
    
    Audita el código en busca de problemas de seguridad.
    """
    
    def __init__(self):
        """Inicializar auditor de seguridad."""
        self.issues: List[SecurityIssue] = []
        self.audit_history: List[Dict[str, Any]] = []
    
    def audit_file(self, file_path: str) -> List[SecurityIssue]:
        """
        Auditar archivo.
        
        Args:
            file_path: Ruta del archivo
            
        Returns:
            Lista de problemas encontrados
        """
        path = Path(file_path)
        if not path.exists():
            return []
        
        issues = []
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
            
            # Buscar patrones de seguridad
            for i, line in enumerate(lines, 1):
                # Hardcoded passwords/secrets
                if any(keyword in line.lower() for keyword in ['password=', 'secret=', 'api_key=', 'token=']):
                    if 'os.getenv' not in line and 'os.environ' not in line:
                        issues.append(SecurityIssue(
                            issue_id=f"{file_path}:{i}",
                            severity="high",
                            category="hardcoded_secret",
                            description=f"Potential hardcoded secret in line {i}",
                            recommendation="Use environment variables or secure configuration",
                            file_path=file_path,
                            line_number=i
                        ))
                
                # SQL injection risks
                if 'execute(' in line and '?' not in line and '%s' not in line:
                    if 'sql' in line.lower() or 'query' in line.lower():
                        issues.append(SecurityIssue(
                            issue_id=f"{file_path}:{i}",
                            severity="high",
                            category="sql_injection",
                            description=f"Potential SQL injection risk in line {i}",
                            recommendation="Use parameterized queries",
                            file_path=file_path,
                            line_number=i
                        ))
                
                # eval/exec usage
                if 'eval(' in line or 'exec(' in line:
                    issues.append(SecurityIssue(
                        issue_id=f"{file_path}:{i}",
                        severity="critical",
                        category="code_injection",
                        description=f"Use of eval/exec in line {i}",
                        recommendation="Avoid eval/exec, use safer alternatives",
                        file_path=file_path,
                        line_number=i
                    ))
                
                # Unsafe deserialization
                if 'pickle.load' in line or 'yaml.load' in line:
                    if 'safe_load' not in line:
                        issues.append(SecurityIssue(
                            issue_id=f"{file_path}:{i}",
                            severity="high",
                            category="unsafe_deserialization",
                            description=f"Unsafe deserialization in line {i}",
                            recommendation="Use safe_load or validate input",
                            file_path=file_path,
                            line_number=i
                        ))
        
        except Exception as e:
            logger.error(f"Error auditing file {file_path}: {e}")
        
        self.issues.extend(issues)
        return issues
    
    def audit_directory(self, directory: str, pattern: str = "*.py") -> List[SecurityIssue]:
        """
        Auditar directorio.
        
        Args:
            directory: Directorio a auditar
            pattern: Patrón de archivos
            
        Returns:
            Lista de problemas encontrados
        """
        path = Path(directory)
        all_issues = []
        
        for file_path in path.rglob(pattern):
            issues = self.audit_file(str(file_path))
            all_issues.extend(issues)
        
        return all_issues
    
    def get_audit_report(self) -> Dict[str, Any]:
        """Obtener reporte de auditoría."""
        if not self.issues:
            return {
                "total_issues": 0,
                "by_severity": {},
                "by_category": {}
            }
        
        by_severity = {}
        by_category = {}
        
        for issue in self.issues:
            by_severity[issue.severity] = by_severity.get(issue.severity, 0) + 1
            by_category[issue.category] = by_category.get(issue.category, 0) + 1
        
        return {
            "total_issues": len(self.issues),
            "by_severity": by_severity,
            "by_category": by_category,
            "issues": [
                {
                    "issue_id": issue.issue_id,
                    "severity": issue.severity,
                    "category": issue.category,
                    "description": issue.description,
                    "recommendation": issue.recommendation,
                    "file_path": issue.file_path,
                    "line_number": issue.line_number
                }
                for issue in self.issues
            ]
        }
    
    def clear_issues(self) -> None:
        """Limpiar problemas encontrados."""
        self.issues.clear()
        logger.info("Security issues cleared")


# Instancia global
_security_auditor: Optional[SecurityAuditor] = None


def get_security_auditor() -> SecurityAuditor:
    """Obtener instancia global del auditor de seguridad."""
    global _security_auditor
    if _security_auditor is None:
        _security_auditor = SecurityAuditor()
    return _security_auditor






