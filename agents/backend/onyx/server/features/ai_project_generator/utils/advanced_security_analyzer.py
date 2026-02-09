"""
Advanced Security Analyzer
==========================

Sistema avanzado de análisis de seguridad.
"""

import ast
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SecurityVulnerability(Enum):
    """Tipos de vulnerabilidades de seguridad."""
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    CSRF = "csrf"
    PATH_TRAVERSAL = "path_traversal"
    COMMAND_INJECTION = "command_injection"
    INSECURE_DESERIALIZATION = "insecure_deserialization"
    HARDCODED_SECRET = "hardcoded_secret"
    WEAK_CRYPTOGRAPHY = "weak_cryptography"
    INSECURE_RANDOM = "insecure_random"
    EXPOSED_CREDENTIALS = "exposed_credentials"


@dataclass
class SecurityIssue:
    """Problema de seguridad."""
    vulnerability_type: SecurityVulnerability
    severity: str  # critical, high, medium, low
    location: str
    line_number: int
    description: str
    code_snippet: str
    fix_suggestion: str
    cwe_id: Optional[str] = None  # Common Weakness Enumeration ID


class AdvancedSecurityAnalyzer:
    """Analizador avanzado de seguridad."""
    
    def __init__(self):
        self.vulnerability_patterns = {
            SecurityVulnerability.SQL_INJECTION: [
                (r'f["\']\s*SELECT.*\{.*\}', "SQL query with f-string (potential injection)"),
                (r'execute\s*\(\s*f["\']', "SQL execution with f-string"),
                (r'query\s*=\s*["\'].*\+.*\+', "String concatenation in SQL"),
            ],
            SecurityVulnerability.XSS: [
                (r'\{(\w+)\}', "Direct variable interpolation in template"),
                (r'innerHTML\s*=\s*(\w+)', "Direct innerHTML assignment"),
                (r'document\.write\s*\(', "Direct document.write"),
            ],
            SecurityVulnerability.PATH_TRAVERSAL: [
                (r'open\s*\(\s*["\'].*\.\./', "Path traversal in file open"),
                (r'os\.path\.join\s*\([^)]*\.\.', "Path traversal in path join"),
            ],
            SecurityVulnerability.COMMAND_INJECTION: [
                (r'os\.system\s*\(', "Use of os.system()"),
                (r'subprocess\.call\s*\([^)]*shell\s*=\s*True', "Shell injection risk"),
                (r'eval\s*\(', "Use of eval()"),
                (r'exec\s*\(', "Use of exec()"),
            ],
            SecurityVulnerability.HARDCODED_SECRET: [
                (r'password\s*=\s*["\'][^"\']+["\']', "Hardcoded password"),
                (r'api_key\s*=\s*["\'][^"\']+["\']', "Hardcoded API key"),
                (r'secret\s*=\s*["\'][^"\']+["\']', "Hardcoded secret"),
                (r'token\s*=\s*["\'][^"\']+["\']', "Hardcoded token"),
            ],
            SecurityVulnerability.WEAK_CRYPTOGRAPHY: [
                (r'hashlib\.md5\s*\(', "Use of weak MD5 hash"),
                (r'hashlib\.sha1\s*\(', "Use of weak SHA1 hash"),
                (r'DES\s*\(', "Use of weak DES encryption"),
            ],
            SecurityVulnerability.INSECURE_RANDOM: [
                (r'random\.randint\s*\(', "Insecure random for security"),
                (r'random\.choice\s*\(', "Insecure random for security"),
            ],
        }
        
        self.cwe_mapping = {
            SecurityVulnerability.SQL_INJECTION: "CWE-89",
            SecurityVulnerability.XSS: "CWE-79",
            SecurityVulnerability.CSRF: "CWE-352",
            SecurityVulnerability.PATH_TRAVERSAL: "CWE-22",
            SecurityVulnerability.COMMAND_INJECTION: "CWE-78",
            SecurityVulnerability.INSECURE_DESERIALIZATION: "CWE-502",
            SecurityVulnerability.HARDCODED_SECRET: "CWE-798",
            SecurityVulnerability.WEAK_CRYPTOGRAPHY: "CWE-327",
            SecurityVulnerability.INSECURE_RANDOM: "CWE-330",
        }
    
    def analyze_security(self, code: str, file_path: str = "unknown") -> Dict[str, Any]:
        """Analiza seguridad del código."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return {
                "file_path": file_path,
                "error": f"Syntax error: {str(e)}",
                "vulnerabilities": []
            }
        
        vulnerabilities = []
        
        # Detectar vulnerabilidades conocidas
        for vuln_type, patterns in self.vulnerability_patterns.items():
            for pattern, description in patterns:
                vulnerabilities.extend(
                    self._detect_pattern(code, pattern, vuln_type, description)
                )
        
        # Análisis AST para vulnerabilidades complejas
        vulnerabilities.extend(self._analyze_ast_security(tree, code))
        
        # Análisis de autenticación y autorización
        vulnerabilities.extend(self._analyze_auth(code))
        
        # Ordenar por severidad
        severity_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        vulnerabilities.sort(
            key=lambda x: severity_order.get(x.severity, 0), reverse=True
        )
        
        return {
            "file_path": file_path,
            "total_vulnerabilities": len(vulnerabilities),
            "by_severity": self._group_by_severity(vulnerabilities),
            "by_type": self._group_by_type(vulnerabilities),
            "security_score": self._calculate_security_score(vulnerabilities),
            "vulnerabilities": [
                {
                    "type": v.vulnerability_type.value,
                    "severity": v.severity,
                    "location": v.location,
                    "line": v.line_number,
                    "description": v.description,
                    "code_snippet": v.code_snippet[:200],
                    "fix_suggestion": v.fix_suggestion,
                    "cwe_id": v.cwe_id
                }
                for v in vulnerabilities
            ],
            "recommendations": self._generate_recommendations(vulnerabilities)
        }
    
    def _detect_pattern(
        self, code: str, pattern: str, vuln_type: SecurityVulnerability, description: str
    ) -> List[SecurityIssue]:
        """Detecta un patrón de vulnerabilidad."""
        issues = []
        
        for match in re.finditer(pattern, code, re.IGNORECASE | re.MULTILINE):
            line_num = code[:match.start()].count('\n') + 1
            code_snippet = match.group(0)
            
            severity = self._determine_severity(vuln_type)
            cwe_id = self.cwe_mapping.get(vuln_type)
            
            issues.append(SecurityIssue(
                vulnerability_type=vuln_type,
                severity=severity,
                location=f"Line {line_num}",
                line_number=line_num,
                description=description,
                code_snippet=code_snippet,
                fix_suggestion=self._get_fix_suggestion(vuln_type),
                cwe_id=cwe_id
            ))
        
        return issues
    
    def _analyze_ast_security(self, tree: ast.AST, code: str) -> List[SecurityIssue]:
        """Analiza AST para vulnerabilidades complejas."""
        issues = []
        
        # Detectar uso de pickle sin verificación
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr == 'loads' and 'pickle' in code:
                        line_num = node.lineno
                        issues.append(SecurityIssue(
                            vulnerability_type=SecurityVulnerability.INSECURE_DESERIALIZATION,
                            severity="critical",
                            location=f"Line {line_num}",
                            line_number=line_num,
                            description="Insecure deserialization with pickle",
                            code_snippet="pickle.loads(...)",
                            fix_suggestion="Use json or safer serialization, or verify data source",
                            cwe_id="CWE-502"
                        ))
        
        # Detectar uso de __import__ dinámico
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == '__import__':
                    line_num = node.lineno
                    issues.append(SecurityIssue(
                        vulnerability_type=SecurityVulnerability.COMMAND_INJECTION,
                        severity="high",
                        location=f"Line {line_num}",
                        line_number=line_num,
                        description="Dynamic import with __import__()",
                        code_snippet="__import__(...)",
                        fix_suggestion="Use static imports or verify input",
                        cwe_id="CWE-78"
                    ))
        
        return issues
    
    def _analyze_auth(self, code: str) -> List[SecurityIssue]:
        """Analiza autenticación y autorización."""
        issues = []
        
        # Detectar falta de verificación de autenticación
        if 'def' in code and 'authenticate' not in code.lower() and 'login' in code.lower():
            issues.append(SecurityIssue(
                vulnerability_type=SecurityVulnerability.EXPOSED_CREDENTIALS,
                severity="high",
                location="File level",
                line_number=1,
                description="Potential missing authentication check",
                code_snippet="Authentication logic",
                fix_suggestion="Ensure all protected endpoints verify authentication",
                cwe_id="CWE-306"
            ))
        
        # Detectar tokens en logs
        if 'log' in code.lower() and ('token' in code.lower() or 'password' in code.lower()):
            issues.append(SecurityIssue(
                vulnerability_type=SecurityVulnerability.EXPOSED_CREDENTIALS,
                severity="critical",
                location="Logging",
                line_number=1,
                description="Potential credential exposure in logs",
                code_snippet="Logging sensitive data",
                fix_suggestion="Never log passwords, tokens, or sensitive data",
                cwe_id="CWE-532"
            ))
        
        return issues
    
    def _determine_severity(self, vuln_type: SecurityVulnerability) -> str:
        """Determina severidad de la vulnerabilidad."""
        critical = [
            SecurityVulnerability.SQL_INJECTION,
            SecurityVulnerability.COMMAND_INJECTION,
            SecurityVulnerability.INSECURE_DESERIALIZATION,
        ]
        high = [
            SecurityVulnerability.XSS,
            SecurityVulnerability.PATH_TRAVERSAL,
            SecurityVulnerability.HARDCODED_SECRET,
        ]
        
        if vuln_type in critical:
            return "critical"
        elif vuln_type in high:
            return "high"
        else:
            return "medium"
    
    def _get_fix_suggestion(self, vuln_type: SecurityVulnerability) -> str:
        """Obtiene sugerencia de fix."""
        suggestions = {
            SecurityVulnerability.SQL_INJECTION: "Use parameterized queries or ORM",
            SecurityVulnerability.XSS: "Escape user input or use template engine",
            SecurityVulnerability.PATH_TRAVERSAL: "Validate and sanitize file paths",
            SecurityVulnerability.COMMAND_INJECTION: "Use subprocess with list arguments, avoid shell=True",
            SecurityVulnerability.INSECURE_DESERIALIZATION: "Use json or verify data source",
            SecurityVulnerability.HARDCODED_SECRET: "Use environment variables or secret management",
            SecurityVulnerability.WEAK_CRYPTOGRAPHY: "Use strong algorithms (SHA-256, AES-256)",
            SecurityVulnerability.INSECURE_RANDOM: "Use secrets module for cryptographic randomness",
            SecurityVulnerability.EXPOSED_CREDENTIALS: "Remove credentials from code, use secure storage",
        }
        return suggestions.get(vuln_type, "Review and fix security issue")
    
    def _calculate_security_score(self, vulnerabilities: List[SecurityIssue]) -> float:
        """Calcula puntuación de seguridad."""
        if not vulnerabilities:
            return 100.0
        
        score = 100.0
        for vuln in vulnerabilities:
            if vuln.severity == "critical":
                score -= 20
            elif vuln.severity == "high":
                score -= 10
            elif vuln.severity == "medium":
                score -= 5
            else:
                score -= 2
        
        return max(0, round(score, 2))
    
    def _group_by_severity(self, vulnerabilities: List[SecurityIssue]) -> Dict[str, int]:
        """Agrupa por severidad."""
        grouped = {}
        for v in vulnerabilities:
            grouped[v.severity] = grouped.get(v.severity, 0) + 1
        return grouped
    
    def _group_by_type(self, vulnerabilities: List[SecurityIssue]) -> Dict[str, int]:
        """Agrupa por tipo."""
        grouped = {}
        for v in vulnerabilities:
            grouped[v.vulnerability_type.value] = grouped.get(v.vulnerability_type.value, 0) + 1
        return grouped
    
    def _generate_recommendations(self, vulnerabilities: List[SecurityIssue]) -> List[str]:
        """Genera recomendaciones."""
        recommendations = []
        
        critical = [v for v in vulnerabilities if v.severity == "critical"]
        if critical:
            recommendations.append(f"Address {len(critical)} critical security vulnerabilities immediately")
        
        hardcoded = [v for v in vulnerabilities if v.vulnerability_type == SecurityVulnerability.HARDCODED_SECRET]
        if hardcoded:
            recommendations.append("Move all secrets to environment variables or secret management")
        
        injection = [v for v in vulnerabilities if 'injection' in v.vulnerability_type.value]
        if injection:
            recommendations.append("Implement input validation and sanitization")
        
        return recommendations


# Factory function
_advanced_security_analyzer = None

def get_advanced_security_analyzer() -> AdvancedSecurityAnalyzer:
    """Obtiene instancia global del analizador."""
    global _advanced_security_analyzer
    if _advanced_security_analyzer is None:
        _advanced_security_analyzer = AdvancedSecurityAnalyzer()
    return _advanced_security_analyzer

