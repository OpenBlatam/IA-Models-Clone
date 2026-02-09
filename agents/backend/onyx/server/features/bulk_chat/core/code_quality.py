"""
Code Quality - Análisis de Calidad de Código
============================================

Sistema de análisis de calidad de código con métricas, detección de code smells y sugerencias de mejora.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


class QualityLevel(Enum):
    """Nivel de calidad."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"


class CodeSmellType(Enum):
    """Tipo de code smell."""
    LONG_METHOD = "long_method"
    DUPLICATE_CODE = "duplicate_code"
    COMPLEX_METHOD = "complex_method"
    LARGE_CLASS = "large_class"
    TOO_MANY_PARAMETERS = "too_many_parameters"
    DEAD_CODE = "dead_code"
    MAGIC_NUMBER = "magic_number"
    LONG_PARAMETER_LIST = "long_parameter_list"


@dataclass
class CodeMetric:
    """Métrica de código."""
    metric_name: str
    value: float
    threshold: float
    status: str  # "pass", "warning", "fail"
    description: str = ""


@dataclass
class CodeSmell:
    """Code smell."""
    smell_id: str
    smell_type: CodeSmellType
    file_path: str
    line_number: int
    severity: str  # "low", "medium", "high"
    description: str
    suggestion: str = ""


@dataclass
class QualityReport:
    """Reporte de calidad."""
    report_id: str
    file_path: str
    overall_score: float
    quality_level: QualityLevel
    metrics: List[CodeMetric] = field(default_factory=list)
    code_smells: List[CodeSmell] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class CodeQuality:
    """Analizador de calidad de código."""
    
    def __init__(self):
        self.reports: Dict[str, QualityReport] = {}
        self.quality_history: List[Dict[str, Any]] = []
        self.thresholds: Dict[str, float] = {
            "cyclomatic_complexity": 10.0,
            "lines_of_code": 100.0,
            "code_coverage": 80.0,
            "duplication": 5.0,
        }
        self._lock = asyncio.Lock()
    
    async def analyze_code(
        self,
        file_path: str,
        code_content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Analizar código."""
        report_id = f"report_{file_path}_{datetime.now().timestamp()}"
        
        # Calcular métricas básicas
        lines = code_content.split('\n')
        lines_of_code = len([l for l in lines if l.strip() and not l.strip().startswith('#')])
        
        # Simular cálculo de complejidad ciclomática
        complexity = self._calculate_complexity(code_content)
        
        # Detectar code smells
        code_smells = self._detect_smells(file_path, code_content, lines)
        
        # Calcular score de calidad
        score = self._calculate_quality_score(lines_of_code, complexity, len(code_smells))
        quality_level = self._get_quality_level(score)
        
        # Crear métricas
        metrics = [
            CodeMetric(
                metric_name="lines_of_code",
                value=lines_of_code,
                threshold=self.thresholds.get("lines_of_code", 100.0),
                status="pass" if lines_of_code <= self.thresholds.get("lines_of_code", 100.0) else "warning",
                description=f"Total lines of code: {lines_of_code}",
            ),
            CodeMetric(
                metric_name="cyclomatic_complexity",
                value=complexity,
                threshold=self.thresholds.get("cyclomatic_complexity", 10.0),
                status="pass" if complexity <= self.thresholds.get("cyclomatic_complexity", 10.0) else "warning",
                description=f"Cyclomatic complexity: {complexity:.1f}",
            ),
            CodeMetric(
                metric_name="code_smells",
                value=len(code_smells),
                threshold=5.0,
                status="pass" if len(code_smells) <= 5 else "warning",
                description=f"Code smells detected: {len(code_smells)}",
            ),
        ]
        
        report = QualityReport(
            report_id=report_id,
            file_path=file_path,
            overall_score=score,
            quality_level=quality_level,
            metrics=metrics,
            code_smells=code_smells,
            metadata=metadata or {},
        )
        
        async with self._lock:
            self.reports[report_id] = report
            self.quality_history.append({
                "report_id": report_id,
                "file_path": file_path,
                "score": score,
                "quality_level": quality_level.value,
                "timestamp": datetime.now(),
            })
        
        logger.info(f"Code analysis completed: {file_path} - Score: {score:.1f}")
        return report_id
    
    def _calculate_complexity(self, code: str) -> float:
        """Calcular complejidad ciclomática (simplificado)."""
        complexity = 1.0  # Base
        
        # Contar estructuras de control
        complexity += code.count('if ')
        complexity += code.count('elif ')
        complexity += code.count('for ')
        complexity += code.count('while ')
        complexity += code.count('except ')
        complexity += code.count('case ')
        
        return complexity
    
    def _detect_smells(self, file_path: str, code: str, lines: List[str]) -> List[CodeSmell]:
        """Detectar code smells."""
        smells = []
        
        # Detectar métodos largos
        for i, line in enumerate(lines, 1):
            if 'def ' in line:
                # Contar líneas hasta siguiente def o fin
                method_lines = 0
                for j in range(i, len(lines)):
                    if j < len(lines) - 1 and 'def ' in lines[j]:
                        break
                    method_lines += 1
                
                if method_lines > 50:
                    smells.append(CodeSmell(
                        smell_id=f"smell_{file_path}_{i}",
                        smell_type=CodeSmellType.LONG_METHOD,
                        file_path=file_path,
                        line_number=i,
                        severity="high",
                        description=f"Method at line {i} has {method_lines} lines",
                        suggestion="Consider breaking down into smaller methods",
                    ))
        
        # Detectar números mágicos
        for i, line in enumerate(lines, 1):
            if any(char.isdigit() for char in line) and '=' in line:
                # Simplificado: buscar números grandes
                import re
                numbers = re.findall(r'\b\d{4,}\b', line)
                if numbers:
                    smells.append(CodeSmell(
                        smell_id=f"smell_{file_path}_{i}_magic",
                        smell_type=CodeSmellType.MAGIC_NUMBER,
                        file_path=file_path,
                        line_number=i,
                        severity="medium",
                        description=f"Magic number found at line {i}",
                        suggestion="Replace with named constant",
                    ))
        
        return smells
    
    def _calculate_quality_score(
        self,
        lines_of_code: int,
        complexity: float,
        smells_count: int,
    ) -> float:
        """Calcular score de calidad."""
        score = 100.0
        
        # Penalizar por líneas de código
        if lines_of_code > 100:
            score -= min(20, (lines_of_code - 100) / 10)
        
        # Penalizar por complejidad
        if complexity > 10:
            score -= min(30, (complexity - 10) * 2)
        
        # Penalizar por code smells
        score -= min(40, smells_count * 5)
        
        return max(0.0, score)
    
    def _get_quality_level(self, score: float) -> QualityLevel:
        """Obtener nivel de calidad."""
        if score >= 90:
            return QualityLevel.EXCELLENT
        elif score >= 75:
            return QualityLevel.GOOD
        elif score >= 60:
            return QualityLevel.FAIR
        elif score >= 40:
            return QualityLevel.POOR
        else:
            return QualityLevel.CRITICAL
    
    def get_report(self, report_id: str) -> Optional[Dict[str, Any]]:
        """Obtener reporte."""
        report = self.reports.get(report_id)
        if not report:
            return None
        
        return {
            "report_id": report.report_id,
            "file_path": report.file_path,
            "overall_score": report.overall_score,
            "quality_level": report.quality_level.value,
            "metrics": [
                {
                    "metric_name": m.metric_name,
                    "value": m.value,
                    "threshold": m.threshold,
                    "status": m.status,
                    "description": m.description,
                }
                for m in report.metrics
            ],
            "code_smells": [
                {
                    "smell_id": s.smell_id,
                    "smell_type": s.smell_type.value,
                    "file_path": s.file_path,
                    "line_number": s.line_number,
                    "severity": s.severity,
                    "description": s.description,
                    "suggestion": s.suggestion,
                }
                for s in report.code_smells
            ],
            "generated_at": report.generated_at.isoformat(),
        }
    
    def get_quality_summary(self) -> Dict[str, Any]:
        """Obtener resumen de calidad."""
        by_level: Dict[str, int] = defaultdict(int)
        by_smell_type: Dict[str, int] = defaultdict(int)
        total_score = 0.0
        
        for report in self.reports.values():
            by_level[report.quality_level.value] += 1
            total_score += report.overall_score
            
            for smell in report.code_smells:
                by_smell_type[smell.smell_type.value] += 1
        
        avg_score = total_score / len(self.reports) if self.reports else 0.0
        
        return {
            "total_reports": len(self.reports),
            "reports_by_level": dict(by_level),
            "average_score": avg_score,
            "total_code_smells": sum(len(r.code_smells) for r in self.reports.values()),
            "code_smells_by_type": dict(by_smell_type),
        }
















