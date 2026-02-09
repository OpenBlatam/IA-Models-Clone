"""
Quality Assurance for Color Grading AI
========================================

Quality assurance and validation for color grading results.
"""

import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class QualityLevel(Enum):
    """Quality levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    FAILED = "failed"


@dataclass
class QualityCheck:
    """Quality check result."""
    check_name: str
    passed: bool
    score: float = 0.0  # 0.0 - 1.0
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityReport:
    """Quality assurance report."""
    overall_score: float  # 0.0 - 1.0
    quality_level: QualityLevel
    checks: List[QualityCheck]
    timestamp: datetime = field(default_factory=datetime.now)
    recommendations: List[str] = field(default_factory=list)


class QualityAssurance:
    """
    Quality assurance service.
    
    Features:
    - Multiple quality checks
    - Scoring system
    - Quality levels
    - Recommendations
    - Validation
    """
    
    def __init__(self):
        """Initialize quality assurance."""
        self._checks: List[Callable] = []
        self._reports: List[QualityReport] = []
        self._max_reports = 1000
    
    def register_check(self, check_func: Callable):
        """
        Register quality check function.
        
        Args:
            check_func: Check function that returns QualityCheck
        """
        self._checks.append(check_func)
        logger.info(f"Registered quality check: {check_func.__name__}")
    
    def assess_quality(
        self,
        input_path: str,
        output_path: str,
        color_params: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> QualityReport:
        """
        Assess quality of color grading result.
        
        Args:
            input_path: Input media path
            output_path: Output media path
            color_params: Color parameters used
            metadata: Optional metadata
            
        Returns:
            Quality report
        """
        checks = []
        
        # Run all registered checks
        for check_func in self._checks:
            try:
                check_result = check_func(input_path, output_path, color_params, metadata)
                if isinstance(check_result, QualityCheck):
                    checks.append(check_result)
            except Exception as e:
                logger.error(f"Error in quality check {check_func.__name__}: {e}")
                checks.append(QualityCheck(
                    check_name=check_func.__name__,
                    passed=False,
                    score=0.0,
                    message=f"Check failed: {e}"
                ))
        
        # Calculate overall score
        if checks:
            overall_score = sum(c.score for c in checks) / len(checks)
        else:
            overall_score = 0.5  # Default if no checks
        
        # Determine quality level
        if overall_score >= 0.9:
            quality_level = QualityLevel.EXCELLENT
        elif overall_score >= 0.7:
            quality_level = QualityLevel.GOOD
        elif overall_score >= 0.5:
            quality_level = QualityLevel.ACCEPTABLE
        elif overall_score >= 0.3:
            quality_level = QualityLevel.POOR
        else:
            quality_level = QualityLevel.FAILED
        
        # Generate recommendations
        recommendations = self._generate_recommendations(checks, overall_score)
        
        report = QualityReport(
            overall_score=overall_score,
            quality_level=quality_level,
            checks=checks,
            recommendations=recommendations
        )
        
        # Store report
        self._reports.append(report)
        if len(self._reports) > self._max_reports:
            self._reports = self._reports[-self._max_reports:]
        
        logger.info(f"Quality assessment: {quality_level.value} (score: {overall_score:.2f})")
        
        return report
    
    def _generate_recommendations(
        self,
        checks: List[QualityCheck],
        overall_score: float
    ) -> List[str]:
        """Generate quality improvement recommendations."""
        recommendations = []
        
        # Check for failed checks
        failed_checks = [c for c in checks if not c.passed]
        if failed_checks:
            recommendations.append(
                f"Address {len(failed_checks)} failed quality checks"
            )
        
        # Low score recommendations
        if overall_score < 0.5:
            recommendations.append("Consider adjusting color parameters")
            recommendations.append("Review input media quality")
        
        # Specific check recommendations
        for check in checks:
            if check.score < 0.5:
                recommendations.append(f"Improve {check.check_name}: {check.message}")
        
        return recommendations
    
    def get_recent_reports(self, limit: int = 10) -> List[QualityReport]:
        """Get recent quality reports."""
        return self._reports[-limit:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get quality assurance statistics."""
        if not self._reports:
            return {
                "total_reports": 0,
                "avg_score": 0.0,
                "quality_distribution": {},
            }
        
        quality_dist = {}
        for level in QualityLevel:
            quality_dist[level.value] = sum(
                1 for r in self._reports if r.quality_level == level
            )
        
        return {
            "total_reports": len(self._reports),
            "avg_score": sum(r.overall_score for r in self._reports) / len(self._reports),
            "quality_distribution": quality_dist,
            "checks_count": len(self._checks),
        }


# Predefined quality checks
def check_color_balance(input_path: str, output_path: str, params: Dict[str, Any], metadata: Any) -> QualityCheck:
    """Check color balance."""
    # Simplified check
    balance = params.get("color_balance", {})
    r = abs(balance.get("r", 0))
    g = abs(balance.get("g", 0))
    b = abs(balance.get("b", 0))
    
    max_balance = max(r, g, b)
    score = 1.0 - min(max_balance * 2, 1.0)  # Penalize extreme balance
    
    return QualityCheck(
        check_name="color_balance",
        passed=score >= 0.5,
        score=score,
        message=f"Color balance: R={r:.2f}, G={g:.2f}, B={b:.2f}"
    )


def check_contrast_range(input_path: str, output_path: str, params: Dict[str, Any], metadata: Any) -> QualityCheck:
    """Check contrast range."""
    contrast = params.get("contrast", 1.0)
    
    # Optimal contrast is around 1.0-1.5
    if 1.0 <= contrast <= 1.5:
        score = 1.0
    elif 0.8 <= contrast <= 2.0:
        score = 0.7
    else:
        score = 0.3
    
    return QualityCheck(
        check_name="contrast_range",
        passed=score >= 0.5,
        score=score,
        message=f"Contrast: {contrast:.2f}"
    )

