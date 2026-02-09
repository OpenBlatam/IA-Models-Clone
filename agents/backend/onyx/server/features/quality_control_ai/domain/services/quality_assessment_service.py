"""
Quality Assessment Service

Domain service for calculating quality scores and assessments.
"""

from typing import List
from ..entities import Defect, Anomaly, QualityScore, DefectSeverity, AnomalySeverity


class QualityAssessmentService:
    """
    Service for assessing quality based on defects and anomalies.
    
    This service implements the business logic for calculating quality scores
    and determining quality status.
    """
    
    # Penalty scores for defects
    DEFECT_PENALTIES = {
        DefectSeverity.CRITICAL: 20.0,
        DefectSeverity.SEVERE: 15.0,
        DefectSeverity.MODERATE: 8.0,
        DefectSeverity.MINOR: 3.0,
    }
    
    # Penalty scores for anomalies
    ANOMALY_PENALTIES = {
        AnomalySeverity.HIGH: 10.0,
        AnomalySeverity.MEDIUM: 5.0,
        AnomalySeverity.LOW: 2.0,
    }
    
    def calculate_quality_score(
        self,
        defects: List[Defect],
        anomalies: List[Anomaly],
        base_score: float = 100.0
    ) -> QualityScore:
        """
        Calculate quality score based on defects and anomalies.
        
        Args:
            defects: List of detected defects
            anomalies: List of detected anomalies
            base_score: Base quality score (default: 100.0)
        
        Returns:
            QualityScore object with calculated score and status
        """
        # Start with base score
        score = base_score
        
        # Subtract penalties from defects
        for defect in defects:
            penalty = self.DEFECT_PENALTIES.get(defect.severity, 0.0)
            score -= penalty
        
        # Subtract penalties from anomalies
        for anomaly in anomalies:
            penalty = self.ANOMALY_PENALTIES.get(anomaly.severity, 0.0)
            score -= penalty
        
        # Ensure score doesn't go below 0
        score = max(0.0, min(100.0, score))
        
        return QualityScore(
            score=score,
            defects_count=len(defects),
            anomalies_count=len(anomalies),
        )
    
    def assess_defect_impact(self, defect: Defect) -> float:
        """
        Assess the impact of a single defect on quality.
        
        Args:
            defect: Defect to assess
        
        Returns:
            Impact score (higher = worse impact)
        """
        base_penalty = self.DEFECT_PENALTIES.get(defect.severity, 0.0)
        
        # Adjust based on confidence
        confidence_factor = defect.confidence
        adjusted_penalty = base_penalty * confidence_factor
        
        # Adjust based on defect area (larger defects have more impact)
        area_factor = min(1.0, defect.location.area / 10000.0)  # Normalize to 100x100 area
        final_impact = adjusted_penalty * (1.0 + area_factor * 0.2)
        
        return final_impact
    
    def assess_anomaly_impact(self, anomaly: Anomaly) -> float:
        """
        Assess the impact of a single anomaly on quality.
        
        Args:
            anomaly: Anomaly to assess
        
        Returns:
            Impact score (higher = worse impact)
        """
        base_penalty = self.ANOMALY_PENALTIES.get(anomaly.severity, 0.0)
        
        # Adjust based on anomaly score
        score_factor = anomaly.score
        adjusted_penalty = base_penalty * score_factor
        
        return adjusted_penalty
    
    def get_quality_recommendation(self, quality_score: QualityScore) -> str:
        """
        Get recommendation based on quality score.
        
        Args:
            quality_score: Quality score to assess
        
        Returns:
            Recommendation string
        """
        return quality_score.recommendation
    
    def should_reject(self, quality_score: QualityScore) -> bool:
        """
        Determine if an item should be rejected based on quality score.
        
        Args:
            quality_score: Quality score to assess
        
        Returns:
            True if item should be rejected
        """
        return quality_score.status.value == "rejected"
    
    def should_review(self, quality_score: QualityScore) -> bool:
        """
        Determine if an item requires review.
        
        Args:
            quality_score: Quality score to assess
        
        Returns:
            True if item requires review
        """
        return quality_score.status.value in ["poor", "acceptable"]



