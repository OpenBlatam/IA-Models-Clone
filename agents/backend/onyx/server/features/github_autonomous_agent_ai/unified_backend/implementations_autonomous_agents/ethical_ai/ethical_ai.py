"""
ETHICAL AI IN SCIENCE
=====================

Paper: "ETHICAL AI IN SCIENCE"

Key concepts:
- Ethical AI principles
- Responsible AI use
- Ethical decision-making
- Bias detection and mitigation
- Fairness and transparency
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from ..common.agent_base import BaseAgent, AgentState, AgentStatus


class EthicalPrinciple(Enum):
    """Ethical principles."""
    AUTONOMY = "autonomy"
    BENEFICENCE = "beneficence"
    NON_MALEFICENCE = "non_maleficence"
    JUSTICE = "justice"
    TRANSPARENCY = "transparency"
    ACCOUNTABILITY = "accountability"


class BiasType(Enum):
    """Types of bias."""
    GENDER = "gender"
    RACIAL = "racial"
    CULTURAL = "cultural"
    SOCIOECONOMIC = "socioeconomic"
    ALGORITHMIC = "algorithmic"


@dataclass
class EthicalAssessment:
    """An ethical assessment."""
    assessment_id: str
    principle: EthicalPrinciple
    compliance_score: float
    issues: List[str]
    recommendations: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class BiasDetection:
    """A bias detection."""
    detection_id: str
    bias_type: BiasType
    severity: float
    evidence: List[str]
    mitigation_strategies: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


class EthicalAIFramework:
    """
    Framework for ethical AI in science.
    
    Ensures responsible and ethical use of AI agents.
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize ethical AI framework.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        self.ethical_assessments: List[EthicalAssessment] = []
        self.bias_detections: List[BiasDetection] = []
        
        # Ethical guidelines
        self.guidelines: Dict[EthicalPrinciple, List[str]] = {
            EthicalPrinciple.AUTONOMY: [
                "Respect user autonomy",
                "Provide informed consent mechanisms"
            ],
            EthicalPrinciple.BENEFICENCE: [
                "Maximize benefits",
                "Minimize harm"
            ],
            EthicalPrinciple.NON_MALEFICENCE: [
                "Do no harm",
                "Prevent negative outcomes"
            ],
            EthicalPrinciple.JUSTICE: [
                "Ensure fairness",
                "Avoid discrimination"
            ],
            EthicalPrinciple.TRANSPARENCY: [
                "Explain decisions",
                "Provide interpretability"
            ],
            EthicalPrinciple.ACCOUNTABILITY: [
                "Take responsibility",
                "Enable auditability"
            ]
        }
    
    def assess_ethical_compliance(
        self,
        agent: BaseAgent,
        action: Dict[str, Any]
    ) -> List[EthicalAssessment]:
        """
        Assess ethical compliance of an action.
        
        Args:
            agent: Agent performing action
            action: Action to assess
            
        Returns:
            List of ethical assessments
        """
        assessments = []
        
        for principle in EthicalPrinciple:
            compliance_score = self._assess_principle(principle, action)
            issues = self._identify_issues(principle, action, compliance_score)
            recommendations = self._generate_recommendations(principle, issues)
            
            assessment = EthicalAssessment(
                assessment_id=f"eth_{datetime.now().timestamp()}",
                principle=principle,
                compliance_score=compliance_score,
                issues=issues,
                recommendations=recommendations
            )
            assessments.append(assessment)
            self.ethical_assessments.append(assessment)
        
        return assessments
    
    def _assess_principle(
        self,
        principle: EthicalPrinciple,
        action: Dict[str, Any]
    ) -> float:
        """Assess compliance with a principle."""
        # Simplified assessment
        # In production, this would use detailed analysis
        
        action_str = str(action).lower()
        
        # Check for violations
        if principle == EthicalPrinciple.NON_MALEFICENCE:
            harmful_keywords = ["harm", "damage", "destroy", "attack"]
            if any(keyword in action_str for keyword in harmful_keywords):
                return 0.3
        
        if principle == EthicalPrinciple.JUSTICE:
            bias_keywords = ["discriminate", "unfair", "biased"]
            if any(keyword in action_str for keyword in bias_keywords):
                return 0.4
        
        if principle == EthicalPrinciple.TRANSPARENCY:
            # Check if action is explainable
            if "explanation" in action_str or "reason" in action_str:
                return 0.8
        
        # Default: moderate compliance
        return 0.7
    
    def _identify_issues(
        self,
        principle: EthicalPrinciple,
        action: Dict[str, Any],
        compliance_score: float
    ) -> List[str]:
        """Identify ethical issues."""
        issues = []
        
        if compliance_score < 0.5:
            issues.append(f"Low compliance with {principle.value} principle")
        
        if principle == EthicalPrinciple.JUSTICE and compliance_score < 0.6:
            issues.append("Potential fairness concerns")
        
        if principle == EthicalPrinciple.TRANSPARENCY and compliance_score < 0.5:
            issues.append("Lack of transparency in decision-making")
        
        return issues
    
    def _generate_recommendations(
        self,
        principle: EthicalPrinciple,
        issues: List[str]
    ) -> List[str]:
        """Generate recommendations."""
        recommendations = []
        
        if issues:
            guidelines = self.guidelines.get(principle, [])
            recommendations.extend(guidelines)
            recommendations.append(f"Address identified issues: {', '.join(issues)}")
        else:
            recommendations.append(f"Maintain compliance with {principle.value}")
        
        return recommendations
    
    def detect_bias(
        self,
        agent: BaseAgent,
        test_cases: Optional[List[Dict[str, Any]]] = None
    ) -> List[BiasDetection]:
        """
        Detect bias in agent behavior.
        
        Args:
            agent: Agent to test
            test_cases: Optional test cases
            
        Returns:
            List of bias detections
        """
        detections = []
        
        # Test with different inputs
        if test_cases:
            for test_case in test_cases:
                result = agent.run(test_case.get("input", ""))
                bias = self._analyze_for_bias(test_case, result)
                if bias:
                    detections.append(bias)
        
        return detections
    
    def _analyze_for_bias(
        self,
        test_case: Dict[str, Any],
        result: Dict[str, Any]
    ) -> Optional[BiasDetection]:
        """Analyze result for bias."""
        # Simplified bias detection
        # In production, this would use statistical analysis
        
        output = str(result.get("action", {}).get("status", ""))
        
        # Check for gender bias
        if any(word in output.lower() for word in ["he", "she", "gender"]):
            return BiasDetection(
                detection_id=f"bias_{datetime.now().timestamp()}",
                bias_type=BiasType.GENDER,
                severity=0.6,
                evidence=[output],
                mitigation_strategies=["Use gender-neutral language", "Review training data"]
            )
        
        return None
    
    def get_ethical_report(self) -> Dict[str, Any]:
        """Get comprehensive ethical report."""
        if not self.ethical_assessments:
            return {}
        
        principle_scores = {}
        for assessment in self.ethical_assessments:
            principle = assessment.principle.value
            if principle not in principle_scores:
                principle_scores[principle] = []
            principle_scores[principle].append(assessment.compliance_score)
        
        avg_scores = {
            principle: sum(scores) / len(scores)
            for principle, scores in principle_scores.items()
        }
        
        return {
            "total_assessments": len(self.ethical_assessments),
            "average_scores_by_principle": avg_scores,
            "overall_compliance": sum(avg_scores.values()) / len(avg_scores) if avg_scores else 0.0,
            "bias_detections": len(self.bias_detections),
            "recommendations": self._get_all_recommendations()
        }
    
    def _get_all_recommendations(self) -> List[str]:
        """Get all recommendations from assessments."""
        all_recommendations = []
        for assessment in self.ethical_assessments:
            all_recommendations.extend(assessment.recommendations)
        return list(set(all_recommendations))  # Remove duplicates
