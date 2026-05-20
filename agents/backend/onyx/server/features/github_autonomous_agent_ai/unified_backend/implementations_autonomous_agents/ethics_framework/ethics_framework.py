"""
Balancing Power and Ethics: A Framework for...
==============================================

Paper: "Balancing Power and Ethics: A Framework for..."

Key concepts:
- Ethical decision-making in AI agents
- Power balance and fairness
- Ethical principles and guidelines
- Bias detection and mitigation
- Responsible AI practices
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from ..common.agent_base import BaseAgent, AgentState, AgentStatus
from ..common.memory import EpisodicMemory, SemanticMemory


class EthicalPrinciple(Enum):
    """Ethical principles for AI agents."""
    FAIRNESS = "fairness"
    TRANSPARENCY = "transparency"
    ACCOUNTABILITY = "accountability"
    PRIVACY = "privacy"
    BENEFICENCE = "beneficence"
    NON_MALEFICENCE = "non_maleficence"
    AUTONOMY = "autonomy"
    JUSTICE = "justice"


class EthicalConcern(Enum):
    """Types of ethical concerns."""
    BIAS = "bias"
    DISCRIMINATION = "discrimination"
    PRIVACY_VIOLATION = "privacy_violation"
    UNFAIR_TREATMENT = "unfair_treatment"
    LACK_OF_TRANSPARENCY = "lack_of_transparency"
    ACCOUNTABILITY_GAP = "accountability_gap"


class EthicalStatus(Enum):
    """Ethical compliance status."""
    COMPLIANT = "compliant"
    MINOR_CONCERN = "minor_concern"
    MODERATE_CONCERN = "moderate_concern"
    MAJOR_VIOLATION = "major_violation"


@dataclass
class EthicalAssessment:
    """Ethical assessment result."""
    assessment_id: str
    principle: EthicalPrinciple
    status: EthicalStatus
    concerns: List[EthicalConcern] = field(default_factory=list)
    score: float = 0.0
    recommendations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class BiasDetection:
    """Bias detection result."""
    bias_id: str
    bias_type: str
    affected_group: Optional[str] = None
    severity: float = 0.0
    description: str = ""
    detected_at: datetime = field(default_factory=datetime.now)


class EthicsFrameworkAgent(BaseAgent):
    """
    Agent with built-in ethical framework.
    
    Implements ethical principles, bias detection, fairness checks,
    and responsible AI practices.
    """
    
    def __init__(
        self,
        name: str,
        ethical_principles: Optional[List[EthicalPrinciple]] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize ethics framework agent.
        
        Args:
            name: Agent name
            ethical_principles: List of ethical principles to follow
            config: Additional configuration
        """
        super().__init__(name, config)
        
        # Ethical principles
        self.ethical_principles = ethical_principles or list(EthicalPrinciple)
        
        # Ethical tracking
        self.ethical_assessments: List[EthicalAssessment] = []
        self.bias_detections: List[BiasDetection] = []
        self.ethical_violations: List[Dict[str, Any]] = []
        
        # Ethics metrics
        self.ethical_decisions_made = 0
        self.biases_detected = 0
        self.biases_mitigated = 0
        
        # Memory systems
        self.episodic_memory = EpisodicMemory()
        self.semantic_memory = SemanticMemory()
        
        # Ethical guidelines
        self.ethical_guidelines: Dict[str, Any] = {
            "fairness_threshold": 0.8,
            "transparency_required": True,
            "bias_detection_enabled": True,
            "privacy_protection": True
        }
    
    def think(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Think about task with ethical considerations.
        
        Args:
            task: Task description
            context: Additional context
            
        Returns:
            Thinking result with ethical assessment
        """
        self.state.status = AgentStatus.THINKING
        self.state.current_task = task
        
        # Perform ethical assessment
        ethical_assessment = self._assess_ethics(task, context)
        
        # Detect biases
        bias_analysis = self._detect_biases(task, context)
        
        # Check fairness
        fairness_check = self._check_fairness(task, context)
        
        # Determine if ethically acceptable
        ethically_acceptable = (
            ethical_assessment["overall_status"] != EthicalStatus.MAJOR_VIOLATION and
            not bias_analysis["biases_found"] and
            fairness_check["is_fair"]
        )
        
        result = {
            "task": task,
            "ethical_assessment": ethical_assessment,
            "bias_analysis": bias_analysis,
            "fairness_check": fairness_check,
            "ethically_acceptable": ethically_acceptable,
            "recommendations": self._generate_ethical_recommendations(
                ethical_assessment, bias_analysis, fairness_check
            )
        }
        
        self.state.add_step("think", result)
        return result
    
    def _assess_ethics(self, task: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess task against ethical principles."""
        assessments = []
        overall_status = EthicalStatus.COMPLIANT
        
        for principle in self.ethical_principles:
            assessment = self._assess_principle(principle, task, context)
            assessments.append(assessment)
            
            # Update overall status
            if assessment.status == EthicalStatus.MAJOR_VIOLATION:
                overall_status = EthicalStatus.MAJOR_VIOLATION
            elif assessment.status == EthicalStatus.MODERATE_CONCERN and overall_status == EthicalStatus.COMPLIANT:
                overall_status = EthicalStatus.MODERATE_CONCERN
            elif assessment.status == EthicalStatus.MINOR_CONCERN and overall_status == EthicalStatus.COMPLIANT:
                overall_status = EthicalStatus.MINOR_CONCERN
        
        self.ethical_assessments.extend(assessments)
        
        return {
            "assessments": [
                {
                    "principle": a.principle.value,
                    "status": a.status.value,
                    "score": a.score,
                    "concerns": [c.value for c in a.concerns]
                }
                for a in assessments
            ],
            "overall_status": overall_status.value,
            "overall_score": sum(a.score for a in assessments) / len(assessments) if assessments else 0.0
        }
    
    def _assess_principle(self, principle: EthicalPrinciple, task: str, context: Optional[Dict[str, Any]]) -> EthicalAssessment:
        """Assess task against a specific ethical principle."""
        concerns = []
        status = EthicalStatus.COMPLIANT
        score = 1.0
        
        if principle == EthicalPrinciple.FAIRNESS:
            # Check for fairness issues
            if any(word in task.lower() for word in ["discriminate", "exclude", "prefer"]):
                concerns.append(EthicalConcern.UNFAIR_TREATMENT)
                status = EthicalStatus.MODERATE_CONCERN
                score = 0.6
        
        elif principle == EthicalPrinciple.PRIVACY:
            # Check for privacy issues
            if any(word in task.lower() for word in ["personal", "private", "sensitive"]):
                if context and not context.get("privacy_protected", False):
                    concerns.append(EthicalConcern.PRIVACY_VIOLATION)
                    status = EthicalStatus.MODERATE_CONCERN
                    score = 0.5
        
        elif principle == EthicalPrinciple.TRANSPARENCY:
            # Check for transparency
            if not self.ethical_guidelines["transparency_required"]:
                concerns.append(EthicalConcern.LACK_OF_TRANSPARENCY)
                status = EthicalStatus.MINOR_CONCERN
                score = 0.7
        
        assessment = EthicalAssessment(
            assessment_id=f"assessment_{datetime.now().timestamp()}",
            principle=principle,
            status=status,
            concerns=concerns,
            score=score,
            recommendations=self._get_principle_recommendations(principle, concerns)
        )
        
        return assessment
    
    def _get_principle_recommendations(self, principle: EthicalPrinciple, concerns: List[EthicalConcern]) -> List[str]:
        """Get recommendations for addressing concerns."""
        recommendations = []
        
        if EthicalConcern.UNFAIR_TREATMENT in concerns:
            recommendations.append("Review decision-making process for fairness")
            recommendations.append("Ensure equal treatment of all groups")
        
        if EthicalConcern.PRIVACY_VIOLATION in concerns:
            recommendations.append("Implement privacy protection measures")
            recommendations.append("Obtain necessary consent")
        
        if EthicalConcern.LACK_OF_TRANSPARENCY in concerns:
            recommendations.append("Provide clear explanations of decisions")
            recommendations.append("Enable audit logging")
        
        return recommendations
    
    def _detect_biases(self, task: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect biases in task."""
        biases = []
        
        # Check for demographic bias
        demographic_keywords = ["gender", "race", "age", "ethnicity", "religion"]
        if any(keyword in task.lower() for keyword in demographic_keywords):
            bias = BiasDetection(
                bias_id=f"bias_{datetime.now().timestamp()}",
                bias_type="demographic",
                severity=0.7,
                description="Potential demographic bias detected"
            )
            biases.append(bias)
            self.bias_detections.append(bias)
            self.biases_detected += 1
        
        # Check for selection bias
        if "select" in task.lower() or "choose" in task.lower():
            if context and context.get("selection_criteria") is None:
                bias = BiasDetection(
                    bias_id=f"bias_{datetime.now().timestamp()}",
                    bias_type="selection",
                    severity=0.5,
                    description="Potential selection bias - criteria not specified"
                )
                biases.append(bias)
                self.bias_detections.append(bias)
                self.biases_detected += 1
        
        return {
            "biases_found": len(biases),
            "biases": [
                {
                    "type": b.bias_type,
                    "severity": b.severity,
                    "description": b.description
                }
                for b in biases
            ]
        }
    
    def _check_fairness(self, task: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Check fairness of task."""
        fairness_score = 1.0
        is_fair = True
        issues = []
        
        # Check fairness threshold
        threshold = self.ethical_guidelines.get("fairness_threshold", 0.8)
        
        # Check for discriminatory language
        discriminatory_keywords = ["discriminate", "exclude", "prefer", "favor"]
        if any(keyword in task.lower() for keyword in discriminatory_keywords):
            fairness_score = 0.5
            is_fair = False
            issues.append("Discriminatory language detected")
        
        # Check context for fairness
        if context:
            if context.get("unequal_treatment", False):
                fairness_score = 0.3
                is_fair = False
                issues.append("Unequal treatment detected")
        
        return {
            "is_fair": is_fair,
            "fairness_score": fairness_score,
            "meets_threshold": fairness_score >= threshold,
            "issues": issues
        }
    
    def _generate_ethical_recommendations(
        self,
        ethical_assessment: Dict[str, Any],
        bias_analysis: Dict[str, Any],
        fairness_check: Dict[str, Any]
    ) -> List[str]:
        """Generate ethical recommendations."""
        recommendations = []
        
        if ethical_assessment["overall_status"] == EthicalStatus.MAJOR_VIOLATION:
            recommendations.append("Review and revise task to address major ethical violations")
        
        if bias_analysis["biases_found"] > 0:
            recommendations.append("Mitigate detected biases")
            recommendations.append("Review data and algorithms for bias")
        
        if not fairness_check["is_fair"]:
            recommendations.append("Ensure fair treatment of all parties")
            recommendations.append("Review decision criteria for fairness")
        
        return recommendations
    
    def act(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute action with ethical checks.
        
        Args:
            action: Action to execute
            
        Returns:
            Action result
        """
        self.state.status = AgentStatus.ACTING
        
        # Perform ethical check
        ethical_check = self._check_action_ethics(action)
        
        if not ethical_check["ethically_acceptable"]:
            self.ethical_violations.append({
                "action": action,
                "violation": ethical_check["violation"],
                "timestamp": datetime.now().isoformat()
            })
            
            return {
                "status": "blocked",
                "reason": "Action violates ethical principles",
                "violation": ethical_check["violation"]
            }
        
        # Execute action
        result = self._execute_ethically(action)
        self.ethical_decisions_made += 1
        
        self.state.add_step("act", {
            "action": action,
            "result": result,
            "ethical_check": ethical_check
        })
        
        return result
    
    def _check_action_ethics(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Check if action is ethically acceptable."""
        action_str = str(action).lower()
        
        # Check for ethical violations
        violation_keywords = ["discriminate", "harm", "exploit", "unfair"]
        if any(keyword in action_str for keyword in violation_keywords):
            return {
                "ethically_acceptable": False,
                "violation": "Action contains unethical elements"
            }
        
        return {
            "ethically_acceptable": True,
            "violation": None
        }
    
    def _execute_ethically(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute action with ethical considerations."""
        return {
            "status": "executed",
            "action": action,
            "ethical_compliance": True,
            "timestamp": datetime.now().isoformat()
        }
    
    def observe(self, observation: Any) -> Dict[str, Any]:
        """
        Process observation and update ethical state.
        
        Args:
            observation: Observation data
            
        Returns:
            Processed observation
        """
        from ..common.agent_utils import standard_observe_pattern
        
        self.state.status = AgentStatus.OBSERVING
        
        # Check for ethical issues in observation
        if isinstance(observation, dict):
            if observation.get("bias_detected"):
                self.biases_detected += 1
        
        # Use standard observe pattern
        return standard_observe_pattern(
            self,
            observation,
            importance=0.7,
            additional_data={
                "ethical_status": self._get_ethical_status()
            }
        )
    
    def run(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run task with ethical framework.
        
        Args:
            task: Task description
            context: Optional context
            
        Returns:
            Final result
        """
        from ..common.agent_utils import standard_run_pattern
        
        # Prepare context
        if context is None:
            context = {}
        
        context["ethical_framework_enabled"] = True
        
        # Use standard pattern
        result = standard_run_pattern(self, task, context)
        
        # Add ethical information
        result["ethics_summary"] = {
            "ethical_decisions": self.ethical_decisions_made,
            "biases_detected": self.biases_detected,
            "biases_mitigated": self.biases_mitigated,
            "ethical_violations": len(self.ethical_violations),
            "ethical_status": self._get_ethical_status()
        }
        
        return result
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        from ..common.agent_utils import create_status_dict
        return create_status_dict(self, {
            "ethical_principles": [p.value for p in self.ethical_principles],
            "ethical_decisions": self.ethical_decisions_made,
            "biases_detected": self.biases_detected,
            "biases_mitigated": self.biases_mitigated,
            "ethical_violations": len(self.ethical_violations),
            "ethical_status": self._get_ethical_status()
        })
    
    def _get_ethical_status(self) -> str:
        """Get overall ethical status."""
        if self.ethical_violations:
            return "violations_detected"
        if self.biases_detected > self.biases_mitigated:
            return "biases_present"
        return "compliant"
    
    def mitigate_bias(self, bias_id: str) -> bool:
        """Mitigate a detected bias."""
        for bias in self.bias_detections:
            if bias.bias_id == bias_id:
                self.bias_detections.remove(bias)
                self.biases_mitigated += 1
                return True
        return False
    
    def get_ethics_report(self) -> Dict[str, Any]:
        """Get comprehensive ethics report."""
        return {
            "ethical_assessments": len(self.ethical_assessments),
            "biases_detected": self.biases_detected,
            "biases_mitigated": self.biases_mitigated,
            "ethical_violations": len(self.ethical_violations),
            "recent_assessments": [
                {
                    "principle": a.principle.value,
                    "status": a.status.value,
                    "score": a.score
                }
                for a in self.ethical_assessments[-10:]
            ],
            "recent_biases": [
                {
                    "type": b.bias_type,
                    "severity": b.severity,
                    "description": b.description
                }
                for b in self.bias_detections[-10:]
            ],
            "ethical_status": self._get_ethical_status()
        }


