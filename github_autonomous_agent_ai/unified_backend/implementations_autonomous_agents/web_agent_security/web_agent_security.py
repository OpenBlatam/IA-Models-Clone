"""
Why Are Web AI Agents More Vulnerable Than Standalone LLMs?
============================================================

Paper: "WHY ARE WEB AI AGENTS MORE VULNERABLE THAN STANDALONE LLMS? A SECURITY ANALYSIS"

Key concepts:
- Security vulnerabilities in web-based AI agents
- Attack vectors and threats
- Vulnerability assessment
- Security hardening
- Threat mitigation
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from ..common.agent_base import BaseAgent, AgentState, AgentStatus
from ..common.memory import EpisodicMemory, SemanticMemory


class VulnerabilityType(Enum):
    """Types of vulnerabilities."""
    INJECTION = "injection"
    XSS = "xss"
    CSRF = "csrf"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_EXPOSURE = "data_exposure"
    MISCONFIGURATION = "misconfiguration"
    LOGGING = "logging"


class ThreatLevel(Enum):
    """Threat severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Vulnerability:
    """Vulnerability assessment."""
    vuln_id: str
    vuln_type: VulnerabilityType
    threat_level: ThreatLevel
    description: str
    affected_component: str
    mitigation: Optional[str] = None
    detected_at: datetime = field(default_factory=datetime.now)


@dataclass
class SecurityEvent:
    """Security event log entry."""
    event_id: str
    event_type: str
    threat_level: ThreatLevel
    description: str
    timestamp: datetime = field(default_factory=datetime.now)
    mitigated: bool = False


class WebAgentSecurityAnalyzer(BaseAgent):
    """
    Agent for analyzing and mitigating security vulnerabilities
    in web-based AI agents.
    
    Identifies vulnerabilities, assesses threats, and implements
    security measures.
    """
    
    def __init__(
        self,
        name: str,
        security_enabled: bool = True,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize web agent security analyzer.
        
        Args:
            name: Agent name
            security_enabled: Whether security checks are enabled
            config: Additional configuration
        """
        super().__init__(name, config)
        self.security_enabled = security_enabled
        
        # Vulnerability tracking
        self.vulnerabilities: List[Vulnerability] = []
        self.security_events: List[SecurityEvent] = []
        
        # Security metrics
        self.attacks_blocked = 0
        self.vulnerabilities_found = 0
        self.vulnerabilities_fixed = 0
        
        # Memory systems
        self.episodic_memory = EpisodicMemory()
        self.semantic_memory = SemanticMemory()
        
        # Security policies
        self.security_policies: Dict[str, Any] = {
            "input_validation": True,
            "output_sanitization": True,
            "authentication_required": True,
            "rate_limiting": True,
            "logging_enabled": True
        }
        
        # Initialize known vulnerabilities
        self._initialize_known_vulnerabilities()
    
    def _initialize_known_vulnerabilities(self):
        """Initialize list of known web agent vulnerabilities."""
        self.vulnerabilities = [
            Vulnerability(
                vuln_id="web_input_injection",
                vuln_type=VulnerabilityType.INJECTION,
                threat_level=ThreatLevel.HIGH,
                description="Web agents are vulnerable to input injection attacks",
                affected_component="input_processing",
                mitigation="Implement input validation and sanitization"
            ),
            Vulnerability(
                vuln_id="external_api_exposure",
                vuln_type=VulnerabilityType.DATA_EXPOSURE,
                threat_level=ThreatLevel.MEDIUM,
                description="Web agents expose APIs that can be exploited",
                affected_component="api_layer",
                mitigation="Implement authentication and rate limiting"
            ),
            Vulnerability(
                vuln_id="session_management",
                vuln_type=VulnerabilityType.AUTHENTICATION,
                threat_level=ThreatLevel.HIGH,
                description="Web agents may have weak session management",
                affected_component="session_handler",
                mitigation="Use secure session tokens and expiration"
            )
        ]
        self.vulnerabilities_found = len(self.vulnerabilities)
    
    def think(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze task for security vulnerabilities.
        
        Args:
            task: Task description
            context: Additional context
            
        Returns:
            Security analysis result
        """
        self.state.status = AgentStatus.THINKING
        self.state.current_task = task
        
        # Analyze for vulnerabilities
        vulnerability_analysis = self._analyze_vulnerabilities(task, context)
        
        # Assess threat level
        threat_assessment = self._assess_threats(task, context)
        
        # Determine security recommendations
        recommendations = self._generate_recommendations(vulnerability_analysis, threat_assessment)
        
        result = {
            "task": task,
            "vulnerability_analysis": vulnerability_analysis,
            "threat_assessment": threat_assessment,
            "recommendations": recommendations,
            "security_status": self._get_security_status()
        }
        
        self.state.add_step("think", result)
        return result
    
    def _analyze_vulnerabilities(self, task: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze task for potential vulnerabilities."""
        found_vulns = []
        
        # Check for injection patterns
        injection_patterns = ["<script>", "javascript:", "eval(", "exec("]
        if any(pattern in task.lower() for pattern in injection_patterns):
            found_vulns.append({
                "type": VulnerabilityType.INJECTION.value,
                "threat_level": ThreatLevel.HIGH.value,
                "description": "Potential injection attack detected"
            })
        
        # Check for XSS patterns
        xss_patterns = ["onerror=", "onclick=", "onload="]
        if any(pattern in task.lower() for pattern in xss_patterns):
            found_vulns.append({
                "type": VulnerabilityType.XSS.value,
                "threat_level": ThreatLevel.MEDIUM.value,
                "description": "Potential XSS attack detected"
            })
        
        # Check context for vulnerabilities
        if context:
            if context.get("external_api_call", False):
                found_vulns.append({
                    "type": VulnerabilityType.DATA_EXPOSURE.value,
                    "threat_level": ThreatLevel.MEDIUM.value,
                    "description": "External API call may expose data"
                })
        
        return {
            "vulnerabilities_found": len(found_vulns),
            "vulnerabilities": found_vulns,
            "risk_score": self._calculate_risk_score(found_vulns)
        }
    
    def _calculate_risk_score(self, vulnerabilities: List[Dict[str, Any]]) -> float:
        """Calculate overall risk score."""
        if not vulnerabilities:
            return 0.0
        
        threat_scores = {
            ThreatLevel.LOW: 0.25,
            ThreatLevel.MEDIUM: 0.5,
            ThreatLevel.HIGH: 0.75,
            ThreatLevel.CRITICAL: 1.0
        }
        
        max_score = max(
            threat_scores.get(ThreatLevel[v["threat_level"].upper()], 0.0)
            for v in vulnerabilities
        )
        
        return max_score
    
    def _assess_threats(self, task: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess threats from task."""
        threat_level = ThreatLevel.LOW
        threats = []
        
        # Check for malicious patterns
        malicious_patterns = ["hack", "exploit", "bypass", "unauthorized"]
        if any(pattern in task.lower() for pattern in malicious_patterns):
            threat_level = ThreatLevel.HIGH
            threats.append("Malicious intent detected")
        
        # Check for data access patterns
        data_access_patterns = ["password", "token", "key", "secret"]
        if any(pattern in task.lower() for pattern in data_access_patterns):
            threat_level = ThreatLevel.MEDIUM
            threats.append("Sensitive data access attempt")
        
        return {
            "threat_level": threat_level.value,
            "threats": threats,
            "requires_mitigation": threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]
        }
    
    def _generate_recommendations(self, vuln_analysis: Dict[str, Any], threat_assessment: Dict[str, Any]) -> List[str]:
        """Generate security recommendations."""
        recommendations = []
        
        if vuln_analysis["vulnerabilities_found"] > 0:
            recommendations.append("Implement input validation and sanitization")
            recommendations.append("Enable output encoding")
        
        if threat_assessment["threat_level"] in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            recommendations.append("Block suspicious requests")
            recommendations.append("Enable enhanced logging")
            recommendations.append("Require additional authentication")
        
        if vuln_analysis["risk_score"] > 0.7:
            recommendations.append("Review security policies")
            recommendations.append("Implement rate limiting")
        
        return recommendations
    
    def _get_security_status(self) -> str:
        """Get overall security status."""
        if not self.vulnerabilities:
            return "secure"
        
        critical_vulns = [v for v in self.vulnerabilities if v.threat_level == ThreatLevel.CRITICAL]
        if critical_vulns:
            return "critical"
        
        high_vulns = [v for v in self.vulnerabilities if v.threat_level == ThreatLevel.HIGH]
        if high_vulns:
            return "high_risk"
        
        return "moderate_risk"
    
    def act(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute action with security checks.
        
        Args:
            action: Action to execute
            
        Returns:
            Action result
        """
        self.state.status = AgentStatus.ACTING
        
        if not self.security_enabled:
            return {"status": "executed", "action": action}
        
        # Check for security threats
        security_check = self._check_security(action)
        
        if not security_check["safe"]:
            self.attacks_blocked += 1
            
            # Log security event
            event = SecurityEvent(
                event_id=f"event_{datetime.now().timestamp()}",
                event_type="attack_blocked",
                threat_level=security_check["threat_level"],
                description=security_check["reason"]
            )
            self.security_events.append(event)
            
            return {
                "status": "blocked",
                "reason": security_check["reason"],
                "threat_level": security_check["threat_level"].value
            }
        
        # Execute action with security measures
        result = self._execute_securely(action)
        
        self.state.add_step("act", {
            "action": action,
            "result": result,
            "security_check": security_check
        })
        
        return result
    
    def _check_security(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Check action for security issues."""
        action_str = str(action).lower()
        
        # Check for injection
        if any(pattern in action_str for pattern in ["<script>", "javascript:", "eval("]):
            return {
                "safe": False,
                "threat_level": ThreatLevel.HIGH,
                "reason": "Potential injection attack detected"
            }
        
        # Check for unauthorized access
        if "unauthorized" in action_str or "bypass" in action_str:
            return {
                "safe": False,
                "threat_level": ThreatLevel.CRITICAL,
                "reason": "Unauthorized access attempt detected"
            }
        
        # Check security policies
        if not self._check_policies(action):
            return {
                "safe": False,
                "threat_level": ThreatLevel.MEDIUM,
                "reason": "Action violates security policies"
            }
        
        return {
            "safe": True,
            "threat_level": ThreatLevel.LOW,
            "reason": "Action passed security checks"
        }
    
    def _check_policies(self, action: Dict[str, Any]) -> bool:
        """Check if action complies with security policies."""
        # Check input validation policy
        if self.security_policies["input_validation"]:
            if not self._validate_input(action):
                return False
        
        # Check authentication policy
        if self.security_policies["authentication_required"]:
            if not action.get("authenticated", False):
                return False
        
        return True
    
    def _validate_input(self, action: Dict[str, Any]) -> bool:
        """Validate action input."""
        # Simple validation - in real implementation would be more thorough
        return True
    
    def _execute_securely(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute action with security measures."""
        # Apply output sanitization if enabled
        if self.security_policies["output_sanitization"]:
            action = self._sanitize_output(action)
        
        return {
            "status": "executed",
            "action": action,
            "security_verified": True,
            "timestamp": datetime.now().isoformat()
        }
    
    def _sanitize_output(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize action output."""
        # Placeholder for output sanitization
        return action
    
    def observe(self, observation: Any) -> Dict[str, Any]:
        """
        Process observation and detect security events.
        
        Args:
            observation: Observation data
            
        Returns:
            Processed observation
        """
        from ..common.agent_utils import standard_observe_pattern
        
        self.state.status = AgentStatus.OBSERVING
        
        # Check for security events in observation
        if isinstance(observation, dict):
            if observation.get("security_alert"):
                event = SecurityEvent(
                    event_id=f"event_{datetime.now().timestamp()}",
                    event_type="security_alert",
                    threat_level=ThreatLevel.MEDIUM,
                    description=str(observation.get("security_alert"))
                )
                self.security_events.append(event)
        
        # Use standard observe pattern
        return standard_observe_pattern(
            self,
            observation,
            importance=0.8,
            additional_data={
                "security_status": self._get_security_status()
            }
        )
    
    def run(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run security analysis on task.
        
        Args:
            task: Task description
            context: Optional context
            
        Returns:
            Security analysis result
        """
        from ..common.agent_utils import standard_run_pattern
        
        # Prepare context
        if context is None:
            context = {}
        
        context["security_enabled"] = self.security_enabled
        
        # Use standard pattern
        result = standard_run_pattern(self, task, context)
        
        # Add security information
        result["security_summary"] = {
            "vulnerabilities_found": self.vulnerabilities_found,
            "vulnerabilities_fixed": self.vulnerabilities_fixed,
            "attacks_blocked": self.attacks_blocked,
            "security_events": len(self.security_events),
            "security_status": self._get_security_status()
        }
        
        return result
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        from ..common.agent_utils import create_status_dict
        return create_status_dict(self, {
            "vulnerabilities_found": self.vulnerabilities_found,
            "vulnerabilities_fixed": self.vulnerabilities_fixed,
            "attacks_blocked": self.attacks_blocked,
            "security_events": len(self.security_events),
            "security_status": self._get_security_status()
        })
    
    def add_vulnerability(self, vulnerability: Vulnerability):
        """Add a new vulnerability."""
        self.vulnerabilities.append(vulnerability)
        self.vulnerabilities_found += 1
    
    def fix_vulnerability(self, vuln_id: str) -> bool:
        """Mark vulnerability as fixed."""
        for vuln in self.vulnerabilities:
            if vuln.vuln_id == vuln_id:
                self.vulnerabilities.remove(vuln)
                self.vulnerabilities_fixed += 1
                return True
        return False
    
    def get_security_report(self) -> Dict[str, Any]:
        """Get comprehensive security report."""
        return {
            "total_vulnerabilities": len(self.vulnerabilities),
            "vulnerabilities_by_type": {
                vuln_type.value: len([v for v in self.vulnerabilities if v.vuln_type == vuln_type])
                for vuln_type in VulnerabilityType
            },
            "threat_distribution": {
                threat.value: len([v for v in self.vulnerabilities if v.threat_level == threat])
                for threat in ThreatLevel
            },
            "attacks_blocked": self.attacks_blocked,
            "security_events": len(self.security_events),
            "recent_events": [
                {
                    "type": event.event_type,
                    "threat_level": event.threat_level.value,
                    "description": event.description
                }
                for event in self.security_events[-10:]
            ],
            "security_status": self._get_security_status()
        }


