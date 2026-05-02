"""
Fully Autonomous Limitations Framework
======================================

Framework for implementing safety constraints and limitations
on autonomous agents to prevent fully autonomous operation.
"""

from .fully_autonomous_limitations import (
    FullyAutonomousLimitationsAgent,
    RiskLevel,
    AutonomyConstraint,
    SafetyCheck,
    ActionLimitation
)

__all__ = [
    "FullyAutonomousLimitationsAgent",
    "RiskLevel",
    "AutonomyConstraint",
    "SafetyCheck",
    "ActionLimitation"
]


