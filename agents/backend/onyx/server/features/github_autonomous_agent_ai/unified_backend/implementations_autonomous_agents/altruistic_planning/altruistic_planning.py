"""
Altruistic Maneuver Planning for Cooperative Autonomous Vehicles
=================================================================

Paper: "Altruistic Maneuver Planning for Cooperative Autonomous Vehicles Using"

Key concepts:
- Altruistic behavior in autonomous agents
- Cooperative planning
- Self-sacrifice for group benefit
- Social utility maximization
- Cooperative game theory
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import math

from ..common.agent_base import BaseAgent, AgentState, AgentStatus


class AltruismLevel(Enum):
    """Levels of altruism."""
    SELFISH = "selfish"  # Maximize own utility
    MODERATE = "moderate"  # Balance own and group utility
    ALTRUISTIC = "altruistic"  # Prioritize group utility
    EXTREME = "extreme"  # Sacrifice for group


@dataclass
class Maneuver:
    """A maneuver plan."""
    maneuver_id: str
    agent_id: str
    maneuver_type: str  # lane_change, speed_adjust, yield, etc.
    utility_self: float  # Utility for self
    utility_group: float  # Utility for group
    cost: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CooperativePlan:
    """A cooperative plan involving multiple agents."""
    plan_id: str
    participants: List[str]
    maneuvers: List[Maneuver]
    total_group_utility: float
    fairness_score: float
    timestamp: datetime = field(default_factory=datetime.now)


class AltruisticPlanner:
    """
    Planner for altruistic maneuvers in cooperative scenarios.
    
    Balances individual and group utility.
    """
    
    def __init__(
        self,
        agents: List[str],
        altruism_levels: Optional[Dict[str, AltruismLevel]] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize altruistic planner.
        
        Args:
            agents: List of agent IDs
            altruism_levels: Altruism level for each agent
            config: Configuration parameters
        """
        self.agents = agents
        self.altruism_levels = altruism_levels or {
            agent_id: AltruismLevel.MODERATE for agent_id in agents
        }
        self.config = config or {}
        
        # Plans
        self.plans: List[CooperativePlan] = []
        
        # Utility weights
        self.self_weight = config.get("self_weight", 0.5)
        self.group_weight = config.get("group_weight", 0.5)
    
    def plan_cooperative_maneuvers(
        self,
        agent_situations: Dict[str, Dict[str, Any]]
    ) -> CooperativePlan:
        """
        Plan cooperative maneuvers for agents.
        
        Args:
            agent_situations: Current situation for each agent
            
        Returns:
            Cooperative plan
        """
        # Generate candidate maneuvers for each agent
        candidate_maneuvers: Dict[str, List[Maneuver]] = {}
        
        for agent_id in self.agents:
            if agent_id in agent_situations:
                maneuvers = self._generate_maneuvers(
                    agent_id,
                    agent_situations[agent_id]
                )
                candidate_maneuvers[agent_id] = maneuvers
        
        # Find optimal cooperative plan
        plan = self._find_optimal_plan(candidate_maneuvers, agent_situations)
        
        self.plans.append(plan)
        return plan
    
    def _generate_maneuvers(
        self,
        agent_id: str,
        situation: Dict[str, Any]
    ) -> List[Maneuver]:
        """Generate candidate maneuvers for an agent."""
        maneuvers = []
        
        # Different maneuver types
        maneuver_types = ["lane_change", "speed_adjust", "yield", "overtake", "maintain"]
        
        for maneuver_type in maneuver_types:
            utility_self, utility_group, cost = self._evaluate_maneuver(
                agent_id,
                maneuver_type,
                situation
            )
            
            maneuver = Maneuver(
                maneuver_id=f"maneuver_{datetime.now().timestamp()}_{maneuver_type}",
                agent_id=agent_id,
                maneuver_type=maneuver_type,
                utility_self=utility_self,
                utility_group=utility_group,
                cost=cost
            )
            
            maneuvers.append(maneuver)
        
        return maneuvers
    
    def _evaluate_maneuver(
        self,
        agent_id: str,
        maneuver_type: str,
        situation: Dict[str, Any]
    ) -> Tuple[float, float, float]:
        """Evaluate a maneuver's utility and cost."""
        # Simplified evaluation
        base_utility = 0.5
        
        # Self utility depends on maneuver type
        if maneuver_type == "maintain":
            utility_self = base_utility
        elif maneuver_type == "speed_adjust":
            utility_self = base_utility + 0.1
        elif maneuver_type == "yield":
            utility_self = base_utility - 0.2  # Cost to self
        else:
            utility_self = base_utility
        
        # Group utility (how much this helps others)
        if maneuver_type == "yield":
            utility_group = 0.8  # High group benefit
        elif maneuver_type == "speed_adjust":
            utility_group = 0.6
        else:
            utility_group = 0.4
        
        # Cost
        if maneuver_type == "yield":
            cost = 0.3  # Higher cost
        else:
            cost = 0.1
        
        return utility_self, utility_group, cost
    
    def _find_optimal_plan(
        self,
        candidate_maneuvers: Dict[str, List[Maneuver]],
        situations: Dict[str, Dict[str, Any]]
    ) -> CooperativePlan:
        """Find optimal cooperative plan."""
        # Select best maneuver for each agent based on altruism level
        selected_maneuvers = []
        
        for agent_id in self.agents:
            if agent_id in candidate_maneuvers:
                altruism = self.altruism_levels.get(agent_id, AltruismLevel.MODERATE)
                best_maneuver = self._select_maneuver_by_altruism(
                    candidate_maneuvers[agent_id],
                    altruism
                )
                selected_maneuvers.append(best_maneuver)
        
        # Calculate total group utility
        total_group_utility = sum(m.utility_group for m in selected_maneuvers)
        
        # Calculate fairness (how evenly distributed are the costs)
        costs = [m.cost for m in selected_maneuvers]
        if costs:
            avg_cost = sum(costs) / len(costs)
            variance = sum((c - avg_cost) ** 2 for c in costs) / len(costs)
            fairness_score = 1.0 / (1.0 + variance)  # Lower variance = higher fairness
        else:
            fairness_score = 0.5
        
        return CooperativePlan(
            plan_id=f"plan_{datetime.now().timestamp()}",
            participants=list(self.agents),
            maneuvers=selected_maneuvers,
            total_group_utility=total_group_utility,
            fairness_score=fairness_score
        )
    
    def _select_maneuver_by_altruism(
        self,
        maneuvers: List[Maneuver],
        altruism: AltruismLevel
    ) -> Maneuver:
        """Select maneuver based on altruism level."""
        if not maneuvers:
            return Maneuver(
                maneuver_id="default",
                agent_id="unknown",
                maneuver_type="maintain",
                utility_self=0.0,
                utility_group=0.0,
                cost=0.0
            )
        
        if altruism == AltruismLevel.SELFISH:
            # Maximize self utility
            return max(maneuvers, key=lambda m: m.utility_self)
        
        elif altruism == AltruismLevel.MODERATE:
            # Balance self and group
            return max(
                maneuvers,
                key=lambda m: self.self_weight * m.utility_self + self.group_weight * m.utility_group
            )
        
        elif altruism == AltruismLevel.ALTRUISTIC:
            # Prioritize group utility
            return max(maneuvers, key=lambda m: m.utility_group)
        
        else:  # EXTREME
            # Maximize group utility, minimize self cost
            return max(
                maneuvers,
                key=lambda m: m.utility_group - m.cost
            )
    
    def set_altruism_level(
        self,
        agent_id: str,
        level: AltruismLevel
    ):
        """
        Set altruism level for an agent.
        
        Args:
            agent_id: Agent ID
            level: Altruism level
        """
        self.altruism_levels[agent_id] = level
    
    def get_plan_history(self) -> List[CooperativePlan]:
        """Get history of cooperative plans."""
        return self.plans
    
    def calculate_social_welfare(self, plan: CooperativePlan) -> float:
        """
        Calculate social welfare of a plan.
        
        Args:
            plan: Cooperative plan
            
        Returns:
            Social welfare score
        """
        # Social welfare = total utility - total cost
        total_utility = sum(m.utility_self + m.utility_group for m in plan.maneuvers)
        total_cost = sum(m.cost for m in plan.maneuvers)
        
        return total_utility - total_cost



