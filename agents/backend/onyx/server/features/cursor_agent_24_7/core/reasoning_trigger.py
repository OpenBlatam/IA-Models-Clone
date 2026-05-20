"""
Reasoning Trigger
=================

Sistema que activa automáticamente el razonamiento antes de decisiones críticas,
siguiendo las mejores prácticas de Devin de razonar antes de acciones importantes.
"""

import logging
from typing import Optional, Dict, Any, List
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


class CriticalActionType(Enum):
    """Tipos de acciones críticas"""
    GIT_BRANCH_DECISION = "git_branch_decision"
    GIT_CHECKOUT = "git_checkout"
    GIT_PR_DECISION = "git_pr_decision"
    CODE_CHANGE_START = "code_change_start"
    COMPLETION_REPORT = "completion_report"
    TEST_FAILURE = "test_failure"
    LINT_FAILURE = "lint_failure"
    CI_FAILURE = "ci_failure"
    ENVIRONMENT_ISSUE = "environment_issue"
    REPO_UNCERTAINTY = "repo_uncertainty"
    NO_CLEAR_NEXT_STEP = "no_clear_next_step"
    MULTIPLE_APPROACHES_FAILED = "multiple_approaches_failed"
    CRITICAL_DECISION = "critical_decision"
    DETAILS_UNCLEAR = "details_unclear"
    UNEXPECTED_DIFFICULTY = "unexpected_difficulty"
    PLANNING_SEARCH_NO_MATCHES = "planning_search_no_matches"


@dataclass
class ReasoningTrigger:
    """Trigger de razonamiento"""
    action_type: CriticalActionType
    context: Dict[str, Any]
    triggered: bool = False
    reasoning_context_id: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            "action_type": self.action_type.value,
            "context": self.context,
            "triggered": self.triggered,
            "reasoning_context_id": self.reasoning_context_id,
            "timestamp": self.timestamp.isoformat()
        }


class ReasoningTriggerSystem:
    """
    Sistema de triggers de razonamiento.
    
    Activa automáticamente el razonamiento antes de decisiones críticas,
    siguiendo las mejores prácticas de Devin.
    """
    
    def __init__(self) -> None:
        """Inicializar sistema de triggers"""
        self.triggers: List[ReasoningTrigger] = []
        self.agent: Optional[Any] = None
        logger.info("🧠 Reasoning trigger system initialized")
    
    def set_agent(self, agent: Any) -> None:
        """Establecer agente"""
        self.agent = agent
    
    def should_reason(
        self,
        action_type: CriticalActionType,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Determinar si debe razonar antes de una acción.
        
        Args:
            action_type: Tipo de acción crítica.
            context: Contexto adicional.
        
        Returns:
            True si debe razonar.
        """
        must_reason = [
            CriticalActionType.GIT_BRANCH_DECISION,
            CriticalActionType.GIT_PR_DECISION,
            CriticalActionType.CODE_CHANGE_START,
            CriticalActionType.COMPLETION_REPORT
        ]
        
        should_reason = [
            CriticalActionType.NO_CLEAR_NEXT_STEP,
            CriticalActionType.DETAILS_UNCLEAR,
            CriticalActionType.UNEXPECTED_DIFFICULTY,
            CriticalActionType.MULTIPLE_APPROACHES_FAILED,
            CriticalActionType.TEST_FAILURE,
            CriticalActionType.LINT_FAILURE,
            CriticalActionType.CI_FAILURE,
            CriticalActionType.ENVIRONMENT_ISSUE,
            CriticalActionType.REPO_UNCERTAINTY,
            CriticalActionType.CRITICAL_DECISION,
            CriticalActionType.PLANNING_SEARCH_NO_MATCHES
        ]
        
        if action_type in must_reason:
            return True
        
        if action_type in should_reason:
            return True
        
        return False
    
    async def trigger_reasoning(
        self,
        action_type: CriticalActionType,
        context: Optional[Dict[str, Any]] = None,
        auto_trigger: bool = True
    ) -> Optional[ReasoningTrigger]:
        """
        Activar razonamiento para una acción crítica.
        
        Args:
            action_type: Tipo de acción crítica.
            context: Contexto adicional.
            auto_trigger: Si activar automáticamente el razonamiento.
        
        Returns:
            Trigger creado o None.
        """
        if not self.should_reason(action_type, context):
            return None
        
        if not self.agent or not hasattr(self.agent, 'devin') or not self.agent.devin:
            logger.warning("Agent or Devin persona not available for reasoning")
            return None
        
        trigger = ReasoningTrigger(
            action_type=action_type,
            context=context or {}
        )
        
        if auto_trigger:
            reasoning_context = self.agent.devin.start_reasoning()
            trigger.reasoning_context_id = len(self.agent.devin.reasoning_contexts) - 1
            trigger.triggered = True
            
            # Agregar observaciones basadas en el tipo de acción
            await self._add_action_observations(reasoning_context, action_type, context)
        
        self.triggers.append(trigger)
        logger.info(f"🧠 Reasoning triggered for: {action_type.value}")
        
        return trigger
    
    async def _add_action_observations(
        self,
        reasoning_context: Any,
        action_type: CriticalActionType,
        context: Optional[Dict[str, Any]]
    ) -> None:
        """Agregar observaciones basadas en el tipo de acción"""
        if action_type == CriticalActionType.GIT_BRANCH_DECISION:
            reasoning_context.add_observation(
                "Deciding on git branch operation - need to ensure correct branch"
            )
            if context:
                reasoning_context.add_consideration(
                    f"Context: {context.get('branch_name', 'unknown')}, "
                    f"operation: {context.get('operation', 'unknown')}"
                )
        
        elif action_type == CriticalActionType.GIT_PR_DECISION:
            reasoning_context.add_observation(
                "Deciding on PR operation - need to ensure correct repo and PR"
            )
            if context:
                reasoning_context.add_consideration(
                    f"Context: repo={context.get('repo', 'unknown')}, "
                    f"PR={context.get('pr_number', 'unknown')}"
                )
        
        elif action_type == CriticalActionType.CODE_CHANGE_START:
            reasoning_context.add_observation(
                "Starting code changes - need to verify all context gathered"
            )
            reasoning_context.add_consideration(
                "Have I gathered all necessary context?"
            )
            reasoning_context.add_consideration(
                "Have I found all locations to edit?"
            )
            reasoning_context.add_consideration(
                "Have I inspected all references, types, and definitions?"
            )
            if context:
                files = context.get('files_to_edit', [])
                reasoning_context.add_observation(
                    f"Files to edit: {len(files)}"
                )
        
        elif action_type == CriticalActionType.COMPLETION_REPORT:
            reasoning_context.add_observation(
                "About to report completion - need to critically examine work"
            )
            reasoning_context.add_consideration(
                "Have I completely fulfilled the user's request?"
            )
            reasoning_context.add_consideration(
                "Have I completed all verification steps?"
            )
            reasoning_context.add_consideration(
                "Have I edited all relevant locations?"
            )
            if context:
                verification = context.get('verification', {})
                reasoning_context.add_observation(
                    f"Verification status: {verification.get('success', 'unknown')}"
                )
        
        elif action_type == CriticalActionType.TEST_FAILURE:
            reasoning_context.add_observation(
                "Tests failed - need to think about root cause"
            )
            reasoning_context.add_consideration(
                "Is the issue in my code or in the test?"
            )
            reasoning_context.add_consideration(
                "What have I changed that could cause this?"
            )
        
        elif action_type == CriticalActionType.NO_CLEAR_NEXT_STEP:
            reasoning_context.add_observation(
                "No clear next step - need to reason about options"
            )
            if context:
                options = context.get('options', [])
                for option in options:
                    reasoning_context.add_consideration(f"Option: {option}")
        
        elif action_type == CriticalActionType.DETAILS_UNCLEAR:
            reasoning_context.add_observation(
                "Details unclear - need to clarify before proceeding"
            )
            if context:
                unclear = context.get('unclear_details', [])
                for detail in unclear:
                    reasoning_context.add_consideration(f"Unclear: {detail}")
        
        elif action_type == CriticalActionType.MULTIPLE_APPROACHES_FAILED:
            reasoning_context.add_observation(
                "Multiple approaches failed - need to reconsider strategy"
            )
            if context:
                approaches = context.get('failed_approaches', [])
                for approach in approaches:
                    reasoning_context.add_consideration(
                        f"Failed approach: {approach}"
                    )
        
        elif action_type == CriticalActionType.ENVIRONMENT_ISSUE:
            reasoning_context.add_observation(
                "Environment issue detected - need to consider reporting"
            )
            if context:
                issue = context.get('issue', {})
                reasoning_context.add_consideration(
                    f"Issue: {issue.get('type', 'unknown')}"
                )
        
        elif action_type == CriticalActionType.PLANNING_SEARCH_NO_MATCHES:
            reasoning_context.add_observation(
                "Search returned no matches - need to try other search terms"
            )
            if context:
                search_term = context.get('search_term', 'unknown')
                reasoning_context.add_consideration(
                    f"Tried: {search_term}"
                )
                reasoning_context.add_next_step(
                    "Try alternative search terms or approaches"
                )
    
    def get_recent_triggers(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Obtener triggers recientes"""
        return [t.to_dict() for t in self.triggers[-limit:]]
    
    def get_triggers_by_type(
        self,
        action_type: CriticalActionType
    ) -> List[Dict[str, Any]]:
        """Obtener triggers por tipo"""
        return [
            t.to_dict() for t in self.triggers
            if t.action_type == action_type
        ]

