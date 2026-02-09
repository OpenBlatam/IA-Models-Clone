"""
Policy Engine - Motor de Políticas
===================================

Sistema avanzado de gestión de políticas con evaluación, aplicación y auditoría.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


class PolicyType(Enum):
    """Tipo de política."""
    ACCESS_CONTROL = "access_control"
    RATE_LIMIT = "rate_limit"
    DATA_RETENTION = "data_retention"
    SECURITY = "security"
    COMPLIANCE = "compliance"
    BUSINESS_RULE = "business_rule"


class PolicyStatus(Enum):
    """Estado de política."""
    ACTIVE = "active"
    DRAFT = "draft"
    DEPRECATED = "deprecated"
    INACTIVE = "inactive"


@dataclass
class Policy:
    """Política."""
    policy_id: str
    name: str
    policy_type: PolicyType
    description: str
    rules: Dict[str, Any]
    status: PolicyStatus = PolicyStatus.ACTIVE
    priority: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyEvaluation:
    """Evaluación de política."""
    evaluation_id: str
    policy_id: str
    context: Dict[str, Any]
    result: bool
    action: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PolicyEngine:
    """Motor de políticas."""
    
    def __init__(self):
        self.policies: Dict[str, Policy] = {}
        self.evaluations: List[PolicyEvaluation] = []
        self.evaluation_handlers: Dict[str, Callable] = {}
        self._lock = asyncio.Lock()
    
    def create_policy(
        self,
        policy_id: str,
        name: str,
        policy_type: PolicyType,
        description: str,
        rules: Dict[str, Any],
        priority: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Crear política."""
        policy = Policy(
            policy_id=policy_id,
            name=name,
            policy_type=policy_type,
            description=description,
            rules=rules,
            priority=priority,
            metadata=metadata or {},
        )
        
        async def save_policy():
            async with self._lock:
                self.policies[policy_id] = policy
        
        asyncio.create_task(save_policy())
        
        logger.info(f"Created policy: {policy_id} - {name}")
        return policy_id
    
    def evaluate_policy(
        self,
        policy_id: str,
        context: Dict[str, Any],
    ) -> Optional[PolicyEvaluation]:
        """Evaluar política."""
        policy = self.policies.get(policy_id)
        if not policy or policy.status != PolicyStatus.ACTIVE:
            return None
        
        # Evaluar reglas
        result = self._evaluate_rules(policy.rules, context)
        
        # Determinar acción
        action = self._determine_action(policy, result, context)
        
        evaluation = PolicyEvaluation(
            evaluation_id=f"eval_{policy_id}_{datetime.now().timestamp()}",
            policy_id=policy_id,
            context=context,
            result=result,
            action=action,
            metadata={
                "policy_name": policy.name,
                "policy_type": policy.policy_type.value,
            },
        )
        
        async def save_evaluation():
            async with self._lock:
                self.evaluations.append(evaluation)
                if len(self.evaluations) > 100000:
                    self.evaluations.pop(0)
        
        asyncio.create_task(save_evaluation())
        
        # Ejecutar handler si existe
        handler = self.evaluation_handlers.get(policy_id)
        if handler:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(evaluation)
                else:
                    handler(evaluation)
            except Exception as e:
                logger.error(f"Error in policy evaluation handler: {e}")
        
        return evaluation
    
    def _evaluate_rules(self, rules: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Evaluar reglas de política."""
        # Evaluación básica de reglas
        # En producción, usaría un motor de reglas más sofisticado
        
        for rule_key, rule_value in rules.items():
            context_value = context.get(rule_key)
            
            if isinstance(rule_value, dict):
                # Regla compleja
                operator = rule_value.get("operator", "eq")
                expected = rule_value.get("value")
                
                if operator == "eq":
                    if context_value != expected:
                        return False
                elif operator == "ne":
                    if context_value == expected:
                        return False
                elif operator == "gt":
                    if context_value <= expected:
                        return False
                elif operator == "lt":
                    if context_value >= expected:
                        return False
                elif operator == "in":
                    if context_value not in expected:
                        return False
                elif operator == "not_in":
                    if context_value in expected:
                        return False
            else:
                # Regla simple: igualdad
                if context_value != rule_value:
                    return False
        
        return True
    
    def _determine_action(
        self,
        policy: Policy,
        result: bool,
        context: Dict[str, Any],
    ) -> str:
        """Determinar acción basada en resultado."""
        if policy.policy_type == PolicyType.ACCESS_CONTROL:
            return "allow" if result else "deny"
        elif policy.policy_type == PolicyType.RATE_LIMIT:
            return "continue" if result else "throttle"
        elif policy.policy_type == PolicyType.DATA_RETENTION:
            return "retain" if result else "delete"
        elif policy.policy_type == PolicyType.SECURITY:
            return "pass" if result else "block"
        else:
            return "approved" if result else "rejected"
    
    def register_evaluation_handler(
        self,
        policy_id: str,
        handler: Callable,
    ):
        """Registrar handler para evaluación de política."""
        self.evaluation_handlers[policy_id] = handler
        logger.info(f"Registered evaluation handler for policy: {policy_id}")
    
    def evaluate_all_policies(
        self,
        context: Dict[str, Any],
        policy_type: Optional[PolicyType] = None,
    ) -> List[PolicyEvaluation]:
        """Evaluar todas las políticas aplicables."""
        evaluations = []
        
        policies = list(self.policies.values())
        
        if policy_type:
            policies = [p for p in policies if p.policy_type == policy_type]
        
        # Ordenar por prioridad
        policies.sort(key=lambda p: p.priority, reverse=True)
        
        for policy in policies:
            if policy.status == PolicyStatus.ACTIVE:
                evaluation = self.evaluate_policy(policy.policy_id, context)
                if evaluation:
                    evaluations.append(evaluation)
        
        return evaluations
    
    def get_policy(self, policy_id: str) -> Optional[Dict[str, Any]]:
        """Obtener política."""
        policy = self.policies.get(policy_id)
        if not policy:
            return None
        
        return {
            "policy_id": policy.policy_id,
            "name": policy.name,
            "policy_type": policy.policy_type.value,
            "description": policy.description,
            "rules": policy.rules,
            "status": policy.status.value,
            "priority": policy.priority,
            "created_at": policy.created_at.isoformat(),
            "updated_at": policy.updated_at.isoformat(),
            "metadata": policy.metadata,
        }
    
    def get_evaluations(
        self,
        policy_id: Optional[str] = None,
        result: Optional[bool] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Obtener evaluaciones."""
        evaluations = self.evaluations
        
        if policy_id:
            evaluations = [e for e in evaluations if e.policy_id == policy_id]
        
        if result is not None:
            evaluations = [e for e in evaluations if e.result == result]
        
        evaluations.sort(key=lambda e: e.timestamp, reverse=True)
        
        return [
            {
                "evaluation_id": e.evaluation_id,
                "policy_id": e.policy_id,
                "result": e.result,
                "action": e.action,
                "timestamp": e.timestamp.isoformat(),
                "metadata": e.metadata,
            }
            for e in evaluations[:limit]
        ]
    
    def get_policy_engine_summary(self) -> Dict[str, Any]:
        """Obtener resumen del motor."""
        by_type: Dict[str, int] = defaultdict(int)
        by_status: Dict[str, int] = defaultdict(int)
        
        for policy in self.policies.values():
            by_type[policy.policy_type.value] += 1
            by_status[policy.status.value] += 1
        
        return {
            "total_policies": len(self.policies),
            "policies_by_type": dict(by_type),
            "policies_by_status": dict(by_status),
            "total_evaluations": len(self.evaluations),
            "active_policies": len([p for p in self.policies.values() if p.status == PolicyStatus.ACTIVE]),
        }



