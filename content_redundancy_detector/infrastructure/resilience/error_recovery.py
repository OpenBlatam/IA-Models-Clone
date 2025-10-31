"""
Advanced Error Recovery System - Automatic recovery and healing
Production-ready error recovery and self-healing mechanisms
"""

import asyncio
import time
import logging
from typing import Any, Callable, Optional, Dict, List
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import traceback

logger = logging.getLogger(__name__)

class RecoveryStrategy(Enum):
    """Recovery strategies"""
    RETRY = "retry"
    RESTART = "restart"
    FALLBACK = "fallback"
    ISOLATE = "isolate"
    ESCALATE = "escalate"

@dataclass
class ErrorPattern:
    """Error pattern for detection"""
    error_type: type
    message_pattern: Optional[str] = None
    frequency_threshold: int = 3
    time_window: float = 300.0  # 5 minutes

@dataclass
class RecoveryAction:
    """Recovery action configuration"""
    strategy: RecoveryStrategy
    max_attempts: int = 3
    delay: float = 5.0
    condition: Optional[Callable[[Exception], bool]] = None
    action_func: Optional[Callable[[], Any]] = None

class ErrorRecoverySystem:
    """Advanced error recovery and self-healing system"""
    
    def __init__(self):
        self.error_history: deque = deque(maxlen=1000)
        self.recovery_rules: Dict[type, RecoveryAction] = {}
        self.error_patterns: List[ErrorPattern] = []
        self.isolation_list: set = set()
        
        # Statistics
        self.total_errors = 0
        self.successful_recoveries = 0
        self.failed_recoveries = 0
        
        # Recovery tracking
        self.recovery_attempts: Dict[str, int] = defaultdict(int)
        self.last_recovery_time: Dict[str, float] = {}

    def register_recovery_rule(
        self,
        error_type: type,
        action: RecoveryAction
    ):
        """Register recovery rule for error type"""
        self.recovery_rules[error_type] = action
        logger.info(f"Registered recovery rule for {error_type.__name__}")

    def register_error_pattern(self, pattern: ErrorPattern):
        """Register error pattern for detection"""
        self.error_patterns.append(pattern)

    def async def recover_from_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Attempt to recover from error"""
        self.total_errors += 1
        error_time = time.time()
        
        # Record error
        self.error_history.append({
            "error": error,
            "type": type(error),
            "message": str(error),
            "time": error_time,
            "context": context or {},
            "traceback": traceback.format_exc()
        })
        
        # Check for error patterns
        pattern_match = self._detect_error_pattern(error)
        if pattern_match:
            logger.warning(f"Error pattern detected: {pattern_match}")
        
        # Find recovery rule
        recovery_action = self._find_recovery_action(error)
        if not recovery_action:
            logger.debug(f"No recovery rule for {type(error).__name__}")
            return None
        
        # Execute recovery
        recovery_id = f"{type(error).__name__}_{int(error_time)}"
        return await self._execute_recovery(recovery_id, error, recovery_action, context)

    async def _execute_recovery(
        self,
        recovery_id: str,
        error: Exception,
        action: RecoveryAction,
        context: Optional[Dict[str, Any]]
    ) -> Any:
        """Execute recovery action"""
        # Check if already attempted too many times
        attempts = self.recovery_attempts.get(recovery_id, 0)
        if attempts >= action.max_attempts:
            logger.error(
                f"Recovery exhausted for {recovery_id} "
                f"({attempts} attempts)"
            )
            self.failed_recoveries += 1
            return None
        
        # Check condition
        if action.condition and not action.condition(error):
            logger.debug(f"Recovery condition not met for {recovery_id}")
            return None
        
        # Increment attempts
        self.recovery_attempts[recovery_id] = attempts + 1
        self.last_recovery_time[recovery_id] = time.time()
        
        logger.info(
            f"Attempting recovery {action.strategy.value} "
            f"for {recovery_id} (attempt {attempts + 1}/{action.max_attempts})"
        )
        
        try:
            # Wait for delay
            if action.delay > 0:
                await asyncio.sleep(action.delay)
            
            # Execute recovery based on strategy
            if action.strategy == RecoveryStrategy.RETRY:
                result = await self._recovery_retry(action, context)
            
            elif action.strategy == RecoveryStrategy.RESTART:
                result = await self._recovery_restart(action, context)
            
            elif action.strategy == RecoveryStrategy.FALLBACK:
                result = await self._recovery_fallback(action, context)
            
            elif action.strategy == RecoveryStrategy.ISOLATE:
                result = await self._recovery_isolate(action, context)
            
            elif action.strategy == RecoveryStrategy.ESCALATE:
                result = await self._recovery_escalate(action, context)
            
            else:
                result = None
            
            if result is not None:
                self.successful_recoveries += 1
                logger.info(f"Recovery successful for {recovery_id}")
            
            return result
            
        except Exception as recovery_error:
            logger.error(
                f"Recovery failed for {recovery_id}: {recovery_error}"
            )
            self.failed_recoveries += 1
            return None

    async def _recovery_retry(
        self,
        action: RecoveryAction,
        context: Optional[Dict[str, Any]]
    ) -> Any:
        """Retry recovery strategy"""
        if action.action_func:
            if asyncio.iscoroutinefunction(action.action_func):
                return await action.action_func()
            else:
                return action.action_func()
        return None

    async def _recovery_restart(
        self,
        action: RecoveryAction,
        context: Optional[Dict[str, Any]]
    ) -> Any:
        """Restart recovery strategy"""
        # This would typically restart a service/component
        if action.action_func:
            if asyncio.iscoroutinefunction(action.action_func):
                return await action.action_func()
            else:
                return action.action_func()
        return None

    async def _recovery_fallback(
        self,
        action: RecoveryAction,
        context: Optional[Dict[str, Any]]
    ) -> Any:
        """Fallback recovery strategy"""
        if action.action_func:
            if asyncio.iscoroutinefunction(action.action_func):
                return await action.action_func()
            else:
                return action.action_func()
        return None

    async def _recovery_isolate(
        self,
        action: RecoveryAction,
        context: Optional[Dict[str, Any]]
    ) -> Any:
        """Isolate recovery strategy"""
        component = context.get("component") if context else None
        if component:
            self.isolation_list.add(component)
            logger.warning(f"Component '{component}' isolated")
        return None

    async def _recovery_escalate(
        self,
        action: RecoveryAction,
        context: Optional[Dict[str, Any]]
    ) -> Any:
        """Escalate recovery strategy"""
        # This would typically send alerts/notifications
        logger.critical(f"Error escalated: {context}")
        return None

    def _find_recovery_action(self, error: Exception) -> Optional[RecoveryAction]:
        """Find recovery action for error"""
        error_type = type(error)
        
        # Check exact match
        if error_type in self.recovery_rules:
            return self.recovery_rules[error_type]
        
        # Check base classes
        for rule_type, action in self.recovery_rules.items():
            if isinstance(error, rule_type):
                return action
        
        return None

    def _detect_error_pattern(self, error: Exception) -> Optional[ErrorPattern]:
        """Detect if error matches a pattern"""
        current_time = time.time()
        
        for pattern in self.error_patterns:
            if not isinstance(error, pattern.error_type):
                continue
            
            # Check message pattern
            if pattern.message_pattern:
                if pattern.message_pattern not in str(error):
                    continue
            
            # Check frequency
            window_start = current_time - pattern.time_window
            recent_errors = [
                e for e in self.error_history
                if (isinstance(e["error"], pattern.error_type) and
                    e["time"] >= window_start)
            ]
            
            if len(recent_errors) >= pattern.frequency_threshold:
                return pattern
        
        return None

    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get recovery statistics"""
        recovery_rate = (
            self.successful_recoveries / max(self.total_errors, 1)
            if self.total_errors > 0 else 0
        )
        
        return {
            "total_errors": self.total_errors,
            "successful_recoveries": self.successful_recoveries,
            "failed_recoveries": self.failed_recoveries,
            "recovery_rate": recovery_rate,
            "recovery_rules": len(self.recovery_rules),
            "isolated_components": list(self.isolation_list),
            "error_patterns": len(self.error_patterns),
            "recent_errors": len([
                e for e in self.error_history
                if time.time() - e["time"] < 3600  # Last hour
            ])
        }

    def clear_error_history(self):
        """Clear error history"""
        self.error_history.clear()
        logger.info("Error history cleared")

    def unisolate_component(self, component: str):
        """Remove component from isolation"""
        if component in self.isolation_list:
            self.isolation_list.remove(component)
            logger.info(f"Component '{component}' unisolated")






