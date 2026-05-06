"""
System 5.9 — Circuit Breaker.

Prevents infinite JSON-validation retry loops by capping retries
and detecting deterministic poison responses from mock/degraded engines.
"""

import json
import time
import logging
from enum import Enum, auto
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger("sys5.circuit_breaker")


# ---------------------------------------------------------------------------
# Poison signatures (deterministic failures that retrying won't solve)
# ---------------------------------------------------------------------------

POISON_PATTERNS: Tuple[str, ...] = (
    "Echo from OpenClaw Agent (Mock)",   # D1: legacy DummyAsyncLLM
    "[EMERGENCY MOCK]",                   # D2: legacy DeepSeek 402
    "DeepSeek API unreachable",           # D2: variant
)


# ---------------------------------------------------------------------------
# Per-trace state
# ---------------------------------------------------------------------------

class CircuitState(Enum):
    CLOSED    = auto()   # Normal — retries allowed
    OPEN      = auto()   # Tripped — block all retries
    HALF_OPEN = auto()   # Cooldown expired — allow one probe


@dataclass
class _TraceState:
    """Mutable retry ledger for a single trace ID."""
    retries: int = 0
    first_fail: float = 0.0
    last_reason: str = ""
    state: CircuitState = CircuitState.CLOSED


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------

class CircuitBreaker:
    """
    Three-state circuit breaker for the LLM → JSON validation loop.

    Usage::

        should_retry, fallback = cb.check(trace_id, raw_llm_output)
        if not should_retry:
            return fallback   # valid AgentAction dict
    """

    def __init__(
        self,
        max_retries: int = 2,
        cooldown_seconds: float = 60.0,
    ) -> None:
        self.max_retries = max_retries
        self.cooldown_seconds = cooldown_seconds
        self._traces: Dict[str, _TraceState] = {}

    # -- public API ---------------------------------------------------------

    def check(
        self, trace_id: str, raw_output: str
    ) -> Tuple[bool, Optional[dict]]:
        """
        Returns ``(should_retry, fallback_dict)``.

        * ``(True, None)``  → safe to retry.
        * ``(False, {...})`` → circuit open; use the fallback.
        """
        ts = self._ensure(trace_id)

        # 1. Poison → immediate open
        if self._is_poisoned(raw_output):
            return self._trip(
                ts, trace_id, "poison_pattern",
                f"Mock engine detectado: {raw_output[:80]}…",
            )

        # 2. If already OPEN, check cooldown before anything else
        if ts.state is CircuitState.OPEN:
            elapsed = time.time() - ts.first_fail
            if elapsed >= self.cooldown_seconds:
                # half-open probe: reset counters and allow one retry
                logger.info("CB [%s]: half-open after %.1fs", trace_id, elapsed)
                ts.state = CircuitState.HALF_OPEN
                ts.retries = 1
                ts.first_fail = time.time()
                return True, None
            return False, self._fallback(trace_id, "Cooldown activo.")

        # 3. Increment & check cap
        ts.retries += 1
        if ts.first_fail == 0.0:
            ts.first_fail = time.time()

        if ts.retries > self.max_retries:
            return self._trip(
                ts, trace_id, "max_retries",
                f"Límite de {self.max_retries} reintentos alcanzado.",
            )

        # 4. Closed — allow
        return True, None

    def reset(self, trace_id: str) -> None:
        """Call after a *successful* LLM response to close the circuit."""
        self._traces.pop(trace_id, None)

    def get_stats(self) -> Dict[str, dict]:
        return {
            tid: {
                "retries": s.retries,
                "state": s.state.name,
                "reason": s.last_reason,
                "age_s": round(time.time() - s.first_fail, 1) if s.first_fail else 0,
            }
            for tid, s in self._traces.items()
        }

    def __repr__(self) -> str:
        open_count = sum(1 for s in self._traces.values() if s.state is CircuitState.OPEN)
        return f"<CircuitBreaker max={self.max_retries} open={open_count}/{len(self._traces)}>"

    # -- internals ----------------------------------------------------------

    def _ensure(self, trace_id: str) -> _TraceState:
        if trace_id not in self._traces:
            self._traces[trace_id] = _TraceState()
        return self._traces[trace_id]

    @staticmethod
    def _is_poisoned(text: str) -> bool:
        return any(p in text for p in POISON_PATTERNS)

    def _trip(
        self,
        ts: _TraceState,
        trace_id: str,
        reason: str,
        user_reason: str,
    ) -> Tuple[bool, dict]:
        ts.state = CircuitState.OPEN
        ts.last_reason = reason
        if ts.first_fail == 0.0:
            ts.first_fail = time.time()
        logger.warning("CB [%s]: OPEN — %s", trace_id, reason)
        return False, self._fallback(trace_id, user_reason)

    @staticmethod
    def _fallback(trace_id: str, reason: str) -> dict:
        """Return a valid AgentAction dict that will pass Pydantic validation."""
        return {
            "thought": f"[CIRCUIT BREAKER] trace={trace_id} — {reason}",
            "tool": None,
            "tool_input": None,
            "final_answer": (
                "⚠️ No fue posible procesar tu solicitud. "
                "El sistema detuvo los reintentos automáticos. "
                f"Detalle: {reason} "
                "Intenta de nuevo más tarde o verifica la configuración."
            ),
            "handoff": None,
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

circuit_breaker = CircuitBreaker(max_retries=2, cooldown_seconds=60.0)
