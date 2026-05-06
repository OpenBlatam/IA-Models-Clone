"""
System 5.9 — Service Registry.

Centralised component discovery, dependency resolution,
and boot sequencing for core system services.
"""

import logging
from typing import Any, Dict, Optional, Type, TypeVar

logger = logging.getLogger("sys5.registry")

T = TypeVar("T")


class Registry:
    """Singleton-friendly service locator for the System 5.9 stack."""

    def __init__(self) -> None:
        self._services: Dict[str, Any] = {}
        self._booted: bool = False

    # -- CRUD ---------------------------------------------------------------

    def register(self, name: str, service: Any) -> None:
        self._services[name] = service
        logger.debug("Registered service: %s", name)

    def get(self, name: str) -> Optional[Any]:
        return self._services.get(name)

    def resolve(self, service_type: Type[T]) -> Optional[T]:
        """Return the first service that is an instance of *service_type*."""
        return next(
            (s for s in self._services.values() if isinstance(s, service_type)),
            None,
        )

    @property
    def service_names(self) -> list:
        return list(self._services)

    # -- boot ---------------------------------------------------------------

    def boot(self) -> None:
        """
        Initialise and register core System 5.9 services.

        Idempotent — safe to call more than once.
        """
        if self._booted:
            return

        self._boot_one("TelemetryService",  self._make_telemetry)
        self._boot_one("CircuitBreaker",    self._make_circuit_breaker)
        self._boot_one("PromptSanitizer",   self._make_prompt_sanitizer)

        self._booted = True
        logger.info("Boot complete. Services: %s", self.service_names)

    # -- repr ---------------------------------------------------------------

    def __repr__(self) -> str:
        return f"<Registry services={self.service_names} booted={self._booted}>"

    # -- private boot helpers -----------------------------------------------

    def _boot_one(self, name: str, factory) -> None:
        try:
            self.register(name, factory())
            logger.info("Booted %s", name)
        except Exception as exc:
            logger.error("Failed to boot %s: %s", name, exc)

    @staticmethod
    def _make_telemetry():
        from .telemetry import TelemetryService
        return TelemetryService()

    @staticmethod
    def _make_circuit_breaker():
        from .circuit_breaker import CircuitBreaker
        return CircuitBreaker(max_retries=2, cooldown_seconds=60.0)

    @staticmethod
    def _make_prompt_sanitizer():
        from .prompt_sanitizer import PromptSanitizer
        return PromptSanitizer()


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

registry = Registry()
