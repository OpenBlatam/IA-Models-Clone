"""
Graceful Shutdown Manager for Color Grading AI
===============================================

Graceful shutdown handling for clean resource cleanup.
"""

import logging
import asyncio
import signal
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class ShutdownPhase(Enum):
    """Shutdown phases."""
    PRE_SHUTDOWN = "pre_shutdown"
    SHUTDOWN = "shutdown"
    POST_SHUTDOWN = "post_shutdown"


@dataclass
class ShutdownHandler:
    """Shutdown handler definition."""
    name: str
    handler_func: Callable
    phase: ShutdownPhase
    timeout: float = 10.0
    critical: bool = True


class GracefulShutdownManager:
    """
    Graceful shutdown manager.
    
    Features:
    - Phased shutdown
    - Handler registration
    - Timeout handling
    - Signal handling
    - Clean resource cleanup
    """
    
    def __init__(self, shutdown_timeout: float = 30.0):
        """
        Initialize graceful shutdown manager.
        
        Args:
            shutdown_timeout: Total shutdown timeout
        """
        self.shutdown_timeout = shutdown_timeout
        self._handlers: Dict[ShutdownPhase, List[ShutdownHandler]] = {
            ShutdownPhase.PRE_SHUTDOWN: [],
            ShutdownPhase.SHUTDOWN: [],
            ShutdownPhase.POST_SHUTDOWN: [],
        }
        self._shutting_down = False
        self._shutdown_event = asyncio.Event()
    
    def register_handler(
        self,
        name: str,
        handler_func: Callable,
        phase: ShutdownPhase = ShutdownPhase.SHUTDOWN,
        timeout: float = 10.0,
        critical: bool = True
    ):
        """
        Register shutdown handler.
        
        Args:
            name: Handler name
            handler_func: Handler function (async)
            phase: Shutdown phase
            timeout: Handler timeout
            critical: Whether handler is critical
        """
        handler = ShutdownHandler(
            name=name,
            handler_func=handler_func,
            phase=phase,
            timeout=timeout,
            critical=critical
        )
        
        self._handlers[phase].append(handler)
        logger.info(f"Registered shutdown handler: {name} (phase: {phase.value})")
    
    async def shutdown(self, reason: str = "manual"):
        """
        Perform graceful shutdown.
        
        Args:
            reason: Shutdown reason
        """
        if self._shutting_down:
            logger.warning("Shutdown already in progress")
            return
        
        self._shutting_down = True
        logger.info(f"Starting graceful shutdown (reason: {reason})")
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Phase 1: Pre-shutdown (stop accepting new requests)
            await self._run_phase(ShutdownPhase.PRE_SHUTDOWN, "pre-shutdown")
            
            # Phase 2: Shutdown (cleanup resources)
            await self._run_phase(ShutdownPhase.SHUTDOWN, "shutdown")
            
            # Phase 3: Post-shutdown (final cleanup)
            await self._run_phase(ShutdownPhase.POST_SHUTDOWN, "post-shutdown")
            
            elapsed = asyncio.get_event_loop().time() - start_time
            logger.info(f"Graceful shutdown completed in {elapsed:.2f}s")
        
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        finally:
            self._shutdown_event.set()
    
    async def _run_phase(self, phase: ShutdownPhase, phase_name: str):
        """Run shutdown phase."""
        handlers = self._handlers[phase]
        if not handlers:
            return
        
        logger.info(f"Running {phase_name} phase ({len(handlers)} handlers)")
        
        tasks = []
        for handler in handlers:
            task = asyncio.create_task(
                self._run_handler(handler, phase_name)
            )
            tasks.append(task)
        
        # Wait for all handlers with timeout
        done, pending = await asyncio.wait(
            tasks,
            timeout=self.shutdown_timeout,
            return_when=asyncio.ALL_COMPLETED
        )
        
        # Check for timeouts
        if pending:
            logger.warning(f"{len(pending)} handlers timed out in {phase_name} phase")
            for task in pending:
                task.cancel()
        
        # Check for failures
        for task in done:
            try:
                await task
            except Exception as e:
                logger.error(f"Handler failed in {phase_name} phase: {e}")
    
    async def _run_handler(self, handler: ShutdownHandler, phase_name: str):
        """Run single handler."""
        try:
            logger.debug(f"Running handler {handler.name} in {phase_name} phase")
            
            if asyncio.iscoroutinefunction(handler.handler_func):
                await asyncio.wait_for(
                    handler.handler_func(),
                    timeout=handler.timeout
                )
            else:
                # Sync function, run in executor
                loop = asyncio.get_event_loop()
                await asyncio.wait_for(
                    loop.run_in_executor(None, handler.handler_func),
                    timeout=handler.timeout
                )
            
            logger.debug(f"Handler {handler.name} completed")
        
        except asyncio.TimeoutError:
            logger.error(f"Handler {handler.name} timed out after {handler.timeout}s")
            if handler.critical:
                raise
        except Exception as e:
            logger.error(f"Handler {handler.name} failed: {e}")
            if handler.critical:
                raise
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown")
            asyncio.create_task(self.shutdown(reason=f"signal_{signum}"))
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        logger.info("Signal handlers registered")
    
    async def wait_for_shutdown(self):
        """Wait for shutdown event."""
        await self._shutdown_event.wait()
    
    def is_shutting_down(self) -> bool:
        """Check if shutdown is in progress."""
        return self._shutting_down




