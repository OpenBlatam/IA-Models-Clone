"""
Backpressure Manager for Color Grading AI
==========================================

Backpressure handling for overload protection.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, Callable
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class BackpressureLevel(Enum):
    """Backpressure levels."""
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    OVERLOAD = "overload"


@dataclass
class BackpressureConfig:
    """Backpressure configuration."""
    warning_threshold: float = 0.7  # 70% capacity
    critical_threshold: float = 0.9  # 90% capacity
    overload_threshold: float = 0.95  # 95% capacity
    check_interval: float = 1.0  # seconds


class BackpressureManager:
    """
    Backpressure manager for overload protection.
    
    Features:
    - Automatic pressure detection
    - Adaptive throttling
    - Graceful degradation
    - Statistics
    """
    
    def __init__(self, config: Optional[BackpressureConfig] = None):
        """
        Initialize backpressure manager.
        
        Args:
            config: Optional backpressure configuration
        """
        self.config = config or BackpressureConfig()
        self._current_pressure: float = 0.0
        self._level = BackpressureLevel.NORMAL
        self._stats: Dict[str, Any] = {
            "checks": 0,
            "warnings": 0,
            "criticals": 0,
            "overloads": 0,
        }
        self._handlers: Dict[BackpressureLevel, Callable] = {}
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
    
    def register_handler(self, level: BackpressureLevel, handler: Callable):
        """
        Register handler for backpressure level.
        
        Args:
            level: Backpressure level
            handler: Handler function
        """
        self._handlers[level] = handler
        logger.info(f"Registered handler for {level.value}")
    
    def get_pressure(self) -> float:
        """
        Get current pressure (0.0 - 1.0).
        
        Returns:
            Current pressure level
        """
        return self._current_pressure
    
    def get_level(self) -> BackpressureLevel:
        """
        Get current backpressure level.
        
        Returns:
            Current level
        """
        return self._level
    
    def update_pressure(self, pressure: float):
        """
        Update current pressure.
        
        Args:
            pressure: Pressure value (0.0 - 1.0)
        """
        self._current_pressure = max(0.0, min(1.0, pressure))
        self._update_level()
    
    def _update_level(self):
        """Update backpressure level based on pressure."""
        old_level = self._level
        
        if self._current_pressure >= self.config.overload_threshold:
            self._level = BackpressureLevel.OVERLOAD
            self._stats["overloads"] += 1
        elif self._current_pressure >= self.config.critical_threshold:
            self._level = BackpressureLevel.CRITICAL
            self._stats["criticals"] += 1
        elif self._current_pressure >= self.config.warning_threshold:
            self._level = BackpressureLevel.WARNING
            self._stats["warnings"] += 1
        else:
            self._level = BackpressureLevel.NORMAL
        
        # Trigger handler if level changed
        if old_level != self._level:
            handler = self._handlers.get(self._level)
            if handler:
                try:
                    handler(self._level, self._current_pressure)
                except Exception as e:
                    logger.error(f"Error in backpressure handler: {e}")
    
    async def start_monitoring(self, pressure_callback: Callable):
        """
        Start automatic pressure monitoring.
        
        Args:
            pressure_callback: Callback to get current pressure
        """
        if self._running:
            return
        
        self._running = True
        self._monitor_task = asyncio.create_task(
            self._monitor_loop(pressure_callback)
        )
        logger.info("Backpressure monitoring started")
    
    async def stop_monitoring(self):
        """Stop automatic pressure monitoring."""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Backpressure monitoring stopped")
    
    async def _monitor_loop(self, pressure_callback: Callable):
        """Monitor pressure in loop."""
        while self._running:
            try:
                pressure = await pressure_callback()
                self.update_pressure(pressure)
                self._stats["checks"] += 1
            except Exception as e:
                logger.error(f"Error in pressure monitoring: {e}")
            
            await asyncio.sleep(self.config.check_interval)
    
    def should_throttle(self) -> bool:
        """
        Check if should throttle requests.
        
        Returns:
            True if should throttle
        """
        return self._level in [
            BackpressureLevel.CRITICAL,
            BackpressureLevel.OVERLOAD
        ]
    
    def get_throttle_factor(self) -> float:
        """
        Get throttle factor (0.0 - 1.0).
        
        Returns:
            Throttle factor
        """
        if self._level == BackpressureLevel.OVERLOAD:
            return 0.1  # 10% of normal capacity
        elif self._level == BackpressureLevel.CRITICAL:
            return 0.3  # 30% of normal capacity
        elif self._level == BackpressureLevel.WARNING:
            return 0.7  # 70% of normal capacity
        else:
            return 1.0  # 100% capacity
    
    def get_stats(self) -> Dict[str, Any]:
        """Get backpressure statistics."""
        return {
            **self._stats,
            "current_pressure": self._current_pressure,
            "current_level": self._level.value,
            "should_throttle": self.should_throttle(),
            "throttle_factor": self.get_throttle_factor(),
        }
    
    def reset_stats(self):
        """Reset statistics."""
        self._stats = {
            "checks": 0,
            "warnings": 0,
            "criticals": 0,
            "overloads": 0,
        }




