"""
Telemetry Processor
===================

IoT telemetry processing.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class TelemetryData:
    """Telemetry data."""
    device_id: str
    timestamp: datetime
    metrics: Dict[str, Any]
    location: Optional[str] = None


class TelemetryProcessor:
    """IoT telemetry processor."""
    
    def __init__(self):
        self._telemetry: List[TelemetryData] = []
        self._processors: List[callable] = []
    
    def register_processor(self, processor: callable):
        """Register telemetry processor."""
        self._processors.append(processor)
        logger.info("Registered telemetry processor")
    
    async def process_telemetry(
        self,
        device_id: str,
        metrics: Dict[str, Any],
        location: Optional[str] = None
    ):
        """Process telemetry data."""
        telemetry = TelemetryData(
            device_id=device_id,
            timestamp=datetime.now(),
            metrics=metrics,
            location=location
        )
        
        self._telemetry.append(telemetry)
        
        # Keep only recent telemetry
        if len(self._telemetry) > 100000:
            self._telemetry = self._telemetry[-50000:]
        
        # Process with registered processors
        for processor in self._processors:
            try:
                if asyncio.iscoroutinefunction(processor):
                    await processor(telemetry)
                else:
                    processor(telemetry)
            except Exception as e:
                logger.error(f"Telemetry processor failed: {e}")
        
        logger.debug(f"Processed telemetry from {device_id}")
    
    def get_telemetry(
        self,
        device_id: Optional[str] = None,
        limit: int = 100
    ) -> List[TelemetryData]:
        """Get telemetry data."""
        telemetry = self._telemetry
        
        if device_id:
            telemetry = [t for t in telemetry if t.device_id == device_id]
        
        return telemetry[-limit:]
    
    def get_telemetry_stats(self) -> Dict[str, Any]:
        """Get telemetry statistics."""
        return {
            "total_telemetry": len(self._telemetry),
            "unique_devices": len(set(t.device_id for t in self._telemetry)),
            "processors": len(self._processors)
        }


# Import asyncio
import asyncio















