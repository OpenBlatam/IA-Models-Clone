"""
Monitoring Script for Document Analyzer
=========================================

Continuous monitoring and health checking script.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.performance_monitor import performance_monitor
from core.health_checker import health_checker
from core.alerting_system import alerting_system, AlertSeverity, AlertChannel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def monitor_loop(interval: float = 60.0):
    """Main monitoring loop"""
    logger.info("Starting monitoring loop...")
    
    # Start performance monitoring
    performance_monitor.start_monitoring(interval=1.0)
    
    # Register health checks
    async def check_system_health():
        return {
            "status": "healthy",
            "message": "System operational"
        }
    
    health_checker.register_check("system", check_system_health)
    
    # Register alerts
    async def check_high_memory(context):
        usage = performance_monitor.get_system_metrics()
        memory_percent = usage.get("memory", {}).get("percent", 0)
        return memory_percent > 90
    
    alerting_system.register_alert(
        name="high_memory",
        condition=check_high_memory,
        severity=AlertSeverity.CRITICAL,
        channels=[AlertChannel.LOG],
        message_template="Memory usage is critically high: {memory_percent}%"
    )
    
    while True:
        try:
            # Run health checks
            health = await health_checker.get_overall_health()
            logger.info(f"Health status: {health['status']}")
            
            # Check alerts
            await alerting_system.check_all_alerts({
                "memory_percent": performance_monitor.get_system_metrics().get("memory", {}).get("percent", 0)
            })
            
            # Get performance stats
            stats = performance_monitor.get_all_stats()
            logger.info(f"Performance stats: {stats}")
            
            await asyncio.sleep(interval)
        
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")
            await asyncio.sleep(interval)

if __name__ == "__main__":
    asyncio.run(monitor_loop())
















