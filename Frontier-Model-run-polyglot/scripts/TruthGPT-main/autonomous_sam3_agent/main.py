"""
Main Entry Point for Autonomous SAM3 Agent
==========================================

Run the autonomous agent in 24/7 continuous mode with:
- Auto-restart on crash (watchdog)
- Exponential backoff on failures
- Graceful shutdown support
- Optional dashboard and components
"""

import asyncio
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Optional, List
import yaml

from core.agent_core import AutonomousSAM3Agent

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("autonomous_agent.log"),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)


class AgentWatchdog:
    """
    Watchdog that monitors and restarts the agent on failures.
    
    Features:
    - Auto-restart on crash
    - Exponential backoff
    - Maximum restart attempts
    - Graceful shutdown support
    """
    
    def __init__(
        self,
        config: dict,
        max_restarts: int = 5,
        backoff_seconds: Optional[List[float]] = None,
    ):
        """
        Initialize watchdog.
        
        Args:
            config: Agent configuration
            max_restarts: Maximum restart attempts before giving up
            backoff_seconds: List of wait times between restarts
        """
        self.config = config
        self.max_restarts = max_restarts
        self.backoff_seconds = backoff_seconds or [5, 10, 30, 60, 300]
        
        self.agent: Optional[AutonomousSAM3Agent] = None
        self.restart_count = 0
        self.last_restart_time = 0.0
        self.running = False
        self._shutdown_event = asyncio.Event()
        
        # Optional components
        self.dashboard = None
        self.health_monitor = None
        self.auto_scaler = None
        self.task_scheduler = None
        self.plugin_manager = None
    
    async def start(self):
        """Start the watchdog and agent."""
        self.running = True
        logger.info("=" * 60)
        logger.info("Starting Autonomous SAM3 Agent with Watchdog")
        logger.info("=" * 60)
        
        while self.running:
            try:
                await self._run_agent()
            except asyncio.CancelledError:
                logger.info("Watchdog cancelled")
                break
            except Exception as e:
                logger.error(f"Agent crashed: {e}", exc_info=True)
                
                if not self.running:
                    break
                
                # Check if we should restart
                if self.restart_count >= self.max_restarts:
                    logger.error(
                        f"Maximum restarts ({self.max_restarts}) exceeded. "
                        "Giving up."
                    )
                    break
                
                # Calculate backoff
                backoff_idx = min(self.restart_count, len(self.backoff_seconds) - 1)
                wait_time = self.backoff_seconds[backoff_idx]
                
                self.restart_count += 1
                logger.info(
                    f"Restarting agent in {wait_time}s "
                    f"(attempt {self.restart_count}/{self.max_restarts})"
                )
                
                await asyncio.sleep(wait_time)
        
        await self._cleanup()
        logger.info("Watchdog stopped")
    
    async def stop(self):
        """Stop the watchdog and agent."""
        logger.info("Stopping watchdog...")
        self.running = False
        self._shutdown_event.set()
        
        if self.agent:
            await self.agent.stop()
    
    async def _run_agent(self):
        """Run the agent with optional components."""
        # Extract configuration
        openrouter_config = self.config.get("openrouter", {})
        sam3_config = self.config.get("sam3", {})
        agent_config = self.config.get("agent", {})
        scaling_config = self.config.get("scaling", {})
        health_config = self.config.get("health", {})
        dashboard_config = self.config.get("dashboard", {})
        
        # Initialize agent
        self.agent = AutonomousSAM3Agent(
            openrouter_api_key=openrouter_config.get("api_key"),
            sam3_model_path=sam3_config.get("model_path"),
            max_parallel_tasks=agent_config.get("max_parallel_tasks", 10),
            output_dir=agent_config.get("output_dir", "autonomous_agent_output"),
            model=openrouter_config.get("model", "anthropic/claude-3.5-sonnet"),
            debug=agent_config.get("debug", False),
        )
        
        # Initialize optional components
        await self._init_optional_components(
            scaling_config, health_config, dashboard_config
        )
        
        # Start agent
        await self.agent.start()
        
        # Reset restart count on successful start
        self.restart_count = 0
        self.last_restart_time = time.time()
        
        # Wait until shutdown
        await self._shutdown_event.wait()
    
    async def _init_optional_components(
        self,
        scaling_config: dict,
        health_config: dict,
        dashboard_config: dict,
    ):
        """Initialize optional components based on configuration."""
        
        # Auto-scaler
        if scaling_config.get("enabled", False):
            try:
                from core.auto_scaler import AutoScaler, ScalingConfig
                
                self.auto_scaler = AutoScaler(
                    config=ScalingConfig(
                        min_workers=scaling_config.get("min_workers", 2),
                        max_workers=scaling_config.get("max_workers", 20),
                        scale_up_threshold=scaling_config.get("scale_up_threshold", 0.8),
                        scale_down_threshold=scaling_config.get("scale_down_threshold", 0.2),
                    )
                )
                await self.auto_scaler.start(self.agent.parallel_executor)
                logger.info("Auto-scaler enabled")
            except ImportError as e:
                logger.warning(f"Could not enable auto-scaler: {e}")
        
        # Health monitor
        if health_config.get("enabled", True):
            try:
                from core.health_monitor import HealthMonitor, HealthConfig
                
                self.health_monitor = HealthMonitor(
                    config=HealthConfig(
                        check_interval_seconds=health_config.get("check_interval_seconds", 30),
                        max_memory_percent=health_config.get("max_memory_percent", 90.0),
                    )
                )
                await self.health_monitor.start()
                logger.info("Health monitor enabled")
            except ImportError as e:
                logger.warning(f"Could not enable health monitor: {e}")
        
        # Dashboard
        if dashboard_config.get("enabled", False):
            try:
                from infrastructure.dashboard import Dashboard
                
                self.dashboard = Dashboard(
                    host=dashboard_config.get("host", "0.0.0.0"),
                    port=dashboard_config.get("port", 8080),
                    agent=self.agent,
                )
                await self.dashboard.start()
                logger.info(f"Dashboard enabled at http://0.0.0.0:{dashboard_config.get('port', 8080)}")
            except ImportError as e:
                logger.warning(f"Could not enable dashboard: {e}")
        
        # Plugin manager
        try:
            from core.plugin_system import PluginManager
            
            self.plugin_manager = PluginManager()
            await self.plugin_manager.load_plugins()
            logger.info(f"Plugin manager enabled ({len(self.plugin_manager.get_plugins())} plugins)")
        except ImportError as e:
            logger.warning(f"Could not enable plugin manager: {e}")
    
    async def _cleanup(self):
        """Cleanup all components."""
        if self.dashboard:
            await self.dashboard.stop()
        
        if self.health_monitor:
            await self.health_monitor.stop()
        
        if self.auto_scaler:
            await self.auto_scaler.stop()
        
        if self.agent:
            await self.agent.stop()


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    config_path = Path(config_path)
    if not config_path.exists():
        logger.warning(f"Config file {config_path} not found, using defaults")
        return {}
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Expand environment variables
    import os
    def expand_env_vars(obj):
        if isinstance(obj, dict):
            return {k: expand_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [expand_env_vars(item) for item in obj]
        elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
            env_var = obj[2:-1]
            return os.getenv(env_var, obj)
        return obj
    
    config = expand_env_vars(config)
    return config


async def main():
    """Main entry point for autonomous agent."""
    # Load configuration
    config = load_config()
    
    # Get watchdog settings
    agent_config = config.get("agent", {})
    
    # Create watchdog
    watchdog = AgentWatchdog(
        config=config,
        max_restarts=agent_config.get("max_restart_attempts", 5),
        backoff_seconds=agent_config.get("restart_backoff_seconds", [5, 10, 30, 60, 300]),
    )
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        logger.info("Received shutdown signal, stopping agent...")
        asyncio.create_task(watchdog.stop())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        await watchdog.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        await watchdog.stop()
        logger.info("Agent stopped")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutdown complete")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
