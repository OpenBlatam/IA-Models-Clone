from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import asyncio
import argparse
import signal
from typing import Optional
from onyx.utils.logger import setup_logger
from onyx.utils.telemetry import TelemetryLogger
from .commands import (
from ..onyx_main import get_system
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Onyx AI Video System - CLI Main Class

Main command-line interface class for the Onyx-adapted AI Video system.
Provides initialization, shutdown, and command routing.
"""


# Onyx imports

# Local imports
    start_system, show_status, show_metrics, handle_config,
    generate_video, handle_plugins, run_tests, health_check
)

logger = setup_logger(__name__)


class OnyxAIVideoCLI:
    """
    Command-line interface for Onyx AI Video System.
    
    Provides comprehensive CLI for system management, configuration,
    and video generation operations.
    """
    
    def __init__(self) -> Any:
        self.logger = setup_logger("onyx_ai_video_cli")
        self.telemetry = TelemetryLogger()
        self.system: Optional[object] = None
    
    async def initialize(self) -> None:
        """Initialize the CLI and system."""
        try:
            self.logger.info("Initializing Onyx AI Video CLI")
            
            # Initialize system
            self.system = await get_system()
            
            self.logger.info("Onyx AI Video CLI initialized successfully")
            
        except Exception as e:
            self.logger.error(f"CLI initialization failed: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown the CLI and system."""
        try:
            if self.system:
                await self.system.shutdown()
            
            self.logger.info("Onyx AI Video CLI shutdown completed")
            
        except Exception as e:
            self.logger.error(f"CLI shutdown failed: {e}")
    
    async def run_command(self, args: argparse.Namespace) -> int:
        """Run CLI command."""
        try:
            command = args.command
            
            if command == "start":
                return await start_system(self, args)
            elif command == "status":
                return await show_status(self, args)
            elif command == "metrics":
                return await show_metrics(self, args)
            elif command == "config":
                return await handle_config(self, args)
            elif command == "generate":
                return await generate_video(self, args)
            elif command == "plugins":
                return await handle_plugins(self, args)
            elif command == "test":
                return await run_tests(self, args)
            elif command == "health":
                return await health_check(self, args)
            else:
                self.logger.error(f"Unknown command: {command}")
                return 1
                
        except Exception as e:
            self.logger.error(f"Command execution failed: {e}")
            return 1 