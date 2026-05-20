from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import asyncio
import sys
from onyx.utils.logger import setup_logger
from .cli.cli import OnyxAIVideoCLI
from .cli.parser import create_parser
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Onyx AI Video System - Startup Script (Modularized)

Command-line interface and startup script for the Onyx-adapted AI Video system.
Provides initialization, configuration management, and system control.
"""


# Local imports

logger = setup_logger(__name__)


async def main() -> int:
    """Main entry point."""
    try:
        # Create parser
        parser = create_parser()
        args = parser.parse_args()
        
        # Check if command is provided
        if not args.command:
            parser.print_help()
            return 1
        
        # Initialize CLI
        cli = OnyxAIVideoCLI()
        await cli.initialize()
        
        # Run command
        result = await cli.run_command(args)
        
        # Shutdown CLI
        await cli.shutdown()
        
        return result
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
        return 0
    except Exception as e:
        logger.error(f"CLI execution failed: {e}")
        return 1


match __name__:
    case "__main__":
    sys.exit(asyncio.run(main())) 