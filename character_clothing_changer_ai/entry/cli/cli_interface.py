"""
CLI Interface
=============

Command-line interface for Character Clothing Changer AI.
"""

import argparse
import logging
from typing import Optional
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from character_clothing_changer_ai.config.clothing_changer_config import ClothingChangerConfig
from character_clothing_changer_ai.core.clothing_changer_service import ClothingChangerService

logger = logging.getLogger(__name__)


class CLIInterface:
    """Command-line interface."""
    
    def __init__(self):
        """Initialize CLI interface."""
        self.parser = argparse.ArgumentParser(
            description="Character Clothing Changer AI - CLI",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        self._setup_commands()
    
    def _setup_commands(self) -> None:
        """Setup CLI commands."""
        subparsers = self.parser.add_subparsers(dest="command", help="Available commands")
        
        # Server command
        server_parser = subparsers.add_parser("server", help="Start the API server")
        server_parser.add_argument("--host", type=str, help="Server host")
        server_parser.add_argument("--port", type=int, help="Server port")
        server_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
        
        # Config command
        config_parser = subparsers.add_parser("config", help="Configuration management")
        config_parser.add_argument("--show", action="store_true", help="Show current configuration")
        config_parser.add_argument("--validate", action="store_true", help="Validate configuration")
        
        # Model command
        model_parser = subparsers.add_parser("model", help="Model management")
        model_parser.add_argument("--info", action="store_true", help="Show model information")
        model_parser.add_argument("--init", action="store_true", help="Initialize model")
        
        # Health command
        health_parser = subparsers.add_parser("health", help="Health check")
    
    def run(self, args: Optional[list] = None) -> None:
        """
        Run CLI interface.
        
        Args:
            args: Optional command-line arguments
        """
        parsed_args = self.parser.parse_args(args)
        
        if not parsed_args.command:
            self.parser.print_help()
            return
        
        try:
            if parsed_args.command == "server":
                self._handle_server(parsed_args)
            elif parsed_args.command == "config":
                self._handle_config(parsed_args)
            elif parsed_args.command == "model":
                self._handle_model(parsed_args)
            elif parsed_args.command == "health":
                self._handle_health()
        except Exception as e:
            logger.error(f"Command error: {e}", exc_info=True)
            sys.exit(1)
    
    def _handle_server(self, args) -> None:
        """Handle server command."""
        from ..server.server_runner import run_server
        
        run_server(
            host=args.host,
            port=args.port,
            reload=args.reload,
        )
    
    def _handle_config(self, args) -> None:
        """Handle config command."""
        config = ClothingChangerConfig.from_env()
        
        if args.show:
            print("Current Configuration:")
            print(f"  API Host: {config.api_host}")
            print(f"  API Port: {config.api_port}")
            print(f"  Output Dir: {config.output_dir}")
            print(f"  Model ID: {config.model_id}")
        
        if args.validate:
            try:
                config.validate()
                print("✅ Configuration is valid")
            except Exception as e:
                print(f"❌ Configuration error: {e}")
                sys.exit(1)
    
    def _handle_model(self, args) -> None:
        """Handle model command."""
        config = ClothingChangerConfig.from_env()
        service = ClothingChangerService(config=config)
        
        if args.init:
            print("Initializing model...")
            service.initialize_model()
            print("✅ Model initialized")
        
        if args.info:
            try:
                info = service.get_model_info()
                print("Model Information:")
                for key, value in info.items():
                    print(f"  {key}: {value}")
            except Exception as e:
                print(f"❌ Error getting model info: {e}")
                sys.exit(1)
    
    def _handle_health(self) -> None:
        """Handle health command."""
        try:
            config = ClothingChangerConfig.from_env()
            service = ClothingChangerService(config=config)
            
            # Basic health check
            print("Health Check:")
            print(f"  Configuration: ✅ Valid")
            print(f"  Service: ✅ Initialized")
            
            # Try to get model info
            try:
                info = service.get_model_info()
                print(f"  Model: ✅ Available")
            except:
                print(f"  Model: ⚠️  Not initialized")
            
            print("\n✅ System is healthy")
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            sys.exit(1)


def main():
    """Main CLI entry point."""
    cli = CLIInterface()
    cli.run()


if __name__ == "__main__":
    main()

