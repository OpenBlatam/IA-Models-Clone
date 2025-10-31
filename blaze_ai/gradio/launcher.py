"""
Gradio demo launcher for Blaze AI module.
"""
from __future__ import annotations

import asyncio
import gradio as gr
import argparse
import sys
from pathlib import Path
from typing import Optional, List

from ..core.interfaces import CoreConfig
from ..utils.logging import get_logger
from .interface import BlazeAIGradioInterface, create_blaze_ai_interface
from .demos import (
    InteractiveModelDemos,
    create_text_generation_demo,
    create_image_generation_demo,
    create_model_comparison_demo,
    create_training_visualization_demo,
    create_performance_analysis_demo,
    create_error_analysis_demo
)

logger = get_logger(__name__)

class GradioLauncher:
    """Launcher for Gradio demos and interfaces."""
    
    def __init__(self, config: Optional[CoreConfig] = None):
        self.config = config or CoreConfig()
        self.logger = get_logger(__name__)
        self.interfaces = {}
        
    def launch_main_interface(self, port: int = 7860, share: bool = False) -> None:
        """Launch the main Blaze AI interface."""
        try:
            self.logger.info("Launching main Blaze AI interface...")
            
            interface = create_blaze_ai_interface(self.config)
            interface.launch(
                server_name="0.0.0.0",
                server_port=port,
                share=share,
                debug=False,
                show_error=True
            )
            
        except Exception as e:
            self.logger.error(f"Failed to launch main interface: {e}")
            raise
    
    def launch_text_generation_demo(self, port: int = 7861, share: bool = False) -> None:
        """Launch text generation demo."""
        try:
            self.logger.info("Launching text generation demo...")
            
            demo = create_text_generation_demo(self.config)
            demo.launch(
                server_name="0.0.0.0",
                server_port=port,
                share=share,
                debug=False,
                show_error=True
            )
            
        except Exception as e:
            self.logger.error(f"Failed to launch text generation demo: {e}")
            raise
    
    def launch_image_generation_demo(self, port: int = 7862, share: bool = False) -> None:
        """Launch image generation demo."""
        try:
            self.logger.info("Launching image generation demo...")
            
            demo = create_image_generation_demo(self.config)
            demo.launch(
                server_name="0.0.0.0",
                server_port=port,
                share=share,
                debug=False,
                show_error=True
            )
            
        except Exception as e:
            self.logger.error(f"Failed to launch image generation demo: {e}")
            raise
    
    def launch_model_comparison_demo(self, port: int = 7863, share: bool = False) -> None:
        """Launch model comparison demo."""
        try:
            self.logger.info("Launching model comparison demo...")
            
            demo = create_model_comparison_demo(self.config)
            demo.launch(
                server_name="0.0.0.0",
                server_port=port,
                share=share,
                debug=False,
                show_error=True
            )
            
        except Exception as e:
            self.logger.error(f"Failed to launch model comparison demo: {e}")
            raise
    
    def launch_training_visualization_demo(self, port: int = 7864, share: bool = False) -> None:
        """Launch training visualization demo."""
        try:
            self.logger.info("Launching training visualization demo...")
            
            demo = create_training_visualization_demo(self.config)
            demo.launch(
                server_name="0.0.0.0",
                server_port=port,
                share=share,
                debug=False,
                show_error=True
            )
            
        except Exception as e:
            self.logger.error(f"Failed to launch training visualization demo: {e}")
            raise
    
    def launch_performance_analysis_demo(self, port: int = 7865, share: bool = False) -> None:
        """Launch performance analysis demo."""
        try:
            self.logger.info("Launching performance analysis demo...")
            
            demo = create_performance_analysis_demo(self.config)
            demo.launch(
                server_name="0.0.0.0",
                server_port=port,
                share=share,
                debug=False,
                show_error=True
            )
            
        except Exception as e:
            self.logger.error(f"Failed to launch performance analysis demo: {e}")
            raise
    
    def launch_error_analysis_demo(self, port: int = 7866, share: bool = False) -> None:
        """Launch error analysis demo."""
        try:
            self.logger.info("Launching error analysis demo...")
            
            demo = create_error_analysis_demo(self.config)
            demo.launch(
                server_name="0.0.0.0",
                server_port=port,
                share=share,
                debug=False,
                show_error=True
            )
            
        except Exception as e:
            self.logger.error(f"Failed to launch error analysis demo: {e}")
            raise
    
    def launch_all_demos(self, base_port: int = 7860, share: bool = False) -> None:
        """Launch all demos on different ports."""
        try:
            self.logger.info("Launching all demos...")
            
            # Create all demos
            demos = [
                ("Main Interface", create_blaze_ai_interface(self.config), base_port),
                ("Text Generation", create_text_generation_demo(self.config), base_port + 1),
                ("Image Generation", create_image_generation_demo(self.config), base_port + 2),
                ("Model Comparison", create_model_comparison_demo(self.config), base_port + 3),
                ("Training Visualization", create_training_visualization_demo(self.config), base_port + 4),
                ("Performance Analysis", create_performance_analysis_demo(self.config), base_port + 5),
                ("Error Analysis", create_error_analysis_demo(self.config), base_port + 6)
            ]
            
            # Launch all demos
            for name, demo, port in demos:
                try:
                    self.logger.info(f"Launching {name} on port {port}...")
                    demo.launch(
                        server_name="0.0.0.0",
                        server_port=port,
                        share=share,
                        debug=False,
                        show_error=True,
                        quiet=True
                    )
                    self.interfaces[name] = {"demo": demo, "port": port}
                    
                except Exception as e:
                    self.logger.error(f"Failed to launch {name}: {e}")
            
            self.logger.info("All demos launched successfully!")
            self._print_demo_urls(base_port)
            
        except Exception as e:
            self.logger.error(f"Failed to launch all demos: {e}")
            raise
    
    def _print_demo_urls(self, base_port: int) -> None:
        """Print demo URLs."""
        print("\n" + "="*60)
        print("🚀 Blaze AI Gradio Demos Launched Successfully!")
        print("="*60)
        print(f"Main Interface:     http://localhost:{base_port}")
        print(f"Text Generation:    http://localhost:{base_port + 1}")
        print(f"Image Generation:   http://localhost:{base_port + 2}")
        print(f"Model Comparison:   http://localhost:{base_port + 3}")
        print(f"Training Viz:       http://localhost:{base_port + 4}")
        print(f"Performance:        http://localhost:{base_port + 5}")
        print(f"Error Analysis:     http://localhost:{base_port + 6}")
        print("="*60)
        print("Press Ctrl+C to stop all demos")
        print("="*60 + "\n")

def create_launcher(config: Optional[CoreConfig] = None) -> GradioLauncher:
    """Create a Gradio launcher instance."""
    return GradioLauncher(config)

def main():
    """Main entry point for the Gradio launcher."""
    parser = argparse.ArgumentParser(description="Launch Blaze AI Gradio demos")
    parser.add_argument(
        "--demo",
        choices=[
            "main", "text", "image", "comparison", 
            "training", "performance", "error", "all"
        ],
        default="main",
        help="Demo to launch (default: main)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Base port for demos (default: 7860)"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Share demos publicly"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = None
        if args.config:
            config_path = Path(args.config)
            if config_path.exists():
                config = CoreConfig.from_yaml(config_path)
                logger.info(f"Loaded configuration from {config_path}")
            else:
                logger.warning(f"Configuration file not found: {config_path}")
        
        # Create launcher
        launcher = create_launcher(config)
        
        # Launch selected demo
        if args.demo == "main":
            launcher.launch_main_interface(args.port, args.share)
        elif args.demo == "text":
            launcher.launch_text_generation_demo(args.port, args.share)
        elif args.demo == "image":
            launcher.launch_image_generation_demo(args.port, args.share)
        elif args.demo == "comparison":
            launcher.launch_model_comparison_demo(args.port, args.share)
        elif args.demo == "training":
            launcher.launch_training_visualization_demo(args.port, args.share)
        elif args.demo == "performance":
            launcher.launch_performance_analysis_demo(args.port, args.share)
        elif args.demo == "error":
            launcher.launch_error_analysis_demo(args.port, args.share)
        elif args.demo == "all":
            launcher.launch_all_demos(args.port, args.share)
            
            # Keep the process running
            try:
                while True:
                    import time
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Shutting down all demos...")
        
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    except Exception as e:
        logger.error(f"Failed to launch demo: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

__all__ = ["GradioLauncher", "create_launcher", "main"]
