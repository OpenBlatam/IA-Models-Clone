from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import os
import sys
import logging
import argparse
import asyncio
from pathlib import Path
from typing import Optional
from gradio_interface import GradioAIVideoApp
import gradio as gr
            import torch
            import shutil
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Gradio Launcher for AI Video System

Launches the Gradio web interface with proper configuration,
error handling, and system checks.
"""


# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent))


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gradio_app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class GradioLauncher:
    """Launcher for the Gradio AI Video application"""
    
    def __init__(self) -> Any:
        self.app = None
        self.server_name = "0.0.0.0"
        self.server_port = 7860
        self.share = False
        self.debug = False
        self.show_error = True
        
    def check_dependencies(self) -> bool:
        """Check if all required dependencies are available"""
        
        required_packages = [
            'gradio',
            'torch',
            'numpy',
            'opencv-python',
            'pillow',
            'plotly',
            'pandas'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            logger.error(f"Missing required packages: {', '.join(missing_packages)}")
            logger.error("Please install them using: pip install " + " ".join(missing_packages))
            return False
        
        logger.info("All dependencies are available")
        return True
    
    def check_gpu_availability(self) -> bool:
        """Check if GPU is available for video processing"""
        
        try:
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"GPU available: {gpu_name} (Count: {gpu_count})")
                return True
            else:
                logger.warning("No GPU available. Video processing will be slower.")
                return False
        except Exception as e:
            logger.warning(f"Could not check GPU availability: {e}")
            return False
    
    def check_disk_space(self) -> bool:
        """Check available disk space"""
        
        try:
            total, used, free = shutil.disk_usage("/")
            free_gb = free // (1024**3)
            
            if free_gb < 10:
                logger.warning(f"Low disk space: {free_gb}GB available")
                return False
            else:
                logger.info(f"Disk space available: {free_gb}GB")
                return True
        except Exception as e:
            logger.warning(f"Could not check disk space: {e}")
            return True
    
    def create_directories(self) -> Any:
        """Create necessary directories for the application"""
        
        directories = [
            "outputs",
            "temp",
            "cache",
            "logs"
        ]
        
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
            logger.info(f"Created directory: {directory}")
    
    def setup_environment(self) -> Any:
        """Setup the environment for the application"""
        
        # Set environment variables
        os.environ['GRADIO_SERVER_NAME'] = self.server_name
        os.environ['GRADIO_SERVER_PORT'] = str(self.server_port)
        os.environ['GRADIO_SHARE'] = str(self.share).lower()
        
        # Create directories
        self.create_directories()
        
        logger.info("Environment setup completed")
    
    def parse_arguments(self) -> Any:
        """Parse command line arguments"""
        
        parser = argparse.ArgumentParser(
            description="Launch AI Video Gradio Interface",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  python gradio_launcher.py
  python gradio_launcher.py --port 8080 --share
  python gradio_launcher.py --host 127.0.0.1 --debug
            """
        )
        
        parser.add_argument(
            '--host',
            type=str,
            default="0.0.0.0",
            help="Host to bind the server to (default: 0.0.0.0)"
        )
        
        parser.add_argument(
            '--port',
            type=int,
            default=7860,
            help="Port to bind the server to (default: 7860)"
        )
        
        parser.add_argument(
            '--share',
            action='store_true',
            help="Create a public link for the interface"
        )
        
        parser.add_argument(
            '--debug',
            action='store_true',
            help="Enable debug mode"
        )
        
        parser.add_argument(
            '--no-error-display',
            action='store_true',
            help="Disable error display in the interface"
        )
        
        parser.add_argument(
            '--skip-checks',
            action='store_true',
            help="Skip dependency and system checks"
        )
        
        args = parser.parse_args()
        
        # Update configuration
        self.server_name = args.host
        self.server_port = args.port
        self.share = args.share
        self.debug = args.debug
        self.show_error = not args.no_error_display
        
        return args
    
    def run_system_checks(self) -> bool:
        """Run all system checks"""
        
        logger.info("Running system checks...")
        
        checks = [
            ("Dependencies", self.check_dependencies),
            ("GPU Availability", self.check_gpu_availability),
            ("Disk Space", self.check_disk_space)
        ]
        
        all_passed = True
        
        for check_name, check_func in checks:
            try:
                if not check_func():
                    logger.warning(f"{check_name} check failed")
                    all_passed = False
            except Exception as e:
                logger.error(f"Error during {check_name} check: {e}")
                all_passed = False
        
        if all_passed:
            logger.info("All system checks passed")
        else:
            logger.warning("Some system checks failed, but continuing...")
        
        return all_passed
    
    def launch_app(self) -> None:
        """Launch the Gradio application"""
        
        try:
            logger.info("Initializing AI Video Gradio App...")
            
            # Create the app
            self.app = GradioAIVideoApp()
            gradio_app = self.app.create_app()
            
            logger.info(f"Launching Gradio app on {self.server_name}:{self.server_port}")
            
            # Launch the app
            gradio_app.launch(
                server_name=self.server_name,
                server_port=self.server_port,
                share=self.share,
                debug=self.debug,
                show_error=self.show_error,
                quiet=False
            )
            
        except Exception as e:
            logger.error(f"Error launching Gradio app: {e}")
            raise
    
    def run(self) -> None:
        """Main run method"""
        
        try:
            # Parse arguments
            args = self.parse_arguments()
            
            # Setup environment
            self.setup_environment()
            
            # Run system checks (unless skipped)
            if not args.skip_checks:
                self.run_system_checks()
            
            # Launch the app
            self.launch_app()
            
        except KeyboardInterrupt:
            logger.info("Application interrupted by user")
        except Exception as e:
            logger.error(f"Application error: {e}")
            sys.exit(1)


def main():
    """Main function"""
    
    launcher = GradioLauncher()
    launcher.run()


match __name__:
    case "__main__":
    main() 