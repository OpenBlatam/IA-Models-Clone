from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import os
import sys
import subprocess
import shutil
from pathlib import Path
import argparse
import json
import yaml
from typing import Dict, Any, List
from onyx_ai_video.plugins.plugin_base import OnyxPluginBase, OnyxPluginContext
import re
                import asyncio
                from onyx_ai_video.api.main import OnyxAIVideoSystem
import sys
import os
from pathlib import Path
from onyx_ai_video.cli.main import main
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Onyx AI Video System - Installation Script

Automated installation and setup script for the Onyx AI Video system.
"""



class OnyxAIVideoInstaller:
    """Installer for Onyx AI Video system."""
    
    def __init__(self, install_dir: str = None, config_path: str = None):
        
    """__init__ function."""
self.install_dir = Path(install_dir) if install_dir else Path.cwd()
        self.config_path = config_path
        self.python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        self.requirements_file = Path(__file__).parent / "requirements.txt"
        
    def check_prerequisites(self) -> bool:
        """Check if prerequisites are met."""
        print("Checking prerequisites...")
        
        # Check Python version
        if sys.version_info < (3, 8):
            print("❌ Python 3.8+ is required")
            return False
        print(f"✅ Python {self.python_version} detected")
        
        # Check pip
        try:
            subprocess.run([sys.executable, "-m", "pip", "--version"], 
                         check=True, capture_output=True)
            print("✅ pip is available")
        except subprocess.CalledProcessError:
            print("❌ pip is not available")
            return False
        
        # Check if we're in a virtual environment
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            print("✅ Virtual environment detected")
        else:
            print("⚠️  No virtual environment detected (recommended)")
        
        return True
    
    def install_dependencies(self) -> bool:
        """Install Python dependencies."""
        print("Installing dependencies...")
        
        try:
            # Upgrade pip
            subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                         check=True)
            print("✅ pip upgraded")
            
            # Install requirements
            if self.requirements_file.exists():
                subprocess.run([
                    sys.executable, "-m", "pip", "install", "-r", str(self.requirements_file)
                ], check=True)
                print("✅ Dependencies installed")
            else:
                print("❌ Requirements file not found")
                return False
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False
        
        return True
    
    def create_directories(self) -> bool:
        """Create necessary directories."""
        print("Creating directories...")
        
        directories = [
            "output",
            "temp",
            "logs",
            "plugins",
            "config",
            "cache",
            "data"
        ]
        
        try:
            for directory in directories:
                dir_path = self.install_dir / directory
                dir_path.mkdir(exist_ok=True)
                print(f"✅ Created directory: {directory}")
                
        except Exception as e:
            print(f"❌ Failed to create directories: {e}")
            return False
        
        return True
    
    def create_config_files(self) -> bool:
        """Create default configuration files."""
        print("Creating configuration files...")
        
        try:
            # Main config
            config_data = {
                "system_name": "Onyx AI Video System",
                "version": "1.0.0",
                "environment": "development",
                "debug": True,
                "logging": {
                    "level": "INFO",
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    "file_path": "./logs/ai_video.log",
                    "max_size": 10,
                    "backup_count": 5,
                    "use_onyx_logging": True
                },
                "llm": {
                    "provider": "openai",
                    "model": "gpt-4",
                    "temperature": 0.7,
                    "max_tokens": 4000,
                    "timeout": 60,
                    "retry_attempts": 3,
                    "use_onyx_llm": True
                },
                "video": {
                    "default_quality": "medium",
                    "default_format": "mp4",
                    "default_duration": 60,
                    "max_duration": 600,
                    "output_directory": "./output",
                    "temp_directory": "./temp",
                    "cleanup_temp": True
                },
                "plugins": {
                    "plugins_directory": "./plugins",
                    "auto_load": True,
                    "enable_all": False,
                    "max_workers": 10,
                    "timeout": 300,
                    "retry_attempts": 3
                },
                "performance": {
                    "enable_monitoring": True,
                    "metrics_interval": 60,
                    "cache_enabled": True,
                    "cache_size": 1000,
                    "cache_ttl": 3600,
                    "gpu_enabled": True,
                    "max_concurrent_requests": 10
                },
                "security": {
                    "enable_encryption": True,
                    "encryption_key": None,
                    "validate_input": True,
                    "max_input_length": 10000,
                    "rate_limit_enabled": True,
                    "rate_limit_requests": 100,
                    "use_onyx_security": True
                },
                "onyx": {
                    "use_onyx_logging": True,
                    "use_onyx_llm": True,
                    "use_onyx_telemetry": True,
                    "use_onyx_encryption": True,
                    "use_onyx_threading": True,
                    "use_onyx_retry": True,
                    "use_onyx_gpu": True,
                    "onyx_config_path": None
                },
                "custom": {}
            }
            
            config_file = self.install_dir / "config" / "config.yaml"
            with open(config_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
            print(f"✅ Created config file: {config_file}")
            
            # Environment file
            env_content = """# Onyx AI Video System Environment Configuration

# System configuration
AI_VIDEO_ENVIRONMENT=development
AI_VIDEO_DEBUG=true

# Logging
AI_VIDEO_LOGGING_LEVEL=INFO
AI_VIDEO_LOGGING_FILE_PATH=./logs/ai_video.log

# LLM configuration
AI_VIDEO_LLM_PROVIDER=openai
AI_VIDEO_LLM_MODEL=gpt-4
AI_VIDEO_LLM_TEMPERATURE=0.7

# Video configuration
AI_VIDEO_OUTPUT_DIRECTORY=./output
AI_VIDEO_TEMP_DIRECTORY=./temp

# Plugin configuration
AI_VIDEO_PLUGINS_DIRECTORY=./plugins
AI_VIDEO_MAX_WORKERS=10

# Performance configuration
AI_VIDEO_CACHE_ENABLED=true
AI_VIDEO_CACHE_SIZE=1000

# Security configuration
AI_VIDEO_ENCRYPTION_KEY=your-secret-key-here
AI_VIDEO_RATE_LIMIT_REQUESTS=100

# Onyx integration
ONYX_USE_LOGGING=true
ONYX_USE_LLM=true
ONYX_USE_TELEMETRY=true
ONYX_USE_ENCRYPTION=true
ONYX_USE_THREADING=true
ONYX_USE_RETRY=true
ONYX_USE_GPU=true
"""
            
            env_file = self.install_dir / ".env"
            with open(env_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                f.write(env_content)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            print(f"✅ Created environment file: {env_file}")
            
        except Exception as e:
            print(f"❌ Failed to create config files: {e}")
            return False
        
        return True
    
    def create_sample_plugins(self) -> bool:
        """Create sample plugins."""
        print("Creating sample plugins...")
        
        try:
            plugins_dir = self.install_dir / "plugins"
            
            # Sample text processing plugin
            text_plugin_dir = plugins_dir / "text_processor"
            text_plugin_dir.mkdir(exist_ok=True)
            
            text_plugin_code = '''"""
Sample Text Processing Plugin

A simple plugin that demonstrates text processing capabilities.
"""


class TextProcessorPlugin(OnyxPluginBase):
    def __init__(self) -> Any:
        super().__init__(
            name="text_processor",
            version="1.0.0",
            description="Sample text processing plugin",
            author="Onyx Team",
            category="custom"
        )
    
    async def execute(self, context: OnyxPluginContext) -> dict:
        """Execute the text processing plugin."""
        input_data = context.input_data
        text = input_data.get("text", "")
        
        # Simple text processing
        processed_text = self._process_text(text)
        
        return {
            "status": "success",
            "processed_text": processed_text,
            "word_count": len(processed_text.split()),
            "character_count": len(processed_text),
            "metadata": {
                "processing_time": 0.1,
                "plugin_version": self.version
            }
        }
    
    def _process_text(self, text: str) -> str:
        """Process input text."""
        # Remove extra whitespace
        text = re.sub(r'\\s+', ' ', text.strip())
        
        # Capitalize first letter of sentences
        sentences = text.split('. ')
        processed_sentences = [s.capitalize() for s in sentences]
        
        return '. '.join(processed_sentences)
'''
            
            with open(text_plugin_dir / "plugin.py", 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                f.write(text_plugin_code)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            # Plugin config
            plugin_config = {
                "name": "text_processor",
                "version": "1.0.0",
                "enabled": True,
                "parameters": {
                    "max_length": 1000,
                    "remove_extra_spaces": True
                },
                "timeout": 30,
                "max_workers": 1,
                "dependencies": [],
                "conflicts": [],
                "gpu_required": False,
                "memory_required": 128,
                "cpu_cores_required": 1
            }
            
            with open(text_plugin_dir / "config.yaml", 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                yaml.dump(plugin_config, f, default_flow_style=False, indent=2)
            
            # Plugin init file
            with open(text_plugin_dir / "__init__.py", 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                f.write('from .plugin import TextProcessorPlugin\n')
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            print("✅ Created sample text processor plugin")
            
        except Exception as e:
            print(f"❌ Failed to create sample plugins: {e}")
            return False
        
        return True
    
    def setup_logging(self) -> bool:
        """Setup logging configuration."""
        print("Setting up logging...")
        
        try:
            logs_dir = self.install_dir / "logs"
            logs_dir.mkdir(exist_ok=True)
            
            # Create log rotation script
            log_script = '''#!/bin/bash
# Log rotation script for Onyx AI Video System

LOG_DIR="./logs"
MAX_SIZE="10M"
BACKUP_COUNT=5

for log_file in "$LOG_DIR"/*.log; do
    if [ -f "$log_file" ]; then
        logrotate -f -s /dev/null <<EOF
"$log_file" {
    size $MAX_SIZE
    rotate $BACKUP_COUNT
    compress
    delaycompress
    missingok
    notifempty
    create 644 root root
}
EOF
    fi
done
'''
            
            log_script_path = self.install_dir / "rotate_logs.sh"
            with open(log_script_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                f.write(log_script)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            # Make executable
            os.chmod(log_script_path, 0o755)
            
            print("✅ Logging setup completed")
            
        except Exception as e:
            print(f"❌ Failed to setup logging: {e}")
            return False
        
        return True
    
    def run_tests(self) -> bool:
        """Run installation tests."""
        print("Running installation tests...")
        
        try:
            # Test imports
            test_imports = [
                "onyx_ai_video",
                "onyx_ai_video.core",
                "onyx_ai_video.workflows",
                "onyx_ai_video.plugins",
                "onyx_ai_video.config",
                "onyx_ai_video.utils"
            ]
            
            for module in test_imports:
                try:
                    __import__(module)
                    print(f"✅ Import test passed: {module}")
                except ImportError as e:
                    print(f"❌ Import test failed: {module} - {e}")
                    return False
            
            # Test system initialization
            try:
                
                async def test_init():
                    
    """test_init function."""
system = OnyxAIVideoSystem()
                    await system.initialize()
                    await system.shutdown()
                
                asyncio.run(test_init())
                print("✅ System initialization test passed")
                
            except Exception as e:
                print(f"❌ System initialization test failed: {e}")
                return False
            
            print("✅ All installation tests passed")
            return True
            
        except Exception as e:
            print(f"❌ Installation tests failed: {e}")
            return False
    
    def create_startup_scripts(self) -> bool:
        """Create startup scripts."""
        print("Creating startup scripts...")
        
        try:
            # Python startup script
            startup_script = f'''#!/usr/bin/env python3
"""
Onyx AI Video System - Startup Script
"""


# Add the installation directory to Python path
install_dir = Path(__file__).parent
sys.path.insert(0, str(install_dir))

# Set environment variables
os.environ.setdefault("AI_VIDEO_ENVIRONMENT", "production")
os.environ.setdefault("AI_VIDEO_CONFIG_PATH", str(install_dir / "config" / "config.yaml"))

# Import and run the CLI

if __name__ == "__main__":
    main()
'''
            
            startup_path = self.install_dir / "start_ai_video.py"
            with open(startup_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                f.write(startup_script)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            os.chmod(startup_path, 0o755)
            
            # Shell startup script
            shell_script = f'''#!/bin/bash
# Onyx AI Video System - Startup Script

# Change to installation directory
cd "{self.install_dir}"

# Activate virtual environment if exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Set environment variables
export AI_VIDEO_ENVIRONMENT=production
export AI_VIDEO_CONFIG_PATH="{self.install_dir}/config/config.yaml"

# Start the system
python start_ai_video.py "$@"
'''
            
            shell_startup_path = self.install_dir / "start_ai_video.sh"
            with open(shell_startup_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                f.write(shell_script)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            os.chmod(shell_startup_path, 0o755)
            
            print("✅ Startup scripts created")
            
        except Exception as e:
            print(f"❌ Failed to create startup scripts: {e}")
            return False
        
        return True
    
    def create_documentation(self) -> bool:
        """Create basic documentation."""
        print("Creating documentation...")
        
        try:
            docs_dir = self.install_dir / "docs"
            docs_dir.mkdir(exist_ok=True)
            
            # Quick start guide
            quick_start = """# Onyx AI Video System - Quick Start Guide

## Getting Started

1. **Initialize the system**:
   ```bash
   python start_ai_video.py init
   ```

2. **Generate your first video**:
   ```bash
   python start_ai_video.py generate --input "Create a video about AI" --user-id user123
   ```

3. **Check system status**:
   ```bash
   python start_ai_video.py status
   ```

4. **Monitor performance**:
   ```bash
   python start_ai_video.py monitor
   ```

## Configuration

Edit the configuration file at `config/config.yaml` to customize the system behavior.

## Environment Variables

Set environment variables in the `.env` file or export them in your shell.

## Plugins

Sample plugins are included in the `plugins/` directory. Create your own plugins by following the plugin development guide.

## Support

For more information, see the main documentation in the `docs/` directory.
"""
            
            with open(docs_dir / "QUICK_START.md", 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                f.write(quick_start)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            print("✅ Documentation created")
            
        except Exception as e:
            print(f"❌ Failed to create documentation: {e}")
            return False
        
        return True
    
    def install(self) -> bool:
        """Run the complete installation process."""
        print("🚀 Starting Onyx AI Video System installation...")
        print(f"Installation directory: {self.install_dir}")
        print()
        
        steps = [
            ("Checking prerequisites", self.check_prerequisites),
            ("Installing dependencies", self.install_dependencies),
            ("Creating directories", self.create_directories),
            ("Creating configuration files", self.create_config_files),
            ("Creating sample plugins", self.create_sample_plugins),
            ("Setting up logging", self.setup_logging),
            ("Creating startup scripts", self.create_startup_scripts),
            ("Creating documentation", self.create_documentation),
            ("Running installation tests", self.run_tests)
        ]
        
        for step_name, step_func in steps:
            print(f"\n📋 {step_name}...")
            if not step_func():
                print(f"\n❌ Installation failed at: {step_name}")
                return False
        
        print("\n🎉 Installation completed successfully!")
        print("\nNext steps:")
        print("1. Edit the configuration file: config/config.yaml")
        print("2. Set your environment variables in .env")
        print("3. Initialize the system: python start_ai_video.py init")
        print("4. Generate your first video: python start_ai_video.py generate --input 'Hello World' --user-id user123")
        
        return True


def main():
    """Main installation function."""
    parser = argparse.ArgumentParser(description="Onyx AI Video System Installer")
    parser.add_argument(
        "--install-dir",
        type=str,
        help="Installation directory (default: current directory)"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--skip-deps",
        action="store_true",
        help="Skip dependency installation"
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip installation tests"
    )
    
    args = parser.parse_args()
    
    installer = OnyxAIVideoInstaller(
        install_dir=args.install_dir,
        config_path=args.config
    )
    
    success = installer.install()
    sys.exit(0 if success else 1)


match __name__:
    case "__main__":
    main() 