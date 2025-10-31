from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

import os
import sys
import subprocess
import shutil
from pathlib import Path
from typing import List, Dict, Any
    from config import create_default_config, CONFIG_TEMPLATE
            from plugins import PluginManager, ManagerConfig
            import asyncio
   from ai_video.plugins import quick_start
from ai_video.plugins import BasePlugin, PluginMetadata
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Plugin System Setup Script

This script sets up the AI Video Plugin System with:
- Dependency installation
- Configuration file creation
- Directory structure setup
- Example plugin installation
- System validation

Usage:
    python setup.py [options]
    
Options:
    --install-deps     Install required dependencies
    --create-config    Create default configuration file
    --setup-dirs       Create plugin directories
    --install-examples Install example plugins
    --validate         Validate the installation
    --all              Run all setup steps
"""


# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
except ImportError:
    # Fallback if config module is not available
    def create_default_config(file_path: str = "ai_video_config.json") -> bool:
        return True
    
    CONFIG_TEMPLATE = {}


class PluginSystemSetup:
    """Setup class for the plugin system."""
    
    def __init__(self) -> Any:
        self.base_dir = Path(__file__).parent
        self.plugin_dirs = [
            "./plugins",
            "./ai_video/plugins", 
            "./extensions",
            "~/.ai_video/plugins"
        ]
        
        # Required dependencies
        self.required_deps = [
            "aiohttp>=3.8.0",
            "beautifulsoup4>=4.9.0",
            "newspaper3k>=0.2.8",
            "trafilatura>=5.0.0",
            "pyyaml>=6.0",
            "requests>=2.25.0",
            "lxml>=4.6.0",
            "selectolax>=0.3.0",
            "extruct>=0.13.0"
        ]
        
        # Optional dependencies
        self.optional_deps = [
            "pandas>=1.3.0",
            "numpy>=1.21.0",
            "matplotlib>=3.4.0",
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
            "dash>=2.0.0",
            "streamlit>=1.0.0"
        ]
    
    def run_setup(self, steps: List[str]):
        """Run the specified setup steps."""
        print("🎯 AI Video Plugin System Setup")
        print("=" * 50)
        
        if "all" in steps or "install-deps" in steps:
            self.install_dependencies()
        
        if "all" in steps or "create-config" in steps:
            self.create_configuration()
        
        if "all" in steps or "setup-dirs" in steps:
            self.setup_directories()
        
        if "all" in steps or "install-examples" in steps:
            self.install_examples()
        
        if "all" in steps or "validate" in steps:
            self.validate_installation()
        
        print("\n✅ Setup completed successfully!")
    
    def install_dependencies(self) -> Any:
        """Install required dependencies."""
        print("\n📦 Installing Dependencies")
        print("-" * 30)
        
        # Check if pip is available
        try:
            subprocess.run([sys.executable, "-m", "pip", "--version"], 
                         check=True, capture_output=True)
        except subprocess.CalledProcessError:
            print("❌ pip is not available. Please install pip first.")
            return False
        
        # Install required dependencies
        print("Installing required dependencies...")
        for dep in self.required_deps:
            try:
                print(f"  Installing {dep}...")
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install", dep
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    print(f"  ✅ {dep} installed successfully")
                else:
                    print(f"  ❌ Failed to install {dep}: {result.stderr}")
                    
            except Exception as e:
                print(f"  ❌ Error installing {dep}: {e}")
        
        # Ask about optional dependencies
        print("\nOptional dependencies:")
        for i, dep in enumerate(self.optional_deps, 1):
            print(f"  {i}. {dep}")
        
        choice = input("\nInstall optional dependencies? (y/n): ").strip().lower()
        if choice in ['y', 'yes']:
            print("Installing optional dependencies...")
            for dep in self.optional_deps:
                try:
                    print(f"  Installing {dep}...")
                    result = subprocess.run([
                        sys.executable, "-m", "pip", "install", dep
                    ], capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        print(f"  ✅ {dep} installed successfully")
                    else:
                        print(f"  ⚠️ Failed to install {dep}: {result.stderr}")
                        
                except Exception as e:
                    print(f"  ⚠️ Error installing {dep}: {e}")
        
        return True
    
    def create_configuration(self) -> Any:
        """Create default configuration files."""
        print("\n⚙️ Creating Configuration")
        print("-" * 25)
        
        # Create main configuration file
        config_files = [
            "ai_video_config.json",
            "ai_video_config.yaml",
            "~/.ai_video/config.json"
        ]
        
        for config_file in config_files:
            try:
                if create_default_config(config_file):
                    print(f"✅ Configuration created: {config_file}")
                else:
                    print(f"⚠️ Failed to create: {config_file}")
            except Exception as e:
                print(f"⚠️ Error creating {config_file}: {e}")
        
        # Create environment file template
        env_template = """# AI Video Plugin System Environment Variables

# Plugin discovery and loading
AI_VIDEO_AUTO_DISCOVER=true
AI_VIDEO_AUTO_LOAD=false
AI_VIDEO_AUTO_INITIALIZE=false

# Validation and security
AI_VIDEO_VALIDATION_LEVEL=standard
AI_VIDEO_ENABLE_SECURITY=true
AI_VIDEO_ENABLE_PERFORMANCE=true

# Plugin directories (comma-separated)
AI_VIDEO_PLUGIN_DIRS=./plugins,./ai_video/plugins,./extensions,~/.ai_video/plugins

# HTTP and networking
AI_VIDEO_HTTP_TIMEOUT=30
AI_VIDEO_MAX_RETRIES=3
AI_VIDEO_USER_AGENT=AI-Video-Plugin-System/1.0

# Performance and monitoring
AI_VIDEO_ENABLE_METRICS=true
AI_VIDEO_ENABLE_EVENTS=true
AI_VIDEO_ENABLE_LOGGING=true
AI_VIDEO_LOG_LEVEL=INFO

# Storage and caching
AI_VIDEO_CACHE_ENABLED=true
AI_VIDEO_CACHE_DIR=~/.ai_video/cache
AI_VIDEO_CACHE_TTL=3600

# Advanced settings
AI_VIDEO_MAX_CONCURRENT=10
AI_VIDEO_LOAD_TIMEOUT=60
AI_VIDEO_AUTO_RECOVERY=true
AI_VIDEO_HEALTH_CHECKS=true
AI_VIDEO_HEALTH_INTERVAL=300
"""
        
        env_file = ".env"
        try:
            with open(env_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                f.write(env_template)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            print(f"✅ Environment template created: {env_file}")
        except Exception as e:
            print(f"⚠️ Failed to create environment template: {e}")
    
    def setup_directories(self) -> Any:
        """Create plugin directories and structure."""
        print("\n📁 Setting up Directories")
        print("-" * 25)
        
        for plugin_dir in self.plugin_dirs:
            try:
                path = Path(plugin_dir).expanduser()
                path.mkdir(parents=True, exist_ok=True)
                print(f"✅ Created directory: {path}")
                
                # Create __init__.py if it doesn't exist
                init_file = path / "__init__.py"
                if not init_file.exists():
                    init_file.touch()
                    print(f"  ✅ Created {init_file}")
                
            except Exception as e:
                print(f"❌ Failed to create directory {plugin_dir}: {e}")
        
        # Create cache directory
        cache_dir = Path("~/.ai_video/cache").expanduser()
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created cache directory: {cache_dir}")
        except Exception as e:
            print(f"⚠️ Failed to create cache directory: {e}")
        
        # Create logs directory
        logs_dir = Path("~/.ai_video/logs").expanduser()
        try:
            logs_dir.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created logs directory: {logs_dir}")
        except Exception as e:
            print(f"⚠️ Failed to create logs directory: {e}")
    
    def install_examples(self) -> Any:
        """Install example plugins."""
        print("\n🔧 Installing Examples")
        print("-" * 20)
        
        # Copy example plugins to plugin directories
        examples_dir = self.base_dir / "examples"
        if not examples_dir.exists():
            print("⚠️ Examples directory not found")
            return
        
        # Find example plugins
        example_plugins = []
        for file_path in examples_dir.rglob("*.py"):
            if file_path.name != "__init__.py":
                example_plugins.append(file_path)
        
        if not example_plugins:
            print("⚠️ No example plugins found")
            return
        
        print(f"Found {len(example_plugins)} example plugins:")
        for plugin_file in example_plugins:
            print(f"  - {plugin_file.name}")
        
        # Ask user which directory to install to
        print("\nAvailable plugin directories:")
        for i, plugin_dir in enumerate(self.plugin_dirs, 1):
            print(f"  {i}. {plugin_dir}")
        
        try:
            choice = int(input("\nSelect directory to install examples (1-4): ")) - 1
            if 0 <= choice < len(self.plugin_dirs):
                target_dir = Path(self.plugin_dirs[choice]).expanduser()
                
                # Copy example plugins
                for plugin_file in example_plugins:
                    try:
                        target_file = target_dir / plugin_file.name
                        shutil.copy2(plugin_file, target_file)
                        print(f"✅ Installed {plugin_file.name} to {target_dir}")
                    except Exception as e:
                        print(f"❌ Failed to install {plugin_file.name}: {e}")
            else:
                print("❌ Invalid choice")
        except ValueError:
            print("❌ Invalid input")
    
    def validate_installation(self) -> bool:
        """Validate the installation."""
        print("\n🔍 Validating Installation")
        print("-" * 25)
        
        # Check Python version
        python_version = sys.version_info
        if python_version >= (3, 8):
            print(f"✅ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
        else:
            print(f"❌ Python version {python_version.major}.{python_version.minor} is too old. Need 3.8+")
        
        # Check required dependencies
        print("\nChecking required dependencies...")
        missing_deps = []
        
        for dep in self.required_deps:
            package_name = dep.split('>=')[0].split('==')[0]
            try:
                __import__(package_name.replace('-', '_'))
                print(f"  ✅ {package_name}")
            except ImportError:
                print(f"  ❌ {package_name} (missing)")
                missing_deps.append(package_name)
        
        if missing_deps:
            print(f"\n⚠️ Missing dependencies: {', '.join(missing_deps)}")
            print("Run 'python setup.py --install-deps' to install them")
        else:
            print("\n✅ All required dependencies are installed")
        
        # Check directories
        print("\nChecking directories...")
        for plugin_dir in self.plugin_dirs:
            path = Path(plugin_dir).expanduser()
            if path.exists():
                print(f"  ✅ {plugin_dir}")
            else:
                print(f"  ❌ {plugin_dir} (missing)")
        
        # Check configuration files
        print("\nChecking configuration files...")
        config_files = ["ai_video_config.json", "ai_video_config.yaml"]
        for config_file in config_files:
            if Path(config_file).exists():
                print(f"  ✅ {config_file}")
            else:
                print(f"  ⚠️ {config_file} (not found)")
        
        # Test plugin system import
        print("\nTesting plugin system...")
        try:
            print("  ✅ Plugin system imports successfully")
            
            # Test basic functionality
            async def test_plugin_system():
                
    """test_plugin_system function."""
try:
                    config = ManagerConfig(auto_discover=False, auto_load=False)
                    manager = PluginManager(config)
                    await manager.start()
                    await manager.shutdown()
                    return True
                except Exception as e:
                    print(f"  ❌ Plugin system test failed: {e}")
                    return False
            
            success = asyncio.run(test_plugin_system())
            if success:
                print("  ✅ Plugin system works correctly")
            else:
                print("  ❌ Plugin system test failed")
                
        except ImportError as e:
            print(f"  ❌ Failed to import plugin system: {e}")
    
    def create_readme(self) -> Any:
        """Create a README file with usage instructions."""
        readme_content = """# AI Video Plugin System

A comprehensive, production-ready plugin system for AI video generation.

## Quick Start

1. **Install the system:**
   ```bash
   python setup.py --all
   ```

2. **Run the demo:**
   ```bash
   python demo.py
   ```

3. **Use in your code:**
   ```python
   
   # Start with recommended settings
   manager = await quick_start()
   
   # Load a plugin
   plugin = await manager.load_plugin("web_extractor", {"timeout": 30})
   
   # Use the plugin
   content = await plugin.extract_content("https://example.com")
   ```

## Configuration

The system can be configured through:
- Configuration files (`ai_video_config.json`, `ai_video_config.yaml`)
- Environment variables (see `.env` template)
- Code configuration

## Plugin Development

To create a plugin, inherit from `BasePlugin`:

```python

class MyPlugin(BasePlugin):
    def __init__(self, config=None) -> Any:
        super().__init__(config)
        self.name = "my_plugin"
        self.version = "1.0.0"
        self.description = "My awesome plugin"
    
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name=self.name,
            version=self.version,
            description=self.description,
            author="Your Name",
            category="extractor"
        )
    
    async def initialize(self) -> Any:
        # Initialize your plugin
        pass
    
    async def cleanup(self) -> Any:
        # Cleanup your plugin
        pass
```

## Features

- 🔌 Plugin discovery and loading
- ✅ Validation and error handling
- 🔄 Lifecycle management
- 📊 Performance monitoring
- 🎯 Event handling
- ⚙️ Configuration management
- 🔒 Security validation

## Documentation

For more information, see the individual module documentation.
"""
        
        try:
            with open("README_PLUGINS.md", 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                f.write(readme_content)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            print("✅ README created: README_PLUGINS.md")
        except Exception as e:
            print(f"⚠️ Failed to create README: {e}")


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print(__doc__)
        return
    
    setup = PluginSystemSetup()
    
    # Parse command line arguments
    steps = []
    for arg in sys.argv[1:]:
        if arg.startswith('--'):
            steps.append(arg[2:])
        else:
            steps.append(arg)
    
    # Run setup
    setup.run_setup(steps)
    
    # Create README if running all steps
    if "all" in steps:
        setup.create_readme()


match __name__:
    case "__main__":
    main() 