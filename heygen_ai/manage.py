#!/usr/bin/env python3
"""
HeyGen AI Management Script

This script provides a command-line interface for managing the HeyGen AI system,
including installation, configuration, testing, and deployment.
"""

import argparse
import sys
import subprocess
from pathlib import Path
from typing import List, Optional

class HeyGenAIManager:
    """Main management class for HeyGen AI."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.scripts_dir = self.project_root / "scripts"
        self.src_dir = self.project_root / "src"
        self.config_dir = self.project_root / "configs"
    
    def install(self, profile: str = "basic"):
        """Install HeyGen AI with specified profile."""
        print(f"🚀 Installing HeyGen AI with profile: {profile}")
        
        installer_script = self.project_root / "install_requirements.py"
        if installer_script.exists():
            try:
                subprocess.run([sys.executable, str(installer_script), profile], check=True)
                print("✅ Installation completed successfully!")
            except subprocess.CalledProcessError as e:
                print(f"❌ Installation failed: {e}")
                return False
        else:
            print("❌ Installer script not found")
            return False
        
        return True
    
    def setup(self, environment: str = "development"):
        """Setup HeyGen AI system."""
        print(f"🔧 Setting up HeyGen AI for {environment} environment")
        
        # Create necessary directories
        directories = [
            "logs",
            "data",
            "models",
            "cache",
            "temp"
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(exist_ok=True)
            print(f"  Created directory: {directory}")
        
        # Copy configuration files
        self._setup_configuration(environment)
        
        print("✅ Setup completed successfully!")
        return True
    
    def test(self, test_type: str = "all"):
        """Run tests for the system."""
        print(f"🧪 Running {test_type} tests...")
        
        if test_type == "all":
            test_dirs = ["tests/unit", "tests/integration"]
        else:
            test_dirs = [f"tests/{test_type}"]
        
        for test_dir in test_dirs:
            test_path = self.project_root / test_dir
            if test_path.exists():
                try:
                    subprocess.run([sys.executable, "-m", "pytest", str(test_path)], check=True)
                    print(f"✅ {test_dir} tests passed")
                except subprocess.CalledProcessError:
                    print(f"❌ {test_dir} tests failed")
            else:
                print(f"⚠️  Test directory not found: {test_dir}")
        
        return True
    
    def run(self, mode: str = "demo", component: Optional[str] = None):
        """Run HeyGen AI system."""
        print(f"🚀 Starting HeyGen AI in {mode} mode")
        
        if mode == "demo":
            demo_script = self.project_root / "launch_demos.py"
            if demo_script.exists():
                try:
                    subprocess.run([sys.executable, str(demo_script)], check=True)
                except subprocess.CalledProcessError as e:
                    print(f"❌ Demo failed: {e}")
                    return False
            else:
                print("❌ Demo launcher not found")
                return False
        
        elif mode == "api":
            api_script = self.src_dir / "api" / "main.py"
            if api_script.exists():
                try:
                    subprocess.run([sys.executable, str(api_script)], check=True)
                except subprocess.CalledProcessError as e:
                    print(f"❌ API failed: {e}")
                    return False
            else:
                print("❌ API script not found")
                return False
        
        elif mode == "plugin":
            plugin_script = self.project_root / "plugin_demo.py"
            if plugin_script.exists():
                try:
                    subprocess.run([sys.executable, str(plugin_script)], check=True)
                except subprocess.CalledProcessError as e:
                    print(f"❌ Plugin demo failed: {e}")
                    return False
            else:
                print("❌ Plugin demo not found")
                return False
        
        return True
    
    def configure(self, action: str, **kwargs):
        """Configure HeyGen AI system."""
        print(f"⚙️  Configuring HeyGen AI: {action}")
        
        if action == "show":
            self._show_configuration()
        elif action == "update":
            self._update_configuration(**kwargs)
        elif action == "validate":
            self._validate_configuration()
        else:
            print(f"❌ Unknown configuration action: {action}")
            return False
        
        return True
    
    def organize(self):
        """Organize project structure."""
        print("📁 Organizing project structure...")
        
        organizer_script = self.project_root / "organize_project.py"
        if organizer_script.exists():
            try:
                subprocess.run([sys.executable, str(organizer_script)], check=True)
                print("✅ Project organization completed!")
            except subprocess.CalledProcessError as e:
                print(f"❌ Project organization failed: {e}")
                return False
        else:
            print("❌ Project organizer script not found")
            return False
        
        return True
    
    def status(self):
        """Show system status."""
        print("📊 HeyGen AI System Status")
        print("=" * 40)
        
        # Check Python version
        python_version = sys.version_info
        print(f"Python: {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        # Check project structure
        print(f"Project Root: {self.project_root}")
        print(f"Source Directory: {'✅' if self.src_dir.exists() else '❌'}")
        print(f"Config Directory: {'✅' if self.config_dir.exists() else '❌'}")
        
        # Check key files
        key_files = [
            "launch_demos.py",
            "plugin_demo.py",
            "install_requirements.py",
            "requirements.txt"
        ]
        
        print("\nKey Files:")
        for file_name in key_files:
            file_path = self.project_root / file_name
            status = "✅" if file_path.exists() else "❌"
            print(f"  {file_name}: {status}")
        
        # Check directories
        print("\nDirectories:")
        directories = ["src", "configs", "requirements", "docs", "tests"]
        for dir_name in directories:
            dir_path = self.project_root / dir_name
            status = "✅" if dir_path.exists() else "❌"
            print(f"  {dir_name}/: {status}")
    
    def _setup_configuration(self, environment: str):
        """Setup configuration files."""
        config_source = self.config_dir / "main" / "heygen_ai_config.yaml"
        config_env = self.config_dir / "environments" / f"{environment}.yaml"
        
        if config_source.exists():
            # Create environments directory
            env_dir = self.config_dir / "environments"
            env_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy and customize environment config
            if not config_env.exists():
                import shutil
                shutil.copy2(config_source, config_env)
                print(f"  Created environment config: {environment}.yaml")
    
    def _show_configuration(self):
        """Show current configuration."""
        config_file = self.config_dir / "main" / "heygen_ai_config.yaml"
        if config_file.exists():
            with open(config_file, 'r') as f:
                print(f.read())
        else:
            print("❌ Configuration file not found")
    
    def _update_configuration(self, **kwargs):
        """Update configuration."""
        print("⚠️  Configuration update not implemented yet")
        print("Please edit the configuration files manually")
    
    def _validate_configuration(self):
        """Validate configuration."""
        print("🔍 Validating configuration...")
        
        # Check if config files exist
        config_files = [
            self.config_dir / "main" / "heygen_ai_config.yaml",
            self.project_root / "requirements.txt"
        ]
        
        for config_file in config_files:
            if config_file.exists():
                print(f"✅ {config_file.name}")
            else:
                print(f"❌ {config_file.name} not found")
        
        print("✅ Configuration validation completed")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="HeyGen AI Management Script")
    parser.add_argument("command", choices=[
        "install", "setup", "test", "run", "configure", "organize", "status"
    ], help="Command to execute")
    
    # Command-specific arguments
    parser.add_argument("--profile", default="basic", 
                       help="Installation profile (minimal, basic, web, enterprise, dev, full)")
    parser.add_argument("--environment", default="development",
                       help="Environment for setup (development, staging, production)")
    parser.add_argument("--mode", default="demo",
                       help="Run mode (demo, api, plugin)")
    parser.add_argument("--test-type", default="all",
                       help="Test type (all, unit, integration)")
    parser.add_argument("--config-action", default="show",
                       help="Configuration action (show, update, validate)")
    
    args = parser.parse_args()
    
    # Create manager instance
    manager = HeyGenAIManager()
    
    # Execute command
    if args.command == "install":
        success = manager.install(args.profile)
    elif args.command == "setup":
        success = manager.setup(args.environment)
    elif args.command == "test":
        success = manager.test(args.test_type)
    elif args.command == "run":
        success = manager.run(args.mode)
    elif args.command == "configure":
        success = manager.configure(args.config_action)
    elif args.command == "organize":
        success = manager.organize()
    elif args.command == "status":
        success = manager.status()
    else:
        print(f"❌ Unknown command: {args.command}")
        success = False
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
