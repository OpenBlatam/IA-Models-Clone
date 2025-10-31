#!/usr/bin/env python3
"""
Installation Script for Refactored Unified AI Interface v3.6
Enhanced installation with improved dependency management and system optimization
"""
import os
import sys
import subprocess
import platform
import shutil
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple

class RefactoredInstaller:
    """Enhanced installer for the refactored AI interface"""
    
    def __init__(self):
        """Initialize the installer"""
        self.system_info = self._get_system_info()
        self.install_path = Path.cwd()
        self.venv_path = self.install_path / "venv"
        self.requirements_file = "requirements_refactored_v3_6.txt"
        self.config_file = "ai_config_v3_6.json"
        
        # Installation status
        self.installation_log = []
        self.errors = []
        self.warnings = []
        
        # Python version requirements
        self.min_python_version = (3, 8)
        self.recommended_python_version = (3, 10)
        
    def _get_system_info(self) -> Dict:
        """Get comprehensive system information"""
        return {
            'platform': platform.system(),
            'platform_version': platform.version(),
            'architecture': platform.architecture()[0],
            'processor': platform.processor(),
            'python_version': sys.version_info,
            'python_executable': sys.executable,
            'working_directory': os.getcwd(),
            'home_directory': str(Path.home())
        }
    
    def print_banner(self):
        """Print installation banner"""
        banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║              REFACTORED UNIFIED AI INTERFACE v3.6 - INSTALLER               ║
║                                                                              ║
║  🚀 Enhanced Installation • 🔧 Smart Dependency Management • ⚡ Optimization  ║
║  📊 System Validation • 🧠 AI Component Setup • 🎯 Performance Tuning      ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """
        print(banner)
    
    def print_system_info(self):
        """Print system information"""
        print("🔍 System Information:")
        print(f"   • Platform: {self.system_info['platform']} {self.system_info['platform_version']}")
        print(f"   • Architecture: {self.system_info['architecture']}")
        print(f"   • Processor: {self.system_info['processor']}")
        print(f"   • Python: {self.system_info['python_version'][0]}.{self.system_info['python_version'][1]}.{self.system_info['python_version'][2]}")
        print(f"   • Python Path: {self.system_info['python_executable']}")
        print(f"   • Working Directory: {self.system_info['working_directory']}")
        print()
    
    def validate_python_version(self) -> bool:
        """Validate Python version requirements"""
        print("🐍 Validating Python Version...")
        
        current_version = self.system_info['python_version']
        
        if current_version < self.min_python_version:
            print(f"   ❌ Python {current_version[0]}.{current_version[1]} is below minimum requirement ({self.min_python_version[0]}.{self.min_python_version[1]})")
            return False
        
        if current_version < self.recommended_python_version:
            print(f"   ⚠️  Python {current_version[0]}.{current_version[1]} is below recommended version ({self.recommended_python_version[0]}.{self.recommended_python_version[1]})")
            self.warnings.append(f"Python version {current_version[0]}.{current_version[1]} is below recommended")
        else:
            print(f"   ✅ Python version {current_version[0]}.{current_version[1]} meets requirements")
        
        return True
    
    def check_pip_availability(self) -> bool:
        """Check if pip is available"""
        print("📦 Checking pip availability...")
        
        try:
            result = subprocess.run([sys.executable, '-m', 'pip', '--version'], 
                                  capture_output=True, text=True, check=True)
            print(f"   ✅ pip is available: {result.stdout.strip()}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("   ❌ pip is not available")
            self.errors.append("pip is not available")
            return False
    
    def check_virtual_environment(self) -> bool:
        """Check if running in virtual environment"""
        print("🏠 Checking Virtual Environment...")
        
        in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
        
        if in_venv:
            print(f"   ✅ Running in virtual environment: {sys.prefix}")
        else:
            print("   ℹ️  Not running in virtual environment")
            self.warnings.append("Not running in virtual environment - consider using one")
        
        return in_venv
    
    def create_virtual_environment(self) -> bool:
        """Create virtual environment if needed"""
        if self.check_virtual_environment():
            return True
        
        print("🏠 Creating Virtual Environment...")
        
        try:
            # Create virtual environment
            subprocess.run([sys.executable, '-m', 'venv', str(self.venv_path)], 
                         check=True, capture_output=True)
            
            print(f"   ✅ Virtual environment created: {self.venv_path}")
            
            # Determine activation script
            if self.system_info['platform'] == 'Windows':
                activate_script = self.venv_path / "Scripts" / "activate.bat"
            else:
                activate_script = self.venv_path / "bin" / "activate"
            
            print(f"   📝 To activate: {activate_script}")
            self.warnings.append(f"Virtual environment created - activate with: {activate_script}")
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Failed to create virtual environment: {e}")
            self.errors.append(f"Failed to create virtual environment: {e}")
            return False
    
    def upgrade_pip(self) -> bool:
        """Upgrade pip to latest version"""
        print("⬆️  Upgrading pip...")
        
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'], 
                         check=True, capture_output=True)
            print("   ✅ pip upgraded successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Failed to upgrade pip: {e}")
            self.warnings.append(f"Failed to upgrade pip: {e}")
            return False
    
    def install_core_dependencies(self) -> bool:
        """Install core dependencies"""
        print("📦 Installing Core Dependencies...")
        
        core_deps = [
            'numpy',
            'pandas',
            'psutil',
            'requests',
            'python-dotenv'
        ]
        
        success_count = 0
        for dep in core_deps:
            try:
                print(f"   📥 Installing {dep}...")
                subprocess.run([sys.executable, '-m', 'pip', 'install', dep], 
                             check=True, capture_output=True)
                print(f"   ✅ {dep} installed successfully")
                success_count += 1
            except subprocess.CalledProcessError as e:
                print(f"   ❌ Failed to install {dep}: {e}")
                self.errors.append(f"Failed to install {dep}: {e}")
        
        print(f"   📊 Core dependencies: {success_count}/{len(core_deps)} installed")
        return success_count == len(core_deps)
    
    def install_requirements(self) -> bool:
        """Install requirements from file"""
        print("📋 Installing Requirements...")
        
        if not Path(self.requirements_file).exists():
            print(f"   ❌ Requirements file not found: {self.requirements_file}")
            self.errors.append(f"Requirements file not found: {self.requirements_file}")
            return False
        
        try:
            print(f"   📥 Installing from {self.requirements_file}...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', self.requirements_file], 
                         check=True, capture_output=True)
            print("   ✅ Requirements installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Failed to install requirements: {e}")
            self.errors.append(f"Failed to install requirements: {e}")
            return False
    
    def create_configuration(self) -> bool:
        """Create default configuration file"""
        print("⚙️  Creating Configuration...")
        
        default_config = {
            "system": {
                "auto_start": True,
                "health_check_interval": 30.0,
                "max_retries": 3,
                "timeout": 60.0
            },
            "monitoring": {
                "enabled": True,
                "interval": 2.0,
                "history_size": 1000,
                "alert_thresholds": {
                    "cpu": 80.0,
                    "memory": 85.0,
                    "disk": 90.0,
                    "gpu": 95.0
                }
            },
            "optimization": {
                "enabled": True,
                "auto_optimize": True,
                "interval": 30.0,
                "threshold": 0.7
            },
            "analytics": {
                "enabled": True,
                "prediction_horizon": 300,
                "confidence_threshold": 0.75
            },
            "ui": {
                "theme": "dark",
                "auto_refresh": True,
                "refresh_interval": 5.0
            }
        }
        
        try:
            with open(self.config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            print(f"   ✅ Configuration created: {self.config_file}")
            return True
        except Exception as e:
            print(f"   ❌ Failed to create configuration: {e}")
            self.errors.append(f"Failed to create configuration: {e}")
            return False
    
    def validate_installation(self) -> bool:
        """Validate the installation"""
        print("✅ Validating Installation...")
        
        # Check if main files exist
        required_files = [
            'refactored_unified_ai_interface_v3_6.py',
            'launch_refactored_ai_v3_6.py',
            'requirements_refactored_v3_6.txt'
        ]
        
        missing_files = []
        for file in required_files:
            if not Path(file).exists():
                missing_files.append(file)
        
        if missing_files:
            print(f"   ❌ Missing required files: {missing_files}")
            self.errors.extend([f"Missing file: {f}" for f in missing_files])
            return False
        
        # Test import of main interface
        try:
            sys.path.insert(0, str(self.install_path))
            from refactored_unified_ai_interface_v3_6 import RefactoredUnifiedAIInterface
            print("   ✅ Main interface import successful")
            
            # Test basic functionality
            interface = RefactoredUnifiedAIInterface()
            status = interface.get_system_status()
            print(f"   ✅ Interface initialization successful: {status['interface']['state']}")
            
            return True
            
        except ImportError as e:
            print(f"   ❌ Failed to import main interface: {e}")
            self.errors.append(f"Import error: {e}")
            return False
        except Exception as e:
            print(f"   ❌ Interface test failed: {e}")
            self.errors.append(f"Interface test failed: {e}")
            return False
    
    def optimize_system(self) -> bool:
        """Perform system optimization"""
        print("⚡ System Optimization...")
        
        optimizations = []
        
        # Check and optimize Python settings
        try:
            import gc
            gc.collect()
            optimizations.append("Garbage collection performed")
        except:
            pass
        
        # Check memory usage
        try:
            import psutil
            memory = psutil.virtual_memory()
            if memory.percent > 80:
                optimizations.append("High memory usage detected - consider optimization")
        except:
            pass
        
        # Check disk space
        try:
            disk = psutil.disk_usage('.')
            if disk.percent > 90:
                optimizations.append("Low disk space - consider cleanup")
        except:
            pass
        
        if optimizations:
            for opt in optimizations:
                print(f"   ℹ️  {opt}")
        else:
            print("   ✅ System appears optimized")
        
        return True
    
    def generate_installation_report(self) -> Dict:
        """Generate comprehensive installation report"""
        return {
            'timestamp': str(Path().cwd()),
            'system_info': self.system_info,
            'installation_path': str(self.install_path),
            'virtual_environment': str(self.venv_path),
            'installation_log': self.installation_log,
            'errors': self.errors,
            'warnings': self.warnings,
            'success': len(self.errors) == 0,
            'recommendations': self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate installation recommendations"""
        recommendations = []
        
        if self.warnings:
            recommendations.append("Review warnings above for potential improvements")
        
        if not self.check_virtual_environment():
            recommendations.append("Consider using a virtual environment for isolation")
        
        if self.system_info['python_version'] < self.recommended_python_version:
            recommendations.append(f"Consider upgrading to Python {self.recommended_python_version[0]}.{self.recommended_python_version[1]}+")
        
        recommendations.append("Run the system validation to ensure everything works correctly")
        recommendations.append("Check the configuration file and adjust settings as needed")
        
        return recommendations
    
    def print_installation_report(self, report: Dict):
        """Print installation report"""
        print("\n" + "="*80)
        print("📊 INSTALLATION REPORT")
        print("="*80)
        
        print(f"Installation Path: {report['installation_path']}")
        print(f"Virtual Environment: {report['virtual_environment']}")
        print(f"Success: {'✅ YES' if report['success'] else '❌ NO'}")
        
        if report['errors']:
            print(f"\n❌ ERRORS ({len(report['errors'])}):")
            for error in report['errors']:
                print(f"   • {error}")
        
        if report['warnings']:
            print(f"\n⚠️  WARNINGS ({len(report['warnings'])}):")
            for warning in report['warnings']:
                print(f"   • {warning}")
        
        if report['recommendations']:
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in report['recommendations']:
                print(f"   • {rec}")
        
        print("\n" + "="*80)
    
    def install(self) -> bool:
        """Perform complete installation"""
        try:
            self.print_banner()
            self.print_system_info()
            
            # Validation steps
            if not self.validate_python_version():
                return False
            
            if not self.check_pip_availability():
                return False
            
            # Environment setup
            self.check_virtual_environment()
            self.create_virtual_environment()
            
            # Installation steps
            self.upgrade_pip()
            self.install_core_dependencies()
            self.install_requirements()
            self.create_configuration()
            
            # Validation and optimization
            if not self.validate_installation():
                return False
            
            self.optimize_system()
            
            # Generate report
            report = self.generate_installation_report()
            self.print_installation_report(report)
            
            if report['success']:
                print("\n🎉 Installation completed successfully!")
                print("   You can now run: python launch_refactored_ai_v3_6.py")
            else:
                print("\n❌ Installation completed with errors!")
                print("   Please review the errors above and try again")
            
            return report['success']
            
        except Exception as e:
            print(f"\n❌ Installation failed with exception: {e}")
            self.errors.append(f"Installation exception: {e}")
            return False

def main():
    """Main installation function"""
    try:
        installer = RefactoredInstaller()
        success = installer.install()
        
        if success:
            print("\n🚀 Ready to launch your Refactored AI Interface!")
        else:
            print("\n🛑 Installation failed. Please check the errors above.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Installation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal installation error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
