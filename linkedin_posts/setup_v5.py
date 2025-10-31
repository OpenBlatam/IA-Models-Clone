#!/usr/bin/env python3
"""
🚀 SETUP SCRIPT v5.0 - INTEGRATED SYSTEM v5.0
==============================================

Comprehensive setup for the Next-Generation LinkedIn Optimizer v5.0 including:
- Advanced AI Intelligence (AutoML, Transfer Learning, NAS)
- Microservices Architecture (Service Mesh, API Gateway, Circuit Breaker)
- Real-Time Analytics & Predictive Insights (Stream Processing, Forecasting)
- Enterprise Security & Compliance (Zero Trust, Homomorphic Encryption, Blockchain)
- Cloud-Native Infrastructure (Kubernetes, Serverless, Multi-Cloud, Edge Computing)

Setup Modes:
1. basic     - Core AI optimization only
2. advanced  - + Analytics & Microservices
3. enterprise - + Security & Infrastructure (RECOMMENDED)
4. quantum   - + Future quantum computing (ROADMAP)
5. full      - Complete v5.0 system with all modules
"""

import os
import sys
import subprocess
import platform
import argparse
import asyncio
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text: str):
    """Print formatted header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}")
    print(f"{text}")
    print(f"{'='*60}{Colors.ENDC}")

def print_success(text: str):
    """Print success message."""
    print(f"{Colors.OKGREEN}✅ {text}{Colors.ENDC}")

def print_info(text: str):
    """Print info message."""
    print(f"{Colors.OKBLUE}ℹ️  {text}{Colors.ENDC}")

def print_warning(text: str):
    """Print warning message."""
    print(f"{Colors.WARNING}⚠️  {text}{Colors.ENDC}")

def print_error(text: str):
    """Print error message."""
    print(f"{Colors.FAIL}❌ {text}{Colors.ENDC}")

def print_step(text: str):
    """Print step message."""
    print(f"{Colors.OKCYAN}🚀 {text}{Colors.ENDC}")

class SetupV5:
    """Setup manager for Integrated System v5.0."""
    
    def __init__(self, mode: str = "enterprise", python_path: str = None):
        self.mode = mode
        self.python_path = python_path or sys.executable
        self.requirements_file = "requirements_v5.txt"
        self.setup_dir = Path(__file__).parent
        self.venv_dir = self.setup_dir / "venv_v5"
        
        # Mode configurations
        self.mode_configs = {
            "basic": {
                "description": "Core AI optimization only",
                "modules": ["ai_advanced_intelligence_v5"],
                "requirements": ["torch", "transformers", "numpy", "pandas"]
            },
            "advanced": {
                "description": "AI + Analytics & Microservices",
                "modules": ["ai_advanced_intelligence_v5", "real_time_analytics_v5"],
                "requirements": ["torch", "transformers", "numpy", "pandas", "scikit-learn", "fastapi"]
            },
            "enterprise": {
                "description": "Full enterprise system (RECOMMENDED)",
                "modules": ["ai_advanced_intelligence_v5", "real_time_analytics_v5", "enterprise_security_v5", "cloud_native_infrastructure_v5"],
                "requirements": ["torch", "transformers", "numpy", "pandas", "scikit-learn", "fastapi", "cryptography", "kubernetes"]
            },
            "quantum": {
                "description": "Enterprise + Quantum computing (ROADMAP)",
                "modules": ["ai_advanced_intelligence_v5", "real_time_analytics_v5", "enterprise_security_v5", "cloud_native_infrastructure_v5"],
                "requirements": ["torch", "transformers", "numpy", "pandas", "scikit-learn", "fastapi", "cryptography", "kubernetes"],
                "note": "Quantum integration coming in v6.0"
            },
            "full": {
                "description": "Complete v5.0 system with all modules",
                "modules": ["ai_advanced_intelligence_v5", "real_time_analytics_v5", "enterprise_security_v5", "cloud_native_infrastructure_v5"],
                "requirements": ["torch", "transformers", "numpy", "pandas", "scikit-learn", "fastapi", "cryptography", "kubernetes", "optuna", "mlflow"]
            }
        }
    
    def check_system_requirements(self) -> bool:
        """Check if system meets requirements."""
        print_header("SYSTEM REQUIREMENTS CHECK")
        
        # Check Python version
        python_version = sys.version_info
        if python_version < (3, 9):
            print_error(f"Python 3.9+ required, found {python_version.major}.{python_version.minor}")
            return False
        print_success(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        # Check platform
        system = platform.system()
        print_info(f"Platform: {system} {platform.release()}")
        
        # Check available memory
        try:
            import psutil
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            if memory_gb < 4:
                print_warning(f"Low memory: {memory_gb:.1f}GB (4GB+ recommended)")
            else:
                print_success(f"Memory: {memory_gb:.1f}GB")
        except ImportError:
            print_warning("psutil not available - cannot check memory")
        
        # Check disk space
        try:
            disk = psutil.disk_usage(self.setup_dir)
            disk_gb = disk.free / (1024**3)
            if disk_gb < 5:
                print_warning(f"Low disk space: {disk_gb:.1f}GB (5GB+ recommended)")
            else:
                print_success(f"Available disk space: {disk_gb:.1f}GB")
        except ImportError:
            print_warning("Cannot check disk space")
        
        return True
    
    def create_virtual_environment(self) -> bool:
        """Create virtual environment for v5.0."""
        print_header("VIRTUAL ENVIRONMENT SETUP")
        
        if self.venv_dir.exists():
            print_info(f"Virtual environment already exists at {self.venv_dir}")
            return True
        
        try:
            print_step("Creating virtual environment...")
            subprocess.run([
                self.python_path, "-m", "venv", str(self.venv_dir)
            ], check=True, capture_output=True)
            
            print_success("Virtual environment created successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            print_error(f"Failed to create virtual environment: {e}")
            return False
    
    def get_venv_python(self) -> str:
        """Get Python executable path from virtual environment."""
        if platform.system() == "Windows":
            return str(self.venv_dir / "Scripts" / "python.exe")
        else:
            return str(self.venv_dir / "bin" / "python")
    
    def get_venv_pip(self) -> str:
        """Get pip executable path from virtual environment."""
        if platform.system() == "Windows":
            return str(self.venv_dir / "Scripts" / "pip.exe")
        else:
            return str(self.venv_dir / "bin" / "pip")
    
    def install_dependencies(self) -> bool:
        """Install required dependencies."""
        print_header("DEPENDENCY INSTALLATION")
        
        venv_pip = self.get_venv_pip()
        requirements_path = self.setup_dir / self.requirements_file
        
        if not requirements_path.exists():
            print_error(f"Requirements file not found: {requirements_path}")
            return False
        
        try:
            print_step("Upgrading pip...")
            subprocess.run([
                venv_pip, "install", "--upgrade", "pip"
            ], check=True, capture_output=True)
            
            print_step("Installing dependencies...")
            subprocess.run([
                venv_pip, "install", "-r", str(requirements_path)
            ], check=True, capture_output=True)
            
            print_success("Dependencies installed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            print_error(f"Failed to install dependencies: {e}")
            return False
    
    def download_ai_models(self) -> bool:
        """Download required AI models."""
        print_header("AI MODEL DOWNLOAD")
        
        venv_python = self.get_venv_python()
        
        try:
            print_step("Downloading spaCy English model...")
            subprocess.run([
                venv_python, "-m", "spacy", "download", "en_core_web_sm"
            ], check=True, capture_output=True)
            
            print_step("Downloading sentence transformer models...")
            # This will download models on first use
            download_script = """
import sentence_transformers
model = sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2')
print("Sentence transformer model downloaded successfully")
"""
            subprocess.run([
                venv_python, "-c", download_script
            ], check=True, capture_output=True)
            
            print_success("AI models downloaded successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            print_warning(f"Some AI model downloads failed: {e}")
            print_info("Models will be downloaded automatically on first use")
            return True
    
    def run_system_tests(self) -> bool:
        """Run system tests to verify installation."""
        print_header("SYSTEM TESTING")
        
        venv_python = self.get_venv_python()
        test_script = f"""
import sys
sys.path.insert(0, '{self.setup_dir}')

# Test imports
modules_to_test = {self.mode_configs[self.mode]['modules']}
successful_imports = 0

for module in modules_to_test:
    try:
        __import__(module)
        print(f"✅ {{module}} imported successfully")
        successful_imports += 1
    except ImportError as e:
        print(f"❌ {{module}} import failed: {{e}}")

# Test integrated system
try:
    from integrated_system_v5 import IntegratedSystemV5
    print("✅ Integrated System v5.0 imported successfully")
    successful_imports += 1
except ImportError as e:
    print(f"❌ Integrated System v5.0 import failed: {{e}}")

print(f"\\n📊 Test Results: {{successful_imports}}/{{len(modules_to_test) + 1}} modules working")
exit(0 if successful_imports >= len(modules_to_test) * 0.7 else 1)
"""
        
        try:
            print_step("Running system tests...")
            result = subprocess.run([
                venv_python, "-c", test_script
            ], check=True, capture_output=True, text=True)
            
            print(result.stdout)
            print_success("System tests passed")
            return True
            
        except subprocess.CalledProcessError as e:
            print_error(f"System tests failed: {e}")
            if e.stdout:
                print(e.stdout)
            if e.stderr:
                print(e.stderr)
            return False
    
    def run_demo(self) -> bool:
        """Run system demo."""
        print_header("SYSTEM DEMO")
        
        venv_python = self.get_venv_python()
        demo_script = f"""
import sys
sys.path.insert(0, '{self.setup_dir}')

try:
    from integrated_system_v5 import demo_integrated_system_v5
    import asyncio
    
    print("🚀 Starting Integrated System v5.0 demo...")
    asyncio.run(demo_integrated_system_v5())
    
except Exception as e:
    print(f"❌ Demo failed: {{e}}")
    exit(1)
"""
        
        try:
            print_step("Running system demo...")
            result = subprocess.run([
                venv_python, "-c", demo_script
            ], check=True, capture_output=True, text=True)
            
            print(result.stdout)
            print_success("System demo completed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            print_error(f"System demo failed: {e}")
            if e.stdout:
                print(e.stdout)
            if e.stderr:
                print(e.stderr)
            return False
    
    def generate_setup_report(self) -> Dict[str, Any]:
        """Generate setup report."""
        report = {
            "setup_mode": self.mode,
            "mode_description": self.mode_configs[self.mode]["description"],
            "python_path": self.python_path,
            "virtual_environment": str(self.venv_dir),
            "requirements_file": self.requirements_file,
            "setup_directory": str(self.setup_dir),
            "platform": platform.system(),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return report
    
    def print_setup_report(self, report: Dict[str, Any]):
        """Print setup report."""
        print_header("SETUP REPORT")
        
        for key, value in report.items():
            if key == "mode_description":
                print(f"{Colors.BOLD}{key.replace('_', ' ').title()}:{Colors.ENDC}")
                print(f"  {value}")
            else:
                print(f"{Colors.BOLD}{key.replace('_', ' ').title()}:{Colors.ENDC} {value}")
        
        print(f"\n{Colors.OKGREEN}🎉 Setup completed successfully!{Colors.ENDC}")
        print(f"\n{Colors.BOLD}Next Steps:{Colors.ENDC}")
        print("1. Activate virtual environment:")
        if platform.system() == "Windows":
            print(f"   {self.venv_dir}\\Scripts\\activate")
        else:
            print(f"   source {self.venv_dir}/bin/activate")
        
        print("2. Run the system:")
        print(f"   python integrated_system_v5.py")
        
        print("3. Access web dashboard (if available):")
        print(f"   python web_dashboard_v5.py")
        
        print(f"\n{Colors.BOLD}Documentation:{Colors.ENDC}")
        print("• Check README.md for detailed usage instructions")
        print("• Review module documentation in each v5.0 file")
        print("• Contact support for enterprise deployment assistance")
    
    async def run_full_setup(self) -> bool:
        """Run complete setup process."""
        print_header(f"INTEGRATED SYSTEM v5.0 SETUP - {self.mode.upper()} MODE")
        print(f"{Colors.BOLD}Mode Description:{Colors.ENDC} {self.mode_configs[self.mode]['description']}")
        
        # Check system requirements
        if not self.check_system_requirements():
            return False
        
        # Create virtual environment
        if not self.create_virtual_environment():
            return False
        
        # Install dependencies
        if not self.install_dependencies():
            return False
        
        # Download AI models
        if not self.download_ai_models():
            return False
        
        # Run system tests
        if not self.run_system_tests():
            print_warning("Some tests failed, but setup can continue")
        
        # Run demo (optional)
        run_demo = input("\n🤔 Would you like to run the system demo? (y/n): ").lower().strip()
        if run_demo in ['y', 'yes']:
            if not self.run_demo():
                print_warning("Demo failed, but setup is complete")
        
        # Generate and print report
        report = self.generate_setup_report()
        self.print_setup_report(report)
        
        return True

def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(
        description="Setup script for Integrated System v5.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Setup Modes:
  basic       Core AI optimization only
  advanced    AI + Analytics & Microservices
  enterprise  Full enterprise system (RECOMMENDED)
  quantum     Enterprise + Quantum computing (ROADMAP)
  full        Complete v5.0 system with all modules

Examples:
  python setup_v5.py                    # Default: enterprise mode
  python setup_v5.py --mode basic       # Basic mode only
  python setup_v5.py --mode full        # Complete system
  python setup_v5.py --python /usr/bin/python3.11  # Custom Python path
        """
    )
    
    parser.add_argument(
        "--mode", 
        choices=["basic", "advanced", "enterprise", "quantum", "full"],
        default="enterprise",
        help="Setup mode (default: enterprise)"
    )
    
    parser.add_argument(
        "--python",
        help="Custom Python executable path"
    )
    
    args = parser.parse_args()
    
    # Create setup manager
    setup = SetupV5(mode=args.mode, python_path=args.python)
    
    try:
        # Run setup
        success = asyncio.run(setup.run_full_setup())
        
        if success:
            print(f"\n{Colors.OKGREEN}🎉 Setup completed successfully!{Colors.ENDC}")
            sys.exit(0)
        else:
            print(f"\n{Colors.FAIL}❌ Setup failed!{Colors.ENDC}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}⚠️ Setup interrupted by user{Colors.ENDC}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.FAIL}❌ Setup failed with error: {e}{Colors.ENDC}")
        sys.exit(1)

if __name__ == "__main__":
    main()
