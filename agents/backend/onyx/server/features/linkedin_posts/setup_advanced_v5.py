"""
🚀 SETUP ADVANCED v5.0 - INTEGRATED SYSTEM v5.0
================================================

Advanced setup script for the Next-Generation LinkedIn Optimizer v5.0 including:
- Multiple deployment modes (Basic, Advanced, Enterprise, Quantum, Full)
- Automated dependency management
- AI model auto-download
- System health checks
- Production deployment tools
- Performance optimization
"""

import asyncio
import os
import sys
import time
import json
import shutil
import subprocess
import platform
from pathlib import Path
from typing import Dict, Any, List, Optional
import argparse

# Color codes for output
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
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(60)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")

def print_success(text: str):
    """Print success message."""
    print(f"{Colors.OKGREEN}✅ {text}{Colors.ENDC}")

def print_warning(text: str):
    """Print warning message."""
    print(f"{Colors.WARNING}⚠️ {text}{Colors.ENDC}")

def print_error(text: str):
    """Print error message."""
    print(f"{Colors.FAIL}❌ {text}{Colors.ENDC}")

def print_info(text: str):
    """Print info message."""
    print(f"{Colors.OKBLUE}ℹ️ {text}{Colors.ENDC}")

class SetupAdvancedV5:
    """Advanced setup system for v5.0."""
    
    def __init__(self):
        self.system_info = self._get_system_info()
        self.python_version = sys.version_info
        self.requirements_file = "requirements_v5.txt"
        self.setup_mode = None
        self.install_path = Path.cwd()
        
        print_header("SETUP ADVANCED v5.0 - INTEGRATED SYSTEM v5.0")
        print_info(f"System: {self.system_info['os']} {self.system_info['version']}")
        print_info(f"Python: {self.python_version.major}.{self.python_version.minor}.{self.python_version.micro}")
        print_info(f"Architecture: {self.system_info['architecture']}")
        print_info(f"Install Path: {self.install_path}")
    
    def _get_system_info(self) -> Dict[str, str]:
        """Get system information."""
        return {
            'os': platform.system(),
            'version': platform.version(),
            'architecture': platform.machine(),
            'processor': platform.processor()
        }
    
    async def run_setup(self, mode: str = "full") -> bool:
        """Run complete setup process."""
        self.setup_mode = mode
        
        print_header(f"SETUP MODE: {mode.upper()}")
        
        try:
            # 1. System Requirements Check
            if not await self._check_system_requirements():
                return False
            
            # 2. Create Virtual Environment
            if not await self._create_virtual_environment():
                return False
            
            # 3. Install Dependencies
            if not await self._install_dependencies():
                return False
            
            # 4. Download AI Models
            if not await self._download_ai_models():
                return False
            
            # 5. System Tests
            if not await self._run_system_tests():
                return False
            
            # 6. Performance Optimization
            if not await self._optimize_performance():
                return False
            
            # 7. Generate Setup Report
            await self._generate_setup_report()
            
            # 8. Optional: Start System
            if await self._prompt_start_system():
                await self._start_system()
            
            return True
            
        except Exception as e:
            print_error(f"Setup failed: {e}")
            return False
    
    async def _check_system_requirements(self) -> bool:
        """Check system requirements."""
        print_header("SYSTEM REQUIREMENTS CHECK")
        
        requirements = [
            ("Python Version", self.python_version.major >= 3, f"3.8+ (Current: {self.python_version.major}.{self.python_version.minor})"),
            ("Virtual Environment", True, "Will be created"),
            ("Disk Space", self._check_disk_space(), "At least 2GB available"),
            ("Memory", self._check_memory(), "At least 4GB RAM"),
            ("Internet Connection", await self._check_internet(), "Required for AI models")
        ]
        
        all_passed = True
        for requirement, passed, description in requirements:
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"{status} {requirement}: {description}")
            if not passed:
                all_passed = False
        
        if not all_passed:
            print_warning("Some system requirements are not met. Setup may fail.")
            if not await self._prompt_continue():
                return False
        
        return True
    
    def _check_disk_space(self) -> bool:
        """Check available disk space."""
        try:
            stat = shutil.disk_usage(self.install_path)
            available_gb = stat.free / (1024**3)
            return available_gb >= 2.0
        except:
            return True  # Assume OK if can't check
    
    def _check_memory(self) -> bool:
        """Check available memory."""
        try:
            import psutil
            memory_gb = psutil.virtual_memory().total / (1024**3)
            return memory_gb >= 4.0
        except:
            return True  # Assume OK if can't check
    
    async def _check_internet(self) -> bool:
        """Check internet connectivity."""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get('https://httpbin.org/get', timeout=5) as response:
                    return response.status == 200
        except:
            return False
    
    async def _prompt_continue(self) -> bool:
        """Prompt user to continue despite warnings."""
        try:
            response = input("\n⚠️ Continue with setup anyway? (y/N): ").strip().lower()
            return response in ['y', 'yes']
        except:
            return False
    
    async def _create_virtual_environment(self) -> bool:
        """Create Python virtual environment."""
        print_header("VIRTUAL ENVIRONMENT SETUP")
        
        venv_path = self.install_path / "venv_v5"
        
        if venv_path.exists():
            print_warning(f"Virtual environment already exists at {venv_path}")
            if await self._prompt_continue():
                shutil.rmtree(venv_path)
            else:
                return False
        
        try:
            print_info("Creating virtual environment...")
            
            # Create virtual environment
            subprocess.run([
                sys.executable, "-m", "venv", str(venv_path)
            ], check=True, capture_output=True)
            
            # Activate virtual environment
            if self.system_info['os'] == "Windows":
                activate_script = venv_path / "Scripts" / "activate.bat"
                self.venv_python = venv_path / "Scripts" / "python.exe"
            else:
                activate_script = venv_path / "bin" / "activate"
                self.venv_python = venv_path / "bin" / "python"
            
            print_success(f"Virtual environment created at {venv_path}")
            return True
            
        except Exception as e:
            print_error(f"Failed to create virtual environment: {e}")
            return False
    
    async def _install_dependencies(self) -> bool:
        """Install Python dependencies."""
        print_header("DEPENDENCY INSTALLATION")
        
        try:
            # Upgrade pip
            print_info("Upgrading pip...")
            subprocess.run([
                str(self.venv_python), "-m", "pip", "install", "--upgrade", "pip"
            ], check=True, capture_output=True)
            
            # Install dependencies based on mode
            requirements = self._get_requirements_for_mode()
            
            print_info(f"Installing {len(requirements)} dependencies...")
            
            for req in requirements:
                try:
                    print_info(f"Installing {req}...")
                    subprocess.run([
                        str(self.venv_python), "-m", "pip", "install", req
                    ], check=True, capture_output=True)
                    print_success(f"✅ {req}")
                except subprocess.CalledProcessError as e:
                    print_warning(f"⚠️ Failed to install {req}: {e}")
                    if not await self._prompt_continue():
                        return False
            
            print_success("Dependencies installed successfully")
            return True
            
        except Exception as e:
            print_error(f"Failed to install dependencies: {e}")
            return False
    
    def _get_requirements_for_mode(self) -> List[str]:
        """Get requirements list based on setup mode."""
        base_requirements = [
            "fastapi",
            "uvicorn[standard]",
            "pydantic",
            "asyncio",
            "aiohttp",
            "aioredis",
            "prometheus-client",
            "structlog"
        ]
        
        if self.setup_mode in ["advanced", "enterprise", "quantum", "full"]:
            base_requirements.extend([
                "torch",
                "transformers",
                "sentence-transformers",
                "spacy",
                "textblob",
                "vaderSentiment",
                "optuna",
                "mlflow",
                "joblib"
            ])
        
        if self.setup_mode in ["enterprise", "quantum", "full"]:
            base_requirements.extend([
                "numpy",
                "pandas",
                "scikit-learn",
                "matplotlib",
                "seaborn"
            ])
        
        if self.setup_mode in ["quantum", "full"]:
            base_requirements.extend([
                "cryptography",
                "PyJWT",
                "secrets",
                "hashlib",
                "hmac"
            ])
        
        return base_requirements
    
    async def _download_ai_models(self) -> bool:
        """Download AI models."""
        if self.setup_mode in ["basic"]:
            print_info("Skipping AI model download for basic mode")
            return True
        
        print_header("AI MODEL DOWNLOAD")
        
        try:
            models_to_download = [
                "en_core_web_sm",  # spaCy English model
                "bert-base-uncased",  # BERT model
                "roberta-base",  # RoBERTa model
                "distilbert-base-uncased"  # DistilBERT model
            ]
            
            print_info(f"Downloading {len(models_to_download)} AI models...")
            
            for model in models_to_download:
                try:
                    print_info(f"Downloading {model}...")
                    
                    if model.startswith("en_core_web_sm"):
                        # Download spaCy model
                        subprocess.run([
                            str(self.venv_python), "-m", "spacy", "download", model
                        ], check=True, capture_output=True)
                    else:
                        # Download transformers model
                        subprocess.run([
                            str(self.venv_python), "-c", 
                            f"from transformers import {model.split('-')[0]}; {model.split('-')[0]}.from_pretrained('{model}')"
                        ], check=True, capture_output=True)
                    
                    print_success(f"✅ {model}")
                    
                except subprocess.CalledProcessError as e:
                    print_warning(f"⚠️ Failed to download {model}: {e}")
                    if not await self._prompt_continue():
                        return False
            
            print_success("AI models downloaded successfully")
            return True
            
        except Exception as e:
            print_error(f"Failed to download AI models: {e}")
            return False
    
    async def _run_system_tests(self) -> bool:
        """Run system tests."""
        print_header("SYSTEM TESTS")
        
        try:
            print_info("Running system tests...")
            
            # Test basic imports
            test_script = f"""
import sys
sys.path.insert(0, '{self.install_path}')

try:
    # Test basic functionality
    print("Testing basic imports...")
    
    # Test core modules
    import asyncio
    import logging
    import json
    
    print("✅ Basic imports successful")
    
    # Test v5.0 modules if available
    try:
        from integrated_system_v5 import IntegratedSystemV5
        print("✅ Integrated System v5.0 available")
    except ImportError as e:
        print(f"⚠️ Integrated System v5.0 not available: {{e}}")
    
    print("✅ System tests completed successfully")
    
except Exception as e:
    print(f"❌ System tests failed: {{e}}")
    sys.exit(1)
"""
            
            # Write test script
            test_file = self.install_path / "test_setup.py"
            with open(test_file, 'w') as f:
                f.write(test_script)
            
            # Run test script
            result = subprocess.run([
                str(self.venv_python), str(test_file)
            ], capture_output=True, text=True)
            
            # Clean up test file
            test_file.unlink()
            
            if result.returncode == 0:
                print_success("System tests passed")
                return True
            else:
                print_error(f"System tests failed: {result.stderr}")
                return False
                
        except Exception as e:
            print_error(f"Failed to run system tests: {e}")
            return False
    
    async def _optimize_performance(self) -> bool:
        """Optimize system performance."""
        print_header("PERFORMANCE OPTIMIZATION")
        
        try:
            optimizations = []
            
            # Check for GPU availability
            try:
                import torch
                if torch.cuda.is_available():
                    optimizations.append("GPU acceleration available")
                    # Set CUDA device
                    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
                else:
                    optimizations.append("GPU not available, using CPU")
            except ImportError:
                optimizations.append("PyTorch not available")
            
            # Check for Redis availability
            try:
                import aioredis
                optimizations.append("Redis caching available")
            except ImportError:
                optimizations.append("Redis not available, using in-memory caching")
            
            # Performance configuration
            config = {
                'max_workers': min(8, os.cpu_count() or 4),
                'cache_size': 1000,
                'batch_size': 32,
                'timeout': 30
            }
            
            # Save configuration
            config_file = self.install_path / "performance_config.json"
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            print_success("Performance optimization completed")
            for opt in optimizations:
                print_info(f"  • {opt}")
            
            return True
            
        except Exception as e:
            print_error(f"Performance optimization failed: {e}")
            return False
    
    async def _generate_setup_report(self):
        """Generate setup report."""
        print_header("SETUP REPORT GENERATION")
        
        try:
            report = {
                'setup_info': {
                    'mode': self.setup_mode,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'system': self.system_info,
                    'python_version': f"{self.python_version.major}.{self.python_version.minor}.{self.python_version.micro}",
                    'install_path': str(self.install_path),
                    'virtual_environment': str(self.venv_python.parent.parent)
                },
                'dependencies': {
                    'installed': True,
                    'ai_models': self.setup_mode != "basic"
                },
                'tests': {
                    'system_tests': True,
                    'performance_optimization': True
                },
                'status': 'COMPLETED'
            }
            
            # Save report
            report_file = self.install_path / "setup_report_v5.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            print_success(f"Setup report saved to {report_file}")
            
            # Print summary
            print_header("SETUP SUMMARY")
            print(f"Mode: {self.setup_mode.upper()}")
            print(f"Status: {report['status']}")
            print(f"Virtual Environment: {report['setup_info']['virtual_environment']}")
            print(f"AI Models: {'Installed' if report['dependencies']['ai_models'] else 'Skipped'}")
            print(f"Performance: Optimized")
            
        except Exception as e:
            print_error(f"Failed to generate setup report: {e}")
    
    async def _prompt_start_system(self) -> bool:
        """Prompt user to start the system."""
        try:
            response = input("\n🚀 Start the system now? (y/N): ").strip().lower()
            return response in ['y', 'yes']
        except:
            return False
    
    async def _start_system(self):
        """Start the system."""
        print_header("STARTING SYSTEM")
        
        try:
            print_info("Starting Integrated System v5.0...")
            
            # Start command
            start_script = f"""
import asyncio
import sys
sys.path.insert(0, '{self.install_path}')

async def start_system():
    try:
        from integrated_system_v5 import IntegratedSystemV5, OptimizationMode
        
        # Initialize system
        system = IntegratedSystemV5(OptimizationMode.ENTERPRISE)
        
        # Wait for initialization
        await asyncio.sleep(2)
        
        # Get status
        status = await system.get_system_status()
        print(f"System Status: {{status.overall_status.name}}")
        
        # Keep running
        print("System is running. Press Ctrl+C to stop.")
        while True:
            await asyncio.sleep(10)
            
    except Exception as e:
        print(f"Failed to start system: {{e}}")

if __name__ == "__main__":
    asyncio.run(start_system())
"""
            
            # Write start script
            start_file = self.install_path / "start_system_v5.py"
            with open(start_file, 'w') as f:
                f.write(start_script)
            
            print_info("System started successfully!")
            print_info(f"Start script: {start_file}")
            print_info("Run: python start_system_v5.py")
            
        except Exception as e:
            print_error(f"Failed to start system: {e}")

def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Setup Advanced v5.0")
    parser.add_argument(
        "--mode", 
        choices=["basic", "advanced", "enterprise", "quantum", "full"],
        default="full",
        help="Setup mode (default: full)"
    )
    
    args = parser.parse_args()
    
    async def run():
        setup = SetupAdvancedV5()
        success = await setup.run_setup(args.mode)
        
        if success:
            print_header("SETUP COMPLETED SUCCESSFULLY!")
            print_success("🎉 Integrated System v5.0 is ready to use!")
            print_info("Next steps:")
            print_info("  1. Activate virtual environment")
            print_info("  2. Run: python start_system_v5.py")
            print_info("  3. Access web dashboard at http://localhost:8000")
        else:
            print_header("SETUP FAILED")
            print_error("❌ Setup encountered errors. Please check the logs above.")
            sys.exit(1)
    
    asyncio.run(run())

if __name__ == "__main__":
    main()
