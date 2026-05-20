from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import subprocess
import sys
import platform
import importlib
import json
from pathlib import Path
            import torch
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Instagram Captions API v8.0 - Installation Script

Automated installation of deep learning dependencies with smart detection
of GPU capabilities and optimal configurations.
"""



class AIInstaller:
    """Smart installer for AI dependencies."""
    
    def __init__(self) -> Any:
        self.system_info = self._get_system_info()
        self.installation_log = []
    
    def _get_system_info(self) -> Optional[Dict[str, Any]]:
        """Get system information for optimal installation."""
        return {
            "platform": platform.system(),
            "architecture": platform.machine(),
            "python_version": sys.version,
            "pip_available": self._check_pip()
        }
    
    def _check_pip(self) -> Any:
        """Check if pip is available."""
        try:
            subprocess.run([sys.executable, "-m", "pip", "--version"], 
                         capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def _run_command(self, command, description) -> Any:
        """Run installation command with logging."""
        print(f"🔄 {description}...")
        self.installation_log.append(f"Running: {' '.join(command)}")
        
        try:
            result = subprocess.run(
                command, 
                capture_output=True, 
                text=True, 
                check=True,
                timeout=300  # 5 minutes timeout
            )
            
            self.installation_log.append(f"✅ Success: {description}")
            print(f"✅ {description} completed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            error_msg = f"❌ Failed: {description}\nError: {e.stderr}"
            self.installation_log.append(error_msg)
            print(error_msg)
            return False
        
        except subprocess.TimeoutExpired:
            timeout_msg = f"⏰ Timeout: {description} took too long"
            self.installation_log.append(timeout_msg)
            print(timeout_msg)
            return False
    
    def detect_gpu_support(self) -> Any:
        """Detect GPU support and recommend installation type."""
        print("🔍 Detecting GPU capabilities...")
        
        gpu_info = {
            "cuda_available": False,
            "recommended_torch": "cpu",
            "gpu_memory": 0
        }
        
        # Try to detect NVIDIA GPU
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True
            )
            gpu_memory = int(result.stdout.strip().split('\n')[0])
            gpu_info.update({
                "cuda_available": True,
                "recommended_torch": "cu121",  # CUDA 12.1
                "gpu_memory": gpu_memory
            })
            print(f"🖥️ NVIDIA GPU detected with {gpu_memory}MB memory")
            
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
            print("💻 No NVIDIA GPU detected, using CPU-only installation")
        
        return gpu_info
    
    def install_core_dependencies(self) -> Any:
        """Install core dependencies first."""
        print("📦 Installing core dependencies...")
        
        core_packages = [
            "pip>=23.0",
            "setuptools>=65.0",
            "wheel>=0.37.0",
            "packaging>=21.0"
        ]
        
        # Upgrade pip first
        if not self._run_command(
            [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
            "Upgrading pip"
        ):
            return False
        
        # Install core packages
        for package in core_packages:
            if not self._run_command(
                [sys.executable, "-m", "pip", "install", package],
                f"Installing {package}"
            ):
                print(f"⚠️ Warning: Failed to install {package}, continuing...")
        
        return True
    
    def install_pytorch(self, gpu_info) -> Any:
        """Install PyTorch with appropriate GPU support."""
        print("🔥 Installing PyTorch...")
        
        if gpu_info["cuda_available"] and gpu_info["gpu_memory"] >= 4000:  # 4GB minimum
            # CUDA installation
            torch_packages = [
                "torch>=2.1.0",
                "torchvision>=0.16.0",
                "torchaudio>=2.1.0",
                "--index-url", "https://download.pytorch.org/whl/cu121"
            ]
            description = "Installing PyTorch with CUDA support"
        else:
            # CPU-only installation
            torch_packages = [
                "torch>=2.1.0",
                "torchvision>=0.16.0", 
                "torchaudio>=2.1.0",
                "--index-url", "https://download.pytorch.org/whl/cpu"
            ]
            description = "Installing PyTorch (CPU-only)"
        
        return self._run_command(
            [sys.executable, "-m", "pip", "install"] + torch_packages,
            description
        )
    
    def install_transformers_stack(self) -> Any:
        """Install Transformers and related packages."""
        print("🤖 Installing Transformers stack...")
        
        transformers_packages = [
            "transformers>=4.36.0",
            "tokenizers>=0.15.0",
            "accelerate>=0.25.0",
            "sentence-transformers>=2.2.2",
            "huggingface-hub>=0.19.0"
        ]
        
        for package in transformers_packages:
            if not self._run_command(
                [sys.executable, "-m", "pip", "install", package],
                f"Installing {package}"
            ):
                return False
        
        return True
    
    async def install_api_dependencies(self) -> Any:
        """Install API and web framework dependencies."""
        print("🌐 Installing API dependencies...")
        
        api_packages = [
            "fastapi>=0.104.0",
            "uvicorn[standard]>=0.23.0",
            "pydantic>=2.5.0",
            "pydantic-settings>=2.1.0",
            "aiohttp>=3.9.0",
            "httpx>=0.25.0"
        ]
        
        # Performance packages
        performance_packages = [
            "orjson>=3.9.0",
            "cachetools>=5.3.0"
        ]
        
        # Try to install uvloop on Unix systems
        if platform.system() != "Windows":
            api_packages.append("uvloop>=0.19.0")
        
        all_packages = api_packages + performance_packages
        
        for package in all_packages:
            if not self._run_command(
                [sys.executable, "-m", "pip", "install", package],
                f"Installing {package}"
            ):
                print(f"⚠️ Warning: Failed to install {package}, continuing...")
        
        return True
    
    def install_gradio_demo(self) -> Any:
        """Install Gradio for interactive demo."""
        print("🎨 Installing Gradio demo dependencies...")
        
        demo_packages = [
            "gradio>=4.8.0",
            "matplotlib>=3.7.0",
            "pandas>=2.1.0"
        ]
        
        for package in demo_packages:
            if not self._run_command(
                [sys.executable, "-m", "pip", "install", package],
                f"Installing {package}"
            ):
                print(f"⚠️ Warning: Failed to install {package}")
        
        return True
    
    def install_monitoring(self) -> Any:
        """Install monitoring and logging dependencies."""
        print("📊 Installing monitoring dependencies...")
        
        monitoring_packages = [
            "prometheus-client>=0.19.0",
            "structlog>=23.2.0",
            "rich>=13.7.0"
        ]
        
        for package in monitoring_packages:
            self._run_command(
                [sys.executable, "-m", "pip", "install", package],
                f"Installing {package}"
            )
        
        return True
    
    def verify_installation(self) -> Any:
        """Verify that key packages are installed correctly."""
        print("🔍 Verifying installation...")
        
        required_packages = [
            ("torch", "PyTorch"),
            ("transformers", "Transformers"),
            ("fastapi", "FastAPI"),
            ("gradio", "Gradio")
        ]
        
        verification_results = {}
        
        for module_name, display_name in required_packages:
            try:
                module = importlib.import_module(module_name)
                version = getattr(module, "__version__", "unknown")
                verification_results[display_name] = f"✅ {version}"
                print(f"✅ {display_name}: {version}")
            except ImportError:
                verification_results[display_name] = "❌ Not installed"
                print(f"❌ {display_name}: Not installed")
        
        # Special check for CUDA
        try:
            cuda_available = torch.cuda.is_available()
            gpu_count = torch.cuda.device_count() if cuda_available else 0
            if cuda_available:
                gpu_name = torch.cuda.get_device_name(0)
                verification_results["CUDA"] = f"✅ Available ({gpu_count} GPU(s), {gpu_name})"
                print(f"✅ CUDA: Available ({gpu_count} GPU(s), {gpu_name})")
            else:
                verification_results["CUDA"] = "⚠️ Not available (CPU mode)"
                print("⚠️ CUDA: Not available (CPU mode)")
        except ImportError:
            verification_results["CUDA"] = "❌ PyTorch not installed"
        
        return verification_results
    
    def save_installation_log(self) -> Any:
        """Save installation log to file."""
        log_file = Path("ai_v8_installation_log.json")
        
        log_data = {
            "timestamp": str(Path().resolve()),
            "system_info": self.system_info,
            "installation_log": self.installation_log,
            "verification": self.verify_installation()
        }
        
        with open(log_file, "w") as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(log_data, f, indent=2)
        
        print(f"📄 Installation log saved to: {log_file}")
    
    def run_full_installation(self) -> Any:
        """Run complete installation process."""
        print("🚀 Starting Instagram Captions API v8.0 - AI Installation")
        print("="*70)
        
        # Detect system capabilities
        gpu_info = self.detect_gpu_support()
        
        # Installation steps
        steps = [
            ("Core Dependencies", self.install_core_dependencies),
            ("PyTorch", lambda: self.install_pytorch(gpu_info)),
            ("Transformers Stack", self.install_transformers_stack),
            ("API Dependencies", self.install_api_dependencies),
            ("Gradio Demo", self.install_gradio_demo),
            ("Monitoring", self.install_monitoring)
        ]
        
        successful_steps = 0
        
        for step_name, step_function in steps:
            print(f"\n📦 Step: {step_name}")
            print("-" * 50)
            
            if step_function():
                successful_steps += 1
                print(f"✅ {step_name} completed successfully")
            else:
                print(f"❌ {step_name} failed")
        
        # Final verification
        print("\n🔍 Final Verification")
        print("-" * 50)
        verification_results = self.verify_installation()
        
        # Save log
        self.save_installation_log()
        
        # Summary
        print("\n" + "="*70)
        print("📊 INSTALLATION SUMMARY")
        print("="*70)
        print(f"Completed Steps: {successful_steps}/{len(steps)}")
        print("Package Status:")
        for package, status in verification_results.items():
            print(f"  {package}: {status}")
        
        if successful_steps == len(steps):
            print("\n🎉 Installation completed successfully!")
            print("Ready to run: py api_ai_v8.py")
            print("Demo available: py gradio_demo_v8.py")
        else:
            print(f"\n⚠️ Installation completed with {len(steps) - successful_steps} issues")
            print("Check the installation log for details")
        
        print("="*70)


def main():
    """Main installation function."""
    installer = AIInstaller()
    installer.run_full_installation()


match __name__:
    case "__main__":
    main() 