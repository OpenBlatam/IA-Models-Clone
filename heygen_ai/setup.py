#!/usr/bin/env python3
"""
HeyGen AI Setup Script

This script helps you set up the HeyGen AI system with all necessary dependencies
and configurations for optimal performance.
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HeyGenAISetup:
    """Setup class for HeyGen AI system."""
    
    def __init__(self):
        self.system = platform.system().lower()
        self.is_windows = self.system == "windows"
        self.is_linux = self.system == "linux"
        self.is_macos = self.system == "darwin"
        
        # Paths
        self.base_path = Path(__file__).parent
        self.requirements_file = self.base_path / "requirements.txt"
        self.config_dir = self.base_path / "config"
        self.data_dir = self.base_path / "data"
        self.logs_dir = self.base_path / "logs"
        self.models_dir = self.base_path / "models"
        self.cache_dir = self.base_path / "cache"
        self.temp_dir = self.base_path / "temp"
        
        # Python info
        self.python_version = sys.version_info
        self.python_executable = sys.executable
        
        logger.info(f"🚀 HeyGen AI Setup for {self.system}")
        logger.info(f"Python: {self.python_version.major}.{self.python_version.minor}.{self.python_version.micro}")
        logger.info(f"Python Path: {self.python_executable}")
    
    def check_python_version(self):
        """Check if Python version meets requirements."""
        logger.info("🔍 Checking Python version...")
        
        if self.python_version.major < 3 or (self.python_version.major == 3 and self.python_version.minor < 8):
            logger.error("❌ Python 3.8+ is required")
            logger.error(f"Current version: {self.python_version.major}.{self.python_version.minor}.{self.python_version.micro}")
            return False
        
        logger.info(f"✅ Python version {self.python_version.major}.{self.python_version.minor}.{self.python_version.micro} is compatible")
        return True
    
    def check_pip(self):
        """Check if pip is available."""
        logger.info("🔍 Checking pip availability...")
        
        try:
            result = subprocess.run([
                self.python_executable, "-m", "pip", "--version"
            ], capture_output=True, text=True, check=True)
            
            pip_version = result.stdout.strip().split()[-1]
            logger.info(f"✅ pip {pip_version} is available")
            return True
            
        except subprocess.CalledProcessError:
            logger.error("❌ pip is not available")
            return False
    
    def check_git(self):
        """Check if git is available."""
        logger.info("🔍 Checking git availability...")
        
        try:
            result = subprocess.run([
                "git", "--version"
            ], capture_output=True, text=True, check=True)
            
            git_version = result.stdout.strip().split()[-1]
            logger.info(f"✅ git {git_version} is available")
            return True
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("⚠️  git is not available (optional)")
            return False
    
    def create_directories(self):
        """Create necessary directories."""
        logger.info("📁 Creating directories...")
        
        directories = [
            self.config_dir,
            self.data_dir,
            self.logs_dir,
            self.models_dir,
            self.cache_dir,
            self.temp_dir
        ]
        
        for directory in directories:
            directory.mkdir(exist_ok=True)
            logger.info(f"✅ Created directory: {directory}")
    
    def install_pytorch(self):
        """Install PyTorch with appropriate CUDA support."""
        logger.info("🔧 Installing PyTorch...")
        
        try:
            # Check if CUDA is available
            cuda_available = self._check_cuda_availability()
            
            if cuda_available:
                logger.info("🚀 Installing PyTorch with CUDA support...")
                
                # Try different CUDA versions
                cuda_versions = ["cu121", "cu118", "cu117"]
                pytorch_installed = False
                
                for cuda_version in cuda_versions:
                    try:
                        logger.info(f"🔄 Trying CUDA {cuda_version}...")
                        
                        subprocess.run([
                            self.python_executable, "-m", "pip", "install",
                            "torch", "torchvision", "torchaudio",
                            "--index-url", f"https://download.pytorch.org/whl/{cuda_version}"
                        ], check=True)
                        
                        logger.info(f"✅ PyTorch installed with CUDA {cuda_version}")
                        pytorch_installed = True
                        break
                        
                    except subprocess.CalledProcessError:
                        logger.warning(f"⚠️  Failed to install with CUDA {cuda_version}")
                        continue
                
                if not pytorch_installed:
                    logger.warning("⚠️  Failed to install PyTorch with CUDA, trying CPU version")
                    self._install_pytorch_cpu()
                
            else:
                logger.info("💻 Installing PyTorch CPU version...")
                self._install_pytorch_cpu()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to install PyTorch: {e}")
            return False
    
    def _check_cuda_availability(self):
        """Check if CUDA is available on the system."""
        try:
            # Check nvidia-smi
            result = subprocess.run([
                "nvidia-smi"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("✅ NVIDIA GPU detected")
                return True
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        
        # Check environment variables
        cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
        if cuda_home:
            logger.info(f"✅ CUDA environment detected: {cuda_home}")
            return True
        
        logger.info("ℹ️  No CUDA detected, will install CPU version")
        return False
    
    def _install_pytorch_cpu(self):
        """Install PyTorch CPU version."""
        try:
            subprocess.run([
                self.python_executable, "-m", "pip", "install",
                "torch", "torchvision", "torchaudio"
            ], check=True)
            
            logger.info("✅ PyTorch CPU version installed")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install PyTorch CPU version: {e}")
            raise
    
    def install_dependencies(self):
        """Install all dependencies from requirements.txt."""
        logger.info("📦 Installing dependencies...")
        
        if not self.requirements_file.exists():
            logger.error("❌ requirements.txt not found")
            return False
        
        try:
            # Install dependencies
            subprocess.run([
                self.python_executable, "-m", "pip", "install", "-r",
                str(self.requirements_file)
            ], check=True)
            
            logger.info("✅ Dependencies installed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install dependencies: {e}")
            return False
    
    def install_optional_dependencies(self):
        """Install optional performance dependencies."""
        logger.info("🚀 Installing optional performance dependencies...")
        
        optional_packages = [
            "flash-attn",  # Flash Attention
            "xformers",    # Memory efficient attention
            "triton",      # Custom CUDA kernels
            "optimum",     # Model optimization
            "onnx",        # ONNX export
            "onnxruntime"  # ONNX runtime
        ]
        
        for package in optional_packages:
            try:
                logger.info(f"🔄 Installing {package}...")
                subprocess.run([
                    self.python_executable, "-m", "pip", "install", package
                ], check=True)
                
                logger.info(f"✅ {package} installed successfully")
                
            except subprocess.CalledProcessError:
                logger.warning(f"⚠️  Failed to install {package} (optional)")
    
    def verify_installation(self):
        """Verify that the installation was successful."""
        logger.info("🔍 Verifying installation...")
        
        try:
            # Test PyTorch
            import torch
            logger.info(f"✅ PyTorch {torch.__version__} imported successfully")
            
            # Test CUDA
            if torch.cuda.is_available():
                logger.info(f"✅ CUDA available: {torch.version.cuda}")
                logger.info(f"✅ GPU device: {torch.cuda.get_device_name(0)}")
            else:
                logger.info("ℹ️  CUDA not available (CPU mode)")
            
            # Test transformers
            import transformers
            logger.info(f"✅ Transformers {transformers.__version__} imported successfully")
            
            # Test diffusers
            import diffusers
            logger.info(f"✅ Diffusers {diffusers.__version__} imported successfully")
            
            # Test gradio
            import gradio
            logger.info(f"✅ Gradio {gradio.__version__} imported successfully")
            
            logger.info("✅ All core dependencies verified successfully")
            return True
            
        except ImportError as e:
            logger.error(f"❌ Import verification failed: {e}")
            return False
    
    def create_sample_config(self):
        """Create sample configuration if it doesn't exist."""
        logger.info("⚙️  Setting up configuration...")
        
        config_file = self.config_dir / "heygen_ai_config.yaml"
        
        if not config_file.exists():
            logger.info("📝 Creating sample configuration...")
            
            # Copy the main config file
            main_config = self.base_path / "config" / "heygen_ai_config.yaml"
            if main_config.exists():
                shutil.copy2(main_config, config_file)
                logger.info("✅ Configuration file created")
            else:
                logger.warning("⚠️  Main configuration file not found")
        else:
            logger.info("✅ Configuration file already exists")
    
    def run_quick_test(self):
        """Run a quick test to verify the system works."""
        logger.info("🧪 Running quick system test...")
        
        try:
            # Test basic imports
            test_script = """
import torch
import transformers
import diffusers
import gradio

print("✅ All imports successful")
print(f"PyTorch: {torch.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"Diffusers: {diffusers.__version__}")
print(f"Gradio: {gradio.__version__}")

if torch.cuda.is_available():
    print(f"CUDA: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA: Not available (CPU mode)")
"""
            
            # Write test script
            test_file = self.temp_dir / "quick_test.py"
            with open(test_file, "w") as f:
                f.write(test_script)
            
            # Run test
            result = subprocess.run([
                self.python_executable, str(test_file)
            ], capture_output=True, text=True, check=True)
            
            logger.info("✅ Quick test completed successfully")
            logger.info("Test output:")
            for line in result.stdout.strip().split("\n"):
                logger.info(f"  {line}")
            
            # Cleanup
            test_file.unlink()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Quick test failed: {e}")
            return False
    
    def display_next_steps(self):
        """Display next steps for the user."""
        logger.info("\n" + "="*60)
        logger.info("🎉 HeyGen AI Setup Completed Successfully!")
        logger.info("="*60)
        
        logger.info("\n🚀 Next Steps:")
        logger.info("1. Launch the demo launcher:")
        logger.info("   python launch_demos.py")
        
        logger.info("\n2. Or run specific demos:")
        logger.info("   python quick_start_ultra_performance.py")
        logger.info("   python run_refactored_demo.py")
        logger.info("   python comprehensive_demo_runner.py")
        
        logger.info("\n3. Check the README.md for detailed usage instructions")
        
        logger.info("\n4. Explore the configuration in config/heygen_ai_config.yaml")
        
        logger.info("\n5. Join the community for support and updates")
        
        logger.info("\n" + "="*60)
        logger.info("Happy AI Development! 🚀✨")
        logger.info("="*60)
    
    def run_setup(self):
        """Run the complete setup process."""
        logger.info("🚀 Starting HeyGen AI Setup...")
        
        try:
            # Check prerequisites
            if not self.check_python_version():
                return False
            
            if not self.check_pip():
                return False
            
            self.check_git()
            
            # Create directories
            self.create_directories()
            
            # Install PyTorch
            if not self.install_pytorch():
                return False
            
            # Install dependencies
            if not self.install_dependencies():
                return False
            
            # Install optional dependencies
            self.install_optional_dependencies()
            
            # Verify installation
            if not self.verify_installation():
                return False
            
            # Setup configuration
            self.create_sample_config()
            
            # Run quick test
            if not self.run_quick_test():
                logger.warning("⚠️  Quick test failed, but setup may still work")
            
            # Display next steps
            self.display_next_steps()
            
            logger.info("🎉 Setup completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Setup failed: {e}")
            return False


def main():
    """Main setup function."""
    try:
        setup = HeyGenAISetup()
        success = setup.run_setup()
        
        if success:
            print("\n🎉 HeyGen AI setup completed successfully!")
            print("🚀 You can now run the demo launcher:")
            print("   python launch_demos.py")
        else:
            print("\n❌ Setup failed. Please check the logs above.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
