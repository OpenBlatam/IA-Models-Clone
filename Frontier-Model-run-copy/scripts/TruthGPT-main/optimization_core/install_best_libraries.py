"""
INSTALL BEST LIBRARIES - Script de instalación automática
Instala las mejores librerías para TruthGPT de forma automática
"""

import subprocess
import sys
import time

class BestLibrariesInstaller:
    """Instalador automático de las mejores librerías."""
    
    def __init__(self):
        self.libraries = self._get_libraries_config()
        
    def _get_libraries_config(self):
        """Obtener configuración de librerías."""
        return {
            'CRITICAL': [
                'torch>=2.1.0',
                'transformers>=4.35.0',
                'numpy>=1.26.0',
            ],
            'HIGH': [
                'accelerate>=0.25.0',
                'peft>=0.6.0',
                'bitsandbytes>=0.41.0',
                'flash-attn>=2.4.0',
                'xformers>=0.0.23',
                'wandb>=0.16.0',
                'deepspeed>=0.12.0',
                'pandas>=2.1.0',
            ],
            'MEDIUM': [
                'gradio>=4.7.0',
                'langchain>=0.1.0',
                'openai>=1.6.0',
                'anthropic>=0.7.0',
                'diffusers>=0.25.0',
                'mlflow>=2.9.0',
                'tensorboard>=2.15.0',
                'triton>=3.0.0',
                'psutil>=5.9.6',
                'tqdm>=4.66.0',
                'rich>=13.7.0',
                'scikit-learn>=1.3.0',
            ],
            'OPTIONAL': [
                'streamlit>=1.28.0',
                'fairscale>=0.4.13',
                'horovod>=0.28.1',
                'pytest>=7.4.3',
                'black>=23.12.0',
                'ruff>=0.1.6',
                'mypy>=1.7.0',
            ]
        }
    
    def install_package(self, package: str) -> bool:
        """Instalar un paquete."""
        try:
            print(f"📦 Installing {package}...")
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', 
                '--upgrade', package
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"✅ Installed {package}")
            return True
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")
            return False
    
    def install_category(self, category: str):
        """Instalar categoría de librerías."""
        if category not in self.libraries:
            print(f"❌ Unknown category: {category}")
            return
        
        print(f"\n{'='*60}")
        print(f"🚀 Installing {category.upper()} libraries...")
        print(f"{'='*60}\n")
        
        packages = self.libraries[category]
        success = 0
        failed = []
        
        for package in packages:
            if self.install_package(package):
                success += 1
            else:
                failed.append(package)
            time.sleep(0.5)  # Small delay between installs
        
        print(f"\n{'='*60}")
        print(f"✅ Successfully installed: {success}/{len(packages)}")
        if failed:
            print(f"❌ Failed: {len(failed)}")
            print(f"Failed packages: {failed}")
        print(f"{'='*60}\n")
    
    def install_all(self):
        """Instalar todas las librerías."""
        print("🚀 INSTALLING ALL BEST LIBRARIES FOR TRUTHGPT")
        print("="*60)
        
        for category in ['CRITICAL', 'HIGH', 'MEDIUM', 'OPTIONAL']:
            self.install_category(category)
        
        print("✅ Installation complete!")
    
    def install_critical_only(self):
        """Instalar solo librerías críticas."""
        print("🚀 INSTALLING CRITICAL LIBRARIES ONLY")
        print("="*60)
        self.install_category('CRITICAL')
    
    def install_essential(self):
        """Instalar librerías esenciales (CRITICAL + HIGH)."""
        print("🚀 INSTALLING ESSENTIAL LIBRARIES")
        print("="*60)
        
        for category in ['CRITICAL', 'HIGH']:
            self.install_category(category)
    
    def verify_installation(self):
        """Verificar instalación de librerías."""
        print("\n🔍 VERIFYING INSTALLATION")
        print("="*60)
        
        import importlib
        
        all_packages = []
        for category in self.libraries.values():
            all_packages.extend(category)
        
        installed = []
        missing = []
        
        for package in all_packages:
            package_name = package.split('>=')[0].split('==')[0]
            if package_name in ['transformers', 'accelerate', 'peft']:
                # Special cases for packages with different import names
                if package_name == 'transformers':
                    try:
                        importlib.import_module('transformers')
                        installed.append(package_name)
                    except ImportError:
                        missing.append(package_name)
                elif package_name == 'accelerate':
                    try:
                        importlib.import_module('accelerate')
                        installed.append(package_name)
                    except ImportError:
                        missing.append(package_name)
                elif package_name == 'peft':
                    try:
                        importlib.import_module('peft')
                        installed.append(package_name)
                    except ImportError:
                        missing.append(package_name)
            else:
                # Try direct import
                try:
                    importlib.import_module(package_name)
                    installed.append(package_name)
                except ImportError:
                    missing.append(package_name)
        
        print(f"✅ Installed: {len(installed)}")
        print(f"❌ Missing: {len(missing)}")
        
        if missing:
            print(f"\nMissing libraries: {missing}")
            print("Run this script again to install missing libraries.")
        
        print("="*60)
        return len(missing) == 0


def main():
    """Función principal."""
    print("🎯 BEST LIBRARIES INSTALLER FOR TRUTHGPT")
    print("="*60)
    print("\nWhat would you like to do?")
    print("1. Install all libraries")
    print("2. Install critical libraries only")
    print("3. Install essential libraries (critical + high)")
    print("4. Verify installation")
    print("5. Exit")
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    installer = BestLibrariesInstaller()
    
    if choice == '1':
        installer.install_all()
    elif choice == '2':
        installer.install_critical_only()
    elif choice == '3':
        installer.install_essential()
    elif choice == '4':
        installer.verify_installation()
    elif choice == '5':
        print("👋 Goodbye!")
        sys.exit(0)
    else:
        print("❌ Invalid choice!")
        sys.exit(1)


if __name__ == "__main__":
    main()

