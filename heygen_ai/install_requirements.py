#!/usr/bin/env python3
"""
HeyGen AI Requirements Installer

This script installs requirements from modular requirements files
based on the selected profile.
"""

import subprocess
import sys
import os
from pathlib import Path
from typing import List, Dict

class RequirementsInstaller:
    """Manages installation of HeyGen AI requirements."""
    
    PROFILES = {
        "minimal": ["base"],
        "basic": ["base", "ml"],
        "web": ["base", "ml", "web"],
        "enterprise": ["base", "ml", "web", "enterprise"],
        "dev": ["base", "ml", "web", "dev"],
        "full": ["base", "ml", "web", "enterprise", "dev"]
    }
    
    def __init__(self):
        self.requirements_dir = Path(__file__).parent / "requirements"
        self.python_executable = self._get_python_executable()
    
    def _get_python_executable(self) -> str:
        """Get the appropriate Python executable."""
        if sys.platform.startswith("win"):
            # Try to find Python in common Windows locations
            python_paths = [
                "python",
                "python3",
                "py",
                r"C:\Users\USER\AppData\Local\Programs\Python\Python311\python.exe",
                r"C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe"
            ]
            
            for path in python_paths:
                try:
                    result = subprocess.run([path, "--version"], 
                                         capture_output=True, text=True, check=True)
                    print(f"✅ Found Python: {path}")
                    return path
                except (subprocess.CalledProcessError, FileNotFoundError):
                    continue
            
            print("❌ No Python executable found")
            return "python"
        else:
            return "python3"
    
    def install_profile(self, profile: str) -> bool:
        """Install requirements for a specific profile."""
        if profile not in self.PROFILES:
            print(f"❌ Unknown profile: {profile}")
            print(f"Available profiles: {', '.join(self.PROFILES.keys())}")
            return False
        
        print(f"🚀 Installing HeyGen AI requirements for profile: {profile}")
        print("=" * 60)
        
        # Install base requirements first
        base_requirements = self.PROFILES[profile]
        
        for req_type in base_requirements:
            req_file = self.requirements_dir / f"{req_type}.txt"
            if req_file.exists():
                print(f"📦 Installing {req_type} requirements...")
                if not self._install_requirements_file(req_file):
                    print(f"❌ Failed to install {req_type} requirements")
                    return False
            else:
                print(f"⚠️  Requirements file not found: {req_file}")
        
        print("=" * 60)
        print(f"✅ Successfully installed {profile} profile requirements!")
        return True
    
    def _install_requirements_file(self, req_file: Path) -> bool:
        """Install requirements from a specific file."""
        try:
            cmd = [self.python_executable, "-m", "pip", "install", "-r", str(req_file)]
            print(f"Running: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"✅ Successfully installed from {req_file.name}")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Error installing from {req_file.name}:")
            print(f"Error code: {e.returncode}")
            if e.stdout:
                print(f"STDOUT: {e.stdout}")
            if e.stderr:
                print(f"STDERR: {e.stderr}")
            return False
    
    def show_available_profiles(self):
        """Display available installation profiles."""
        print("🎯 Available HeyGen AI Installation Profiles:")
        print("=" * 50)
        
        for profile, reqs in self.PROFILES.items():
            print(f"\n🔧 {profile.upper()}:")
            print(f"   Requirements: {', '.join(reqs)}")
            
            if profile == "minimal":
                print("   Description: Core dependencies only")
            elif profile == "basic":
                print("   Description: Core + Machine Learning")
            elif profile == "web":
                print("   Description: Core + ML + Web Framework")
            elif profile == "enterprise":
                print("   Description: Core + ML + Web + Enterprise")
            elif profile == "dev":
                print("   Description: Core + ML + Web + Development")
            elif profile == "full":
                print("   Description: All dependencies")
    
    def check_system_requirements(self) -> bool:
        """Check if system meets basic requirements."""
        print("🔍 Checking system requirements...")
        
        # Check Python version
        try:
            result = subprocess.run([self.python_executable, "--version"], 
                                 capture_output=True, text=True, check=True)
            version = result.stdout.strip()
            print(f"✅ Python: {version}")
            
            # Check if version is 3.8+
            if "3.8" in version or "3.9" in version or "3.10" in version or "3.11" in version or "3.12" in version:
                print("✅ Python version is compatible")
            else:
                print("⚠️  Python 3.8+ recommended")
                
        except subprocess.CalledProcessError:
            print("❌ Could not determine Python version")
            return False
        
        # Check pip
        try:
            result = subprocess.run([self.python_executable, "-m", "pip", "--version"], 
                                 capture_output=True, text=True, check=True)
            print("✅ pip is available")
        except subprocess.CalledProcessError:
            print("❌ pip not available")
            return False
        
        return True

def main():
    """Main installation function."""
    installer = RequirementsInstaller()
    
    if len(sys.argv) > 1:
        profile = sys.argv[1].lower()
        if profile == "list":
            installer.show_available_profiles()
            return
        elif profile == "check":
            installer.check_system_requirements()
            return
        elif profile in installer.PROFILES:
            if installer.check_system_requirements():
                installer.install_profile(profile)
            else:
                print("❌ System requirements not met")
                sys.exit(1)
        else:
            print(f"❌ Unknown profile: {profile}")
            print("Use 'list' to see available profiles")
            sys.exit(1)
    else:
        # Interactive mode
        print("🚀 HeyGen AI Requirements Installer")
        print("=" * 40)
        
        installer.show_available_profiles()
        print("\n" + "=" * 40)
        
        if installer.check_system_requirements():
            print("\nSelect installation profile:")
            for i, profile in enumerate(installer.PROFILES.keys(), 1):
                print(f"{i}. {profile}")
            
            try:
                choice = input("\nEnter your choice (1-6): ").strip()
                if choice.isdigit() and 1 <= int(choice) <= len(installer.PROFILES):
                    profile = list(installer.PROFILES.keys())[int(choice) - 1]
                    installer.install_profile(profile)
                else:
                    print("❌ Invalid choice")
            except KeyboardInterrupt:
                print("\n\n👋 Installation cancelled")
        else:
            print("❌ System requirements not met")
            sys.exit(1)

if __name__ == "__main__":
    main()
