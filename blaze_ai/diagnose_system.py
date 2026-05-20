#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
System Diagnostic Script for Blaze AI.
Helps identify and resolve common issues.
"""

import sys
import os
import subprocess
import platform
import shutil
from pathlib import Path
from typing import Dict, List, Any

class SystemDiagnostic:
    """System diagnostic and troubleshooting class."""
    
    def __init__(self):
        self.issues = []
        self.suggestions = []
        self.system_info = {}
    
    def collect_system_info(self):
        """Collect basic system information."""
        print("🔍 Collecting system information...")
        
        self.system_info = {
            "platform": platform.platform(),
            "python_version": sys.version,
            "python_executable": sys.executable,
            "current_directory": os.getcwd(),
            "python_path": sys.path[:5],  # First 5 entries
        }
        
        print(f"  ✅ Platform: {self.system_info['platform']}")
        print(f"  ✅ Python: {self.system_info['python_version'].split()[0]}")
        print(f"  ✅ Executable: {self.system_info['python_executable']}")
        print(f"  ✅ Current Dir: {self.system_info['current_directory']}")
    
    def check_disk_space(self):
        """Check available disk space."""
        print("\n💾 Checking disk space...")
        
        try:
            current_dir = Path.cwd()
            total, used, free = shutil.disk_usage(current_dir)
            
            free_gb = free / (1024**3)
            used_gb = used / (1024**3)
            total_gb = total / (1024**3)
            
            print(f"  📊 Total: {total_gb:.2f} GB")
            print(f"  📊 Used: {used_gb:.2f} GB")
            print(f"  📊 Free: {free_gb:.2f} GB")
            
            if free_gb < 1.0:
                self.issues.append(f"Low disk space: {free_gb:.2f} GB available")
                self.suggestions.append("Free up disk space by removing unnecessary files")
            elif free_gb < 5.0:
                self.issues.append(f"Moderate disk space: {free_gb:.2f} GB available")
                self.suggestions.append("Consider freeing up some disk space")
            else:
                print(f"  ✅ Sufficient disk space available")
                
        except Exception as e:
            self.issues.append(f"Could not check disk space: {e}")
    
    def check_python_installation(self):
        """Check Python installation and accessibility."""
        print("\n🐍 Checking Python installation...")
        
        # Check if python command works
        try:
            result = subprocess.run([sys.executable, "--version"], 
                                  capture_output=True, text=True, check=True)
            print(f"  ✅ Python version check: {result.stdout.strip()}")
        except Exception as e:
            self.issues.append(f"Python version check failed: {e}")
        
        # Check if we can import common modules
        common_modules = ['os', 'sys', 'pathlib', 'subprocess', 'shutil']
        for module in common_modules:
            try:
                __import__(module)
                print(f"  ✅ Module '{module}' available")
            except ImportError as e:
                self.issues.append(f"Module '{module}' not available: {e}")
    
    def check_file_permissions(self):
        """Check file permissions and accessibility."""
        print("\n🔐 Checking file permissions...")
        
        test_files = [
            "engines/__init__.py",
            "engines/plugins.py",
            "simple_test.py"
        ]
        
        for file_path in test_files:
            if os.path.exists(file_path):
                try:
                    # Try to read the file
                    with open(file_path, 'r') as f:
                        f.read(100)  # Read first 100 characters
                    print(f"  ✅ {file_path} is readable")
                    
                    # Try to write to a test file
                    test_write_path = f"{file_path}.test_write"
                    try:
                        with open(test_write_path, 'w') as f:
                            f.write("test")
                        os.remove(test_write_path)
                        print(f"  ✅ {file_path} directory is writable")
                    except Exception as e:
                        self.issues.append(f"Directory not writable: {e}")
                        
                except Exception as e:
                    self.issues.append(f"File {file_path} not readable: {e}")
            else:
                print(f"  ⚠️  {file_path} not found")
    
    def check_import_issues(self):
        """Check for common import issues."""
        print("\n📦 Checking import issues...")
        
        # Test relative imports
        try:
            # Create a simple test module
            test_module = """
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from engines.plugins import PluginConfig
    print("  ✅ Plugin imports working")
except Exception as e:
    print(f"  ❌ Plugin imports failed: {e}")
"""
            
            test_file = "test_imports.py"
            with open(test_file, 'w') as f:
                f.write(test_module)
            
            # Run the test
            result = subprocess.run([sys.executable, test_file], 
                                  capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                print(result.stdout.strip())
            else:
                print(f"  ❌ Import test failed: {result.stderr.strip()}")
                self.issues.append("Plugin imports not working")
                self.suggestions.append("Check import paths and module structure")
            
            # Clean up
            os.remove(test_file)
            
        except Exception as e:
            self.issues.append(f"Import test failed: {e}")
    
    def check_network_connectivity(self):
        """Check basic network connectivity."""
        print("\n🌐 Checking network connectivity...")
        
        try:
            # Try to ping localhost
            result = subprocess.run(["ping", "-n", "1", "127.0.0.1"], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print("  ✅ Local network connectivity working")
            else:
                print("  ⚠️  Local network connectivity issues")
                
        except Exception as e:
            print(f"  ⚠️  Could not check network: {e}")
    
    def generate_report(self):
        """Generate diagnostic report."""
        print("\n" + "=" * 70)
        print("📋 DIAGNOSTIC REPORT")
        print("=" * 70)
        
        if not self.issues:
            print("🎉 No issues detected! Your system appears to be working correctly.")
            return
        
        print(f"❌ Found {len(self.issues)} issue(s):")
        for i, issue in enumerate(self.issues, 1):
            print(f"  {i}. {issue}")
        
        if self.suggestions:
            print(f"\n💡 Suggestions to resolve issues:")
            for i, suggestion in enumerate(self.suggestions, 1):
                print(f"  {i}. {suggestion}")
        
        print(f"\n🔧 System Information:")
        for key, value in self.system_info.items():
            if isinstance(value, list):
                print(f"  {key}: {value[:3]}...")  # Show first 3 items
            else:
                print(f"  {key}: {value}")
    
    def run_full_diagnostic(self):
        """Run complete system diagnostic."""
        print("🚀 Blaze AI System Diagnostic")
        print("=" * 70)
        
        self.collect_system_info()
        self.check_disk_space()
        self.check_python_installation()
        self.check_file_permissions()
        self.check_import_issues()
        self.check_network_connectivity()
        self.generate_report()
        
        return len(self.issues) == 0

def main():
    """Main diagnostic function."""
    try:
        diagnostic = SystemDiagnostic()
        success = diagnostic.run_full_diagnostic()
        
        if success:
            print("\n🎉 System diagnostic completed successfully!")
            sys.exit(0)
        else:
            print("\n⚠️  System diagnostic found issues that need attention.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Diagnostic interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error during diagnostic: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
