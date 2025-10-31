#!/usr/bin/env python3
"""
Enhanced Blaze AI System Validation Script

This script performs comprehensive validation of the entire enhanced Blaze AI system
including dependencies, configuration, security, and operational readiness.
"""

import asyncio
import sys
import os
import importlib
import yaml
import json
import subprocess
import platform
import psutil
from pathlib import Path
from typing import Dict, Any, List, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SystemValidator:
    """Comprehensive system validation for Enhanced Blaze AI."""
    
    def __init__(self):
        self.validation_results = {}
        self.critical_errors = []
        self.warnings = []
        self.success_count = 0
        self.total_checks = 0
        
    def print_header(self, title: str):
        """Print a formatted header."""
        print("\n" + "=" * 60)
        print(f"🔍 {title}")
        print("=" * 60)
    
    def print_section(self, title: str):
        """Print a formatted section header."""
        print(f"\n📋 {title}")
        print("-" * 40)
    
    def log_result(self, check_name: str, success: bool, details: str = "", critical: bool = False):
        """Log validation result."""
        self.total_checks += 1
        if success:
            self.success_count += 1
            status = "✅ PASS"
            print(f"{status} {check_name}: {details}")
        else:
            status = "❌ FAIL"
            print(f"{status} {check_name}: {details}")
            if critical:
                self.critical_errors.append(f"{check_name}: {details}")
            else:
                self.warnings.append(f"{check_name}: {details}")
        
        self.validation_results[check_name] = {
            "success": success,
            "details": details,
            "critical": critical
        }
    
    def validate_python_environment(self) -> bool:
        """Validate Python environment and version."""
        self.print_section("Python Environment Validation")
        
        # Check Python version
        python_version = sys.version_info
        if python_version >= (3, 8):
            self.log_result("Python Version", True, f"Python {python_version.major}.{python_version.minor}.{python_version.micro}")
        else:
            self.log_result("Python Version", False, f"Python {python_version.major}.{python_version.minor}.{python_version.micro} (requires 3.8+)", critical=True)
            return False
        
        # Check virtual environment
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            self.log_result("Virtual Environment", True, "Virtual environment is active")
        else:
            self.log_result("Virtual Environment", False, "No virtual environment detected (recommended)")
        
        return True
    
    def validate_dependencies(self) -> bool:
        """Validate all required dependencies."""
        self.print_section("Dependencies Validation")
        
        required_packages = [
            "fastapi",
            "uvicorn",
            "pydantic",
            "redis",
            "prometheus_client",
            "PyJWT",
            "bcrypt",
            "cryptography",
            "passlib",
            "tenacity",
            "circuitbreaker"
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                importlib.import_module(package.replace("-", "_"))
                self.log_result(f"Package: {package}", True, "Available")
            except ImportError:
                missing_packages.append(package)
                self.log_result(f"Package: {package}", False, "Missing", critical=True)
        
        if missing_packages:
            print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
            print("Install with: pip install -r requirements.txt")
            return False
        
        return True
    
    def validate_configuration_files(self) -> bool:
        """Validate configuration files and structure."""
        self.print_section("Configuration Files Validation")
        
        config_files = [
            "config-enhanced.yaml",
            "requirements.txt",
            "main.py",
            "test_enhanced_features.py",
            "demo_enhanced_features.py",
            "Dockerfile",
            "docker-compose.yml",
            "deploy.sh",
            "README_FINAL.md"
        ]
        
        missing_files = []
        for file_path in config_files:
            if os.path.exists(file_path):
                self.log_result(f"Config File: {file_path}", True, "Found")
            else:
                missing_files.append(file_path)
                self.log_result(f"Config File: {file_path}", False, "Missing", critical=True)
        
        if missing_files:
            return False
        
        # Validate YAML configuration
        try:
            with open("config-enhanced.yaml", 'r') as f:
                config = yaml.safe_load(f)
            
            required_sections = ["app", "security", "rate_limiting", "monitoring", "error_handling"]
            for section in required_sections:
                if section in config:
                    self.log_result(f"Config Section: {section}", True, "Present")
                else:
                    self.log_result(f"Config Section: {section}", False, "Missing", critical=True)
            
            self.log_result("YAML Syntax", True, "Valid YAML format")
            
        except Exception as e:
            self.log_result("YAML Syntax", False, f"Invalid YAML: {str(e)}", critical=True)
            return False
        
        return True
    
    def validate_file_permissions(self) -> bool:
        """Validate file permissions and security."""
        self.print_section("File Permissions Validation")
        
        # Check if deploy script is executable
        deploy_script = "deploy.sh"
        if os.path.exists(deploy_script):
            if os.access(deploy_script, os.X_OK):
                self.log_result("Deploy Script Permissions", True, "Executable")
            else:
                self.log_result("Deploy Script Permissions", False, "Not executable (run: chmod +x deploy.sh)")
        else:
            self.log_result("Deploy Script Permissions", False, "File not found", critical=True)
        
        # Check sensitive files
        sensitive_files = [".env", "config-enhanced.yaml"]
        for file_path in sensitive_files:
            if os.path.exists(file_path):
                stat_info = os.stat(file_path)
                mode = stat_info.st_mode & 0o777
                if mode <= 0o644:  # Readable by owner, readable by group and others
                    self.log_result(f"File Permissions: {file_path}", True, f"Secure ({oct(mode)})")
                else:
                    self.log_result(f"File Permissions: {file_path}", False, f"Too permissive ({oct(mode)})")
            else:
                if file_path == ".env":
                    self.log_result(f"File Permissions: {file_path}", False, "File not found (will be created by deploy script)")
                else:
                    self.log_result(f"File Permissions: {file_path}", False, "File not found", critical=True)
        
        return True
    
    def validate_system_resources(self) -> bool:
        """Validate system resources and requirements."""
        self.print_section("System Resources Validation")
        
        # Check memory
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        if memory_gb >= 4:
            self.log_result("Memory", True, f"{memory_gb:.1f}GB available (>= 4GB required)")
        else:
            self.log_result("Memory", False, f"{memory_gb:.1f}GB available (< 4GB required)", critical=True)
        
        # Check disk space
        disk = psutil.disk_usage('.')
        disk_gb = disk.free / (1024**3)
        if disk_gb >= 10:
            self.log_result("Disk Space", True, f"{disk_gb:.1f}GB free (>= 10GB required)")
        else:
            self.log_result("Disk Space", False, f"{disk_gb:.1f}GB free (< 10GB required)", critical=True)
        
        # Check CPU cores
        cpu_count = psutil.cpu_count()
        if cpu_count >= 2:
            self.log_result("CPU Cores", True, f"{cpu_count} cores (>= 2 required)")
        else:
            self.log_result("CPU Cores", False, f"{cpu_count} cores (< 2 required)", critical=True)
        
        return True
    
    def validate_network_connectivity(self) -> bool:
        """Validate network connectivity and ports."""
        self.print_section("Network Connectivity Validation")
        
        # Check if ports are available
        ports_to_check = [8000, 9090, 6379, 3000]
        for port in ports_to_check:
            try:
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                result = sock.connect_ex(('localhost', port))
                sock.close()
                
                if result == 0:
                    self.log_result(f"Port {port}", False, f"Port {port} is already in use")
                else:
                    self.log_result(f"Port {port}", True, f"Port {port} is available")
            except Exception as e:
                self.log_result(f"Port {port}", False, f"Error checking port: {str(e)}")
        
        return True
    
    def validate_docker_environment(self) -> bool:
        """Validate Docker environment if available."""
        self.print_section("Docker Environment Validation")
        
        try:
            # Check if Docker is installed
            result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                self.log_result("Docker Installation", True, "Docker is available")
                
                # Check if Docker daemon is running
                result = subprocess.run(['docker', 'info'], capture_output=True, text=True)
                if result.returncode == 0:
                    self.log_result("Docker Daemon", True, "Docker daemon is running")
                else:
                    self.log_result("Docker Daemon", False, "Docker daemon is not running")
                
                # Check Docker Compose
                result = subprocess.run(['docker-compose', '--version'], capture_output=True, text=True)
                if result.returncode == 0:
                    self.log_result("Docker Compose", True, "Docker Compose is available")
                else:
                    self.log_result("Docker Compose", False, "Docker Compose is not available")
                    
            else:
                self.log_result("Docker Installation", False, "Docker is not installed")
                
        except FileNotFoundError:
            self.log_result("Docker Installation", False, "Docker command not found")
        except Exception as e:
            self.log_result("Docker Installation", False, f"Error checking Docker: {str(e)}")
        
        return True
    
    def validate_code_quality(self) -> bool:
        """Validate code quality and structure."""
        self.print_section("Code Quality Validation")
        
        # Check Python syntax
        python_files = ["main.py", "test_enhanced_features.py", "demo_enhanced_features.py"]
        for file_path in python_files:
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        compile(f.read(), file_path, 'exec')
                    self.log_result(f"Python Syntax: {file_path}", True, "Valid syntax")
                except SyntaxError as e:
                    self.log_result(f"Python Syntax: {file_path}", False, f"Syntax error: {str(e)}", critical=True)
                except Exception as e:
                    self.log_result(f"Python Syntax: {file_path}", False, f"Error: {str(e)}")
            else:
                self.log_result(f"Python Syntax: {file_path}", False, "File not found", critical=True)
        
        # Check for common security issues
        security_patterns = [
            ("Hardcoded Secrets", ["password", "secret", "key", "token"]),
            ("Debug Code", ["print(", "debug", "console.log"]),
            ("SQL Injection", ["execute(", "executescript("]),
            ("Command Injection", ["os.system", "subprocess.call", "eval("])
        ]
        
        for pattern_name, patterns in security_patterns:
            found_issues = []
            for file_path in python_files:
                if os.path.exists(file_path):
                    try:
                        with open(file_path, 'r') as f:
                            content = f.read().lower()
                            for pattern in patterns:
                                if pattern.lower() in content:
                                    found_issues.append(f"{pattern} in {file_path}")
                    except Exception:
                        continue
            
            if found_issues:
                self.log_result(f"Security Check: {pattern_name}", False, f"Potential issues: {', '.join(found_issues[:3])}")
            else:
                self.log_result(f"Security Check: {pattern_name}", True, "No obvious issues found")
        
        return True
    
    def validate_documentation(self) -> bool:
        """Validate documentation completeness."""
        self.print_section("Documentation Validation")
        
        docs_to_check = [
            ("README_FINAL.md", "Main documentation"),
            ("DEPLOYMENT_GUIDE.md", "Deployment guide"),
            ("config-enhanced.yaml", "Configuration reference"),
            ("docker-compose.yml", "Docker configuration"),
            ("Dockerfile", "Container configuration")
        ]
        
        for file_path, description in docs_to_check:
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                        if len(content.strip()) > 100:  # Basic content check
                            self.log_result(f"Documentation: {description}", True, "Comprehensive")
                        else:
                            self.log_result(f"Documentation: {description}", False, "Minimal content")
                except Exception as e:
                    self.log_result(f"Documentation: {description}", False, f"Error reading: {str(e)}")
            else:
                self.log_result(f"Documentation: {description}", False, "File not found", critical=True)
        
        return True
    
    def generate_validation_report(self) -> str:
        """Generate comprehensive validation report."""
        report = []
        report.append("# Enhanced Blaze AI System Validation Report")
        report.append("")
        report.append(f"**Validation Date:** {asyncio.get_event_loop().time()}")
        report.append(f"**System:** {platform.system()} {platform.release()}")
        report.append(f"**Python:** {platform.python_version()}")
        report.append("")
        
        # Summary
        success_rate = (self.success_count / self.total_checks * 100) if self.total_checks > 0 else 0
        report.append("## Summary")
        report.append(f"- **Total Checks:** {self.total_checks}")
        report.append(f"- **Passed:** {self.success_count}")
        report.append(f"- **Failed:** {self.total_checks - self.success_count}")
        report.append(f"- **Success Rate:** {success_rate:.1f}%")
        report.append("")
        
        # Critical Errors
        if self.critical_errors:
            report.append("## 🚨 Critical Errors")
            for error in self.critical_errors:
                report.append(f"- {error}")
            report.append("")
        
        # Warnings
        if self.warnings:
            report.append("## ⚠️ Warnings")
            for warning in self.warnings:
                report.append(f"- {warning}")
            report.append("")
        
        # Detailed Results
        report.append("## Detailed Results")
        for check_name, result in self.validation_results.items():
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            critical = " (CRITICAL)" if result['critical'] else ""
            report.append(f"### {check_name}")
            report.append(f"- **Status:** {status}{critical}")
            report.append(f"- **Details:** {result['details']}")
            report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        if self.critical_errors:
            report.append("1. **Fix all critical errors before proceeding**")
            report.append("2. Review system requirements and dependencies")
            report.append("3. Ensure proper configuration files are present")
        elif self.warnings:
            report.append("1. **System is ready for deployment**")
            report.append("2. Consider addressing warnings for optimal performance")
            report.append("3. Run the demo script to verify functionality")
        else:
            report.append("1. **🎉 System is fully validated and ready!**")
            report.append("2. Proceed with deployment using `./deploy.sh`")
            report.append("3. Run the demo script to showcase features")
        
        return "\n".join(report)
    
    def run_full_validation(self) -> bool:
        """Run complete system validation."""
        self.print_header("Enhanced Blaze AI System Validation")
        print("Performing comprehensive system validation...")
        
        # Run all validation checks
        validations = [
            ("Python Environment", self.validate_python_environment),
            ("Dependencies", self.validate_dependencies),
            ("Configuration Files", self.validate_configuration_files),
            ("File Permissions", self.validate_file_permissions),
            ("System Resources", self.validate_system_resources),
            ("Network Connectivity", self.validate_network_connectivity),
            ("Docker Environment", self.validate_docker_environment),
            ("Code Quality", self.validate_code_quality),
            ("Documentation", self.validate_documentation)
        ]
        
        all_passed = True
        for validation_name, validation_func in validations:
            try:
                self.print_header(f"Validation: {validation_name}")
                if not validation_func():
                    all_passed = False
            except Exception as e:
                self.log_result(f"Validation: {validation_name}", False, f"Validation crashed: {str(e)}", critical=True)
                all_passed = False
        
        # Generate and save report
        report = self.generate_validation_report()
        with open("system_validation_report.md", "w") as f:
            f.write(report)
        
        # Show final results
        self.print_header("Validation Complete")
        print(f"✅ Passed: {self.success_count}/{self.total_checks}")
        print(f"❌ Failed: {self.total_checks - self.success_count}")
        print(f"📊 Success Rate: {(self.success_count/self.total_checks*100):.1f}%")
        
        if self.critical_errors:
            print(f"\n🚨 Critical Errors: {len(self.critical_errors)}")
            print("System is NOT ready for production!")
        elif self.warnings:
            print(f"\n⚠️  Warnings: {len(self.warnings)}")
            print("System is ready but has warnings to consider.")
        else:
            print("\n🎉 All validations passed! System is ready for production!")
        
        print(f"\n📋 Detailed report saved to: system_validation_report.md")
        
        return all_passed and len(self.critical_errors) == 0


async def main():
    """Main validation execution function."""
    validator = SystemValidator()
    success = validator.run_full_validation()
    
    if success:
        print("\n🎯 System validation completed successfully!")
        print("🚀 Ready to deploy with: ./deploy.sh")
        return 0
    else:
        print("\n⚠️  System validation failed. Please fix critical errors before proceeding.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
