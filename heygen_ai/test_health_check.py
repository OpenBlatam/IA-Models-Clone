#!/usr/bin/env python3
"""
Test Health Check for HeyGen AI
==============================

Comprehensive health check script that validates the entire test suite
and provides detailed diagnostics.
"""

import sys
import os
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

class TestHealthChecker:
    """Comprehensive test health checker for HeyGen AI"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.test_dir = self.base_dir / "tests"
        self.core_dir = self.base_dir / "core"
        self.results = {
            "imports": {},
            "files": {},
            "dependencies": {},
            "configuration": {},
            "overall_health": "unknown"
        }
    
    def check_file_structure(self) -> Dict[str, Any]:
        """Check if all required files exist"""
        print("🔍 Checking file structure...")
        
        required_files = {
            "test_files": [
                "tests/__init__.py",
                "tests/conftest.py", 
                "tests/test_basic_imports.py",
                "tests/test_core_structures.py",
                "tests/test_enterprise_features.py",
                "tests/test_lifecycle_management.py"
            ],
            "config_files": [
                "pytest.ini",
                "requirements-test.txt"
            ],
            "runner_files": [
                "run_tests.py",
                "validate_tests.py"
            ],
            "documentation": [
                "TESTING_GUIDE.md",
                "TEST_FIXES_SUMMARY.md",
                "FINAL_TEST_SUMMARY.md"
            ]
        }
        
        file_status = {}
        all_exist = True
        
        for category, files in required_files.items():
            file_status[category] = {}
            for file_path in files:
                full_path = self.base_dir / file_path
                exists = full_path.exists()
                file_status[category][file_path] = exists
                if not exists:
                    all_exist = False
                    print(f"  ❌ Missing: {file_path}")
                else:
                    print(f"  ✅ Found: {file_path}")
        
        self.results["files"] = {
            "status": "healthy" if all_exist else "issues",
            "details": file_status,
            "all_exist": all_exist
        }
        
        return file_status
    
    def check_core_modules(self) -> Dict[str, Any]:
        """Check if core modules exist and are importable"""
        print("\n🔍 Checking core modules...")
        
        core_modules = [
            "base_service",
            "dependency_manager", 
            "error_handler",
            "config_manager",
            "logging_service",
            "enterprise_features"
        ]
        
        module_status = {}
        all_importable = True
        
        for module in core_modules:
            module_path = self.core_dir / f"{module}.py"
            if module_path.exists():
                print(f"  ✅ Module exists: {module}.py")
                module_status[module] = "exists"
            else:
                print(f"  ❌ Missing module: {module}.py")
                module_status[module] = "missing"
                all_importable = False
        
        self.results["imports"] = {
            "status": "healthy" if all_importable else "issues",
            "details": module_status,
            "all_importable": all_importable
        }
        
        return module_status
    
    def check_dependencies(self) -> Dict[str, Any]:
        """Check if required dependencies are available"""
        print("\n🔍 Checking dependencies...")
        
        dependencies = {
            "core": ["pytest", "pytest-asyncio"],
            "optional": ["cryptography", "PyJWT"],
            "testing": ["factory-boy", "faker", "responses"]
        }
        
        dep_status = {}
        all_available = True
        
        for category, deps in dependencies.items():
            dep_status[category] = {}
            for dep in deps:
                try:
                    __import__(dep.replace("-", "_"))
                    print(f"  ✅ {dep} is available")
                    dep_status[category][dep] = "available"
                except ImportError:
                    if category == "core":
                        print(f"  ❌ Required: {dep} is missing")
                        all_available = False
                    else:
                        print(f"  ⚠️  Optional: {dep} is missing")
                    dep_status[category][dep] = "missing"
        
        self.results["dependencies"] = {
            "status": "healthy" if all_available else "issues",
            "details": dep_status,
            "all_available": all_available
        }
        
        return dep_status
    
    def check_configuration(self) -> Dict[str, Any]:
        """Check configuration files"""
        print("\n🔍 Checking configuration...")
        
        config_status = {}
        
        # Check pytest.ini
        pytest_ini = self.base_dir / "pytest.ini"
        if pytest_ini.exists():
            print("  ✅ pytest.ini exists")
            config_status["pytest_ini"] = "exists"
        else:
            print("  ❌ pytest.ini missing")
            config_status["pytest_ini"] = "missing"
        
        # Check requirements-test.txt
        req_file = self.base_dir / "requirements-test.txt"
        if req_file.exists():
            print("  ✅ requirements-test.txt exists")
            config_status["requirements_test"] = "exists"
        else:
            print("  ❌ requirements-test.txt missing")
            config_status["requirements_test"] = "missing"
        
        # Check test directory structure
        if self.test_dir.exists():
            test_files = list(self.test_dir.glob("test_*.py"))
            print(f"  ✅ Found {len(test_files)} test files")
            config_status["test_files_count"] = len(test_files)
        else:
            print("  ❌ tests directory missing")
            config_status["test_files_count"] = 0
        
        self.results["configuration"] = {
            "status": "healthy" if config_status.get("pytest_ini") == "exists" else "issues",
            "details": config_status
        }
        
        return config_status
    
    def run_import_validation(self) -> bool:
        """Run the import validation script"""
        print("\n🔍 Running import validation...")
        
        try:
            result = subprocess.run([
                sys.executable, "validate_tests.py"
            ], capture_output=True, text=True, cwd=self.base_dir, timeout=30)
            
            if result.returncode == 0:
                print("  ✅ Import validation passed")
                return True
            else:
                print("  ❌ Import validation failed")
                print(f"  Error: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            print("  ❌ Import validation timed out")
            return False
        except Exception as e:
            print(f"  ❌ Import validation error: {e}")
            return False
    
    def run_linter_check(self) -> bool:
        """Run linter check on test files"""
        print("\n🔍 Running linter check...")
        
        test_files = list(self.test_dir.glob("test_*.py"))
        all_clean = True
        
        for test_file in test_files:
            try:
                # Simple syntax check
                with open(test_file, 'r', encoding='utf-8') as f:
                    compile(f.read(), str(test_file), 'exec')
                print(f"  ✅ {test_file.name} syntax OK")
            except SyntaxError as e:
                print(f"  ❌ {test_file.name} syntax error: {e}")
                all_clean = False
            except Exception as e:
                print(f"  ⚠️  {test_file.name} check error: {e}")
        
        return all_clean
    
    def generate_health_report(self) -> str:
        """Generate comprehensive health report"""
        report = []
        report.append("🏥 HeyGen AI Test Suite Health Report")
        report.append("=" * 60)
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Overall health
        health_score = 0
        total_checks = 0
        
        # File structure
        files_healthy = self.results["files"]["all_exist"]
        report.append("📁 File Structure:")
        report.append(f"  Status: {'✅ Healthy' if files_healthy else '❌ Issues'}")
        if files_healthy:
            health_score += 1
        total_checks += 1
        
        # Core modules
        imports_healthy = self.results["imports"]["all_importable"]
        report.append("\n🔧 Core Modules:")
        report.append(f"  Status: {'✅ Healthy' if imports_healthy else '❌ Issues'}")
        if imports_healthy:
            health_score += 1
        total_checks += 1
        
        # Dependencies
        deps_healthy = self.results["dependencies"]["all_available"]
        report.append("\n📦 Dependencies:")
        report.append(f"  Status: {'✅ Healthy' if deps_healthy else '❌ Issues'}")
        if deps_healthy:
            health_score += 1
        total_checks += 1
        
        # Configuration
        config_healthy = self.results["configuration"]["status"] == "healthy"
        report.append("\n⚙️  Configuration:")
        report.append(f"  Status: {'✅ Healthy' if config_healthy else '❌ Issues'}")
        if config_healthy:
            health_score += 1
        total_checks += 1
        
        # Overall score
        health_percentage = (health_score / total_checks) * 100
        report.append(f"\n🎯 Overall Health Score: {health_percentage:.1f}%")
        
        if health_percentage >= 90:
            report.append("🏆 Status: EXCELLENT - Test suite is in perfect condition")
            self.results["overall_health"] = "excellent"
        elif health_percentage >= 75:
            report.append("✅ Status: GOOD - Test suite is functional with minor issues")
            self.results["overall_health"] = "good"
        elif health_percentage >= 50:
            report.append("⚠️  Status: FAIR - Test suite has some issues that need attention")
            self.results["overall_health"] = "fair"
        else:
            report.append("❌ Status: POOR - Test suite has significant issues")
            self.results["overall_health"] = "poor"
        
        return "\n".join(report)
    
    def run_full_health_check(self) -> Dict[str, Any]:
        """Run complete health check"""
        print("🏥 Starting HeyGen AI Test Suite Health Check")
        print("=" * 60)
        
        start_time = time.time()
        
        # Run all checks
        self.check_file_structure()
        self.check_core_modules()
        self.check_dependencies()
        self.check_configuration()
        
        # Run validation tests
        import_validation = self.run_import_validation()
        linter_check = self.run_linter_check()
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Generate report
        report = self.generate_health_report()
        print(f"\n{report}")
        
        # Add timing info
        self.results["timing"] = {
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration
        }
        
        self.results["validation"] = {
            "import_validation": import_validation,
            "linter_check": linter_check
        }
        
        return self.results
    
    def save_health_report(self, filename: str = "test_health_report.txt"):
        """Save health report to file"""
        report = self.generate_health_report()
        report_file = self.base_dir / filename
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n📄 Health report saved to: {report_file}")

def main():
    """Main health check function"""
    checker = TestHealthChecker()
    results = checker.run_full_health_check()
    
    # Save report
    checker.save_health_report()
    
    # Return appropriate exit code
    if results["overall_health"] in ["excellent", "good"]:
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main())





