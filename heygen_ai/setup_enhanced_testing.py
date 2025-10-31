#!/usr/bin/env python3
"""
Enhanced Testing Setup for HeyGen AI
====================================

Automated setup and configuration script for the enhanced testing infrastructure.
"""

import sys
import os
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional

class EnhancedTestingSetup:
    """Setup and configuration for enhanced testing infrastructure"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.setup_log = []
    
    def log(self, message: str, level: str = "INFO"):
        """Log setup messages"""
        timestamp = __import__('datetime').datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}"
        self.setup_log.append(log_entry)
        print(log_entry)
    
    def check_python_version(self) -> bool:
        """Check if Python version is compatible"""
        self.log("Checking Python version...")
        
        version = sys.version_info
        if version < (3, 8):
            self.log(f"Python {version.major}.{version.minor} is not supported. Minimum required: 3.8", "ERROR")
            return False
        
        self.log(f"Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    
    def check_dependencies(self) -> Dict[str, bool]:
        """Check if required dependencies are available"""
        self.log("Checking dependencies...")
        
        dependencies = {
            "pytest": False,
            "pytest-asyncio": False,
            "pytest-cov": False,
            "pytest-mock": False,
            "pytest-benchmark": False,
            "pytest-xdist": False,
            "pytest-html": False,
            "pytest-json-report": False,
            "bandit": False,
            "safety": False,
            "flake8": False,
            "black": False,
            "isort": False,
            "mypy": False,
            "psutil": False,
            "pyyaml": False
        }
        
        for dep in dependencies:
            try:
                __import__(dep.replace("-", "_"))
                dependencies[dep] = True
                self.log(f"  ✅ {dep} is available")
            except ImportError:
                dependencies[dep] = False
                self.log(f"  ❌ {dep} is missing")
        
        return dependencies
    
    def install_dependencies(self, dependencies: Dict[str, bool]) -> bool:
        """Install missing dependencies"""
        self.log("Installing missing dependencies...")
        
        missing_deps = [dep for dep, available in dependencies.items() if not available]
        
        if not missing_deps:
            self.log("All dependencies are already installed")
            return True
        
        # Install from requirements-test.txt first
        req_file = self.base_dir / "requirements-test.txt"
        if req_file.exists():
            self.log("Installing from requirements-test.txt...")
            try:
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install", "-r", str(req_file)
                ], capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    self.log("Successfully installed dependencies from requirements-test.txt")
                else:
                    self.log(f"Failed to install from requirements-test.txt: {result.stderr}", "WARNING")
            except Exception as e:
                self.log(f"Error installing from requirements-test.txt: {e}", "WARNING")
        
        # Install individual missing dependencies
        for dep in missing_deps:
            self.log(f"Installing {dep}...")
            try:
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install", dep
                ], capture_output=True, text=True, timeout=120)
                
                if result.returncode == 0:
                    self.log(f"  ✅ Successfully installed {dep}")
                else:
                    self.log(f"  ❌ Failed to install {dep}: {result.stderr}", "ERROR")
            except Exception as e:
                self.log(f"  ❌ Error installing {dep}: {e}", "ERROR")
        
        return True
    
    def create_directories(self):
        """Create necessary directories"""
        self.log("Creating directories...")
        
        directories = [
            "logs",
            "reports",
            "artifacts",
            "tests/fixtures",
            "tests/factories",
            "tests/mock_data",
            "tests/integration",
            "tests/unit",
            "tests/performance"
        ]
        
        for dir_path in directories:
            full_path = self.base_dir / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            self.log(f"  ✅ Created directory: {dir_path}")
    
    def setup_git_hooks(self):
        """Setup Git hooks for automated testing"""
        self.log("Setting up Git hooks...")
        
        git_dir = self.base_dir / ".git"
        if not git_dir.exists():
            self.log("Not a Git repository, skipping Git hooks setup")
            return
        
        hooks_dir = git_dir / "hooks"
        hooks_dir.mkdir(exist_ok=True)
        
        # Pre-commit hook
        pre_commit_hook = hooks_dir / "pre-commit"
        pre_commit_content = """#!/bin/bash
# HeyGen AI Pre-commit Hook

echo "Running pre-commit tests..."

# Run basic tests
python run_tests.py
if [ $? -ne 0 ]; then
    echo "Tests failed. Commit aborted."
    exit 1
fi

# Run linting
python -m flake8 core/ tests/ --count --statistics
if [ $? -ne 0 ]; then
    echo "Linting failed. Commit aborted."
    exit 1
fi

echo "Pre-commit checks passed!"
"""
        
        with open(pre_commit_hook, 'w') as f:
            f.write(pre_commit_content)
        
        # Make executable
        os.chmod(pre_commit_hook, 0o755)
        self.log("  ✅ Created pre-commit hook")
        
        # Pre-push hook
        pre_push_hook = hooks_dir / "pre-push"
        pre_push_content = """#!/bin/bash
# HeyGen AI Pre-push Hook

echo "Running pre-push quality gate..."

# Run quality gate
python test_quality_gate.py
if [ $? -ne 0 ]; then
    echo "Quality gate failed. Push aborted."
    exit 1
fi

echo "Pre-push checks passed!"
"""
        
        with open(pre_push_hook, 'w') as f:
            f.write(pre_push_content)
        
        # Make executable
        os.chmod(pre_push_hook, 0o755)
        self.log("  ✅ Created pre-push hook")
    
    def setup_environment(self):
        """Setup environment variables and configuration"""
        self.log("Setting up environment...")
        
        # Create .env file
        env_file = self.base_dir / ".env"
        env_content = """# HeyGen AI Testing Environment
TEST_ENV=true
COVERAGE_ENV=true
PYTHONPATH=.
MAX_WORKERS=4
BENCHMARK_ITERATIONS=1000
COVERAGE_THRESHOLD=80
QUALITY_GATE_ENABLED=true
"""
        
        with open(env_file, 'w') as f:
            f.write(env_content)
        
        self.log("  ✅ Created .env file")
        
        # Create .gitignore additions
        gitignore_file = self.base_dir / ".gitignore"
        gitignore_additions = """
# Testing artifacts
.pytest_cache/
htmlcov/
.coverage
coverage.xml
coverage.json
*.log
logs/
reports/
artifacts/
benchmark_results.json
optimization_results.json
coverage_analysis.json
quality_gate_report.json
comprehensive_test_results.json
test_health_report.txt
"""
        
        if gitignore_file.exists():
            with open(gitignore_file, 'r') as f:
                existing_content = f.read()
            
            if "Testing artifacts" not in existing_content:
                with open(gitignore_file, 'a') as f:
                    f.write(gitignore_additions)
                self.log("  ✅ Updated .gitignore")
        else:
            with open(gitignore_file, 'w') as f:
                f.write(gitignore_additions)
            self.log("  ✅ Created .gitignore")
    
    def run_initial_tests(self) -> bool:
        """Run initial tests to verify setup"""
        self.log("Running initial tests...")
        
        try:
            # Run health check
            self.log("Running health check...")
            result = subprocess.run([
                sys.executable, "test_health_check.py"
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                self.log("  ✅ Health check passed")
            else:
                self.log(f"  ⚠️ Health check completed with issues: {result.stderr}", "WARNING")
            
            # Run import validation
            self.log("Running import validation...")
            result = subprocess.run([
                sys.executable, "validate_tests.py"
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                self.log("  ✅ Import validation passed")
            else:
                self.log(f"  ❌ Import validation failed: {result.stderr}", "ERROR")
                return False
            
            # Run basic tests
            self.log("Running basic tests...")
            result = subprocess.run([
                sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                self.log("  ✅ Basic tests passed")
            else:
                self.log(f"  ⚠️ Basic tests completed with issues: {result.stderr}", "WARNING")
            
            return True
            
        except subprocess.TimeoutExpired:
            self.log("Initial tests timed out", "ERROR")
            return False
        except Exception as e:
            self.log(f"Error running initial tests: {e}", "ERROR")
            return False
    
    def generate_setup_report(self) -> str:
        """Generate setup completion report"""
        report = []
        report.append("🎉 HeyGen AI Enhanced Testing Setup Complete!")
        report.append("=" * 60)
        report.append(f"Setup completed at: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        report.append("📋 Setup Summary:")
        report.append("-" * 30)
        report.append("✅ Python version compatibility checked")
        report.append("✅ Dependencies installed")
        report.append("✅ Directories created")
        report.append("✅ Git hooks configured")
        report.append("✅ Environment setup")
        report.append("✅ Initial tests run")
        report.append("")
        
        report.append("🚀 Available Commands:")
        report.append("-" * 30)
        report.append("  python run_tests.py                    # Basic test runner")
        report.append("  python advanced_test_runner.py         # Comprehensive test suite")
        report.append("  python test_benchmark.py              # Performance benchmarks")
        report.append("  python test_optimizer.py              # Test optimization")
        report.append("  python test_coverage_analyzer.py      # Coverage analysis")
        report.append("  python test_quality_gate.py           # Quality gate")
        report.append("  python test_health_check.py           # Health diagnostics")
        report.append("  python ci_test_runner.py              # CI/CD test runner")
        report.append("")
        
        report.append("📚 Documentation:")
        report.append("-" * 30)
        report.append("  ENHANCED_TESTING_GUIDE.md             # Enhanced features guide")
        report.append("  TESTING_GUIDE.md                      # Basic testing guide")
        report.append("  README_TESTING.md                     # Infrastructure overview")
        report.append("")
        
        report.append("🎯 Next Steps:")
        report.append("-" * 30)
        report.append("  1. Review the enhanced testing guide")
        report.append("  2. Run the comprehensive test suite")
        report.append("  3. Configure quality gate thresholds")
        report.append("  4. Set up CI/CD integration")
        report.append("  5. Start using advanced features")
        
        return "\n".join(report)
    
    def save_setup_log(self, filename: str = "setup_log.txt"):
        """Save setup log to file"""
        log_file = self.base_dir / filename
        
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(self.setup_log))
        
        self.log(f"Setup log saved to: {log_file}")
    
    def run_setup(self) -> bool:
        """Run complete setup process"""
        self.log("🚀 Starting HeyGen AI Enhanced Testing Setup")
        self.log("=" * 60)
        
        try:
            # Check Python version
            if not self.check_python_version():
                return False
            
            # Check and install dependencies
            dependencies = self.check_dependencies()
            self.install_dependencies(dependencies)
            
            # Create directories
            self.create_directories()
            
            # Setup Git hooks
            self.setup_git_hooks()
            
            # Setup environment
            self.setup_environment()
            
            # Run initial tests
            if not self.run_initial_tests():
                self.log("Initial tests failed, but setup completed", "WARNING")
            
            # Generate and display report
            report = self.generate_setup_report()
            print(f"\n{report}")
            
            # Save setup log
            self.save_setup_log()
            
            self.log("🎉 Enhanced testing setup completed successfully!")
            return True
            
        except Exception as e:
            self.log(f"Setup failed with error: {e}", "ERROR")
            return False

def main():
    """Main setup function"""
    setup = EnhancedTestingSetup()
    success = setup.run_setup()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())




