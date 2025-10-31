from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import subprocess
import sys
import pkg_resources
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime
import logging
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
            import torch
    import argparse
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Advanced Dependency Manager for NotebookLM AI Project
=====================================================

This module provides comprehensive dependency management capabilities including:
- Dependency analysis and validation
- Automated installation and updates
- Conflict resolution
- Performance optimization recommendations
- Security vulnerability scanning
"""


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DependencyInfo:
    """Information about a dependency."""
    name: str
    version: str
    category: str
    description: str
    critical: bool = False
    security_risk: bool = False
    performance_impact: str = "low"
    size_mb: Optional[float] = None

class AdvancedDependencyManager:
    """Advanced dependency management for NotebookLM AI project."""
    
    def __init__(self, project_root: Optional[Path] = None):
        
    """__init__ function."""
self.project_root = project_root or Path.cwd()
        self.requirements_dir = self.project_root / "requirements"
        self.dependency_cache = {}
        self.security_vulnerabilities = {}
        
        # Define dependency categories with metadata
        self.dependency_categories = {
            "core": {
                "description": "Essential utilities required for all environments",
                "critical": True,
                "file": "base.txt"
            },
            "ai_ml": {
                "description": "Machine learning and AI model capabilities",
                "critical": True,
                "file": "ai-ml.txt"
            },
            "web_api": {
                "description": "FastAPI and web framework dependencies",
                "critical": True,
                "file": "web-api.txt"
            },
            "document_processing": {
                "description": "Document parsing and processing capabilities",
                "critical": False,
                "file": "document-processing.txt"
            },
            "multimedia": {
                "description": "Image, audio, and video processing",
                "critical": False,
                "file": "multimedia.txt"
            },
            "cloud_deployment": {
                "description": "Cloud and deployment tools",
                "critical": False,
                "file": "cloud-deployment.txt"
            },
            "development": {
                "description": "Testing and development tools",
                "critical": False,
                "file": "development.txt"
            },
            "production": {
                "description": "Production-specific optimizations",
                "critical": False,
                "file": "production.txt"
            }
        }
    
    def analyze_dependencies(self) -> Dict[str, any]:
        """Comprehensive dependency analysis."""
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "total_dependencies": 0,
            "categories": {},
            "conflicts": [],
            "security_issues": [],
            "performance_issues": [],
            "recommendations": []
        }
        
        # Analyze each category
        for category, config in self.dependency_categories.items():
            category_deps = self._parse_requirements_file(config["file"])
            analysis["categories"][category] = {
                "count": len(category_deps),
                "dependencies": category_deps,
                "critical": config["critical"],
                "description": config["description"]
            }
            analysis["total_dependencies"] += len(category_deps)
        
        # Check for conflicts
        analysis["conflicts"] = self._detect_conflicts()
        
        # Check security vulnerabilities
        analysis["security_issues"] = await self._check_security_vulnerabilities()
        
        # Performance analysis
        analysis["performance_issues"] = self._analyze_performance_impact()
        
        # Generate recommendations
        analysis["recommendations"] = self._generate_recommendations(analysis)
        
        return analysis
    
    def _parse_requirements_file(self, filename: str) -> List[Dict[str, str]]:
        """Parse a requirements file and extract dependency information."""
        file_path = self.requirements_dir / filename
        dependencies = []
        
        if not file_path.exists():
            logger.warning(f"Requirements file not found: {filename}")
            return dependencies
        
        with open(file_path, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue
                
                # Parse dependency line
                dep_info = self._parse_dependency_line(line)
                if dep_info:
                    dep_info["file"] = filename
                    dep_info["line"] = line_num
                    dependencies.append(dep_info)
        
        return dependencies
    
    def _parse_dependency_line(self, line: str) -> Optional[Dict[str, str]]:
        """Parse a single dependency line."""
        try:
            # Handle different formats: package==version, package>=version, etc.
            if '==' in line:
                name, version = line.split('==', 1)
            elif '>=' in line:
                name, version = line.split('>=', 1)
                version = f">={version}"
            elif '<=' in line:
                name, version = line.split('<=', 1)
                version = f"<={version}"
            else:
                name = line
                version = "latest"
            
            return {
                "name": name.strip(),
                "version": version.strip(),
                "original_line": line
            }
        except Exception as e:
            logger.warning(f"Failed to parse dependency line: {line} - {e}")
            return None
    
    def _detect_conflicts(self) -> List[Dict[str, str]]:
        """Detect dependency conflicts."""
        conflicts = []
        installed_packages = self._get_installed_packages()
        
        # Check for version conflicts
        for category, config in self.dependency_categories.items():
            deps = self._parse_requirements_file(config["file"])
            for dep in deps:
                name = dep["name"]
                required_version = dep["version"]
                
                if name in installed_packages:
                    installed_version = installed_packages[name]
                    if not self._versions_compatible(required_version, installed_version):
                        conflicts.append({
                            "package": name,
                            "required": required_version,
                            "installed": installed_version,
                            "category": category
                        })
        
        return conflicts
    
    def _versions_compatible(self, required: str, installed: str) -> bool:
        """Check if versions are compatible."""
        try:
            # Simple version compatibility check
            if required.startswith(">="):
                min_version = required[2:]
                return pkg_resources.parse_version(installed) >= pkg_resources.parse_version(min_version)
            elif required.startswith("<="):
                max_version = required[2:]
                return pkg_resources.parse_version(installed) <= pkg_resources.parse_version(max_version)
            elif "==" in required:
                exact_version = required.split("==")[1]
                return installed == exact_version
            else:
                return True
        except Exception:
            return False
    
    def _get_installed_packages(self) -> Dict[str, str]:
        """Get currently installed packages."""
        installed = {}
        for package in pkg_resources.working_set:
            installed[package.project_name] = package.version
        return installed
    
    async def _check_security_vulnerabilities(self) -> List[Dict[str, str]]:
        """Check for known security vulnerabilities."""
        vulnerabilities = []
        
        # This would integrate with security databases like NVD
        # For now, we'll check for some common vulnerable packages
        vulnerable_packages = {
            "requests": ["2.28.0", "2.28.1"],  # Example vulnerable versions
            "urllib3": ["1.26.0", "1.26.1"],   # Example vulnerable versions
        }
        
        installed_packages = self._get_installed_packages()
        
        for package, vulnerable_versions in vulnerable_packages.items():
            if package in installed_packages:
                installed_version = installed_packages[package]
                if installed_version in vulnerable_versions:
                    vulnerabilities.append({
                        "package": package,
                        "version": installed_version,
                        "severity": "high",
                        "description": f"Known security vulnerability in {package} {installed_version}"
                    })
        
        return vulnerabilities
    
    def _analyze_performance_impact(self) -> List[Dict[str, str]]:
        """Analyze performance impact of dependencies."""
        performance_issues = []
        
        # Check for heavy dependencies that might impact startup time
        heavy_packages = ["torch", "transformers", "tensorflow", "opencv-python"]
        installed_packages = self._get_installed_packages()
        
        for package in heavy_packages:
            if package in installed_packages:
                performance_issues.append({
                    "package": package,
                    "issue": "heavy_dependency",
                    "impact": "startup_time",
                    "recommendation": f"Consider lazy loading for {package}"
                })
        
        return performance_issues
    
    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        # Security recommendations
        if analysis["security_issues"]:
            recommendations.append("Update packages with security vulnerabilities")
        
        # Performance recommendations
        if analysis["performance_issues"]:
            recommendations.append("Consider implementing lazy loading for heavy dependencies")
        
        # Conflict recommendations
        if analysis["conflicts"]:
            recommendations.append("Resolve dependency conflicts before deployment")
        
        # General recommendations
        if analysis["total_dependencies"] > 100:
            recommendations.append("Consider splitting dependencies into smaller modules")
        
        return recommendations
    
    def install_category(self, category: str, upgrade: bool = False) -> bool:
        """Install dependencies for a specific category."""
        if category not in self.dependency_categories:
            logger.error(f"Unknown category: {category}")
            return False
        
        config = self.dependency_categories[category]
        file_path = self.requirements_dir / config["file"]
        
        if not file_path.exists():
            logger.error(f"Requirements file not found: {config['file']}")
            return False
        
        try:
            cmd = [sys.executable, "-m", "pip", "install", "-r", str(file_path)]
            if upgrade:
                cmd.append("--upgrade")
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(f"Successfully installed category: {category}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install category {category}: {e.stderr}")
            return False
    
    def install_all(self, include_optional: bool = True) -> bool:
        """Install all dependencies."""
        success = True
        
        # Install core categories first
        core_categories = [cat for cat, config in self.dependency_categories.items() 
                          if config["critical"]]
        
        for category in core_categories:
            if not self.install_category(category):
                success = False
                logger.error(f"Failed to install core category: {category}")
        
        # Install optional categories if requested
        if include_optional and success:
            optional_categories = [cat for cat, config in self.dependency_categories.items() 
                                 if not config["critical"]]
            
            for category in optional_categories:
                if not self.install_category(category):
                    logger.warning(f"Failed to install optional category: {category}")
        
        return success
    
    def create_requirements_summary(self, output_file: str = "dependencies_summary.json") -> None:
        """Create a comprehensive dependencies summary."""
        analysis = self.analyze_dependencies()
        
        with open(output_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            json.dump(analysis, f, indent=2)
        
        logger.info(f"Dependencies summary created: {output_file}")
    
    def check_gpu_compatibility(self) -> Dict[str, any]:
        """Check GPU compatibility for AI/ML dependencies."""
        gpu_info = {
            "cuda_available": False,
            "cudnn_available": False,
            "mps_available": False,
            "gpu_memory": None,
            "recommendations": []
        }
        
        try:
            gpu_info["cuda_available"] = torch.cuda.is_available()
            gpu_info["cudnn_available"] = torch.backends.cudnn.is_available()
            gpu_info["mps_available"] = torch.backends.mps.is_available()
            
            if torch.cuda.is_available():
                gpu_info["gpu_memory"] = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                
                if gpu_info["gpu_memory"] < 8:
                    gpu_info["recommendations"].append("Consider using smaller models or quantization")
                elif gpu_info["gpu_memory"] >= 16:
                    gpu_info["recommendations"].append("GPU memory sufficient for large models")
            
        except ImportError:
            gpu_info["recommendations"].append("PyTorch not installed")
        
        return gpu_info
    
    def optimize_installation_order(self) -> List[str]:
        """Determine optimal installation order for dependencies."""
        # Install base dependencies first
        order = ["core"]
        
        # Then install AI/ML (heaviest dependencies)
        order.append("ai_ml")
        
        # Then web API
        order.append("web_api")
        
        # Then optional categories
        optional = [cat for cat, config in self.dependency_categories.items() 
                   if not config["critical"] and cat not in order]
        order.extend(optional)
        
        return order

class DependencyValidator:
    """Validates dependency configurations and provides recommendations."""
    
    @staticmethod
    def validate_requirements_file(file_path: Path) -> List[str]:
        """Validate a requirements file for common issues."""
        issues = []
        
        if not file_path.exists():
            issues.append(f"File does not exist: {file_path}")
            return issues
        
        with open(file_path, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                if not line or line.startswith('#'):
                    continue
                
                # Check for common issues
                if ' ' in line and not line.startswith('#'):
                    issues.append(f"Line {line_num}: Contains spaces in package specification")
                
                if '==' in line and line.count('==') > 1:
                    issues.append(f"Line {line_num}: Multiple version specifiers")
        
        return issues
    
    @staticmethod
    def check_for_duplicates(requirements_files: List[Path]) -> Dict[str, List[str]]:
        """Check for duplicate dependencies across requirements files."""
        all_dependencies = {}
        duplicates = {}
        
        for file_path in requirements_files:
            if not file_path.exists():
                continue
            
            with open(file_path, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        package_name = line.split('==')[0].split('>=')[0].split('<=')[0].strip()
                        
                        if package_name in all_dependencies:
                            if package_name not in duplicates:
                                duplicates[package_name] = []
                            duplicates[package_name].append(str(file_path))
                        else:
                            all_dependencies[package_name] = str(file_path)
        
        return duplicates

# Example usage and CLI interface
async def main():
    """Main function for dependency management."""
    
    parser = argparse.ArgumentParser(description="NotebookLM AI Dependency Manager")
    parser.add_argument("--analyze", action="store_true", help="Analyze dependencies")
    parser.add_argument("--install", choices=["all", "core", "ai_ml", "web_api"], 
                       help="Install dependencies")
    parser.add_argument("--validate", action="store_true", help="Validate requirements files")
    parser.add_argument("--gpu-check", action="store_true", help="Check GPU compatibility")
    parser.add_argument("--summary", action="store_true", help="Create dependencies summary")
    
    args = parser.parse_args()
    
    manager = AdvancedDependencyManager()
    
    if args.analyze:
        analysis = manager.analyze_dependencies()
        print(json.dumps(analysis, indent=2))
    
    if args.install:
        if args.install == "all":
            success = manager.install_all()
        else:
            success = manager.install_category(args.install)
        
        if success:
            print(f"Successfully installed {args.install} dependencies")
        else:
            print(f"Failed to install {args.install} dependencies")
    
    if args.validate:
        validator = DependencyValidator()
        requirements_files = list(manager.requirements_dir.glob("*.txt"))
        
        for file_path in requirements_files:
            issues = validator.validate_requirements_file(file_path)
            if issues:
                print(f"Issues in {file_path.name}:")
                for issue in issues:
                    print(f"  - {issue}")
            else:
                print(f"✓ {file_path.name} is valid")
    
    if args.gpu_check:
        gpu_info = manager.check_gpu_compatibility()
        print(json.dumps(gpu_info, indent=2))
    
    if args.summary:
        manager.create_requirements_summary()

match __name__:
    case "__main__":
    asyncio.run(main()) 