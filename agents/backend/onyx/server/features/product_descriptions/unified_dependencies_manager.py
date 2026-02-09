from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

import asyncio
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urlparse
import pkg_resources
import requests
import yaml
import structlog
from typing import Any, List, Dict, Optional
"""
Unified Dependencies Management System

This module provides comprehensive dependency management for the entire
product descriptions system, consolidating all requirements from various
modules and providing:

- Unified dependency resolution and management
- Version compatibility checking
- Automatic dependency installation
- Environment management
- Dependency conflict resolution
- Security vulnerability scanning
- Performance optimization recommendations
"""



logger = structlog.get_logger(__name__)


class DependencyCategory(Enum):
    """Dependency category enumeration."""
    CORE = "core"
    PROFILING = "profiling"
    TRAINING = "training"
    SECURITY = "security"
    VISUALIZATION = "visualization"
    MONITORING = "monitoring"
    TESTING = "testing"
    DEVELOPMENT = "development"
    OPTIONAL = "optional"
    EXPERIMENTAL = "experimental"


class DependencyPriority(Enum):
    """Dependency priority enumeration."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    OPTIONAL = "optional"


class PlatformType(Enum):
    """Platform type enumeration."""
    WINDOWS = "windows"
    LINUX = "linux"
    MACOS = "macos"
    ALL = "all"


@dataclass
class DependencyInfo:
    """Dependency information structure."""
    
    name: str
    version: str
    category: DependencyCategory
    priority: DependencyPriority
    platforms: List[PlatformType] = field(default_factory=lambda: [PlatformType.ALL])
    description: str = ""
    url: str = ""
    license: str = ""
    security_issues: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    alternatives: List[str] = field(default_factory=list)
    performance_impact: str = "low"
    memory_usage: str = "low"
    cpu_usage: str = "low"
    
    def __post_init__(self) -> Any:
        """Post-initialization validation."""
        if not self.name:
            raise ValueError("Dependency name cannot be empty")
        if not self.version:
            raise ValueError("Dependency version cannot be empty")


@dataclass
class DependencyGroup:
    """Dependency group structure."""
    
    name: str
    description: str
    dependencies: List[DependencyInfo]
    category: DependencyCategory
    priority: DependencyPriority
    required: bool = True
    auto_install: bool = True
    
    def __post_init__(self) -> Any:
        """Post-initialization validation."""
        if not self.name:
            raise ValueError("Group name cannot be empty")
        if not self.dependencies:
            raise ValueError("Group must contain at least one dependency")


class UnifiedDependenciesManager:
    """Unified dependencies manager for the entire system."""
    
    def __init__(self, config_path: Optional[str] = None):
        
    """__init__ function."""
self.config_path = config_path or "dependencies_config.yaml"
        self.dependencies: Dict[str, DependencyInfo] = {}
        self.groups: Dict[str, DependencyGroup] = {}
        self.installed_dependencies: Set[str] = set()
        self.conflicts: List[Tuple[str, str]] = []
        self.security_issues: List[Dict[str, str]] = []
        
        # Load dependencies configuration
        self._load_dependencies_config()
        self._scan_installed_dependencies()
    
    def _load_dependencies_config(self) -> Any:
        """Load dependencies configuration from file."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    config = yaml.safe_load(f)
                
                # Load dependencies
                for dep_config in config.get('dependencies', []):
                    dep = DependencyInfo(
                        name=dep_config['name'],
                        version=dep_config['version'],
                        category=DependencyCategory(dep_config['category']),
                        priority=DependencyPriority(dep_config['priority']),
                        platforms=[PlatformType(p) for p in dep_config.get('platforms', ['all'])],
                        description=dep_config.get('description', ''),
                        url=dep_config.get('url', ''),
                        license=dep_config.get('license', ''),
                        security_issues=dep_config.get('security_issues', []),
                        conflicts=dep_config.get('conflicts', []),
                        alternatives=dep_config.get('alternatives', []),
                        performance_impact=dep_config.get('performance_impact', 'low'),
                        memory_usage=dep_config.get('memory_usage', 'low'),
                        cpu_usage=dep_config.get('cpu_usage', 'low')
                    )
                    self.dependencies[dep.name] = dep
                
                # Load groups
                for group_config in config.get('groups', []):
                    group_deps = []
                    for dep_name in group_config.get('dependencies', []):
                        if dep_name in self.dependencies:
                            group_deps.append(self.dependencies[dep_name])
                    
                    group = DependencyGroup(
                        name=group_config['name'],
                        description=group_config.get('description', ''),
                        dependencies=group_deps,
                        category=DependencyCategory(group_config['category']),
                        priority=DependencyPriority(group_config['priority']),
                        required=group_config.get('required', True),
                        auto_install=group_config.get('auto_install', True)
                    )
                    self.groups[group.name] = group
                    
            else:
                # Create default configuration
                self._create_default_config()
                
        except Exception as e:
            logger.error(f"Failed to load dependencies config: {e}")
            self._create_default_config()
    
    def _create_default_config(self) -> Any:
        """Create default dependencies configuration."""
        logger.info("Creating default dependencies configuration")
        
        # Core dependencies
        core_deps = [
            DependencyInfo(
                name="torch",
                version=">=2.0.0",
                category=DependencyCategory.CORE,
                priority=DependencyPriority.CRITICAL,
                description="PyTorch deep learning framework",
                url="https://pytorch.org/",
                license="BSD-3-Clause",
                performance_impact="medium",
                memory_usage="high",
                cpu_usage="medium"
            ),
            DependencyInfo(
                name="numpy",
                version=">=1.21.0",
                category=DependencyCategory.CORE,
                priority=DependencyPriority.CRITICAL,
                description="Numerical computing library",
                url="https://numpy.org/",
                license="BSD-3-Clause",
                performance_impact="low",
                memory_usage="medium",
                cpu_usage="low"
            ),
            DependencyInfo(
                name="pandas",
                version=">=1.3.0",
                category=DependencyCategory.CORE,
                priority=DependencyPriority.HIGH,
                description="Data manipulation and analysis",
                url="https://pandas.pydata.org/",
                license="BSD-3-Clause",
                performance_impact="low",
                memory_usage="medium",
                cpu_usage="low"
            )
        ]
        
        # Profiling dependencies
        profiling_deps = [
            DependencyInfo(
                name="psutil",
                version=">=5.9.0",
                category=DependencyCategory.PROFILING,
                priority=DependencyPriority.HIGH,
                description="System and process utilities",
                url="https://psutil.readthedocs.io/",
                license="BSD-3-Clause",
                performance_impact="low",
                memory_usage="low",
                cpu_usage="low"
            ),
            DependencyInfo(
                name="memory-profiler",
                version=">=0.60.0",
                category=DependencyCategory.PROFILING,
                priority=DependencyPriority.MEDIUM,
                description="Memory profiling utilities",
                url="https://pypi.org/project/memory-profiler/",
                license="BSD-3-Clause",
                performance_impact="medium",
                memory_usage="low",
                cpu_usage="low"
            ),
            DependencyInfo(
                name="GPUtil",
                version=">=1.4.0",
                category=DependencyCategory.PROFILING,
                priority=DependencyPriority.MEDIUM,
                platforms=[PlatformType.LINUX, PlatformType.MACOS],
                description="GPU monitoring utilities",
                url="https://github.com/anderskm/gputil",
                license="MIT",
                performance_impact="low",
                memory_usage="low",
                cpu_usage="low"
            )
        ]
        
        # Training dependencies
        training_deps = [
            DependencyInfo(
                name="torchvision",
                version=">=0.15.0",
                category=DependencyCategory.TRAINING,
                priority=DependencyPriority.HIGH,
                description="Computer vision utilities for PyTorch",
                url="https://pytorch.org/vision/",
                license="BSD-3-Clause",
                performance_impact="medium",
                memory_usage="medium",
                cpu_usage="medium"
            ),
            DependencyInfo(
                name="torchaudio",
                version=">=2.0.0",
                category=DependencyCategory.TRAINING,
                priority=DependencyPriority.MEDIUM,
                description="Audio utilities for PyTorch",
                url="https://pytorch.org/audio/",
                license="BSD-3-Clause",
                performance_impact="medium",
                memory_usage="medium",
                cpu_usage="medium"
            ),
            DependencyInfo(
                name="transformers",
                version=">=4.20.0",
                category=DependencyCategory.TRAINING,
                priority=DependencyPriority.MEDIUM,
                description="Hugging Face Transformers library",
                url="https://huggingface.co/transformers/",
                license="Apache-2.0",
                performance_impact="high",
                memory_usage="high",
                cpu_usage="medium"
            )
        ]
        
        # Visualization dependencies
        viz_deps = [
            DependencyInfo(
                name="matplotlib",
                version=">=3.5.0",
                category=DependencyCategory.VISUALIZATION,
                priority=DependencyPriority.MEDIUM,
                description="Plotting library",
                url="https://matplotlib.org/",
                license="PSF",
                performance_impact="low",
                memory_usage="medium",
                cpu_usage="low"
            ),
            DependencyInfo(
                name="seaborn",
                version=">=0.11.0",
                category=DependencyCategory.VISUALIZATION,
                priority=DependencyPriority.LOW,
                description="Statistical data visualization",
                url="https://seaborn.pydata.org/",
                license="BSD-3-Clause",
                performance_impact="low",
                memory_usage="low",
                cpu_usage="low"
            ),
            DependencyInfo(
                name="plotly",
                version=">=5.0.0",
                category=DependencyCategory.VISUALIZATION,
                priority=DependencyPriority.LOW,
                description="Interactive plotting library",
                url="https://plotly.com/python/",
                license="MIT",
                performance_impact="medium",
                memory_usage="medium",
                cpu_usage="low"
            )
        ]
        
        # Testing dependencies
        testing_deps = [
            DependencyInfo(
                name="pytest",
                version=">=7.0.0",
                category=DependencyCategory.TESTING,
                priority=DependencyPriority.HIGH,
                description="Testing framework",
                url="https://pytest.org/",
                license="MIT",
                performance_impact="low",
                memory_usage="low",
                cpu_usage="low"
            ),
            DependencyInfo(
                name="pytest-asyncio",
                version=">=0.20.0",
                category=DependencyCategory.TESTING,
                priority=DependencyPriority.MEDIUM,
                description="Async testing support for pytest",
                url="https://pytest-asyncio.readthedocs.io/",
                license="Apache-2.0",
                performance_impact="low",
                memory_usage="low",
                cpu_usage="low"
            )
        ]
        
        # Add all dependencies
        all_deps = core_deps + profiling_deps + training_deps + viz_deps + testing_deps
        for dep in all_deps:
            self.dependencies[dep.name] = dep
        
        # Create groups
        self.groups = {
            "core": DependencyGroup(
                name="core",
                description="Core system dependencies",
                dependencies=core_deps,
                category=DependencyCategory.CORE,
                priority=DependencyPriority.CRITICAL,
                required=True,
                auto_install=True
            ),
            "profiling": DependencyGroup(
                name="profiling",
                description="Code profiling and optimization dependencies",
                dependencies=profiling_deps,
                category=DependencyCategory.PROFILING,
                priority=DependencyPriority.HIGH,
                required=True,
                auto_install=True
            ),
            "training": DependencyGroup(
                name="training",
                description="Machine learning training dependencies",
                dependencies=training_deps,
                category=DependencyCategory.TRAINING,
                priority=DependencyPriority.HIGH,
                required=True,
                auto_install=True
            ),
            "visualization": DependencyGroup(
                name="visualization",
                description="Data visualization dependencies",
                dependencies=viz_deps,
                category=DependencyCategory.VISUALIZATION,
                priority=DependencyPriority.MEDIUM,
                required=False,
                auto_install=True
            ),
            "testing": DependencyGroup(
                name="testing",
                description="Testing framework dependencies",
                dependencies=testing_deps,
                category=DependencyCategory.TESTING,
                priority=DependencyPriority.HIGH,
                required=True,
                auto_install=True
            )
        }
        
        # Save configuration
        self._save_config()
    
    def _save_config(self) -> Any:
        """Save dependencies configuration to file."""
        try:
            config = {
                'dependencies': [
                    {
                        'name': dep.name,
                        'version': dep.version,
                        'category': dep.category.value,
                        'priority': dep.priority.value,
                        'platforms': [p.value for p in dep.platforms],
                        'description': dep.description,
                        'url': dep.url,
                        'license': dep.license,
                        'security_issues': dep.security_issues,
                        'conflicts': dep.conflicts,
                        'alternatives': dep.alternatives,
                        'performance_impact': dep.performance_impact,
                        'memory_usage': dep.memory_usage,
                        'cpu_usage': dep.cpu_usage
                    }
                    for dep in self.dependencies.values()
                ],
                'groups': [
                    {
                        'name': group.name,
                        'description': group.description,
                        'dependencies': [dep.name for dep in group.dependencies],
                        'category': group.category.value,
                        'priority': group.priority.value,
                        'required': group.required,
                        'auto_install': group.auto_install
                    }
                    for group in self.groups.values()
                ]
            }
            
            with open(self.config_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                yaml.dump(config, f, default_flow_style=False, indent=2)
                
            logger.info(f"Dependencies configuration saved to {self.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to save dependencies config: {e}")
    
    def _scan_installed_dependencies(self) -> Any:
        """Scan currently installed dependencies."""
        try:
            installed_packages = pkg_resources.working_set
            for package in installed_packages:
                self.installed_dependencies.add(package.project_name.lower())
            
            logger.info(f"Found {len(self.installed_dependencies)} installed packages")
            
        except Exception as e:
            logger.error(f"Failed to scan installed dependencies: {e}")
    
    def get_dependency_info(self, name: str) -> Optional[DependencyInfo]:
        """Get dependency information by name."""
        return self.dependencies.get(name.lower())
    
    def get_group_dependencies(self, group_name: str) -> List[DependencyInfo]:
        """Get dependencies for a specific group."""
        group = self.groups.get(group_name.lower())
        return group.dependencies if group else []
    
    def check_dependency_installed(self, name: str) -> bool:
        """Check if a dependency is installed."""
        return name.lower() in self.installed_dependencies
    
    def check_dependency_version(self, name: str, required_version: str) -> bool:
        """Check if installed dependency version meets requirements."""
        try:
            installed_version = pkg_resources.get_distribution(name).version
            return pkg_resources.require(f"{name}{required_version}")
        except (pkg_resources.DistributionNotFound, pkg_resources.VersionConflict):
            return False
    
    def get_missing_dependencies(self, group_name: Optional[str] = None) -> List[DependencyInfo]:
        """Get list of missing dependencies."""
        missing = []
        
        if group_name:
            deps = self.get_group_dependencies(group_name)
        else:
            deps = list(self.dependencies.values())
        
        for dep in deps:
            if not self.check_dependency_installed(dep.name):
                missing.append(dep)
            elif not self.check_dependency_version(dep.name, dep.version):
                missing.append(dep)
        
        return missing
    
    def get_outdated_dependencies(self) -> List[Tuple[DependencyInfo, str, str]]:
        """Get list of outdated dependencies with current and latest versions."""
        outdated = []
        
        for dep in self.dependencies.values():
            if self.check_dependency_installed(dep.name):
                try:
                    current_version = pkg_resources.get_distribution(dep.name).version
                    latest_version = self._get_latest_version(dep.name)
                    
                    if latest_version and current_version != latest_version:
                        outdated.append((dep, current_version, latest_version))
                        
                except Exception as e:
                    logger.warning(f"Failed to check version for {dep.name}: {e}")
        
        return outdated
    
    def _get_latest_version(self, package_name: str) -> Optional[str]:
        """Get latest version of a package from PyPI."""
        try:
            url = f"https://pypi.org/pypi/{package_name}/json"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data['info']['version']
        except Exception as e:
            logger.debug(f"Failed to get latest version for {package_name}: {e}")
        
        return None
    
    def check_dependency_conflicts(self) -> List[Tuple[str, str]]:
        """Check for dependency conflicts."""
        conflicts = []
        
        for dep in self.dependencies.values():
            for conflict in dep.conflicts:
                if self.check_dependency_installed(conflict):
                    conflicts.append((dep.name, conflict))
        
        return conflicts
    
    def check_security_vulnerabilities(self) -> List[Dict[str, str]]:
        """Check for known security vulnerabilities."""
        vulnerabilities = []
        
        for dep in self.dependencies.values():
            if dep.security_issues:
                for issue in dep.security_issues:
                    vulnerabilities.append({
                        'package': dep.name,
                        'issue': issue,
                        'severity': 'high'  # Could be enhanced with actual severity
                    })
        
        return vulnerabilities
    
    def generate_requirements_file(self, group_name: Optional[str] = None, 
                                 include_optional: bool = False) -> str:
        """Generate requirements.txt file content."""
        requirements = []
        
        if group_name:
            deps = self.get_group_dependencies(group_name)
        else:
            deps = list(self.dependencies.values())
        
        for dep in deps:
            if include_optional or dep.priority != DependencyPriority.OPTIONAL:
                requirements.append(f"{dep.name}{dep.version}")
        
        return "\n".join(sorted(requirements))
    
    def install_dependencies(self, group_name: Optional[str] = None, 
                           upgrade: bool = False) -> Dict[str, bool]:
        """Install dependencies for a group or all dependencies."""
        results = {}
        
        if group_name:
            deps = self.get_group_dependencies(group_name)
        else:
            deps = list(self.dependencies.values())
        
        for dep in deps:
            try:
                if upgrade:
                    cmd = [sys.executable, "-m", "pip", "install", "--upgrade", f"{dep.name}{dep.version}"]
                else:
                    cmd = [sys.executable, "-m", "pip", "install", f"{dep.name}{dep.version}"]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                results[dep.name] = result.returncode == 0
                
                if result.returncode == 0:
                    logger.info(f"Successfully installed {dep.name}")
                else:
                    logger.error(f"Failed to install {dep.name}: {result.stderr}")
                    
            except Exception as e:
                logger.error(f"Failed to install {dep.name}: {e}")
                results[dep.name] = False
        
        # Update installed dependencies list
        self._scan_installed_dependencies()
        
        return results
    
    def uninstall_dependencies(self, group_name: Optional[str] = None) -> Dict[str, bool]:
        """Uninstall dependencies for a group or all dependencies."""
        results = {}
        
        if group_name:
            deps = self.get_group_dependencies(group_name)
        else:
            deps = list(self.dependencies.values())
        
        for dep in deps:
            if self.check_dependency_installed(dep.name):
                try:
                    cmd = [sys.executable, "-m", "pip", "uninstall", "-y", dep.name]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                    results[dep.name] = result.returncode == 0
                    
                    if result.returncode == 0:
                        logger.info(f"Successfully uninstalled {dep.name}")
                    else:
                        logger.error(f"Failed to uninstall {dep.name}: {result.stderr}")
                        
                except Exception as e:
                    logger.error(f"Failed to uninstall {dep.name}: {e}")
                    results[dep.name] = False
        
        # Update installed dependencies list
        self._scan_installed_dependencies()
        
        return results
    
    def get_dependency_report(self) -> Dict[str, any]:
        """Generate comprehensive dependency report."""
        missing = self.get_missing_dependencies()
        outdated = self.get_outdated_dependencies()
        conflicts = self.check_dependency_conflicts()
        vulnerabilities = self.check_security_vulnerabilities()
        
        # Calculate statistics
        total_deps = len(self.dependencies)
        installed_deps = len([d for d in self.dependencies.values() 
                            if self.check_dependency_installed(d.name)])
        missing_deps = len(missing)
        outdated_deps = len(outdated)
        
        # Group by category
        by_category = {}
        for dep in self.dependencies.values():
            cat = dep.category.value
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append({
                'name': dep.name,
                'version': dep.version,
                'installed': self.check_dependency_installed(dep.name),
                'priority': dep.priority.value
            })
        
        return {
            'summary': {
                'total_dependencies': total_deps,
                'installed_dependencies': installed_deps,
                'missing_dependencies': missing_deps,
                'outdated_dependencies': outdated_deps,
                'conflicts': len(conflicts),
                'vulnerabilities': len(vulnerabilities),
                'installation_rate': installed_deps / total_deps if total_deps > 0 else 0
            },
            'missing_dependencies': [
                {
                    'name': dep.name,
                    'version': dep.version,
                    'category': dep.category.value,
                    'priority': dep.priority.value,
                    'description': dep.description
                }
                for dep in missing
            ],
            'outdated_dependencies': [
                {
                    'name': dep.name,
                    'current_version': current,
                    'latest_version': latest,
                    'category': dep.category.value,
                    'priority': dep.priority.value
                }
                for dep, current, latest in outdated
            ],
            'conflicts': [
                {
                    'package1': pkg1,
                    'package2': pkg2,
                    'description': f"Conflict between {pkg1} and {pkg2}"
                }
                for pkg1, pkg2 in conflicts
            ],
            'vulnerabilities': vulnerabilities,
            'by_category': by_category,
            'groups': {
                name: {
                    'description': group.description,
                    'dependencies': len(group.dependencies),
                    'installed': len([d for d in group.dependencies 
                                    if self.check_dependency_installed(d.name)]),
                    'required': group.required,
                    'auto_install': group.auto_install
                }
                for name, group in self.groups.items()
            }
        }
    
    def validate_environment(self) -> Dict[str, any]:
        """Validate the current environment against requirements."""
        report = self.get_dependency_report()
        
        # Check critical dependencies
        critical_missing = [d for d in report['missing_dependencies'] 
                          if d['priority'] == 'critical']
        
        # Check for conflicts
        has_conflicts = len(report['conflicts']) > 0
        
        # Check for vulnerabilities
        has_vulnerabilities = len(report['vulnerabilities']) > 0
        
        # Overall validation
        is_valid = (len(critical_missing) == 0 and 
                   not has_conflicts and 
                   not has_vulnerabilities)
        
        return {
            'is_valid': is_valid,
            'critical_missing': critical_missing,
            'has_conflicts': has_conflicts,
            'has_vulnerabilities': has_vulnerabilities,
            'warnings': [
                f"Missing {len(critical_missing)} critical dependencies",
                f"Found {len(report['conflicts'])} dependency conflicts",
                f"Found {len(report['vulnerabilities'])} security vulnerabilities"
            ] if not is_valid else [],
            'recommendations': self._generate_recommendations(report)
        }
    
    def _generate_recommendations(self, report: Dict[str, any]) -> List[str]:
        """Generate recommendations based on dependency report."""
        recommendations = []
        
        if report['missing_dependencies']:
            recommendations.append(
                f"Install {len(report['missing_dependencies'])} missing dependencies"
            )
        
        if report['outdated_dependencies']:
            recommendations.append(
                f"Update {len(report['outdated_dependencies'])} outdated dependencies"
            )
        
        if report['conflicts']:
            recommendations.append(
                f"Resolve {len(report['conflicts'])} dependency conflicts"
            )
        
        if report['vulnerabilities']:
            recommendations.append(
                f"Address {len(report['vulnerabilities'])} security vulnerabilities"
            )
        
        return recommendations


# Utility functions
def create_requirements_file(group_name: str, output_path: str = "requirements.txt"):
    """Create requirements.txt file for a specific group."""
    manager = UnifiedDependenciesManager()
    requirements = manager.generate_requirements_file(group_name)
    
    with open(output_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        f.write(requirements)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
    
    logger.info(f"Requirements file created: {output_path}")
    return output_path


def install_group_dependencies(group_name: str, upgrade: bool = False):
    """Install dependencies for a specific group."""
    manager = UnifiedDependenciesManager()
    results = manager.install_dependencies(group_name, upgrade)
    
    success_count = sum(1 for success in results.values() if success)
    total_count = len(results)
    
    logger.info(f"Installed {success_count}/{total_count} dependencies for group '{group_name}'")
    return results


def validate_system_dependencies():
    """Validate system dependencies and return status."""
    manager = UnifiedDependenciesManager()
    validation = manager.validate_environment()
    
    if validation['is_valid']:
        logger.info("✅ All system dependencies are valid")
    else:
        logger.warning("⚠️ System dependencies have issues:")
        for warning in validation['warnings']:
            logger.warning(f"  - {warning}")
    
    return validation


async def demo_unified_dependencies():
    """Demonstrate unified dependencies management."""
    logger.info("Starting Unified Dependencies Management Demo")
    
    # Create manager
    manager = UnifiedDependenciesManager()
    
    # Get dependency report
    report = manager.get_dependency_report()
    
    # Print summary
    summary = report['summary']
    logger.info(f"Total dependencies: {summary['total_dependencies']}")
    logger.info(f"Installed: {summary['installed_dependencies']}")
    logger.info(f"Missing: {summary['missing_dependencies']}")
    logger.info(f"Outdated: {summary['outdated_dependencies']}")
    
    # Validate environment
    validation = manager.validate_environment()
    
    if validation['is_valid']:
        logger.info("✅ Environment validation passed")
    else:
        logger.warning("⚠️ Environment validation failed:")
        for warning in validation['warnings']:
            logger.warning(f"  - {warning}")
        
        logger.info("Recommendations:")
        for rec in validation['recommendations']:
            logger.info(f"  - {rec}")
    
    # Generate requirements files for each group
    for group_name in manager.groups.keys():
        requirements = manager.generate_requirements_file(group_name)
        output_file = f"requirements-{group_name}.txt"
        
        with open(output_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write(requirements)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        
        logger.info(f"Generated requirements file: {output_file}")
    
    return {
        'report': report,
        'validation': validation,
        'groups': list(manager.groups.keys())
    }


if __name__ == "__main__":
    # Run demo
    asyncio.run(demo_unified_dependencies()) 