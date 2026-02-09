#!/usr/bin/env python3
"""
Dependency Manager for Video-OpusClip
Comprehensive dependency management system
"""

import subprocess
import sys
import os
import json
import yaml
import toml
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class DependencyType(Enum):
    """Dependency type enumeration"""
    PRODUCTION = "production"
    DEVELOPMENT = "development"
    TESTING = "testing"
    OPTIONAL = "optional"


class DependencyStatus(Enum):
    """Dependency status enumeration"""
    INSTALLED = "installed"
    OUTDATED = "outdated"
    MISSING = "missing"
    CONFLICT = "conflict"


@dataclass
class Dependency:
    """Dependency information"""
    name: str
    version: str
    type: DependencyType
    description: Optional[str] = None
    url: Optional[str] = None
    license: Optional[str] = None
    status: DependencyStatus = DependencyStatus.MISSING
    installed_version: Optional[str] = None
    latest_version: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)


@dataclass
class DependencyGroup:
    """Dependency group information"""
    name: str
    type: DependencyType
    dependencies: List[Dependency] = field(default_factory=list)
    description: Optional[str] = None


class DependencyManager:
    """Comprehensive dependency manager"""
    
    def __init__(self, project_root: Optional[str] = None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.requirements_dir = self.project_root / "requirements"
        self.dependencies: Dict[str, Dependency] = {}
        self.groups: Dict[str, DependencyGroup] = {}
        self._load_dependencies()
    
    def _load_dependencies(self):
        """Load dependencies from requirements files"""
        # Load main requirements
        main_req_file = self.requirements_dir / "requirements.txt"
        if main_req_file.exists():
            self._load_requirements_file(main_req_file, DependencyType.PRODUCTION)
        
        # Load development requirements
        dev_req_file = self.requirements_dir / "requirements-dev.txt"
        if dev_req_file.exists():
            self._load_requirements_file(dev_req_file, DependencyType.DEVELOPMENT)
        
        # Load testing requirements
        test_req_file = self.requirements_dir / "requirements-test.txt"
        if test_req_file.exists():
            self._load_requirements_file(test_req_file, DependencyType.TESTING)
        
        # Load production requirements
        prod_req_file = self.requirements_dir / "requirements-prod.txt"
        if prod_req_file.exists():
            self._load_requirements_file(prod_req_file, DependencyType.PRODUCTION)
    
    def _load_requirements_file(self, file_path: Path, dep_type: DependencyType):
        """Load dependencies from a requirements file"""
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Parse dependency line
                    if '==' in line:
                        name, version = line.split('==', 1)
                    elif '>=' in line:
                        name, version = line.split('>=', 1)
                        version = f">={version}"
                    elif '<=' in line:
                        name, version = line.split('<=', 1)
                        version = f"<={version}"
                    elif '~=' in line:
                        name, version = line.split('~=', 1)
                        version = f"~={version}"
                    else:
                        name, version = line, "*"
                    
                    name = name.strip()
                    version = version.strip()
                    
                    dependency = Dependency(
                        name=name,
                        version=version,
                        type=dep_type
                    )
                    
                    self.dependencies[name] = dependency
                    
        except Exception as e:
            logger.error(f"Failed to load requirements file {file_path}: {e}")
    
    def install_dependencies(
        self,
        dep_type: Optional[DependencyType] = None,
        upgrade: bool = False,
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Install dependencies
        
        Args:
            dep_type: Type of dependencies to install
            upgrade: Whether to upgrade existing packages
            force: Whether to force reinstall
            
        Returns:
            Installation results
        """
        results = {
            "success": [],
            "failed": [],
            "skipped": [],
            "upgraded": []
        }
        
        # Determine which requirements file to use
        if dep_type == DependencyType.DEVELOPMENT:
            req_file = self.requirements_dir / "requirements-dev.txt"
        elif dep_type == DependencyType.TESTING:
            req_file = self.requirements_dir / "requirements-test.txt"
        elif dep_type == DependencyType.PRODUCTION:
            req_file = self.requirements_dir / "requirements-prod.txt"
        else:
            req_file = self.requirements_dir / "requirements.txt"
        
        if not req_file.exists():
            results["failed"].append(f"Requirements file not found: {req_file}")
            return results
        
        # Build pip command
        cmd = [sys.executable, "-m", "pip", "install"]
        
        if upgrade:
            cmd.append("--upgrade")
        
        if force:
            cmd.append("--force-reinstall")
        
        cmd.extend(["-r", str(req_file)])
        
        try:
            # Run pip install
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            if result.returncode == 0:
                results["success"].append(f"Successfully installed dependencies from {req_file.name}")
                logger.info(f"Successfully installed dependencies from {req_file.name}")
            else:
                results["failed"].append(f"Failed to install dependencies: {result.stderr}")
                logger.error(f"Failed to install dependencies: {result.stderr}")
            
        except Exception as e:
            results["failed"].append(f"Exception during installation: {e}")
            logger.error(f"Exception during installation: {e}")
        
        return results
    
    def uninstall_dependencies(
        self,
        package_names: List[str],
        yes: bool = False
    ) -> Dict[str, Any]:
        """
        Uninstall dependencies
        
        Args:
            package_names: List of package names to uninstall
            yes: Whether to skip confirmation
            
        Returns:
            Uninstallation results
        """
        results = {
            "success": [],
            "failed": [],
            "skipped": []
        }
        
        for package in package_names:
            cmd = [sys.executable, "-m", "pip", "uninstall"]
            
            if yes:
                cmd.append("-y")
            
            cmd.append(package)
            
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=self.project_root
                )
                
                if result.returncode == 0:
                    results["success"].append(f"Successfully uninstalled {package}")
                    logger.info(f"Successfully uninstalled {package}")
                else:
                    results["failed"].append(f"Failed to uninstall {package}: {result.stderr}")
                    logger.error(f"Failed to uninstall {package}: {result.stderr}")
                
            except Exception as e:
                results["failed"].append(f"Exception during uninstallation of {package}: {e}")
                logger.error(f"Exception during uninstallation of {package}: {e}")
        
        return results
    
    def check_dependencies(self) -> Dict[str, Any]:
        """
        Check dependency status
        
        Returns:
            Dependency status information
        """
        results = {
            "installed": [],
            "outdated": [],
            "missing": [],
            "conflicts": [],
            "summary": {}
        }
        
        for name, dependency in self.dependencies.items():
            try:
                # Check if package is installed
                cmd = [sys.executable, "-m", "pip", "show", name]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    # Parse installed version
                    for line in result.stdout.split('\n'):
                        if line.startswith('Version:'):
                            installed_version = line.split(':', 1)[1].strip()
                            dependency.installed_version = installed_version
                            dependency.status = DependencyStatus.INSTALLED
                            
                            # Check if outdated
                            if self._is_outdated(installed_version, dependency.version):
                                dependency.status = DependencyStatus.OUTDATED
                                results["outdated"].append(dependency)
                            else:
                                results["installed"].append(dependency)
                            break
                else:
                    dependency.status = DependencyStatus.MISSING
                    results["missing"].append(dependency)
                
            except Exception as e:
                logger.error(f"Error checking dependency {name}: {e}")
                dependency.status = DependencyStatus.MISSING
                results["missing"].append(dependency)
        
        # Generate summary
        results["summary"] = {
            "total": len(self.dependencies),
            "installed": len(results["installed"]),
            "outdated": len(results["outdated"]),
            "missing": len(results["missing"]),
            "conflicts": len(results["conflicts"])
        }
        
        return results
    
    def _is_outdated(self, installed_version: str, required_version: str) -> bool:
        """Check if installed version is outdated"""
        try:
            from packaging import version
            
            installed = version.parse(installed_version)
            
            if required_version.startswith(">="):
                required = version.parse(required_version[2:])
                return installed < required
            elif required_version.startswith("<="):
                required = version.parse(required_version[2:])
                return installed > required
            elif required_version.startswith("~="):
                required = version.parse(required_version[2:])
                return installed < required
            elif required_version == "*":
                return False
            else:
                required = version.parse(required_version)
                return installed < required
                
        except Exception:
            return False
    
    def update_dependencies(
        self,
        dep_type: Optional[DependencyType] = None,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Update dependencies
        
        Args:
            dep_type: Type of dependencies to update
            dry_run: Whether to perform a dry run
            
        Returns:
            Update results
        """
        results = {
            "updated": [],
            "failed": [],
            "skipped": [],
            "dry_run": dry_run
        }
        
        # Check current status
        status = self.check_dependencies()
        
        # Update outdated dependencies
        for dependency in status["outdated"]:
            if dep_type and dependency.type != dep_type:
                results["skipped"].append(dependency.name)
                continue
            
            if dry_run:
                results["updated"].append(f"Would update {dependency.name} from {dependency.installed_version} to latest")
            else:
                try:
                    cmd = [sys.executable, "-m", "pip", "install", "--upgrade", dependency.name]
                    result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
                    
                    if result.returncode == 0:
                        results["updated"].append(f"Updated {dependency.name}")
                        logger.info(f"Updated {dependency.name}")
                    else:
                        results["failed"].append(f"Failed to update {dependency.name}: {result.stderr}")
                        logger.error(f"Failed to update {dependency.name}: {result.stderr}")
                
                except Exception as e:
                    results["failed"].append(f"Exception updating {dependency.name}: {e}")
                    logger.error(f"Exception updating {dependency.name}: {e}")
        
        return results
    
    def generate_requirements_file(
        self,
        dep_type: DependencyType,
        output_file: Optional[str] = None
    ) -> str:
        """
        Generate requirements file for specific dependency type
        
        Args:
            dep_type: Type of dependencies
            output_file: Output file path
            
        Returns:
            Generated requirements content
        """
        content = []
        content.append(f"# {dep_type.value.title()} Requirements for Video-OpusClip")
        content.append("# Generated by DependencyManager")
        content.append("")
        
        for name, dependency in self.dependencies.items():
            if dependency.type == dep_type:
                content.append(f"{name}=={dependency.version}")
        
        content = "\n".join(content)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(content)
        
        return content
    
    def export_dependencies(self, format: str = "json") -> str:
        """
        Export dependencies in specified format
        
        Args:
            format: Export format (json, yaml, toml)
            
        Returns:
            Exported dependencies
        """
        data = {
            "project": "Video-OpusClip",
            "dependencies": {
                name: {
                    "version": dep.version,
                    "type": dep.type.value,
                    "description": dep.description,
                    "status": dep.status.value,
                    "installed_version": dep.installed_version
                }
                for name, dep in self.dependencies.items()
            }
        }
        
        if format.lower() == "json":
            return json.dumps(data, indent=2)
        elif format.lower() == "yaml":
            return yaml.dump(data, default_flow_style=False)
        elif format.lower() == "toml":
            return toml.dumps(data)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def validate_dependencies(self) -> Dict[str, Any]:
        """
        Validate dependencies for conflicts and issues
        
        Returns:
            Validation results
        """
        results = {
            "valid": True,
            "conflicts": [],
            "warnings": [],
            "errors": []
        }
        
        # Check for version conflicts
        for name, dependency in self.dependencies.items():
            # Check if package can be installed
            try:
                cmd = [sys.executable, "-m", "pip", "check"]
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
                
                if result.returncode != 0:
                    results["conflicts"].append(f"Conflict detected for {name}: {result.stderr}")
                    results["valid"] = False
                
            except Exception as e:
                results["errors"].append(f"Error validating {name}: {e}")
                results["valid"] = False
        
        return results
    
    def get_dependency_info(self, package_name: str) -> Optional[Dependency]:
        """Get information about a specific dependency"""
        return self.dependencies.get(package_name)
    
    def add_dependency(
        self,
        name: str,
        version: str,
        dep_type: DependencyType = DependencyType.PRODUCTION,
        description: Optional[str] = None
    ) -> bool:
        """
        Add a new dependency
        
        Args:
            name: Package name
            version: Package version
            dep_type: Dependency type
            description: Package description
            
        Returns:
            Success status
        """
        try:
            dependency = Dependency(
                name=name,
                version=version,
                type=dep_type,
                description=description
            )
            
            self.dependencies[name] = dependency
            
            # Update requirements file
            self._update_requirements_file(dep_type)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to add dependency {name}: {e}")
            return False
    
    def remove_dependency(self, name: str) -> bool:
        """
        Remove a dependency
        
        Args:
            name: Package name
            
        Returns:
            Success status
        """
        try:
            if name in self.dependencies:
                dep_type = self.dependencies[name].type
                del self.dependencies[name]
                
                # Update requirements file
                self._update_requirements_file(dep_type)
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to remove dependency {name}: {e}")
            return False
    
    def _update_requirements_file(self, dep_type: DependencyType):
        """Update requirements file for dependency type"""
        if dep_type == DependencyType.DEVELOPMENT:
            req_file = self.requirements_dir / "requirements-dev.txt"
        elif dep_type == DependencyType.TESTING:
            req_file = self.requirements_dir / "requirements-test.txt"
        elif dep_type == DependencyType.PRODUCTION:
            req_file = self.requirements_dir / "requirements-prod.txt"
        else:
            req_file = self.requirements_dir / "requirements.txt"
        
        # Regenerate requirements file
        content = self.generate_requirements_file(dep_type)
        
        with open(req_file, 'w') as f:
            f.write(content)


# Example usage
if __name__ == "__main__":
    # Example dependency management
    print("📦 Dependency Manager Example")
    
    # Create dependency manager
    manager = DependencyManager()
    
    # Check dependencies
    print("\n" + "="*60)
    print("CHECKING DEPENDENCIES")
    print("="*60)
    
    status = manager.check_dependencies()
    print(f"Total dependencies: {status['summary']['total']}")
    print(f"Installed: {status['summary']['installed']}")
    print(f"Outdated: {status['summary']['outdated']}")
    print(f"Missing: {status['summary']['missing']}")
    
    # Show some dependency details
    print("\n" + "="*60)
    print("DEPENDENCY DETAILS")
    print("="*60)
    
    for name, dependency in list(manager.dependencies.items())[:5]:
        print(f"{name}: {dependency.version} ({dependency.type.value}) - {dependency.status.value}")
    
    # Export dependencies
    print("\n" + "="*60)
    print("EXPORTING DEPENDENCIES")
    print("="*60)
    
    json_export = manager.export_dependencies("json")
    print(f"JSON export length: {len(json_export)} characters")
    
    # Validate dependencies
    print("\n" + "="*60)
    print("VALIDATING DEPENDENCIES")
    print("="*60)
    
    validation = manager.validate_dependencies()
    print(f"Valid: {validation['valid']}")
    print(f"Conflicts: {len(validation['conflicts'])}")
    print(f"Warnings: {len(validation['warnings'])}")
    print(f"Errors: {len(validation['errors'])}")
    
    # Generate requirements file
    print("\n" + "="*60)
    print("GENERATING REQUIREMENTS")
    print("="*60)
    
    prod_requirements = manager.generate_requirements_file(DependencyType.PRODUCTION)
    print(f"Production requirements generated: {len(prod_requirements)} characters")
    
    print("\n✅ Dependency manager example completed!") 