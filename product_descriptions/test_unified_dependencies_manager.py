from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import asyncio
import json
import os
import tempfile
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Optional
import pytest
import yaml
from unified_dependencies_manager import (
        import psutil
        import gc
from typing import Any, List, Dict, Optional
import logging
"""
Comprehensive Tests for Unified Dependencies Management System

This test suite covers:
- Dependency information and group management
- Environment validation and analysis
- Requirements file generation
- Installation and uninstallation
- Security vulnerability scanning
- Performance impact analysis
- Edge cases and error handling
- Integration testing
"""



    UnifiedDependenciesManager, DependencyInfo, DependencyGroup,
    DependencyCategory, DependencyPriority, PlatformType,
    create_requirements_file, install_group_dependencies, validate_system_dependencies
)


class TestDependencyInfo:
    """Test DependencyInfo class."""
    
    def test_initialization(self) -> Any:
        """Test dependency info initialization."""
        dep = DependencyInfo(
            name="test-package",
            version=">=1.0.0",
            category=DependencyCategory.CORE,
            priority=DependencyPriority.CRITICAL
        )
        
        assert dep.name == "test-package"
        assert dep.version == ">=1.0.0"
        assert dep.category == DependencyCategory.CORE
        assert dep.priority == DependencyPriority.CRITICAL
        assert dep.platforms == [PlatformType.ALL]
        assert dep.description == ""
        assert dep.url == ""
        assert dep.license == ""
        assert len(dep.security_issues) == 0
        assert len(dep.conflicts) == 0
        assert len(dep.alternatives) == 0
        assert dep.performance_impact == "low"
        assert dep.memory_usage == "low"
        assert dep.cpu_usage == "low"
    
    def test_custom_initialization(self) -> Any:
        """Test dependency info with custom values."""
        dep = DependencyInfo(
            name="custom-package",
            version="==2.1.0",
            category=DependencyCategory.PROFILING,
            priority=DependencyPriority.HIGH,
            platforms=[PlatformType.LINUX, PlatformType.MACOS],
            description="Custom package description",
            url="https://example.com",
            license="MIT",
            security_issues=["CVE-2023-1234"],
            conflicts=["conflicting-package"],
            alternatives=["alternative-package"],
            performance_impact="high",
            memory_usage="medium",
            cpu_usage="high"
        )
        
        assert dep.name == "custom-package"
        assert dep.version == "==2.1.0"
        assert dep.category == DependencyCategory.PROFILING
        assert dep.priority == DependencyPriority.HIGH
        assert dep.platforms == [PlatformType.LINUX, PlatformType.MACOS]
        assert dep.description == "Custom package description"
        assert dep.url == "https://example.com"
        assert dep.license == "MIT"
        assert dep.security_issues == ["CVE-2023-1234"]
        assert dep.conflicts == ["conflicting-package"]
        assert dep.alternatives == ["alternative-package"]
        assert dep.performance_impact == "high"
        assert dep.memory_usage == "medium"
        assert dep.cpu_usage == "high"
    
    def test_invalid_initialization(self) -> Any:
        """Test dependency info with invalid values."""
        with pytest.raises(ValueError):
            DependencyInfo(
                name="",  # Empty name
                version=">=1.0.0",
                category=DependencyCategory.CORE,
                priority=DependencyPriority.CRITICAL
            )
        
        with pytest.raises(ValueError):
            DependencyInfo(
                name="test-package",
                version="",  # Empty version
                category=DependencyCategory.CORE,
                priority=DependencyPriority.CRITICAL
            )


class TestDependencyGroup:
    """Test DependencyGroup class."""
    
    @pytest.fixture
    def sample_dependencies(self) -> Any:
        """Create sample dependencies for testing."""
        return [
            DependencyInfo(
                name="dep1",
                version=">=1.0.0",
                category=DependencyCategory.CORE,
                priority=DependencyPriority.CRITICAL
            ),
            DependencyInfo(
                name="dep2",
                version=">=2.0.0",
                category=DependencyCategory.PROFILING,
                priority=DependencyPriority.HIGH
            )
        ]
    
    def test_initialization(self, sample_dependencies) -> Any:
        """Test dependency group initialization."""
        group = DependencyGroup(
            name="test-group",
            description="Test group description",
            dependencies=sample_dependencies,
            category=DependencyCategory.CORE,
            priority=DependencyPriority.CRITICAL
        )
        
        assert group.name == "test-group"
        assert group.description == "Test group description"
        assert group.dependencies == sample_dependencies
        assert group.category == DependencyCategory.CORE
        assert group.priority == DependencyPriority.CRITICAL
        assert group.required is True
        assert group.auto_install is True
    
    def test_custom_initialization(self, sample_dependencies) -> Any:
        """Test dependency group with custom values."""
        group = DependencyGroup(
            name="custom-group",
            description="Custom group description",
            dependencies=sample_dependencies,
            category=DependencyCategory.PROFILING,
            priority=DependencyPriority.MEDIUM,
            required=False,
            auto_install=False
        )
        
        assert group.name == "custom-group"
        assert group.description == "Custom group description"
        assert group.dependencies == sample_dependencies
        assert group.category == DependencyCategory.PROFILING
        assert group.priority == DependencyPriority.MEDIUM
        assert group.required is False
        assert group.auto_install is False
    
    def test_invalid_initialization(self) -> Any:
        """Test dependency group with invalid values."""
        with pytest.raises(ValueError):
            DependencyGroup(
                name="",  # Empty name
                description="Test description",
                dependencies=[],
                category=DependencyCategory.CORE,
                priority=DependencyPriority.CRITICAL
            )
        
        with pytest.raises(ValueError):
            DependencyGroup(
                name="test-group",
                description="Test description",
                dependencies=[],  # Empty dependencies
                category=DependencyCategory.CORE,
                priority=DependencyPriority.CRITICAL
            )


class TestUnifiedDependenciesManager:
    """Test UnifiedDependenciesManager class."""
    
    @pytest.fixture
    def temp_config_file(self) -> Any:
        """Create temporary config file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config = {
                'dependencies': [
                    {
                        'name': 'test-package',
                        'version': '>=1.0.0',
                        'category': 'core',
                        'priority': 'critical',
                        'platforms': ['all'],
                        'description': 'Test package',
                        'url': 'https://example.com',
                        'license': 'MIT',
                        'security_issues': [],
                        'conflicts': [],
                        'alternatives': [],
                        'performance_impact': 'low',
                        'memory_usage': 'low',
                        'cpu_usage': 'low'
                    }
                ],
                'groups': [
                    {
                        'name': 'test-group',
                        'description': 'Test group',
                        'dependencies': ['test-package'],
                        'category': 'core',
                        'priority': 'critical',
                        'required': True,
                        'auto_install': True
                    }
                ]
            }
            yaml.dump(config, f)
            return f.name
    
    @pytest.fixture
    def manager(self, temp_config_file) -> Any:
        """Create manager instance with temp config."""
        return UnifiedDependenciesManager(temp_config_file)
    
    def test_initialization_with_config(self, temp_config_file) -> Any:
        """Test manager initialization with config file."""
        manager = UnifiedDependenciesManager(temp_config_file)
        
        assert len(manager.dependencies) == 1
        assert 'test-package' in manager.dependencies
        assert len(manager.groups) == 1
        assert 'test-group' in manager.groups
        
        # Clean up
        os.unlink(temp_config_file)
    
    def test_initialization_without_config(self) -> Any:
        """Test manager initialization without config file."""
        manager = UnifiedDependenciesManager("nonexistent.yaml")
        
        # Should create default configuration
        assert len(manager.dependencies) > 0
        assert len(manager.groups) > 0
        
        # Check for core dependencies
        assert 'torch' in manager.dependencies
        assert 'numpy' in manager.dependencies
        assert 'pandas' in manager.dependencies
    
    def test_get_dependency_info(self, manager) -> Optional[Dict[str, Any]]:
        """Test getting dependency information."""
        dep_info = manager.get_dependency_info('test-package')
        
        assert dep_info is not None
        assert dep_info.name == 'test-package'
        assert dep_info.version == '>=1.0.0'
        assert dep_info.category == DependencyCategory.CORE
        assert dep_info.priority == DependencyPriority.CRITICAL
    
    def test_get_dependency_info_nonexistent(self, manager) -> Optional[Dict[str, Any]]:
        """Test getting nonexistent dependency information."""
        dep_info = manager.get_dependency_info('nonexistent-package')
        assert dep_info is None
    
    def test_get_group_dependencies(self, manager) -> Optional[Dict[str, Any]]:
        """Test getting group dependencies."""
        deps = manager.get_group_dependencies('test-group')
        
        assert len(deps) == 1
        assert deps[0].name == 'test-package'
    
    def test_get_group_dependencies_nonexistent(self, manager) -> Optional[Dict[str, Any]]:
        """Test getting dependencies for nonexistent group."""
        deps = manager.get_group_dependencies('nonexistent-group')
        assert len(deps) == 0
    
    @patch('pkg_resources.working_set')
    def test_check_dependency_installed(self, mock_working_set, manager) -> Any:
        """Test checking if dependency is installed."""
        # Mock installed packages
        mock_package = Mock()
        mock_package.project_name = 'test-package'
        mock_working_set.__iter__.return_value = [mock_package]
        
        # Update installed dependencies
        manager._scan_installed_dependencies()
        
        assert manager.check_dependency_installed('test-package') is True
        assert manager.check_dependency_installed('nonexistent-package') is False
    
    @patch('pkg_resources.require')
    @patch('pkg_resources.get_distribution')
    def test_check_dependency_version(self, mock_get_dist, mock_require, manager) -> Any:
        """Test checking dependency version."""
        # Mock successful version check
        mock_require.return_value = True
        
        assert manager.check_dependency_version('test-package', '>=1.0.0') is True
        
        # Mock version conflict
        mock_require.side_effect = pkg_resources.VersionConflict("test", "1.0.0")
        
        assert manager.check_dependency_version('test-package', '>=2.0.0') is False
    
    def test_get_missing_dependencies(self, manager) -> Optional[Dict[str, Any]]:
        """Test getting missing dependencies."""
        missing = manager.get_missing_dependencies()
        
        # Should include test-package since it's not actually installed
        assert len(missing) > 0
        assert any(dep.name == 'test-package' for dep in missing)
    
    def test_get_missing_dependencies_for_group(self, manager) -> Optional[Dict[str, Any]]:
        """Test getting missing dependencies for specific group."""
        missing = manager.get_missing_dependencies('test-group')
        
        assert len(missing) > 0
        assert any(dep.name == 'test-package' for dep in missing)
    
    @patch('requests.get')
    def test_get_latest_version(self, mock_get, manager) -> Optional[Dict[str, Any]]:
        """Test getting latest version from PyPI."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'info': {'version': '2.0.0'}}
        mock_get.return_value = mock_response
        
        latest_version = manager._get_latest_version('test-package')
        assert latest_version == '2.0.0'
        
        # Mock failed response
        mock_response.status_code = 404
        latest_version = manager._get_latest_version('nonexistent-package')
        assert latest_version is None
    
    def test_check_dependency_conflicts(self, manager) -> Any:
        """Test checking dependency conflicts."""
        # Add a dependency with conflicts
        conflicting_dep = DependencyInfo(
            name='conflicting-package',
            version='>=1.0.0',
            category=DependencyCategory.CORE,
            priority=DependencyPriority.CRITICAL,
            conflicts=['test-package']
        )
        manager.dependencies['conflicting-package'] = conflicting_dep
        
        # Mock that conflicting package is installed
        manager.installed_dependencies.add('conflicting-package')
        
        conflicts = manager.check_dependency_conflicts()
        assert len(conflicts) == 1
        assert ('conflicting-package', 'test-package') in conflicts
    
    def test_check_security_vulnerabilities(self, manager) -> Any:
        """Test checking security vulnerabilities."""
        # Add a dependency with security issues
        vulnerable_dep = DependencyInfo(
            name='vulnerable-package',
            version='>=1.0.0',
            category=DependencyCategory.CORE,
            priority=DependencyPriority.CRITICAL,
            security_issues=['CVE-2023-1234', 'CVE-2023-5678']
        )
        manager.dependencies['vulnerable-package'] = vulnerable_dep
        
        vulnerabilities = manager.check_security_vulnerabilities()
        assert len(vulnerabilities) == 2
        assert all(vuln['package'] == 'vulnerable-package' for vuln in vulnerabilities)
    
    def test_generate_requirements_file(self, manager) -> Any:
        """Test generating requirements file."""
        requirements = manager.generate_requirements_file()
        
        assert isinstance(requirements, str)
        assert 'test-package>=1.0.0' in requirements
    
    def test_generate_requirements_file_for_group(self, manager) -> Any:
        """Test generating requirements file for specific group."""
        requirements = manager.generate_requirements_file('test-group')
        
        assert isinstance(requirements, str)
        assert 'test-package>=1.0.0' in requirements
    
    def test_generate_requirements_file_with_optional(self, manager) -> Any:
        """Test generating requirements file with optional dependencies."""
        # Add optional dependency
        optional_dep = DependencyInfo(
            name='optional-package',
            version='>=1.0.0',
            category=DependencyCategory.OPTIONAL,
            priority=DependencyPriority.OPTIONAL
        )
        manager.dependencies['optional-package'] = optional_dep
        
        # Without optional
        requirements = manager.generate_requirements_file(include_optional=False)
        assert 'optional-package' not in requirements
        
        # With optional
        requirements = manager.generate_requirements_file(include_optional=True)
        assert 'optional-package>=1.0.0' in requirements
    
    @patch('subprocess.run')
    def test_install_dependencies(self, mock_run, manager) -> Any:
        """Test installing dependencies."""
        # Mock successful installation
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Successfully installed"
        mock_run.return_value = mock_result
        
        results = manager.install_dependencies()
        
        assert isinstance(results, dict)
        assert len(results) > 0
    
    @patch('subprocess.run')
    def test_install_dependencies_with_upgrade(self, mock_run, manager) -> Any:
        """Test installing dependencies with upgrade."""
        # Mock successful installation
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Successfully upgraded"
        mock_run.return_value = mock_result
        
        results = manager.install_dependencies(upgrade=True)
        
        assert isinstance(results, dict)
        assert len(results) > 0
        
        # Check that upgrade flag was used
        mock_run.assert_called()
        call_args = mock_run.call_args[0][0]
        assert '--upgrade' in call_args
    
    @patch('subprocess.run')
    def test_uninstall_dependencies(self, mock_run, manager) -> Any:
        """Test uninstalling dependencies."""
        # Mock successful uninstallation
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Successfully uninstalled"
        mock_run.return_value = mock_result
        
        # Mock that package is installed
        manager.installed_dependencies.add('test-package')
        
        results = manager.uninstall_dependencies()
        
        assert isinstance(results, dict)
        assert len(results) > 0
    
    def test_get_dependency_report(self, manager) -> Optional[Dict[str, Any]]:
        """Test generating dependency report."""
        report = manager.get_dependency_report()
        
        assert isinstance(report, dict)
        assert 'summary' in report
        assert 'missing_dependencies' in report
        assert 'outdated_dependencies' in report
        assert 'conflicts' in report
        assert 'vulnerabilities' in report
        assert 'by_category' in report
        assert 'groups' in report
        
        # Check summary structure
        summary = report['summary']
        assert 'total_dependencies' in summary
        assert 'installed_dependencies' in summary
        assert 'missing_dependencies' in summary
        assert 'outdated_dependencies' in summary
        assert 'conflicts' in summary
        assert 'vulnerabilities' in summary
        assert 'installation_rate' in summary
    
    def test_validate_environment(self, manager) -> bool:
        """Test environment validation."""
        validation = manager.validate_environment()
        
        assert isinstance(validation, dict)
        assert 'is_valid' in validation
        assert 'critical_missing' in validation
        assert 'has_conflicts' in validation
        assert 'has_vulnerabilities' in validation
        assert 'warnings' in validation
        assert 'recommendations' in validation
    
    def test_calculate_security_score(self, manager) -> Any:
        """Test security score calculation."""
        # Test with no issues
        vulnerabilities = []
        outdated = []
        conflicts = []
        
        score = manager._calculate_security_score(vulnerabilities, outdated, conflicts)
        assert score == 100.0
        
        # Test with issues
        vulnerabilities = [{'package': 'test', 'issue': 'CVE-2023-1234'}]
        outdated = [('dep1', '1.0.0', '2.0.0')]
        conflicts = [('dep1', 'dep2')]
        
        score = manager._calculate_security_score(vulnerabilities, outdated, conflicts)
        assert score < 100.0
        assert score >= 0.0
    
    def test_generate_recommendations(self, manager) -> Any:
        """Test generating recommendations."""
        report = {
            'missing_dependencies': [{'name': 'dep1'}, {'name': 'dep2'}],
            'outdated_dependencies': [{'name': 'dep3'}],
            'conflicts': [{'package1': 'dep1', 'package2': 'dep2'}],
            'vulnerabilities': [{'package': 'dep4'}]
        }
        
        recommendations = manager._generate_recommendations(report)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) == 4
        assert any('Install 2 missing dependencies' in rec for rec in recommendations)
        assert any('Update 1 outdated dependencies' in rec for rec in recommendations)
        assert any('Resolve 1 dependency conflicts' in rec for rec in recommendations)
        assert any('Address 1 security vulnerabilities' in rec for rec in recommendations)


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_create_requirements_file(self) -> Any:
        """Test creating requirements file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "requirements.txt")
            
            result = create_requirements_file('core', output_path)
            
            assert result == output_path
            assert os.path.exists(output_path)
            
            # Check file content
            with open(output_path, 'r') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                content = f.read()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                assert len(content) > 0
    
    @patch('subprocess.run')
    def test_install_group_dependencies(self, mock_run) -> Any:
        """Test installing group dependencies."""
        # Mock successful installation
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Successfully installed"
        mock_run.return_value = mock_result
        
        results = install_group_dependencies('core')
        
        assert isinstance(results, dict)
        assert len(results) > 0
    
    def test_validate_system_dependencies(self) -> bool:
        """Test validating system dependencies."""
        validation = validate_system_dependencies()
        
        assert isinstance(validation, dict)
        assert 'is_valid' in validation
        assert 'critical_missing' in validation
        assert 'has_conflicts' in validation
        assert 'has_vulnerabilities' in validation
        assert 'warnings' in validation
        assert 'recommendations' in validation


class TestIntegration:
    """Integration tests for the unified dependencies management system."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self) -> Any:
        """Test end-to-end dependency management workflow."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config = {
                'dependencies': [
                    {
                        'name': 'test-package-1',
                        'version': '>=1.0.0',
                        'category': 'core',
                        'priority': 'critical',
                        'platforms': ['all'],
                        'description': 'Test package 1',
                        'url': 'https://example.com',
                        'license': 'MIT',
                        'security_issues': [],
                        'conflicts': [],
                        'alternatives': [],
                        'performance_impact': 'low',
                        'memory_usage': 'low',
                        'cpu_usage': 'low'
                    },
                    {
                        'name': 'test-package-2',
                        'version': '>=2.0.0',
                        'category': 'profiling',
                        'priority': 'high',
                        'platforms': ['all'],
                        'description': 'Test package 2',
                        'url': 'https://example.com',
                        'license': 'MIT',
                        'security_issues': [],
                        'conflicts': [],
                        'alternatives': [],
                        'performance_impact': 'medium',
                        'memory_usage': 'medium',
                        'cpu_usage': 'low'
                    }
                ],
                'groups': [
                    {
                        'name': 'test-group-1',
                        'description': 'Test group 1',
                        'dependencies': ['test-package-1'],
                        'category': 'core',
                        'priority': 'critical',
                        'required': True,
                        'auto_install': True
                    },
                    {
                        'name': 'test-group-2',
                        'description': 'Test group 2',
                        'dependencies': ['test-package-2'],
                        'category': 'profiling',
                        'priority': 'high',
                        'required': False,
                        'auto_install': True
                    }
                ]
            }
            yaml.dump(config, f)
            config_path = f.name
        
        try:
            # Create manager
            manager = UnifiedDependenciesManager(config_path)
            
            # Test dependency retrieval
            dep1 = manager.get_dependency_info('test-package-1')
            dep2 = manager.get_dependency_info('test-package-2')
            
            assert dep1 is not None
            assert dep2 is not None
            assert dep1.category == DependencyCategory.CORE
            assert dep2.category == DependencyCategory.PROFILING
            
            # Test group dependencies
            group1_deps = manager.get_group_dependencies('test-group-1')
            group2_deps = manager.get_group_dependencies('test-group-2')
            
            assert len(group1_deps) == 1
            assert len(group2_deps) == 1
            assert group1_deps[0].name == 'test-package-1'
            assert group2_deps[0].name == 'test-package-2'
            
            # Test missing dependencies
            missing = manager.get_missing_dependencies()
            assert len(missing) >= 2  # Both packages should be missing
            
            # Test requirements generation
            requirements = manager.generate_requirements_file()
            assert 'test-package-1>=1.0.0' in requirements
            assert 'test-package-2>=2.0.0' in requirements
            
            # Test dependency report
            report = manager.get_dependency_report()
            assert report['summary']['total_dependencies'] == 2
            assert report['summary']['missing_dependencies'] >= 2
            
            # Test environment validation
            validation = manager.validate_environment()
            assert not validation['is_valid']  # Should be invalid due to missing dependencies
            assert len(validation['critical_missing']) > 0
            
        finally:
            # Clean up
            os.unlink(config_path)
    
    def test_dependency_conflict_resolution(self) -> Any:
        """Test dependency conflict resolution."""
        manager = UnifiedDependenciesManager()
        
        # Add conflicting dependencies
        dep1 = DependencyInfo(
            name='package-a',
            version='>=1.0.0',
            category=DependencyCategory.CORE,
            priority=DependencyPriority.CRITICAL,
            conflicts=['package-b']
        )
        dep2 = DependencyInfo(
            name='package-b',
            version='>=2.0.0',
            category=DependencyCategory.CORE,
            priority=DependencyPriority.CRITICAL,
            conflicts=['package-a']
        )
        
        manager.dependencies['package-a'] = dep1
        manager.dependencies['package-b'] = dep2
        
        # Mock that both packages are installed
        manager.installed_dependencies.add('package-a')
        manager.installed_dependencies.add('package-b')
        
        # Check for conflicts
        conflicts = manager.check_dependency_conflicts()
        assert len(conflicts) == 2  # Both directions should be detected
        
        # Validate environment
        validation = manager.validate_environment()
        assert validation['has_conflicts'] is True
        assert not validation['is_valid']
    
    def test_security_vulnerability_scanning(self) -> Any:
        """Test security vulnerability scanning."""
        manager = UnifiedDependenciesManager()
        
        # Add vulnerable dependency
        vulnerable_dep = DependencyInfo(
            name='vulnerable-package',
            version='>=1.0.0',
            category=DependencyCategory.CORE,
            priority=DependencyPriority.CRITICAL,
            security_issues=['CVE-2023-1234', 'CVE-2023-5678']
        )
        manager.dependencies['vulnerable-package'] = vulnerable_dep
        
        # Check vulnerabilities
        vulnerabilities = manager.check_security_vulnerabilities()
        assert len(vulnerabilities) == 2
        
        # Validate environment
        validation = manager.validate_environment()
        assert validation['has_vulnerabilities'] is True
        assert not validation['is_valid']
    
    def test_performance_impact_analysis(self) -> Any:
        """Test performance impact analysis."""
        manager = UnifiedDependenciesManager()
        
        # Add dependencies with different performance impacts
        low_impact = DependencyInfo(
            name='low-impact-package',
            version='>=1.0.0',
            category=DependencyCategory.CORE,
            priority=DependencyPriority.CRITICAL,
            performance_impact='low',
            memory_usage='low',
            cpu_usage='low'
        )
        
        high_impact = DependencyInfo(
            name='high-impact-package',
            version='>=1.0.0',
            category=DependencyCategory.PROFILING,
            priority=DependencyPriority.HIGH,
            performance_impact='high',
            memory_usage='high',
            cpu_usage='high'
        )
        
        manager.dependencies['low-impact-package'] = low_impact
        manager.dependencies['high-impact-package'] = high_impact
        
        # Mock that both are installed
        manager.installed_dependencies.add('low-impact-package')
        manager.installed_dependencies.add('high-impact-package')
        
        # Get dependency report
        report = manager.get_dependency_report()
        
        # Check that both dependencies are included
        assert report['summary']['total_dependencies'] >= 2
        assert report['summary']['installed_dependencies'] >= 2


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_config_file(self) -> Any:
        """Test handling empty config file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({}, f)
            config_path = f.name
        
        try:
            manager = UnifiedDependenciesManager(config_path)
            
            # Should create default configuration
            assert len(manager.dependencies) > 0
            assert len(manager.groups) > 0
            
        finally:
            os.unlink(config_path)
    
    def test_invalid_config_file(self) -> Any:
        """Test handling invalid config file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            config_path = f.name
        
        try:
            manager = UnifiedDependenciesManager(config_path)
            
            # Should create default configuration
            assert len(manager.dependencies) > 0
            assert len(manager.groups) > 0
            
        finally:
            os.unlink(config_path)
    
    def test_nonexistent_config_file(self) -> Any:
        """Test handling nonexistent config file."""
        manager = UnifiedDependenciesManager("nonexistent.yaml")
        
        # Should create default configuration
        assert len(manager.dependencies) > 0
        assert len(manager.groups) > 0
    
    def test_dependency_with_no_platforms(self) -> Any:
        """Test dependency with no platforms specified."""
        dep = DependencyInfo(
            name='test-package',
            version='>=1.0.0',
            category=DependencyCategory.CORE,
            priority=DependencyPriority.CRITICAL,
            platforms=[]  # No platforms
        )
        
        # Should default to ALL platforms
        assert dep.platforms == [PlatformType.ALL]
    
    def test_group_with_no_dependencies(self) -> Any:
        """Test group with no dependencies."""
        with pytest.raises(ValueError):
            DependencyGroup(
                name='empty-group',
                description='Empty group',
                dependencies=[],  # No dependencies
                category=DependencyCategory.CORE,
                priority=DependencyPriority.CRITICAL
            )
    
    def test_duplicate_dependencies(self) -> Any:
        """Test handling duplicate dependencies."""
        manager = UnifiedDependenciesManager()
        
        # Add same dependency twice
        dep1 = DependencyInfo(
            name='duplicate-package',
            version='>=1.0.0',
            category=DependencyCategory.CORE,
            priority=DependencyPriority.CRITICAL
        )
        dep2 = DependencyInfo(
            name='duplicate-package',
            version='>=2.0.0',
            category=DependencyCategory.PROFILING,
            priority=DependencyPriority.HIGH
        )
        
        manager.dependencies['duplicate-package'] = dep1
        manager.dependencies['duplicate-package'] = dep2  # Overwrites first
        
        # Should only have one dependency with second version
        assert len(manager.dependencies) >= 1
        assert manager.dependencies['duplicate-package'].version == '>=2.0.0'


class TestPerformance:
    """Performance tests."""
    
    def test_large_dependency_set(self) -> Any:
        """Test performance with large number of dependencies."""
        manager = UnifiedDependenciesManager()
        
        # Add many dependencies
        for i in range(100):
            dep = DependencyInfo(
                name=f'package-{i}',
                version=f'>={i}.0.0',
                category=DependencyCategory.CORE,
                priority=DependencyPriority.MEDIUM
            )
            manager.dependencies[f'package-{i}'] = dep
        
        # Test performance of operations
        start_time = time.time()
        
        # Get missing dependencies
        missing = manager.get_missing_dependencies()
        
        # Get dependency report
        report = manager.get_dependency_report()
        
        # Validate environment
        validation = manager.validate_environment()
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should complete within reasonable time
        assert execution_time < 5.0  # Less than 5 seconds
        
        # Check results
        assert len(missing) == 100  # All should be missing
        assert report['summary']['total_dependencies'] == 100
        assert not validation['is_valid']  # Should be invalid due to missing dependencies
    
    def test_memory_usage(self) -> Any:
        """Test memory usage with large dependency set."""
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Create manager with many dependencies
        manager = UnifiedDependenciesManager()
        
        for i in range(1000):
            dep = DependencyInfo(
                name=f'package-{i}',
                version=f'>={i}.0.0',
                category=DependencyCategory.CORE,
                priority=DependencyPriority.MEDIUM
            )
            manager.dependencies[f'package-{i}'] = dep
        
        # Force garbage collection
        gc.collect()
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (< 100MB)
        assert memory_increase < 100 * 1024 * 1024


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 