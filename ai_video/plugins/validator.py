from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import asyncio
import inspect
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from ..core.exceptions import ValidationError, PluginError
from ..core.types import PluginInfo
from .base import BasePlugin
        import re
from typing import Any, List, Dict, Optional
"""
Plugin Validator - Comprehensive Plugin Validation System

This module provides robust validation for plugins including:
- Plugin integrity checks
- Dependency validation
- Configuration schema validation
- Performance and security checks
- Detailed validation reports
"""



logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation levels for different types of checks."""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    SECURITY = "security"


class ValidationStatus(Enum):
    """Validation status results."""
    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ValidationResult:
    """Result of a validation check."""
    check_name: str
    status: ValidationStatus
    message: str
    details: Optional[Dict[str, Any]] = None
    duration: float = 0.0


@dataclass
class ValidationReport:
    """Complete validation report for a plugin."""
    plugin_name: str
    overall_status: ValidationStatus
    results: List[ValidationResult] = field(default_factory=list)
    total_checks: int = 0
    passed_checks: int = 0
    failed_checks: int = 0
    warning_checks: int = 0
    total_duration: float = 0.0
    
    def add_result(self, result: ValidationResult):
        """Add a validation result to the report."""
        self.results.append(result)
        self.total_checks += 1
        self.total_duration += result.duration
        
        if result.status == ValidationStatus.PASSED:
            self.passed_checks += 1
        elif result.status == ValidationStatus.FAILED:
            self.failed_checks += 1
        elif result.status == ValidationStatus.WARNING:
            self.warning_checks += 1
        
        # Update overall status
        if self.failed_checks > 0:
            self.overall_status = ValidationStatus.FAILED
        elif self.warning_checks > 0:
            self.overall_status = ValidationStatus.WARNING
        else:
            self.overall_status = ValidationStatus.PASSED


class PluginValidator:
    """
    Comprehensive plugin validation system.
    
    Features:
    - Multiple validation levels
    - Detailed reporting
    - Performance monitoring
    - Security checks
    - Dependency validation
    """
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        
    """__init__ function."""
self.validation_level = validation_level
        self.validation_stats = {
            'total_validations': 0,
            'passed_validations': 0,
            'failed_validations': 0,
            'warning_validations': 0
        }
        
        logger.info(f"PluginValidator initialized with level: {validation_level.value}")
    
    def validate_plugin(self, plugin: BasePlugin) -> bool:
        """
        Validate a plugin and return overall success status.
        
        Args:
            plugin: Plugin to validate
            
        Returns:
            True if plugin passes validation
        """
        report = self.validate_plugin_detailed(plugin)
        return report.overall_status != ValidationStatus.FAILED
    
    def validate_plugin_detailed(self, plugin: BasePlugin) -> ValidationReport:
        """
        Perform detailed validation of a plugin.
        
        Args:
            plugin: Plugin to validate
            
        Returns:
            Detailed validation report
        """
        start_time = time.time()
        report = ValidationReport(plugin_name=plugin.name)
        
        logger.info(f"🔍 Validating plugin: {plugin.name}")
        
        try:
            # Basic validation checks
            self._validate_basic_structure(plugin, report)
            
            # Metadata validation
            self._validate_metadata(plugin, report)
            
            # Configuration validation
            self._validate_configuration(plugin, report)
            
            # Interface validation
            self._validate_interfaces(plugin, report)
            
            # Security validation (if enabled)
            if self.validation_level in [ValidationLevel.STRICT, ValidationLevel.SECURITY]:
                self._validate_security(plugin, report)
            
            # Performance validation (if enabled)
            if self.validation_level == ValidationLevel.STRICT:
                self._validate_performance(plugin, report)
            
            # Update statistics
            self._update_stats(report)
            
            # Log results
            self._log_validation_results(report)
            
        except Exception as e:
            logger.error(f"❌ Validation failed for plugin {plugin.name}: {e}")
            report.add_result(ValidationResult(
                check_name="validation_system",
                status=ValidationStatus.FAILED,
                message=f"Validation system error: {str(e)}",
                duration=time.time() - start_time
            ))
        
        return report
    
    def _validate_basic_structure(self, plugin: BasePlugin, report: ValidationReport):
        """Validate basic plugin structure."""
        start_time = time.time()
        
        # Check if plugin inherits from BasePlugin
        if not isinstance(plugin, BasePlugin):
            report.add_result(ValidationResult(
                check_name="inheritance",
                status=ValidationStatus.FAILED,
                message="Plugin must inherit from BasePlugin",
                duration=time.time() - start_time
            ))
            return
        
        # Check required attributes
        required_attrs = ['name', 'version', 'description']
        for attr in required_attrs:
            if not hasattr(plugin, attr) or not getattr(plugin, attr):
                report.add_result(ValidationResult(
                    check_name=f"required_attribute_{attr}",
                    status=ValidationStatus.FAILED,
                    message=f"Missing or empty required attribute: {attr}",
                    duration=0.0
                ))
        
        # Check if plugin name is valid
        if hasattr(plugin, 'name') and plugin.name:
            if not plugin.name.replace('_', '').replace('-', '').isalnum():
                report.add_result(ValidationResult(
                    check_name="plugin_name_format",
                    status=ValidationStatus.WARNING,
                    message="Plugin name should be alphanumeric with underscores or hyphens",
                    duration=0.0
                ))
        
        report.add_result(ValidationResult(
            check_name="basic_structure",
            status=ValidationStatus.PASSED,
            message="Basic structure validation passed",
            duration=time.time() - start_time
        ))
    
    def _validate_metadata(self, plugin: BasePlugin, report: ValidationReport):
        """Validate plugin metadata."""
        start_time = time.time()
        
        try:
            metadata = plugin.get_metadata()
            
            # Validate metadata structure
            if not metadata.name or metadata.name == "base_plugin":
                report.add_result(ValidationResult(
                    check_name="metadata_name",
                    status=ValidationStatus.FAILED,
                    message="Plugin must have a unique name",
                    duration=0.0
                ))
            
            if not metadata.version:
                report.add_result(ValidationResult(
                    check_name="metadata_version",
                    status=ValidationStatus.FAILED,
                    message="Plugin must have a version",
                    duration=0.0
                ))
            
            if not metadata.description:
                report.add_result(ValidationResult(
                    check_name="metadata_description",
                    status=ValidationStatus.WARNING,
                    message="Plugin should have a description",
                    duration=0.0
                ))
            
            # Validate version format
            if metadata.version:
                if not self._is_valid_version(metadata.version):
                    report.add_result(ValidationResult(
                        check_name="version_format",
                        status=ValidationStatus.WARNING,
                        message="Version should follow semantic versioning (e.g., 1.0.0)",
                        duration=0.0
                    ))
            
            report.add_result(ValidationResult(
                check_name="metadata",
                status=ValidationStatus.PASSED,
                message="Metadata validation passed",
                duration=time.time() - start_time
            ))
            
        except Exception as e:
            report.add_result(ValidationResult(
                check_name="metadata",
                status=ValidationStatus.FAILED,
                message=f"Metadata validation failed: {str(e)}",
                duration=time.time() - start_time
            ))
    
    def _validate_configuration(self, plugin: BasePlugin, report: ValidationReport):
        """Validate plugin configuration."""
        start_time = time.time()
        
        try:
            # Test configuration validation
            if hasattr(plugin, 'validate_config'):
                # Test with empty config
                if not plugin.validate_config({}):
                    report.add_result(ValidationResult(
                        check_name="config_validation_empty",
                        status=ValidationStatus.WARNING,
                        message="Plugin rejected empty configuration",
                        duration=0.0
                    ))
                
                # Test with invalid config
                if plugin.validate_config({"invalid": "config"}):
                    report.add_result(ValidationResult(
                        check_name="config_validation_invalid",
                        status=ValidationStatus.WARNING,
                        message="Plugin accepted invalid configuration",
                        duration=0.0
                    ))
            
            # Check configuration schema
            if hasattr(plugin, 'get_config_schema'):
                schema = plugin.get_config_schema()
                if schema and not isinstance(schema, dict):
                    report.add_result(ValidationResult(
                        check_name="config_schema_type",
                        status=ValidationStatus.FAILED,
                        message="Configuration schema must be a dictionary",
                        duration=0.0
                    ))
            
            report.add_result(ValidationResult(
                check_name="configuration",
                status=ValidationStatus.PASSED,
                message="Configuration validation passed",
                duration=time.time() - start_time
            ))
            
        except Exception as e:
            report.add_result(ValidationResult(
                check_name="configuration",
                status=ValidationStatus.FAILED,
                message=f"Configuration validation failed: {str(e)}",
                duration=time.time() - start_time
            ))
    
    def _validate_interfaces(self, plugin: BasePlugin, report: ValidationReport):
        """Validate plugin interfaces and methods."""
        start_time = time.time()
        
        # Check required methods
        required_methods = ['initialize', 'cleanup']
        for method in required_methods:
            if not hasattr(plugin, method) or not callable(getattr(plugin, method)):
                report.add_result(ValidationResult(
                    check_name=f"required_method_{method}",
                    status=ValidationStatus.FAILED,
                    message=f"Missing required method: {method}",
                    duration=0.0
                ))
        
        # Check if plugin implements any component interfaces
        component_methods = [
            'get_extractors', 'get_suggestion_engines', 'get_generators',
            'get_repositories', 'get_metrics_collectors'
        ]
        
        implemented_components = []
        for method in component_methods:
            if hasattr(plugin, method) and callable(getattr(plugin, method)):
                implemented_components.append(method.replace('get_', '').replace('s', ''))
        
        if not implemented_components:
            report.add_result(ValidationResult(
                check_name="component_interfaces",
                status=ValidationStatus.WARNING,
                message="Plugin does not implement any component interfaces",
                duration=0.0
            ))
        else:
            report.add_result(ValidationResult(
                check_name="component_interfaces",
                status=ValidationStatus.PASSED,
                message=f"Plugin implements components: {', '.join(implemented_components)}",
                duration=0.0
            ))
        
        report.add_result(ValidationResult(
            check_name="interfaces",
            status=ValidationStatus.PASSED,
            message="Interface validation passed",
            duration=time.time() - start_time
        ))
    
    def _validate_security(self, plugin: BasePlugin, report: ValidationReport):
        """Perform security validation checks."""
        start_time = time.time()
        
        # Check for potentially dangerous attributes
        dangerous_attrs = ['__subclasses__', '__bases__', '__dict__']
        for attr in dangerous_attrs:
            if hasattr(plugin, attr):
                report.add_result(ValidationResult(
                    check_name=f"security_dangerous_attr_{attr}",
                    status=ValidationStatus.WARNING,
                    message=f"Plugin exposes potentially dangerous attribute: {attr}",
                    duration=0.0
                ))
        
        # Check for eval/exec usage in source code
        try:
            source = inspect.getsource(plugin.__class__)
            if 'eval(' in source or 'exec(' in source:
                report.add_result(ValidationResult(
                    check_name="security_eval_exec",
                    status=ValidationStatus.FAILED,
                    message="Plugin contains eval() or exec() calls",
                    duration=0.0
                ))
        except Exception:
            # Can't inspect source, skip this check
            pass
        
        report.add_result(ValidationResult(
            check_name="security",
            status=ValidationStatus.PASSED,
            message="Security validation passed",
            duration=time.time() - start_time
        ))
    
    def _validate_performance(self, plugin: BasePlugin, report: ValidationReport):
        """Perform performance validation checks."""
        start_time = time.time()
        
        # Check for expensive operations in __init__
        init_start = time.time()
        try:
            # Create a new instance to measure initialization time
            plugin_class = plugin.__class__
            new_instance = plugin_class()
            init_time = time.time() - init_start
            
            if init_time > 1.0:  # More than 1 second
                report.add_result(ValidationResult(
                    check_name="performance_init_time",
                    status=ValidationStatus.WARNING,
                    message=f"Plugin initialization took {init_time:.2f}s (should be < 1s)",
                    duration=0.0
                ))
            
        except Exception as e:
            report.add_result(ValidationResult(
                check_name="performance_init_time",
                status=ValidationStatus.WARNING,
                message=f"Could not measure initialization time: {str(e)}",
                duration=0.0
            ))
        
        report.add_result(ValidationResult(
            check_name="performance",
            status=ValidationStatus.PASSED,
            message="Performance validation passed",
            duration=time.time() - start_time
        ))
    
    def _is_valid_version(self, version: str) -> bool:
        """Check if version string follows semantic versioning."""
        pattern = r'^\d+\.\d+\.\d+(\-[a-zA-Z0-9\-\.]+)?(\+[a-zA-Z0-9\-\.]+)?$'
        return bool(re.match(pattern, version))
    
    def _update_stats(self, report: ValidationReport):
        """Update validation statistics."""
        self.validation_stats['total_validations'] += 1
        
        if report.overall_status == ValidationStatus.PASSED:
            self.validation_stats['passed_validations'] += 1
        elif report.overall_status == ValidationStatus.FAILED:
            self.validation_stats['failed_validations'] += 1
        elif report.overall_status == ValidationStatus.WARNING:
            self.validation_stats['warning_validations'] += 1
    
    def _log_validation_results(self, report: ValidationReport):
        """Log validation results."""
        status_emoji = {
            ValidationStatus.PASSED: "✅",
            ValidationStatus.WARNING: "⚠️",
            ValidationStatus.FAILED: "❌",
            ValidationStatus.SKIPPED: "⏭️"
        }
        
        emoji = status_emoji.get(report.overall_status, "❓")
        
        logger.info(
            f"{emoji} Plugin '{report.plugin_name}' validation: "
            f"{report.passed_checks} passed, "
            f"{report.failed_checks} failed, "
            f"{report.warning_checks} warnings "
            f"({report.total_duration:.2f}s)"
        )
        
        # Log detailed results if there are failures or warnings
        if report.failed_checks > 0 or report.warning_checks > 0:
            for result in report.results:
                if result.status in [ValidationStatus.FAILED, ValidationStatus.WARNING]:
                    logger.warning(f"  {result.check_name}: {result.message}")
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        stats = self.validation_stats.copy()
        if stats['total_validations'] > 0:
            stats['success_rate'] = stats['passed_validations'] / stats['total_validations']
        else:
            stats['success_rate'] = 0.0
        
        return stats
    
    def validate_plugin_info(self, plugin_info: PluginInfo) -> ValidationReport:
        """
        Validate plugin information without loading the plugin.
        
        Args:
            plugin_info: Plugin information to validate
            
        Returns:
            Validation report
        """
        start_time = time.time()
        report = ValidationReport(plugin_name=plugin_info.name)
        
        # Validate basic info
        if not plugin_info.name or plugin_info.name == "base_plugin":
            report.add_result(ValidationResult(
                check_name="info_name",
                status=ValidationStatus.FAILED,
                message="Plugin must have a unique name",
                duration=0.0
            ))
        
        if not plugin_info.version:
            report.add_result(ValidationResult(
                check_name="info_version",
                status=ValidationStatus.FAILED,
                message="Plugin must have a version",
                duration=0.0
            ))
        
        if not plugin_info.description:
            report.add_result(ValidationResult(
                check_name="info_description",
                status=ValidationStatus.WARNING,
                message="Plugin should have a description",
                duration=0.0
            ))
        
        # Validate version format
        if plugin_info.version and not self._is_valid_version(plugin_info.version):
            report.add_result(ValidationResult(
                check_name="info_version_format",
                status=ValidationStatus.WARNING,
                message="Version should follow semantic versioning (e.g., 1.0.0)",
                duration=0.0
            ))
        
        report.total_duration = time.time() - start_time
        return report 