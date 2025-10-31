"""
Organized Configuration Loader
Highly organized configuration loader for the modular TruthGPT system
"""

import yaml
import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import sys
from dataclasses import dataclass, field
from enum import Enum
import time

# Add the TruthGPT path to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent / "Frontier-Model-run" / "scripts" / "TruthGPT-main"))

logger = logging.getLogger(__name__)

class ConfigSection(Enum):
    """Configuration sections"""
    METADATA = "metadata"
    CORE_SYSTEM = "core_system"
    MICRO_MODULES = "micro_modules"
    PLUGINS = "plugins"
    SYSTEM = "system"
    API = "api"
    DATABASE = "database"
    REDIS = "redis"
    LOGGING = "logging"
    TESTING = "testing"
    DEPLOYMENT = "deployment"
    SECURITY = "security"
    PERFORMANCE = "performance"
    FEATURE_FLAGS = "feature_flags"

@dataclass
class ConfigValidationResult:
    """Configuration validation result"""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)

@dataclass
class ModuleConfig:
    """Module configuration"""
    name: str
    class_name: str
    config: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    scope: str = "singleton"
    enabled: bool = True

class OrganizedConfigLoader:
    """Organized configuration loader"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = {}
        self.sections = {}
        self.modules = {}
        self.plugins = {}
        self.validation_results = {}
        
    def load_config(self) -> Dict[str, Any]:
        """Load organized configuration"""
        try:
            with open(self.config_path, 'r') as f:
                if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                    self.config = yaml.safe_load(f)
                elif self.config_path.endswith('.json'):
                    self.config = json.load(f)
                else:
                    raise ValueError(f"Unsupported config format: {self.config_path}")
            
            # Organize configuration by sections
            self._organize_sections()
            
            logger.info(f"✅ Loaded organized configuration from {self.config_path}")
            return self.config
        except Exception as e:
            logger.error(f"❌ Failed to load configuration: {e}")
            raise
    
    def _organize_sections(self) -> None:
        """Organize configuration into sections"""
        for section in ConfigSection:
            if section.value in self.config:
                self.sections[section] = self.config[section.value]
                logger.info(f"📋 Loaded section: {section.value}")
        
        # Extract modules and plugins
        self._extract_modules()
        self._extract_plugins()
    
    def _extract_modules(self) -> None:
        """Extract module configurations"""
        micro_modules = self.sections.get(ConfigSection.MICRO_MODULES, {})
        
        for module_type, modules in micro_modules.items():
            if isinstance(modules, dict):
                for module_name, module_config in modules.items():
                    if isinstance(module_config, dict):
                        module = ModuleConfig(
                            name=module_name,
                            class_name=module_config.get('class', ''),
                            config=module_config.get('config', {}),
                            dependencies=module_config.get('dependencies', []),
                            scope=module_config.get('scope', 'singleton'),
                            enabled=module_config.get('enabled', True)
                        )
                        self.modules[module_name] = module
                        logger.info(f"🔧 Extracted module: {module_name} ({module_type})")
    
    def _extract_plugins(self) -> None:
        """Extract plugin configurations"""
        plugins = self.sections.get(ConfigSection.PLUGINS, {})
        
        for plugin_type, plugin_list in plugins.items():
            if isinstance(plugin_list, dict):
                for plugin_name, plugin_config in plugin_list.items():
                    if isinstance(plugin_config, dict):
                        plugin = ModuleConfig(
                            name=plugin_name,
                            class_name=plugin_config.get('class', ''),
                            config=plugin_config.get('config', {}),
                            dependencies=plugin_config.get('dependencies', []),
                            scope=plugin_config.get('scope', 'singleton'),
                            enabled=plugin_config.get('enabled', True)
                        )
                        self.plugins[plugin_name] = plugin
                        logger.info(f"🔌 Extracted plugin: {plugin_name} ({plugin_type})")
    
    def validate_config(self) -> Dict[str, ConfigValidationResult]:
        """Validate organized configuration"""
        validation_results = {}
        
        # Validate metadata
        validation_results['metadata'] = self._validate_metadata()
        
        # Validate core system
        validation_results['core_system'] = self._validate_core_system()
        
        # Validate micro-modules
        validation_results['micro_modules'] = self._validate_micro_modules()
        
        # Validate plugins
        validation_results['plugins'] = self._validate_plugins()
        
        # Validate system configuration
        validation_results['system'] = self._validate_system()
        
        # Validate security
        validation_results['security'] = self._validate_security()
        
        # Validate performance
        validation_results['performance'] = self._validate_performance()
        
        # Validate feature flags
        validation_results['feature_flags'] = self._validate_feature_flags()
        
        self.validation_results = validation_results
        return validation_results
    
    def _validate_metadata(self) -> ConfigValidationResult:
        """Validate metadata section"""
        result = ConfigValidationResult(is_valid=True)
        metadata = self.sections.get(ConfigSection.METADATA, {})
        
        required_fields = ['name', 'version', 'environment']
        for field in required_fields:
            if field not in metadata:
                result.errors.append(f"Missing required metadata field: {field}")
                result.is_valid = False
        
        if not result.is_valid:
            result.suggestions.append("Add missing metadata fields: name, version, environment")
        
        return result
    
    def _validate_core_system(self) -> ConfigValidationResult:
        """Validate core system section"""
        result = ConfigValidationResult(is_valid=True)
        core_system = self.sections.get(ConfigSection.CORE_SYSTEM, {})
        
        # Validate AI optimization
        ai_optimization = core_system.get('ai_optimization', {})
        if ai_optimization.get('enabled', False):
            required_ai_fields = ['auto_tuning', 'adaptive_configuration', 'machine_learning_config']
            for field in required_ai_fields:
                if field not in ai_optimization:
                    result.warnings.append(f"AI optimization enabled but missing {field}")
        
        # Validate modular architecture
        modular_architecture = core_system.get('modular_architecture', {})
        if not modular_architecture:
            result.errors.append("Missing modular_architecture configuration")
            result.is_valid = False
        
        return result
    
    def _validate_micro_modules(self) -> ConfigValidationResult:
        """Validate micro-modules section"""
        result = ConfigValidationResult(is_valid=True)
        
        if not self.modules:
            result.warnings.append("No micro-modules configured")
            return result
        
        # Check for required modules
        required_module_types = ['optimizers', 'models', 'trainers', 'inferencers', 'monitors']
        micro_modules = self.sections.get(ConfigSection.MICRO_MODULES, {})
        
        for module_type in required_module_types:
            if module_type not in micro_modules:
                result.warnings.append(f"Missing recommended module type: {module_type}")
        
        # Validate individual modules
        for module_name, module in self.modules.items():
            if not module.class_name:
                result.errors.append(f"Module {module_name} missing class name")
                result.is_valid = False
            
            if not module.config:
                result.warnings.append(f"Module {module_name} has no configuration")
        
        return result
    
    def _validate_plugins(self) -> ConfigValidationResult:
        """Validate plugins section"""
        result = ConfigValidationResult(is_valid=True)
        
        if not self.plugins:
            result.warnings.append("No plugins configured")
            return result
        
        # Validate individual plugins
        for plugin_name, plugin in self.plugins.items():
            if not plugin.class_name:
                result.errors.append(f"Plugin {plugin_name} missing class name")
                result.is_valid = False
            
            if not plugin.config:
                result.warnings.append(f"Plugin {plugin_name} has no configuration")
        
        return result
    
    def _validate_system(self) -> ConfigValidationResult:
        """Validate system section"""
        result = ConfigValidationResult(is_valid=True)
        system = self.sections.get(ConfigSection.SYSTEM, {})
        
        required_fields = ['max_concurrent_generations', 'batch_size', 'max_workers']
        for field in required_fields:
            if field not in system:
                result.errors.append(f"Missing required system field: {field}")
                result.is_valid = False
        
        return result
    
    def _validate_security(self) -> ConfigValidationResult:
        """Validate security section"""
        result = ConfigValidationResult(is_valid=True)
        security = self.sections.get(ConfigSection.SECURITY, {})
        
        if not security:
            result.warnings.append("No security configuration found")
            return result
        
        # Check for basic security features
        if not security.get('enable_ssl', False):
            result.warnings.append("SSL not enabled")
        
        if not security.get('authentication', {}).get('type'):
            result.warnings.append("No authentication type specified")
        
        return result
    
    def _validate_performance(self) -> ConfigValidationResult:
        """Validate performance section"""
        result = ConfigValidationResult(is_valid=True)
        performance = self.sections.get(ConfigSection.PERFORMANCE, {})
        
        if not performance:
            result.warnings.append("No performance configuration found")
            return result
        
        return result
    
    def _validate_feature_flags(self) -> ConfigValidationResult:
        """Validate feature flags section"""
        result = ConfigValidationResult(is_valid=True)
        feature_flags = self.sections.get(ConfigSection.FEATURE_FLAGS, {})
        
        if not feature_flags:
            result.warnings.append("No feature flags configured")
            return result
        
        # Check for core features
        core_features = feature_flags.get('core_features', {})
        if not core_features.get('modular_architecture', False):
            result.warnings.append("Modular architecture not enabled in feature flags")
        
        return result
    
    def get_section(self, section: ConfigSection) -> Dict[str, Any]:
        """Get configuration section"""
        return self.sections.get(section, {})
    
    def get_module(self, module_name: str) -> Optional[ModuleConfig]:
        """Get module configuration"""
        return self.modules.get(module_name)
    
    def get_plugin(self, plugin_name: str) -> Optional[ModuleConfig]:
        """Get plugin configuration"""
        return self.plugins.get(plugin_name)
    
    def get_modules_by_type(self, module_type: str) -> List[ModuleConfig]:
        """Get modules by type"""
        modules = []
        micro_modules = self.sections.get(ConfigSection.MICRO_MODULES, {})
        
        if module_type in micro_modules:
            for module_name, module_config in micro_modules[module_type].items():
                if isinstance(module_config, dict):
                    module = ModuleConfig(
                        name=module_name,
                        class_name=module_config.get('class', ''),
                        config=module_config.get('config', {}),
                        dependencies=module_config.get('dependencies', []),
                        scope=module_config.get('scope', 'singleton'),
                        enabled=module_config.get('enabled', True)
                    )
                    modules.append(module)
        
        return modules
    
    def get_plugins_by_type(self, plugin_type: str) -> List[ModuleConfig]:
        """Get plugins by type"""
        plugins = []
        plugins_section = self.sections.get(ConfigSection.PLUGINS, {})
        
        if plugin_type in plugins_section:
            for plugin_name, plugin_config in plugins_section[plugin_type].items():
                if isinstance(plugin_config, dict):
                    plugin = ModuleConfig(
                        name=plugin_name,
                        class_name=plugin_config.get('class', ''),
                        config=plugin_config.get('config', {}),
                        dependencies=plugin_config.get('dependencies', []),
                        scope=plugin_config.get('scope', 'singleton'),
                        enabled=plugin_config.get('enabled', True)
                    )
                    plugins.append(plugin)
        
        return plugins
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get validation summary"""
        total_sections = len(self.validation_results)
        valid_sections = sum(1 for result in self.validation_results.values() if result.is_valid)
        total_errors = sum(len(result.errors) for result in self.validation_results.values())
        total_warnings = sum(len(result.warnings) for result in self.validation_results.values())
        total_suggestions = sum(len(result.suggestions) for result in self.validation_results.values())
        
        return {
            "total_sections": total_sections,
            "valid_sections": valid_sections,
            "invalid_sections": total_sections - valid_sections,
            "total_errors": total_errors,
            "total_warnings": total_warnings,
            "total_suggestions": total_suggestions,
            "validation_score": (valid_sections / total_sections * 100) if total_sections > 0 else 0
        }
    
    def generate_config_report(self) -> str:
        """Generate configuration report"""
        report = []
        report.append("=" * 80)
        report.append("ORGANIZED CONFIGURATION REPORT")
        report.append("=" * 80)
        report.append(f"Configuration File: {self.config_path}")
        report.append(f"Load Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Metadata
        metadata = self.get_section(ConfigSection.METADATA)
        if metadata:
            report.append("📋 METADATA")
            report.append("-" * 40)
            for key, value in metadata.items():
                report.append(f"  {key}: {value}")
            report.append("")
        
        # Sections summary
        report.append("📊 SECTIONS SUMMARY")
        report.append("-" * 40)
        for section in ConfigSection:
            if section in self.sections:
                report.append(f"  ✅ {section.value}: Loaded")
            else:
                report.append(f"  ❌ {section.value}: Missing")
        report.append("")
        
        # Modules summary
        report.append("🔧 MODULES SUMMARY")
        report.append("-" * 40)
        report.append(f"  Total Modules: {len(self.modules)}")
        for module_name, module in self.modules.items():
            status = "✅" if module.enabled else "❌"
            report.append(f"  {status} {module_name}: {module.class_name}")
        report.append("")
        
        # Plugins summary
        report.append("🔌 PLUGINS SUMMARY")
        report.append("-" * 40)
        report.append(f"  Total Plugins: {len(self.plugins)}")
        for plugin_name, plugin in self.plugins.items():
            status = "✅" if plugin.enabled else "❌"
            report.append(f"  {status} {plugin_name}: {plugin.class_name}")
        report.append("")
        
        # Validation summary
        validation_summary = self.get_validation_summary()
        report.append("✅ VALIDATION SUMMARY")
        report.append("-" * 40)
        report.append(f"  Validation Score: {validation_summary['validation_score']:.1f}%")
        report.append(f"  Valid Sections: {validation_summary['valid_sections']}/{validation_summary['total_sections']}")
        report.append(f"  Errors: {validation_summary['total_errors']}")
        report.append(f"  Warnings: {validation_summary['total_warnings']}")
        report.append(f"  Suggestions: {validation_summary['total_suggestions']}")
        report.append("")
        
        # Detailed validation results
        if self.validation_results:
            report.append("🔍 DETAILED VALIDATION")
            report.append("-" * 40)
            for section_name, result in self.validation_results.items():
                status = "✅" if result.is_valid else "❌"
                report.append(f"  {status} {section_name.upper()}")
                
                if result.errors:
                    for error in result.errors:
                        report.append(f"    ❌ Error: {error}")
                
                if result.warnings:
                    for warning in result.warnings:
                        report.append(f"    ⚠️  Warning: {warning}")
                
                if result.suggestions:
                    for suggestion in result.suggestions:
                        report.append(f"    💡 Suggestion: {suggestion}")
                report.append("")
        
        report.append("=" * 80)
        return "\n".join(report)
    
    def save_report(self, output_path: str) -> bool:
        """Save configuration report to file"""
        try:
            report = self.generate_config_report()
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"✅ Configuration report saved to {output_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to save report: {e}")
            return False

def main():
    """Main function to demonstrate organized configuration loading"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Organized Configuration Loader')
    parser.add_argument('--config', default='organized_modular_config.yaml', 
                       help='Configuration file path')
    parser.add_argument('--action', choices=['load', 'validate', 'report'], 
                       default='load', help='Action to perform')
    parser.add_argument('--output', help='Output file for report')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Create loader
        loader = OrganizedConfigLoader(args.config)
        
        if args.action == 'load':
            # Load configuration
            config = loader.load_config()
            logger.info(f"✅ Loaded organized configuration with {len(config)} sections")
            
            # Show sections
            for section in ConfigSection:
                if section in loader.sections:
                    logger.info(f"📋 Section {section.value}: {len(loader.sections[section])} items")
        
        elif args.action == 'validate':
            # Load and validate
            loader.load_config()
            validation_results = loader.validate_config()
            
            # Show validation results
            for section_name, result in validation_results.items():
                status = "✅" if result.is_valid else "❌"
                logger.info(f"{status} {section_name}: {len(result.errors)} errors, {len(result.warnings)} warnings")
        
        elif args.action == 'report':
            # Load, validate, and generate report
            loader.load_config()
            loader.validate_config()
            
            if args.output:
                loader.save_report(args.output)
            else:
                report = loader.generate_config_report()
                print(report)
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())

