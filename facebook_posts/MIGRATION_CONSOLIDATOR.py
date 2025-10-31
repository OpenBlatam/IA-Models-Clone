#!/usr/bin/env python3
"""
MIGRATION AND CONSOLIDATION SYSTEM
==================================

Consolidates all existing Facebook Posts system versions (v3.1-v3.7)
into the new Ultimate Consolidated System v4.0

Features:
- Automatic migration from any previous version
- Data preservation and transformation
- Configuration consolidation
- Performance optimization
- Backward compatibility
"""

import asyncio
import json
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import importlib.util
import sys

logger = logging.getLogger(__name__)


class MigrationConsolidator:
    """
    Consolidates all Facebook Posts system versions
    """
    
    def __init__(self, base_path: str = "."):
        """Initialize the migration consolidator"""
        self.base_path = Path(base_path)
        self.versions_found = []
        self.migration_log = []
        self.consolidated_config = {}
        
        logger.info("Migration Consolidator initialized")
    
    async def discover_versions(self) -> List[Dict[str, Any]]:
        """Discover all available system versions"""
        logger.info("Discovering system versions...")
        
        version_patterns = [
            "ultra_modular_ai_interface_v3_7.py",
            "refactored_unified_ai_interface_v3_6.py",
            "enhanced_unified_ai_interface_v3_5.py",
            "unified_ai_interface_v3_4.py",
            "unified_ai_interface_v3_3.py",
            "autonomous_agents_interface.py",
            "advanced_learning_interface_v3_1.py",
            "advanced_predictive_interface.py"
        ]
        
        for pattern in version_patterns:
            file_path = self.base_path / pattern
            if file_path.exists():
                version_info = await self._analyze_version_file(file_path)
                if version_info:
                    self.versions_found.append(version_info)
                    logger.info(f"Found version: {version_info['name']} v{version_info['version']}")
        
        logger.info(f"Discovered {len(self.versions_found)} versions")
        return self.versions_found
    
    async def _analyze_version_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Analyze a version file to extract metadata"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract version information
            version_info = {
                "file_path": str(file_path),
                "name": file_path.stem,
                "version": self._extract_version(content),
                "size": file_path.stat().st_size,
                "last_modified": datetime.fromtimestamp(file_path.stat().st_mtime),
                "features": self._extract_features(content),
                "dependencies": self._extract_dependencies(content),
                "config_sections": self._extract_config_sections(content)
            }
            
            return version_info
            
        except Exception as e:
            logger.error(f"Failed to analyze {file_path}: {e}")
            return None
    
    def _extract_version(self, content: str) -> str:
        """Extract version number from content"""
        import re
        
        # Look for version patterns
        version_patterns = [
            r'v(\d+\.\d+\.\d+)',
            r'version["\']?\s*[:=]\s*["\']?(\d+\.\d+\.\d+)',
            r'Version\s+(\d+\.\d+\.\d+)'
        ]
        
        for pattern in version_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return "unknown"
    
    def _extract_features(self, content: str) -> List[str]:
        """Extract features from content"""
        features = []
        
        feature_keywords = [
            "async", "ai", "optimization", "analytics", "caching", "monitoring",
            "performance", "security", "authentication", "rate_limiting",
            "batch_processing", "real_time", "machine_learning", "neural_network",
            "quantum", "federated_learning", "autonomous", "predictive"
        ]
        
        content_lower = content.lower()
        for keyword in feature_keywords:
            if keyword in content_lower:
                features.append(keyword)
        
        return features
    
    def _extract_dependencies(self, content: str) -> List[str]:
        """Extract dependencies from content"""
        dependencies = []
        
        # Look for import statements
        import re
        import_matches = re.findall(r'import\s+(\w+)', content)
        from_matches = re.findall(r'from\s+(\w+)', content)
        
        all_imports = set(import_matches + from_matches)
        
        # Filter out standard library modules
        stdlib_modules = {
            'os', 'sys', 'json', 'time', 'datetime', 'pathlib', 'typing',
            'asyncio', 'logging', 'uuid', 'hashlib', 'math', 'random'
        }
        
        for module in all_imports:
            if module not in stdlib_modules and len(module) > 2:
                dependencies.append(module)
        
        return list(set(dependencies))
    
    def _extract_config_sections(self, content: str) -> List[str]:
        """Extract configuration sections from content"""
        config_sections = []
        
        # Look for config-related patterns
        import re
        config_patterns = [
            r'config\[["\'](\w+)["\']\]',
            r'self\.config\.(\w+)',
            r'(\w+)_config',
            r'(\w+)_settings'
        ]
        
        for pattern in config_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            config_sections.extend(matches)
        
        return list(set(config_sections))
    
    async def consolidate_configurations(self) -> Dict[str, Any]:
        """Consolidate configurations from all versions"""
        logger.info("Consolidating configurations...")
        
        consolidated = {
            "api": {
                "title": "Ultimate Facebook Posts API",
                "version": "4.0.0",
                "description": "Consolidated AI-powered Facebook post generation system"
            },
            "performance": {
                "max_concurrent_requests": 1000,
                "request_timeout": 30,
                "enable_caching": True,
                "enable_metrics": True
            },
            "ai": {
                "providers": ["openai", "anthropic", "local"],
                "default_model": "gpt-4",
                "fallback_model": "gpt-3.5-turbo",
                "max_tokens": 2000,
                "temperature": 0.7
            },
            "caching": {
                "provider": "redis",
                "ttl": 3600,
                "max_connections": 100,
                "compression": True
            },
            "monitoring": {
                "enable_prometheus": True,
                "enable_health_checks": True,
                "log_level": "INFO",
                "metrics_retention": "7d"
            },
            "security": {
                "api_key_required": True,
                "rate_limiting": True,
                "cors_enabled": True,
                "input_validation": True
            },
            "features": {
                "batch_processing": True,
                "real_time_analytics": True,
                "content_optimization": True,
                "multi_language": True,
                "audience_targeting": True
            }
        }
        
        # Merge configurations from all versions
        for version in self.versions_found:
            if "config_sections" in version:
                for section in version["config_sections"]:
                    if section not in consolidated:
                        consolidated[section] = {}
        
        self.consolidated_config = consolidated
        logger.info("Configuration consolidation completed")
        return consolidated
    
    async def migrate_data(self) -> Dict[str, Any]:
        """Migrate data from all versions"""
        logger.info("Starting data migration...")
        
        migration_results = {
            "posts_migrated": 0,
            "configs_migrated": 0,
            "errors": [],
            "warnings": []
        }
        
        # Migrate configuration files
        config_files = [
            "ai_config_v3_6.json",
            "system_config.json",
            "config.json",
            "settings.json"
        ]
        
        for config_file in config_files:
            config_path = self.base_path / config_file
            if config_path.exists():
                try:
                    await self._migrate_config_file(config_path)
                    migration_results["configs_migrated"] += 1
                except Exception as e:
                    migration_results["errors"].append(f"Config migration failed for {config_file}: {e}")
        
        # Migrate data directories
        data_dirs = ["data", "posts", "cache", "logs"]
        for data_dir in data_dirs:
            dir_path = self.base_path / data_dir
            if dir_path.exists():
                try:
                    await self._migrate_data_directory(dir_path)
                except Exception as e:
                    migration_results["warnings"].append(f"Data directory migration warning for {data_dir}: {e}")
        
        logger.info("Data migration completed", **migration_results)
        return migration_results
    
    async def _migrate_config_file(self, config_path: Path) -> None:
        """Migrate a configuration file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # Transform configuration to new format
            transformed_config = self._transform_config(config_data)
            
            # Save to new location
            new_config_path = self.base_path / "config" / "consolidated_config.json"
            new_config_path.parent.mkdir(exist_ok=True)
            
            with open(new_config_path, 'w', encoding='utf-8') as f:
                json.dump(transformed_config, f, indent=2)
            
            logger.info(f"Migrated config file: {config_path.name}")
            
        except Exception as e:
            logger.error(f"Failed to migrate config file {config_path}: {e}")
            raise
    
    def _transform_config(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform configuration to new format"""
        # This would contain the logic to transform old config formats to new format
        # For now, return the data as-is
        return config_data
    
    async def _migrate_data_directory(self, data_dir: Path) -> None:
        """Migrate a data directory"""
        # Create new data structure
        new_data_dir = self.base_path / "data" / "migrated" / data_dir.name
        new_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy files
        for item in data_dir.iterdir():
            if item.is_file():
                shutil.copy2(item, new_data_dir / item.name)
            elif item.is_dir():
                shutil.copytree(item, new_data_dir / item.name, dirs_exist_ok=True)
        
        logger.info(f"Migrated data directory: {data_dir.name}")
    
    async def create_migration_report(self) -> Dict[str, Any]:
        """Create a comprehensive migration report"""
        report = {
            "migration_timestamp": datetime.utcnow().isoformat(),
            "versions_discovered": len(self.versions_found),
            "versions_details": self.versions_found,
            "consolidated_config": self.consolidated_config,
            "migration_log": self.migration_log,
            "recommendations": self._generate_recommendations()
        }
        
        # Save report
        report_path = self.base_path / "MIGRATION_REPORT.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Migration report saved to: {report_path}")
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate migration recommendations"""
        recommendations = [
            "Use the new Ultimate Consolidated System v4.0 for all new development",
            "Gradually migrate existing integrations to the new API endpoints",
            "Update client applications to use the new response formats",
            "Configure monitoring and alerting for the new system",
            "Test all critical workflows with the new system before full migration",
            "Keep backup of original systems until migration is verified",
            "Update documentation to reflect the new system architecture"
        ]
        
        return recommendations
    
    async def run_full_migration(self) -> Dict[str, Any]:
        """Run the complete migration process"""
        logger.info("Starting full migration process...")
        
        try:
            # Step 1: Discover versions
            await self.discover_versions()
            
            # Step 2: Consolidate configurations
            await self.consolidate_configurations()
            
            # Step 3: Migrate data
            migration_results = await self.migrate_data()
            
            # Step 4: Create migration report
            report = await self.create_migration_report()
            
            logger.info("Full migration process completed successfully")
            
            return {
                "success": True,
                "versions_found": len(self.versions_found),
                "migration_results": migration_results,
                "report_path": "MIGRATION_REPORT.json"
            }
            
        except Exception as e:
            logger.error(f"Migration process failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }


# CLI interface
async def main():
    """Main CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Facebook Posts System Migration Consolidator")
    parser.add_argument("--path", default=".", help="Base path for migration")
    parser.add_argument("--discover-only", action="store_true", help="Only discover versions")
    parser.add_argument("--config-only", action="store_true", help="Only consolidate configs")
    
    args = parser.parse_args()
    
    consolidator = MigrationConsolidator(args.path)
    
    if args.discover_only:
        versions = await consolidator.discover_versions()
        print(f"Found {len(versions)} versions:")
        for version in versions:
            print(f"  - {version['name']} v{version['version']}")
    
    elif args.config_only:
        await consolidator.discover_versions()
        config = await consolidator.consolidate_configurations()
        print("Consolidated configuration:")
        print(json.dumps(config, indent=2))
    
    else:
        result = await consolidator.run_full_migration()
        print("Migration result:")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())

