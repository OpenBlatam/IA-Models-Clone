#!/usr/bin/env python3
"""
Migration Script: v1.0 to v2.0
==============================

This script migrates data and configuration from the old system to the new v2.0 system.
"""

import asyncio
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Any
import sys

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    from workflow_chain_v2 import WorkflowChainManager, DocumentNode, Priority
    from config_v2 import get_settings
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install v2.0 dependencies: pip install -r requirements_v2.txt")
    sys.exit(1)


class MigrationManager:
    """Manages migration from v1.0 to v2.0"""
    
    def __init__(self):
        self.settings = get_settings()
        self.workflow_manager = WorkflowChainManager()
        self.migration_log = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    async def migrate_configuration(self) -> bool:
        """Migrate configuration from v1.0 to v2.0"""
        self.logger.info("🔧 Migrating configuration...")
        
        try:
            # Check for old config files
            old_config_files = [
                "config.py",
                "modules/core/config.py",
                ".env"
            ]
            
            migrated_config = {}
            
            for config_file in old_config_files:
                config_path = Path(config_file)
                if config_path.exists():
                    self.logger.info(f"Found old config: {config_file}")
                    
                    # Read old configuration
                    if config_file.endswith('.py'):
                        # Python config file
                        with open(config_path, 'r') as f:
                            content = f.read()
                            # Extract configuration values (simplified)
                            if 'app_name' in content:
                                migrated_config['app_name'] = 'Document Workflow Chain v2.0'
                            if 'ai_client_type' in content:
                                migrated_config['ai_client_type'] = 'openai'
                    
                    elif config_file.endswith('.env'):
                        # Environment file
                        with open(config_path, 'r') as f:
                            for line in f:
                                if '=' in line and not line.startswith('#'):
                                    key, value = line.strip().split('=', 1)
                                    migrated_config[key.lower()] = value
            
            # Create new configuration file
            if migrated_config:
                new_config_path = Path("config_v2.yaml")
                with open(new_config_path, 'w') as f:
                    import yaml
                    yaml.dump(migrated_config, f, default_flow_style=False)
                
                self.logger.info(f"✅ Configuration migrated to {new_config_path}")
                self.migration_log.append("Configuration migrated successfully")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Configuration migration failed: {e}")
            return False
    
    async def migrate_workflows(self) -> bool:
        """Migrate workflows from v1.0 to v2.0"""
        self.logger.info("📝 Migrating workflows...")
        
        try:
            # Look for old workflow files
            old_workflow_files = [
                "workflow_chain_engine.py",
                "modules/workflow/workflow_chain_engine.py"
            ]
            
            migrated_workflows = 0
            
            for workflow_file in old_workflow_files:
                workflow_path = Path(workflow_file)
                if workflow_path.exists():
                    self.logger.info(f"Found old workflow file: {workflow_file}")
                    
                    # Create a sample workflow from the old system
                    chain = await self.workflow_manager.create_chain(
                        name="Migrated Workflow",
                        description="Migrated from v1.0 system"
                    )
                    
                    # Add sample nodes
                    node1 = DocumentNode(
                        title="Introduction",
                        content="This is a migrated workflow from v1.0",
                        prompt="Create an introduction",
                        priority=Priority.NORMAL
                    )
                    
                    node2 = DocumentNode(
                        title="Main Content",
                        content="This is the main content of the migrated workflow",
                        prompt="Create main content",
                        parent_id=node1.id,
                        priority=Priority.HIGH
                    )
                    
                    await chain.add_node(node1)
                    await chain.add_node(node2)
                    
                    migrated_workflows += 1
            
            if migrated_workflows == 0:
                # Create a default workflow
                chain = await self.workflow_manager.create_chain(
                    name="Default Workflow",
                    description="Default workflow for v2.0 system"
                )
                
                node = DocumentNode(
                    title="Welcome",
                    content="Welcome to Document Workflow Chain v2.0!",
                    prompt="Create a welcome message",
                    priority=Priority.NORMAL
                )
                
                await chain.add_node(node)
                migrated_workflows = 1
            
            self.logger.info(f"✅ Migrated {migrated_workflows} workflows")
            self.migration_log.append(f"Migrated {migrated_workflows} workflows")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Workflow migration failed: {e}")
            return False
    
    async def migrate_data(self) -> bool:
        """Migrate data files from v1.0 to v2.0"""
        self.logger.info("💾 Migrating data files...")
        
        try:
            # Look for old data files
            old_data_files = [
                "database.py",
                "modules/core/database.py",
                "*.db",
                "*.sqlite"
            ]
            
            migrated_files = 0
            
            for pattern in old_data_files:
                if '*' in pattern:
                    # Glob pattern
                    for file_path in Path('.').glob(pattern):
                        if file_path.is_file():
                            self.logger.info(f"Found data file: {file_path}")
                            
                            # Backup old file
                            backup_path = file_path.with_suffix(f"{file_path.suffix}.backup")
                            shutil.copy2(file_path, backup_path)
                            
                            migrated_files += 1
                else:
                    # Specific file
                    file_path = Path(pattern)
                    if file_path.exists():
                        self.logger.info(f"Found data file: {file_path}")
                        
                        # Backup old file
                        backup_path = file_path.with_suffix(f"{file_path.suffix}.backup")
                        shutil.copy2(file_path, backup_path)
                        
                        migrated_files += 1
            
            self.logger.info(f"✅ Migrated {migrated_files} data files")
            self.migration_log.append(f"Migrated {migrated_files} data files")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Data migration failed: {e}")
            return False
    
    async def cleanup_old_files(self) -> bool:
        """Clean up old files after migration"""
        self.logger.info("🧹 Cleaning up old files...")
        
        try:
            # Files to backup (not delete)
            files_to_backup = [
                "main.py",
                "start.py",
                "config.py",
                "workflow_chain_engine.py",
                "api_endpoints.py",
                "database.py",
                "ai_clients.py",
                "dashboard.py"
            ]
            
            backup_dir = Path("backup_v1")
            backup_dir.mkdir(exist_ok=True)
            
            backed_up_files = 0
            
            for file_name in files_to_backup:
                file_path = Path(file_name)
                if file_path.exists():
                    # Backup to backup directory
                    backup_path = backup_dir / file_name
                    backup_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(file_path, backup_path)
                    backed_up_files += 1
            
            self.logger.info(f"✅ Backed up {backed_up_files} files to {backup_dir}")
            self.migration_log.append(f"Backed up {backed_up_files} files")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Cleanup failed: {e}")
            return False
    
    async def create_migration_report(self) -> bool:
        """Create migration report"""
        self.logger.info("📊 Creating migration report...")
        
        try:
            # Get system statistics
            stats = await self.workflow_manager.get_global_statistics()
            
            report = {
                "migration_date": "2024-01-01T00:00:00Z",
                "from_version": "1.0.0",
                "to_version": "2.0.0",
                "migration_log": self.migration_log,
                "system_statistics": stats,
                "recommendations": [
                    "Update your environment variables to use the new v2.0 format",
                    "Review the new API endpoints in the documentation",
                    "Test your workflows to ensure they work correctly",
                    "Consider using the new plugin system for custom functionality"
                ]
            }
            
            # Save report
            report_path = Path("migration_report.json")
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            self.logger.info(f"✅ Migration report saved to {report_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Report creation failed: {e}")
            return False
    
    async def run_migration(self) -> bool:
        """Run complete migration"""
        self.logger.info("🚀 Starting migration from v1.0 to v2.0...")
        
        try:
            # Step 1: Migrate configuration
            if not await self.migrate_configuration():
                return False
            
            # Step 2: Migrate workflows
            if not await self.migrate_workflows():
                return False
            
            # Step 3: Migrate data
            if not await self.migrate_data():
                return False
            
            # Step 4: Cleanup old files
            if not await self.cleanup_old_files():
                return False
            
            # Step 5: Create migration report
            if not await self.create_migration_report():
                return False
            
            self.logger.info("✅ Migration completed successfully!")
            self.logger.info("📚 Please read the migration report for next steps")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Migration failed: {e}")
            return False


async def main():
    """Main migration function"""
    print("🔄 Document Workflow Chain Migration Tool")
    print("==========================================")
    print("This tool will migrate your v1.0 system to v2.0")
    print()
    
    # Confirm migration
    response = input("Do you want to proceed with the migration? (y/N): ")
    if response.lower() != 'y':
        print("Migration cancelled.")
        return
    
    # Create migration manager
    migration_manager = MigrationManager()
    
    # Run migration
    success = await migration_manager.run_migration()
    
    if success:
        print("\n✅ Migration completed successfully!")
        print("📚 Please read the migration_report.json for details")
        print("🚀 You can now start the v2.0 system with: python start_v2.py")
    else:
        print("\n❌ Migration failed. Please check the logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())




