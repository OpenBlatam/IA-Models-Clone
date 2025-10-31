#!/usr/bin/env python3
"""
Migration Script
================

Script to migrate from the original Bulk TruthGPT system to the refactored version.
"""

import os
import shutil
import sys
from pathlib import Path
from typing import List, Dict, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MigrationManager:
    """Manages the migration process."""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.backup_path = self.base_path / "backup_original"
        self.migration_log = []
        
    def log_migration(self, action: str, details: str):
        """Log migration action."""
        log_entry = f"[{action}] {details}"
        self.migration_log.append(log_entry)
        logger.info(log_entry)
    
    def create_backup(self):
        """Create backup of original files."""
        try:
            logger.info("Creating backup of original files...")
            
            if self.backup_path.exists():
                shutil.rmtree(self.backup_path)
            
            self.backup_path.mkdir(parents=True, exist_ok=True)
            
            # Files to backup
            files_to_backup = [
                "main.py",
                "requirements.txt",
                "Dockerfile",
                "docker-compose.yml",
                "nginx.conf",
                "prometheus.yml",
                "start_production.py",
                "README.md"
            ]
            
            for file_name in files_to_backup:
                source_file = self.base_path / file_name
                if source_file.exists():
                    backup_file = self.backup_path / file_name
                    shutil.copy2(source_file, backup_file)
                    self.log_migration("BACKUP", f"Backed up {file_name}")
            
            logger.info("Backup created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create backup: {str(e)}")
            return False
    
    def migrate_main_file(self):
        """Migrate main.py to refactored version."""
        try:
            logger.info("Migrating main.py to refactored version...")
            
            # Replace main.py with refactored version
            original_main = self.base_path / "main.py"
            refactored_main = self.base_path / "main_refactored.py"
            
            if refactored_main.exists():
                shutil.copy2(refactored_main, original_main)
                self.log_migration("MIGRATE", "Replaced main.py with refactored version")
                return True
            else:
                logger.error("Refactored main.py not found")
                return False
                
        except Exception as e:
            logger.error(f"Failed to migrate main.py: {str(e)}")
            return False
    
    def update_requirements(self):
        """Update requirements.txt with new dependencies."""
        try:
            logger.info("Updating requirements.txt...")
            
            requirements_file = self.base_path / "requirements.txt"
            
            # New dependencies for refactored system
            new_dependencies = [
                "structlog==23.2.0",
                "pydantic-settings==2.1.0",
                "prometheus-client==0.19.0",
                "psutil==5.9.6",
                "numpy==1.24.3",
                "pandas==2.1.3",
                "scikit-learn==1.3.2",
                "scipy==1.11.4"
            ]
            
            # Read existing requirements
            existing_requirements = []
            if requirements_file.exists():
                with open(requirements_file, 'r') as f:
                    existing_requirements = f.readlines()
            
            # Add new dependencies
            updated_requirements = existing_requirements.copy()
            for dep in new_dependencies:
                if not any(dep.split('==')[0] in line for line in existing_requirements):
                    updated_requirements.append(f"{dep}\n")
            
            # Write updated requirements
            with open(requirements_file, 'w') as f:
                f.writelines(updated_requirements)
            
            self.log_migration("UPDATE", "Updated requirements.txt with new dependencies")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update requirements: {str(e)}")
            return False
    
    def create_new_directories(self):
        """Create new directory structure."""
        try:
            logger.info("Creating new directory structure...")
            
            new_dirs = [
                "tests",
                "config",
                "utils",
                "core",
                "services",
                "models"
            ]
            
            for dir_name in new_dirs:
                dir_path = self.base_path / dir_name
                if not dir_path.exists():
                    dir_path.mkdir(parents=True, exist_ok=True)
                    self.log_migration("CREATE", f"Created directory: {dir_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create directories: {str(e)}")
            return False
    
    def update_dockerfile(self):
        """Update Dockerfile for refactored system."""
        try:
            logger.info("Updating Dockerfile...")
            
            dockerfile_path = self.base_path / "Dockerfile"
            
            new_dockerfile_content = '''# Bulk TruthGPT Production Dockerfile - Refactored
# ================================================

FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    software-properties-common \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/storage /app/templates /app/models /app/knowledge_base /app/logs /app/tests

# Set permissions
RUN chmod -R 755 /app

# Create non-root user
RUN useradd --create-home --shell /bin/bash app && \\
    chown -R app:app /app
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
'''
            
            with open(dockerfile_path, 'w') as f:
                f.write(new_dockerfile_content)
            
            self.log_migration("UPDATE", "Updated Dockerfile for refactored system")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update Dockerfile: {str(e)}")
            return False
    
    def create_test_runner(self):
        """Create test runner script."""
        try:
            logger.info("Creating test runner script...")
            
            test_runner_content = '''#!/usr/bin/env python3
"""
Test Runner
===========

Script to run tests for the refactored Bulk TruthGPT system.
"""

import pytest
import sys
import os
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def run_tests():
    """Run all tests."""
    try:
        # Run tests
        result = pytest.main([
            "tests/",
            "-v",
            "--tb=short",
            "--cov=.",
            "--cov-report=html",
            "--cov-report=term"
        ])
        
        return result == 0
        
    except Exception as e:
        print(f"Error running tests: {str(e)}")
        return False

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
'''
            
            test_runner_path = self.base_path / "run_tests.py"
            with open(test_runner_path, 'w') as f:
                f.write(test_runner_content)
            
            # Make executable
            os.chmod(test_runner_path, 0o755)
            
            self.log_migration("CREATE", "Created test runner script")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create test runner: {str(e)}")
            return False
    
    def create_migration_report(self):
        """Create migration report."""
        try:
            logger.info("Creating migration report...")
            
            report_content = f"""# Migration Report

## Migration Summary
- **Date**: {datetime.now().isoformat()}
- **Original System**: Bulk TruthGPT v1.0.0
- **Refactored System**: Bulk TruthGPT v2.0.0

## Changes Made

### Architecture Improvements
1. **Base Component System**: Implemented base classes for all components
2. **Component Registry**: Centralized component management
3. **Dependency Injection**: Improved component dependencies
4. **Error Handling**: Comprehensive exception handling system
5. **Logging**: Structured logging with context
6. **Metrics**: Advanced metrics collection and monitoring
7. **Configuration**: Centralized settings management

### New Features
1. **Health Checks**: Comprehensive system health monitoring
2. **Metrics**: Prometheus integration and custom metrics
3. **Testing**: Complete test suite with integration tests
4. **Error Recovery**: Improved error handling and recovery
5. **Performance**: Optimized for better performance
6. **Monitoring**: Enhanced monitoring and observability

### Files Modified
{chr(10).join(f"- {entry}" for entry in self.migration_log)}

### New Files Created
- `core/base.py` - Base component system
- `config/settings.py` - Centralized configuration
- `utils/logging.py` - Advanced logging system
- `utils/exceptions.py` - Exception handling
- `utils/metrics.py` - Metrics collection
- `tests/test_system.py` - Comprehensive test suite
- `main_refactored.py` - Refactored main application

### Migration Steps
1. ✅ Created backup of original files
2. ✅ Migrated main.py to refactored version
3. ✅ Updated requirements.txt
4. ✅ Created new directory structure
5. ✅ Updated Dockerfile
6. ✅ Created test runner script

## Post-Migration Steps
1. **Install Dependencies**: `pip install -r requirements.txt`
2. **Run Tests**: `python run_tests.py`
3. **Start System**: `python main.py`
4. **Verify Health**: `curl http://localhost:8000/health`

## Rollback Instructions
If rollback is needed:
1. Stop the refactored system
2. Copy files from `backup_original/` to root directory
3. Restart the original system

## Support
For issues with the refactored system, check:
1. Test results: `python run_tests.py`
2. System health: `curl http://localhost:8000/health`
3. Logs: Check `./logs/` directory
4. Metrics: `curl http://localhost:8000/metrics`
"""
            
            report_path = self.base_path / "MIGRATION_REPORT.md"
            with open(report_path, 'w') as f:
                f.write(report_content)
            
            self.log_migration("CREATE", "Created migration report")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create migration report: {str(e)}")
            return False
    
    def run_migration(self):
        """Run the complete migration process."""
        try:
            logger.info("Starting migration process...")
            
            # Step 1: Create backup
            if not self.create_backup():
                return False
            
            # Step 2: Create new directories
            if not self.create_new_directories():
                return False
            
            # Step 3: Migrate main file
            if not self.migrate_main_file():
                return False
            
            # Step 4: Update requirements
            if not self.update_requirements():
                return False
            
            # Step 5: Update Dockerfile
            if not self.update_dockerfile():
                return False
            
            # Step 6: Create test runner
            if not self.create_test_runner():
                return False
            
            # Step 7: Create migration report
            if not self.create_migration_report():
                return False
            
            logger.info("Migration completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Migration failed: {str(e)}")
            return False

def main():
    """Main migration function."""
    import argparse
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description="Migrate Bulk TruthGPT to refactored version")
    parser.add_argument("--path", default=".", help="Path to Bulk TruthGPT directory")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    
    args = parser.parse_args()
    
    if args.dry_run:
        logger.info("DRY RUN MODE - No changes will be made")
        logger.info(f"Would migrate system in: {args.path}")
        return
    
    # Run migration
    migration_manager = MigrationManager(args.path)
    success = migration_manager.run_migration()
    
    if success:
        logger.info("Migration completed successfully!")
        logger.info("Next steps:")
        logger.info("1. Install dependencies: pip install -r requirements.txt")
        logger.info("2. Run tests: python run_tests.py")
        logger.info("3. Start system: python main.py")
        logger.info("4. Check health: curl http://localhost:8000/health")
    else:
        logger.error("Migration failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()











