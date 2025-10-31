"""
Content Lifecycle Engine - Advanced Content Lifecycle Management
==============================================================

This module provides comprehensive content lifecycle management including:
- Content versioning and history tracking
- Automated content archiving and deletion
- Content migration and transformation
- Content dependency management
- Content expiration and renewal
- Content backup and recovery
- Content lifecycle analytics
- Automated content workflows
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
import hashlib
import shutil
import os
import zipfile
import tarfile
from collections import defaultdict, deque
import redis
import sqlalchemy as sa
from sqlalchemy import create_engine, text
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import requests
import boto3
from google.cloud import storage
import openai
import anthropic
from cryptography.fernet import Fernet
import schedule
import threading
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContentStatus(Enum):
    """Content status enumeration"""
    DRAFT = "draft"
    REVIEW = "review"
    APPROVED = "approved"
    PUBLISHED = "published"
    ARCHIVED = "archived"
    DELETED = "deleted"
    EXPIRED = "expired"
    MIGRATED = "migrated"

class LifecycleStage(Enum):
    """Lifecycle stage enumeration"""
    CREATION = "creation"
    DEVELOPMENT = "development"
    REVIEW = "review"
    PUBLICATION = "publication"
    MAINTENANCE = "maintenance"
    ARCHIVAL = "archival"
    DELETION = "deletion"

class ContentType(Enum):
    """Content type enumeration"""
    ARTICLE = "article"
    BLOG_POST = "blog_post"
    DOCUMENT = "document"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    PRESENTATION = "presentation"
    SPREADSHEET = "spreadsheet"
    CODE = "code"
    CONFIGURATION = "configuration"

class MigrationType(Enum):
    """Migration type enumeration"""
    FORMAT_CHANGE = "format_change"
    PLATFORM_MIGRATION = "platform_migration"
    VERSION_UPGRADE = "version_upgrade"
    STRUCTURE_REORGANIZATION = "structure_reorganization"
    ENCODING_CHANGE = "encoding_change"

@dataclass
class ContentVersion:
    """Content version data structure"""
    version_id: str
    content_id: str
    version_number: str
    content_data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = ""
    change_summary: str = ""
    file_size: int = 0
    checksum: str = ""

@dataclass
class ContentLifecycle:
    """Content lifecycle data structure"""
    lifecycle_id: str
    content_id: str
    current_stage: LifecycleStage
    status: ContentStatus
    stages_history: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    archived_at: Optional[datetime] = None
    deleted_at: Optional[datetime] = None

@dataclass
class ContentDependency:
    """Content dependency data structure"""
    dependency_id: str
    source_content_id: str
    target_content_id: str
    dependency_type: str  # references, includes, links_to, depends_on
    strength: float = 1.0  # 0.0 to 1.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    is_active: bool = True

@dataclass
class ContentMigration:
    """Content migration data structure"""
    migration_id: str
    content_id: str
    migration_type: MigrationType
    source_format: str
    target_format: str
    migration_rules: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"  # pending, in_progress, completed, failed
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: str = ""
    backup_location: str = ""

@dataclass
class ContentBackup:
    """Content backup data structure"""
    backup_id: str
    content_id: str
    backup_type: str  # full, incremental, differential
    backup_location: str
    file_size: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    is_encrypted: bool = True
    checksum: str = ""

@dataclass
class LifecycleRule:
    """Lifecycle rule data structure"""
    rule_id: str
    name: str
    description: str
    content_type: ContentType
    conditions: Dict[str, Any] = field(default_factory=dict)
    actions: List[Dict[str, Any]] = field(default_factory=list)
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = ""

class ContentLifecycleEngine:
    """
    Advanced Content Lifecycle Engine
    
    Provides comprehensive content lifecycle management capabilities
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the Content Lifecycle Engine"""
        self.config = config
        self.content_versions = {}
        self.content_lifecycles = {}
        self.content_dependencies = {}
        self.content_migrations = {}
        self.content_backups = {}
        self.lifecycle_rules = {}
        self.redis_client = None
        self.database_engine = None
        self.scheduler = None
        
        # Initialize components
        self._initialize_database()
        self._initialize_redis()
        self._initialize_scheduler()
        self._initialize_storage()
        
        # Start background tasks
        self._start_background_tasks()
        
        logger.info("Content Lifecycle Engine initialized successfully")
    
    def _initialize_database(self):
        """Initialize database connection"""
        try:
            if self.config.get("database_url"):
                self.database_engine = create_engine(self.config["database_url"])
                logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
    
    def _initialize_redis(self):
        """Initialize Redis connection"""
        try:
            if self.config.get("redis_url"):
                self.redis_client = redis.Redis.from_url(self.config["redis_url"])
                logger.info("Redis connection established")
        except Exception as e:
            logger.error(f"Error initializing Redis: {e}")
    
    def _initialize_scheduler(self):
        """Initialize task scheduler"""
        try:
            self.scheduler = schedule
            logger.info("Task scheduler initialized")
        except Exception as e:
            logger.error(f"Error initializing scheduler: {e}")
    
    def _initialize_storage(self):
        """Initialize storage backends"""
        try:
            # Initialize AWS S3 if configured
            if self.config.get("aws_access_key_id"):
                self.s3_client = boto3.client(
                    's3',
                    aws_access_key_id=self.config["aws_access_key_id"],
                    aws_secret_access_key=self.config["aws_secret_access_key"],
                    region_name=self.config.get("aws_region", "us-east-1")
                )
                logger.info("AWS S3 client initialized")
            
            # Initialize Google Cloud Storage if configured
            if self.config.get("gcp_project_id"):
                self.gcs_client = storage.Client(project=self.config["gcp_project_id"])
                logger.info("Google Cloud Storage client initialized")
            
            # Initialize local storage
            self.local_storage_path = self.config.get("local_storage_path", "./storage")
            os.makedirs(self.local_storage_path, exist_ok=True)
            logger.info("Local storage initialized")
            
        except Exception as e:
            logger.error(f"Error initializing storage: {e}")
    
    def _start_background_tasks(self):
        """Start background tasks"""
        try:
            # Start lifecycle monitoring task
            asyncio.create_task(self._monitor_lifecycle_periodically())
            
            # Start backup task
            asyncio.create_task(self._backup_content_periodically())
            
            # Start cleanup task
            asyncio.create_task(self._cleanup_expired_content_periodically())
            
            # Start migration task
            asyncio.create_task(self._process_migrations_periodically())
            
            logger.info("Background tasks started")
            
        except Exception as e:
            logger.error(f"Error starting background tasks: {e}")
    
    async def create_content_version(self, content_id: str, content_data: Dict[str, Any], 
                                   created_by: str, change_summary: str = "") -> ContentVersion:
        """Create a new content version"""
        try:
            version_id = str(uuid.uuid4())
            
            # Calculate version number
            existing_versions = [v for v in self.content_versions.values() if v.content_id == content_id]
            version_number = f"v{len(existing_versions) + 1}"
            
            # Calculate file size and checksum
            content_str = json.dumps(content_data, sort_keys=True)
            file_size = len(content_str.encode('utf-8'))
            checksum = hashlib.sha256(content_str.encode('utf-8')).hexdigest()
            
            version = ContentVersion(
                version_id=version_id,
                content_id=content_id,
                version_number=version_number,
                content_data=content_data,
                created_by=created_by,
                change_summary=change_summary,
                file_size=file_size,
                checksum=checksum
            )
            
            # Store version
            self.content_versions[version_id] = version
            
            # Update lifecycle
            await self._update_content_lifecycle(content_id, LifecycleStage.DEVELOPMENT, ContentStatus.DRAFT)
            
            logger.info(f"Content version {version_id} created for content {content_id}")
            
            return version
            
        except Exception as e:
            logger.error(f"Error creating content version: {e}")
            raise
    
    async def get_content_history(self, content_id: str) -> List[ContentVersion]:
        """Get content version history"""
        try:
            versions = [v for v in self.content_versions.values() if v.content_id == content_id]
            versions.sort(key=lambda x: x.created_at, reverse=True)
            return versions
            
        except Exception as e:
            logger.error(f"Error getting content history: {e}")
            return []
    
    async def restore_content_version(self, version_id: str, restored_by: str) -> bool:
        """Restore content to a specific version"""
        try:
            if version_id not in self.content_versions:
                raise ValueError(f"Version {version_id} not found")
            
            version = self.content_versions[version_id]
            
            # Create new version with restored content
            restored_version = await self.create_content_version(
                version.content_id,
                version.content_data,
                restored_by,
                f"Restored from version {version.version_number}"
            )
            
            # Update lifecycle
            await self._update_content_lifecycle(version.content_id, LifecycleStage.DEVELOPMENT, ContentStatus.DRAFT)
            
            logger.info(f"Content {version.content_id} restored to version {version.version_number}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error restoring content version: {e}")
            return False
    
    async def create_content_dependency(self, source_content_id: str, target_content_id: str, 
                                      dependency_type: str, strength: float = 1.0) -> ContentDependency:
        """Create content dependency"""
        try:
            dependency_id = str(uuid.uuid4())
            
            dependency = ContentDependency(
                dependency_id=dependency_id,
                source_content_id=source_content_id,
                target_content_id=target_content_id,
                dependency_type=dependency_type,
                strength=strength
            )
            
            # Store dependency
            self.content_dependencies[dependency_id] = dependency
            
            logger.info(f"Content dependency {dependency_id} created: {source_content_id} -> {target_content_id}")
            
            return dependency
            
        except Exception as e:
            logger.error(f"Error creating content dependency: {e}")
            raise
    
    async def get_content_dependencies(self, content_id: str, dependency_type: str = None) -> List[ContentDependency]:
        """Get content dependencies"""
        try:
            dependencies = []
            
            for dep in self.content_dependencies.values():
                if dep.is_active:
                    if dep.source_content_id == content_id:
                        if dependency_type is None or dep.dependency_type == dependency_type:
                            dependencies.append(dep)
                    elif dep.target_content_id == content_id:
                        if dependency_type is None or dep.dependency_type == dependency_type:
                            dependencies.append(dep)
            
            return dependencies
            
        except Exception as e:
            logger.error(f"Error getting content dependencies: {e}")
            return []
    
    async def analyze_dependency_impact(self, content_id: str) -> Dict[str, Any]:
        """Analyze impact of content changes on dependencies"""
        try:
            impact_analysis = {
                "direct_dependencies": [],
                "indirect_dependencies": [],
                "impact_score": 0.0,
                "affected_content": [],
                "recommendations": []
            }
            
            # Get direct dependencies
            direct_deps = await self.get_content_dependencies(content_id)
            impact_analysis["direct_dependencies"] = [
                {
                    "content_id": dep.target_content_id if dep.source_content_id == content_id else dep.source_content_id,
                    "dependency_type": dep.dependency_type,
                    "strength": dep.strength
                }
                for dep in direct_deps
            ]
            
            # Calculate impact score
            if direct_deps:
                impact_analysis["impact_score"] = sum(dep.strength for dep in direct_deps) / len(direct_deps)
            
            # Get affected content
            affected_content = set()
            for dep in direct_deps:
                affected_content.add(dep.target_content_id if dep.source_content_id == content_id else dep.source_content_id)
            
            impact_analysis["affected_content"] = list(affected_content)
            
            # Generate recommendations
            if impact_analysis["impact_score"] > 0.7:
                impact_analysis["recommendations"].append("High impact change - notify all affected content owners")
                impact_analysis["recommendations"].append("Consider staged rollout to minimize disruption")
            elif impact_analysis["impact_score"] > 0.3:
                impact_analysis["recommendations"].append("Medium impact change - review affected content")
                impact_analysis["recommendations"].append("Test changes in staging environment")
            else:
                impact_analysis["recommendations"].append("Low impact change - proceed with normal process")
            
            return impact_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing dependency impact: {e}")
            return {}
    
    async def create_content_migration(self, content_id: str, migration_type: MigrationType, 
                                     source_format: str, target_format: str, 
                                     migration_rules: Dict[str, Any] = None) -> ContentMigration:
        """Create content migration"""
        try:
            migration_id = str(uuid.uuid4())
            
            migration = ContentMigration(
                migration_id=migration_id,
                content_id=content_id,
                migration_type=migration_type,
                source_format=source_format,
                target_format=target_format,
                migration_rules=migration_rules or {}
            )
            
            # Store migration
            self.content_migrations[migration_id] = migration
            
            logger.info(f"Content migration {migration_id} created for content {content_id}")
            
            return migration
            
        except Exception as e:
            logger.error(f"Error creating content migration: {e}")
            raise
    
    async def execute_content_migration(self, migration_id: str) -> bool:
        """Execute content migration"""
        try:
            if migration_id not in self.content_migrations:
                raise ValueError(f"Migration {migration_id} not found")
            
            migration = self.content_migrations[migration_id]
            
            # Update migration status
            migration.status = "in_progress"
            migration.started_at = datetime.utcnow()
            
            # Create backup before migration
            backup = await self._create_content_backup(migration.content_id, "full")
            migration.backup_location = backup.backup_location
            
            # Execute migration based on type
            success = False
            
            if migration.migration_type == MigrationType.FORMAT_CHANGE:
                success = await self._migrate_content_format(migration)
            elif migration.migration_type == MigrationType.PLATFORM_MIGRATION:
                success = await self._migrate_content_platform(migration)
            elif migration.migration_type == MigrationType.VERSION_UPGRADE:
                success = await self._migrate_content_version(migration)
            elif migration.migration_type == MigrationType.STRUCTURE_REORGANIZATION:
                success = await self._migrate_content_structure(migration)
            elif migration.migration_type == MigrationType.ENCODING_CHANGE:
                success = await self._migrate_content_encoding(migration)
            
            # Update migration status
            if success:
                migration.status = "completed"
                migration.completed_at = datetime.utcnow()
                
                # Update content lifecycle
                await self._update_content_lifecycle(migration.content_id, LifecycleStage.MAINTENANCE, ContentStatus.PUBLISHED)
            else:
                migration.status = "failed"
                migration.error_message = "Migration execution failed"
            
            logger.info(f"Content migration {migration_id} {'completed' if success else 'failed'}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error executing content migration: {e}")
            if migration_id in self.content_migrations:
                self.content_migrations[migration_id].status = "failed"
                self.content_migrations[migration_id].error_message = str(e)
            return False
    
    async def _migrate_content_format(self, migration: ContentMigration) -> bool:
        """Migrate content format"""
        try:
            # Get latest content version
            content_versions = await self.get_content_history(migration.content_id)
            if not content_versions:
                return False
            
            latest_version = content_versions[0]
            content_data = latest_version.content_data
            
            # Apply format migration rules
            migrated_data = await self._apply_migration_rules(content_data, migration.migration_rules)
            
            # Create new version with migrated content
            await self.create_content_version(
                migration.content_id,
                migrated_data,
                "system",
                f"Format migration: {migration.source_format} -> {migration.target_format}"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error migrating content format: {e}")
            return False
    
    async def _migrate_content_platform(self, migration: ContentMigration) -> bool:
        """Migrate content platform"""
        try:
            # Platform-specific migration logic would go here
            # This is a placeholder for actual platform migration
            logger.info(f"Platform migration: {migration.source_format} -> {migration.target_format}")
            return True
            
        except Exception as e:
            logger.error(f"Error migrating content platform: {e}")
            return False
    
    async def _migrate_content_version(self, migration: ContentMigration) -> bool:
        """Migrate content version"""
        try:
            # Version upgrade logic would go here
            logger.info(f"Version migration: {migration.source_format} -> {migration.target_format}")
            return True
            
        except Exception as e:
            logger.error(f"Error migrating content version: {e}")
            return False
    
    async def _migrate_content_structure(self, migration: ContentMigration) -> bool:
        """Migrate content structure"""
        try:
            # Structure reorganization logic would go here
            logger.info(f"Structure migration: {migration.source_format} -> {migration.target_format}")
            return True
            
        except Exception as e:
            logger.error(f"Error migrating content structure: {e}")
            return False
    
    async def _migrate_content_encoding(self, migration: ContentMigration) -> bool:
        """Migrate content encoding"""
        try:
            # Encoding change logic would go here
            logger.info(f"Encoding migration: {migration.source_format} -> {migration.target_format}")
            return True
            
        except Exception as e:
            logger.error(f"Error migrating content encoding: {e}")
            return False
    
    async def _apply_migration_rules(self, content_data: Dict[str, Any], rules: Dict[str, Any]) -> Dict[str, Any]:
        """Apply migration rules to content data"""
        try:
            migrated_data = content_data.copy()
            
            # Apply field mapping rules
            if "field_mapping" in rules:
                for old_field, new_field in rules["field_mapping"].items():
                    if old_field in migrated_data:
                        migrated_data[new_field] = migrated_data.pop(old_field)
            
            # Apply transformation rules
            if "transformations" in rules:
                for field, transformation in rules["transformations"].items():
                    if field in migrated_data:
                        if transformation["type"] == "format_change":
                            # Apply format change
                            pass
                        elif transformation["type"] == "value_mapping":
                            # Apply value mapping
                            if migrated_data[field] in transformation["mapping"]:
                                migrated_data[field] = transformation["mapping"][migrated_data[field]]
            
            return migrated_data
            
        except Exception as e:
            logger.error(f"Error applying migration rules: {e}")
            return content_data
    
    async def create_content_backup(self, content_id: str, backup_type: str = "full") -> ContentBackup:
        """Create content backup"""
        try:
            backup = await self._create_content_backup(content_id, backup_type)
            return backup
            
        except Exception as e:
            logger.error(f"Error creating content backup: {e}")
            raise
    
    async def _create_content_backup(self, content_id: str, backup_type: str) -> ContentBackup:
        """Internal method to create content backup"""
        try:
            backup_id = str(uuid.uuid4())
            
            # Get all content versions
            versions = await self.get_content_history(content_id)
            if not versions:
                raise ValueError(f"No versions found for content {content_id}")
            
            # Create backup data
            backup_data = {
                "content_id": content_id,
                "backup_type": backup_type,
                "versions": [v.__dict__ for v in versions],
                "created_at": datetime.utcnow().isoformat()
            }
            
            # Save backup to storage
            backup_location = await self._save_backup_to_storage(backup_id, backup_data)
            
            # Calculate file size and checksum
            backup_str = json.dumps(backup_data, sort_keys=True)
            file_size = len(backup_str.encode('utf-8'))
            checksum = hashlib.sha256(backup_str.encode('utf-8')).hexdigest()
            
            # Set expiration date
            expires_at = datetime.utcnow() + timedelta(days=365)  # 1 year retention
            
            backup = ContentBackup(
                backup_id=backup_id,
                content_id=content_id,
                backup_type=backup_type,
                backup_location=backup_location,
                file_size=file_size,
                expires_at=expires_at,
                checksum=checksum
            )
            
            # Store backup
            self.content_backups[backup_id] = backup
            
            logger.info(f"Content backup {backup_id} created for content {content_id}")
            
            return backup
            
        except Exception as e:
            logger.error(f"Error creating content backup: {e}")
            raise
    
    async def _save_backup_to_storage(self, backup_id: str, backup_data: Dict[str, Any]) -> str:
        """Save backup to storage backend"""
        try:
            # Save to local storage
            local_path = os.path.join(self.local_storage_path, f"{backup_id}.json")
            with open(local_path, 'w') as f:
                json.dump(backup_data, f, indent=2)
            
            # Save to cloud storage if configured
            if hasattr(self, 's3_client'):
                s3_key = f"backups/{backup_id}.json"
                self.s3_client.put_object(
                    Bucket=self.config.get("s3_bucket", "content-backups"),
                    Key=s3_key,
                    Body=json.dumps(backup_data, indent=2)
                )
                return f"s3://{self.config.get('s3_bucket', 'content-backups')}/{s3_key}"
            
            return local_path
            
        except Exception as e:
            logger.error(f"Error saving backup to storage: {e}")
            return ""
    
    async def restore_content_from_backup(self, backup_id: str, restored_by: str) -> bool:
        """Restore content from backup"""
        try:
            if backup_id not in self.content_backups:
                raise ValueError(f"Backup {backup_id} not found")
            
            backup = self.content_backups[backup_id]
            
            # Load backup data
            backup_data = await self._load_backup_from_storage(backup.backup_location)
            if not backup_data:
                return False
            
            # Restore content versions
            for version_data in backup_data["versions"]:
                version = ContentVersion(**version_data)
                self.content_versions[version.version_id] = version
            
            # Update content lifecycle
            await self._update_content_lifecycle(backup.content_id, LifecycleStage.MAINTENANCE, ContentStatus.PUBLISHED)
            
            logger.info(f"Content {backup.content_id} restored from backup {backup_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error restoring content from backup: {e}")
            return False
    
    async def _load_backup_from_storage(self, backup_location: str) -> Optional[Dict[str, Any]]:
        """Load backup from storage backend"""
        try:
            if backup_location.startswith("s3://"):
                # Load from S3
                bucket, key = backup_location[5:].split("/", 1)
                response = self.s3_client.get_object(Bucket=bucket, Key=key)
                return json.loads(response['Body'].read().decode('utf-8'))
            else:
                # Load from local storage
                with open(backup_location, 'r') as f:
                    return json.load(f)
            
        except Exception as e:
            logger.error(f"Error loading backup from storage: {e}")
            return None
    
    async def create_lifecycle_rule(self, rule_data: Dict[str, Any]) -> LifecycleRule:
        """Create lifecycle rule"""
        try:
            rule_id = str(uuid.uuid4())
            
            rule = LifecycleRule(
                rule_id=rule_id,
                name=rule_data["name"],
                description=rule_data["description"],
                content_type=ContentType(rule_data["content_type"]),
                conditions=rule_data.get("conditions", {}),
                actions=rule_data.get("actions", []),
                created_by=rule_data.get("created_by", "system")
            )
            
            # Store rule
            self.lifecycle_rules[rule_id] = rule
            
            logger.info(f"Lifecycle rule {rule_id} created: {rule.name}")
            
            return rule
            
        except Exception as e:
            logger.error(f"Error creating lifecycle rule: {e}")
            raise
    
    async def apply_lifecycle_rules(self, content_id: str, content_type: ContentType) -> List[Dict[str, Any]]:
        """Apply lifecycle rules to content"""
        try:
            applied_actions = []
            
            # Get applicable rules
            applicable_rules = [
                rule for rule in self.lifecycle_rules.values()
                if rule.is_active and rule.content_type == content_type
            ]
            
            for rule in applicable_rules:
                # Check if conditions are met
                if await self._evaluate_rule_conditions(content_id, rule.conditions):
                    # Execute actions
                    for action in rule.actions:
                        result = await self._execute_lifecycle_action(content_id, action)
                        applied_actions.append({
                            "rule_id": rule.rule_id,
                            "rule_name": rule.name,
                            "action": action,
                            "result": result
                        })
            
            return applied_actions
            
        except Exception as e:
            logger.error(f"Error applying lifecycle rules: {e}")
            return []
    
    async def _evaluate_rule_conditions(self, content_id: str, conditions: Dict[str, Any]) -> bool:
        """Evaluate lifecycle rule conditions"""
        try:
            # Get content lifecycle
            lifecycle = self.content_lifecycles.get(content_id)
            if not lifecycle:
                return False
            
            # Check age condition
            if "max_age_days" in conditions:
                age_days = (datetime.utcnow() - lifecycle.created_at).days
                if age_days < conditions["max_age_days"]:
                    return False
            
            # Check status condition
            if "status" in conditions:
                if lifecycle.status.value != conditions["status"]:
                    return False
            
            # Check stage condition
            if "stage" in conditions:
                if lifecycle.current_stage.value != conditions["stage"]:
                    return False
            
            # Check access condition
            if "last_accessed_days" in conditions:
                # This would require access tracking
                pass
            
            return True
            
        except Exception as e:
            logger.error(f"Error evaluating rule conditions: {e}")
            return False
    
    async def _execute_lifecycle_action(self, content_id: str, action: Dict[str, Any]) -> bool:
        """Execute lifecycle action"""
        try:
            action_type = action.get("type")
            
            if action_type == "archive":
                return await self._archive_content(content_id)
            elif action_type == "delete":
                return await self._delete_content(content_id)
            elif action_type == "backup":
                return await self._create_content_backup(content_id, "full") is not None
            elif action_type == "migrate":
                return await self._migrate_content_automatically(content_id, action.get("migration_type"))
            elif action_type == "notify":
                return await self._notify_content_owners(content_id, action.get("message"))
            
            return False
            
        except Exception as e:
            logger.error(f"Error executing lifecycle action: {e}")
            return False
    
    async def _archive_content(self, content_id: str) -> bool:
        """Archive content"""
        try:
            await self._update_content_lifecycle(content_id, LifecycleStage.ARCHIVAL, ContentStatus.ARCHIVED)
            logger.info(f"Content {content_id} archived")
            return True
            
        except Exception as e:
            logger.error(f"Error archiving content: {e}")
            return False
    
    async def _delete_content(self, content_id: str) -> bool:
        """Delete content"""
        try:
            await self._update_content_lifecycle(content_id, LifecycleStage.DELETION, ContentStatus.DELETED)
            logger.info(f"Content {content_id} deleted")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting content: {e}")
            return False
    
    async def _migrate_content_automatically(self, content_id: str, migration_type: str) -> bool:
        """Automatically migrate content"""
        try:
            migration = await self.create_content_migration(
                content_id,
                MigrationType(migration_type),
                "auto",
                "auto"
            )
            
            return await self.execute_content_migration(migration.migration_id)
            
        except Exception as e:
            logger.error(f"Error automatically migrating content: {e}")
            return False
    
    async def _notify_content_owners(self, content_id: str, message: str) -> bool:
        """Notify content owners"""
        try:
            # This would integrate with notification system
            logger.info(f"Notification sent for content {content_id}: {message}")
            return True
            
        except Exception as e:
            logger.error(f"Error notifying content owners: {e}")
            return False
    
    async def _update_content_lifecycle(self, content_id: str, stage: LifecycleStage, status: ContentStatus):
        """Update content lifecycle"""
        try:
            if content_id not in self.content_lifecycles:
                lifecycle = ContentLifecycle(
                    lifecycle_id=str(uuid.uuid4()),
                    content_id=content_id,
                    current_stage=stage,
                    status=status
                )
                self.content_lifecycles[content_id] = lifecycle
            else:
                lifecycle = self.content_lifecycles[content_id]
                lifecycle.current_stage = stage
                lifecycle.status = status
                lifecycle.updated_at = datetime.utcnow()
            
            # Add to stages history
            lifecycle.stages_history.append({
                "stage": stage.value,
                "status": status.value,
                "timestamp": datetime.utcnow().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error updating content lifecycle: {e}")
    
    async def get_content_lifecycle_analytics(self, time_period: str = "30d") -> Dict[str, Any]:
        """Get content lifecycle analytics"""
        try:
            # Calculate time range
            end_date = datetime.utcnow()
            if time_period == "7d":
                start_date = end_date - timedelta(days=7)
            elif time_period == "30d":
                start_date = end_date - timedelta(days=30)
            elif time_period == "90d":
                start_date = end_date - timedelta(days=90)
            else:
                start_date = end_date - timedelta(days=30)
            
            analytics = {
                "time_period": time_period,
                "total_content": len(self.content_lifecycles),
                "content_by_status": {},
                "content_by_stage": {},
                "lifecycle_metrics": {},
                "trends": {}
            }
            
            # Content by status
            for lifecycle in self.content_lifecycles.values():
                status = lifecycle.status.value
                analytics["content_by_status"][status] = analytics["content_by_status"].get(status, 0) + 1
            
            # Content by stage
            for lifecycle in self.content_lifecycles.values():
                stage = lifecycle.current_stage.value
                analytics["content_by_stage"][stage] = analytics["content_by_stage"].get(stage, 0) + 1
            
            # Lifecycle metrics
            analytics["lifecycle_metrics"] = {
                "average_lifecycle_duration": await self._calculate_average_lifecycle_duration(),
                "content_creation_rate": await self._calculate_content_creation_rate(start_date, end_date),
                "content_archival_rate": await self._calculate_content_archival_rate(start_date, end_date),
                "content_deletion_rate": await self._calculate_content_deletion_rate(start_date, end_date)
            }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting content lifecycle analytics: {e}")
            return {}
    
    async def _calculate_average_lifecycle_duration(self) -> float:
        """Calculate average lifecycle duration"""
        try:
            durations = []
            for lifecycle in self.content_lifecycles.values():
                if lifecycle.archived_at or lifecycle.deleted_at:
                    end_date = lifecycle.archived_at or lifecycle.deleted_at
                    duration = (end_date - lifecycle.created_at).days
                    durations.append(duration)
            
            return np.mean(durations) if durations else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating average lifecycle duration: {e}")
            return 0.0
    
    async def _calculate_content_creation_rate(self, start_date: datetime, end_date: datetime) -> float:
        """Calculate content creation rate"""
        try:
            created_content = [
                lifecycle for lifecycle in self.content_lifecycles.values()
                if start_date <= lifecycle.created_at <= end_date
            ]
            
            days = (end_date - start_date).days
            return len(created_content) / days if days > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating content creation rate: {e}")
            return 0.0
    
    async def _calculate_content_archival_rate(self, start_date: datetime, end_date: datetime) -> float:
        """Calculate content archival rate"""
        try:
            archived_content = [
                lifecycle for lifecycle in self.content_lifecycles.values()
                if lifecycle.archived_at and start_date <= lifecycle.archived_at <= end_date
            ]
            
            days = (end_date - start_date).days
            return len(archived_content) / days if days > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating content archival rate: {e}")
            return 0.0
    
    async def _calculate_content_deletion_rate(self, start_date: datetime, end_date: datetime) -> float:
        """Calculate content deletion rate"""
        try:
            deleted_content = [
                lifecycle for lifecycle in self.content_lifecycles.values()
                if lifecycle.deleted_at and start_date <= lifecycle.deleted_at <= end_date
            ]
            
            days = (end_date - start_date).days
            return len(deleted_content) / days if days > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating content deletion rate: {e}")
            return 0.0
    
    async def _monitor_lifecycle_periodically(self):
        """Monitor content lifecycle periodically"""
        while True:
            try:
                await asyncio.sleep(3600)  # Monitor every hour
                
                # Apply lifecycle rules to all content
                for content_id, lifecycle in self.content_lifecycles.items():
                    # Determine content type (this would be stored in content metadata)
                    content_type = ContentType.ARTICLE  # Placeholder
                    
                    await self.apply_lifecycle_rules(content_id, content_type)
                
                logger.info("Content lifecycle monitoring completed")
                
            except Exception as e:
                logger.error(f"Error in lifecycle monitoring: {e}")
                await asyncio.sleep(3600)
    
    async def _backup_content_periodically(self):
        """Backup content periodically"""
        while True:
            try:
                await asyncio.sleep(86400)  # Backup daily
                
                # Backup content that hasn't been backed up recently
                for content_id in self.content_lifecycles.keys():
                    recent_backups = [
                        backup for backup in self.content_backups.values()
                        if backup.content_id == content_id and 
                        (datetime.utcnow() - backup.created_at).days < 7
                    ]
                    
                    if not recent_backups:
                        await self._create_content_backup(content_id, "full")
                
                logger.info("Content backup completed")
                
            except Exception as e:
                logger.error(f"Error in content backup: {e}")
                await asyncio.sleep(86400)
    
    async def _cleanup_expired_content_periodically(self):
        """Cleanup expired content periodically"""
        while True:
            try:
                await asyncio.sleep(86400)  # Cleanup daily
                
                # Cleanup expired backups
                expired_backups = [
                    backup for backup in self.content_backups.values()
                    if backup.expires_at and backup.expires_at < datetime.utcnow()
                ]
                
                for backup in expired_backups:
                    await self._delete_backup(backup.backup_id)
                
                logger.info("Expired content cleanup completed")
                
            except Exception as e:
                logger.error(f"Error in content cleanup: {e}")
                await asyncio.sleep(86400)
    
    async def _process_migrations_periodically(self):
        """Process pending migrations periodically"""
        while True:
            try:
                await asyncio.sleep(1800)  # Process every 30 minutes
                
                # Process pending migrations
                pending_migrations = [
                    migration for migration in self.content_migrations.values()
                    if migration.status == "pending"
                ]
                
                for migration in pending_migrations:
                    await self.execute_content_migration(migration.migration_id)
                
                logger.info("Content migration processing completed")
                
            except Exception as e:
                logger.error(f"Error in migration processing: {e}")
                await asyncio.sleep(1800)
    
    async def _delete_backup(self, backup_id: str) -> bool:
        """Delete backup"""
        try:
            if backup_id not in self.content_backups:
                return False
            
            backup = self.content_backups[backup_id]
            
            # Delete from storage
            if backup.backup_location.startswith("s3://"):
                bucket, key = backup.backup_location[5:].split("/", 1)
                self.s3_client.delete_object(Bucket=bucket, Key=key)
            else:
                if os.path.exists(backup.backup_location):
                    os.remove(backup.backup_location)
            
            # Remove from memory
            del self.content_backups[backup_id]
            
            logger.info(f"Backup {backup_id} deleted")
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting backup: {e}")
            return False

# Example usage and testing
async def main():
    """Example usage of the Content Lifecycle Engine"""
    try:
        # Initialize engine
        config = {
            "database_url": "postgresql://user:password@localhost/lifecycledb",
            "redis_url": "redis://localhost:6379",
            "local_storage_path": "./storage"
        }
        
        engine = ContentLifecycleEngine(config)
        
        # Create content version
        print("Creating content version...")
        content_data = {
            "title": "Sample Article",
            "content": "This is a sample article content.",
            "author": "John Doe",
            "tags": ["sample", "article"]
        }
        
        version = await engine.create_content_version(
            "content_001",
            content_data,
            "author1",
            "Initial version"
        )
        print(f"Content version created: {version.version_id}")
        
        # Get content history
        print("Getting content history...")
        history = await engine.get_content_history("content_001")
        print(f"Content history: {len(history)} versions")
        
        # Create content dependency
        print("Creating content dependency...")
        dependency = await engine.create_content_dependency(
            "content_001",
            "content_002",
            "references",
            0.8
        )
        print(f"Content dependency created: {dependency.dependency_id}")
        
        # Analyze dependency impact
        print("Analyzing dependency impact...")
        impact = await engine.analyze_dependency_impact("content_001")
        print(f"Impact score: {impact['impact_score']}")
        print(f"Affected content: {impact['affected_content']}")
        
        # Create content migration
        print("Creating content migration...")
        migration = await engine.create_content_migration(
            "content_001",
            MigrationType.FORMAT_CHANGE,
            "json",
            "xml",
            {"field_mapping": {"title": "headline", "content": "body"}}
        )
        print(f"Content migration created: {migration.migration_id}")
        
        # Execute migration
        print("Executing content migration...")
        success = await engine.execute_content_migration(migration.migration_id)
        print(f"Migration {'successful' if success else 'failed'}")
        
        # Create content backup
        print("Creating content backup...")
        backup = await engine.create_content_backup("content_001", "full")
        print(f"Content backup created: {backup.backup_id}")
        
        # Create lifecycle rule
        print("Creating lifecycle rule...")
        rule = await engine.create_lifecycle_rule({
            "name": "Auto Archive Rule",
            "description": "Automatically archive content after 90 days",
            "content_type": "article",
            "conditions": {"max_age_days": 90, "status": "published"},
            "actions": [{"type": "archive"}],
            "created_by": "admin"
        })
        print(f"Lifecycle rule created: {rule.rule_id}")
        
        # Get lifecycle analytics
        print("Getting lifecycle analytics...")
        analytics = await engine.get_content_lifecycle_analytics("30d")
        print(f"Total content: {analytics['total_content']}")
        print(f"Content by status: {analytics['content_by_status']}")
        
        print("\nContent Lifecycle Engine demo completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main demo: {e}")

if __name__ == "__main__":
    asyncio.run(main())
























