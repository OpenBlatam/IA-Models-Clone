"""
Advanced Content Management System for OpusClip Improved
=====================================================

Comprehensive content management with versioning, collaboration, and workflow automation.
"""

import asyncio
import logging
import json
import hashlib
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from uuid import UUID, uuid4
from pathlib import Path
import aiofiles
import magic
from PIL import Image
import cv2
import numpy as np

from .schemas import get_settings
from .exceptions import ContentManagementError, create_content_management_error

logger = logging.getLogger(__name__)


class ContentType(str, Enum):
    """Content types"""
    VIDEO = "video"
    AUDIO = "audio"
    IMAGE = "image"
    DOCUMENT = "document"
    TEXT = "text"
    ASSET = "asset"


class ContentStatus(str, Enum):
    """Content status"""
    DRAFT = "draft"
    REVIEW = "review"
    APPROVED = "approved"
    PUBLISHED = "published"
    ARCHIVED = "archived"
    DELETED = "deleted"


class ApprovalStatus(str, Enum):
    """Approval status"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    CHANGES_REQUESTED = "changes_requested"


class WorkflowStage(str, Enum):
    """Workflow stages"""
    CREATION = "creation"
    REVIEW = "review"
    APPROVAL = "approval"
    PUBLISHING = "publishing"
    ARCHIVING = "archiving"


@dataclass
class ContentMetadata:
    """Content metadata"""
    content_id: str
    title: str
    description: str
    content_type: ContentType
    file_path: str
    file_size: int
    mime_type: str
    duration: Optional[float] = None
    resolution: Optional[str] = None
    fps: Optional[float] = None
    bitrate: Optional[int] = None
    codec: Optional[str] = None
    thumbnail_path: Optional[str] = None
    tags: List[str] = None
    categories: List[str] = None
    created_at: datetime = None
    updated_at: datetime = None
    created_by: str = None
    updated_by: str = None


@dataclass
class ContentVersion:
    """Content version"""
    version_id: str
    content_id: str
    version_number: int
    file_path: str
    file_hash: str
    changes_description: str
    created_at: datetime
    created_by: str
    is_current: bool = False


@dataclass
class ContentWorkflow:
    """Content workflow"""
    workflow_id: str
    content_id: str
    current_stage: WorkflowStage
    stages: List[Dict[str, Any]]
    approvers: List[str]
    reviewers: List[str]
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None


@dataclass
class ContentApproval:
    """Content approval"""
    approval_id: str
    content_id: str
    approver_id: str
    status: ApprovalStatus
    comments: str
    created_at: datetime
    updated_at: datetime


class ContentAnalyzer:
    """Advanced content analysis"""
    
    def __init__(self):
        self.settings = get_settings()
    
    async def analyze_video_content(self, file_path: str) -> Dict[str, Any]:
        """Analyze video content"""
        try:
            # Open video file
            cap = cv2.VideoCapture(file_path)
            
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {file_path}")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            # Analyze frames
            brightness_values = []
            contrast_values = []
            motion_values = []
            
            prev_frame = None
            frame_idx = 0
            sample_rate = max(1, frame_count // 100)  # Sample every 1% of frames
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % sample_rate == 0:
                    # Convert to grayscale for analysis
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    # Calculate brightness
                    brightness = np.mean(gray)
                    brightness_values.append(brightness)
                    
                    # Calculate contrast
                    contrast = np.std(gray)
                    contrast_values.append(contrast)
                    
                    # Calculate motion (if previous frame exists)
                    if prev_frame is not None:
                        diff = cv2.absdiff(gray, prev_frame)
                        motion = np.mean(diff)
                        motion_values.append(motion)
                    
                    prev_frame = gray
                
                frame_idx += 1
            
            cap.release()
            
            # Calculate statistics
            analysis = {
                "duration": duration,
                "fps": fps,
                "resolution": f"{width}x{height}",
                "frame_count": frame_count,
                "brightness": {
                    "mean": np.mean(brightness_values) if brightness_values else 0,
                    "std": np.std(brightness_values) if brightness_values else 0,
                    "min": np.min(brightness_values) if brightness_values else 0,
                    "max": np.max(brightness_values) if brightness_values else 0
                },
                "contrast": {
                    "mean": np.mean(contrast_values) if contrast_values else 0,
                    "std": np.std(contrast_values) if contrast_values else 0,
                    "min": np.min(contrast_values) if contrast_values else 0,
                    "max": np.max(contrast_values) if contrast_values else 0
                },
                "motion": {
                    "mean": np.mean(motion_values) if motion_values else 0,
                    "std": np.std(motion_values) if motion_values else 0,
                    "min": np.min(motion_values) if motion_values else 0,
                    "max": np.max(motion_values) if motion_values else 0
                }
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Video content analysis failed: {e}")
            raise create_content_management_error("video_analysis", file_path, e)
    
    async def analyze_image_content(self, file_path: str) -> Dict[str, Any]:
        """Analyze image content"""
        try:
            # Open image
            image = Image.open(file_path)
            
            # Get basic properties
            width, height = image.size
            mode = image.mode
            format_name = image.format
            
            # Convert to RGB for analysis
            if mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to numpy array
            img_array = np.array(image)
            
            # Calculate color statistics
            r_mean, g_mean, b_mean = np.mean(img_array, axis=(0, 1))
            r_std, g_std, b_std = np.std(img_array, axis=(0, 1))
            
            # Calculate brightness and contrast
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            brightness = np.mean(gray)
            contrast = np.std(gray)
            
            # Detect edges
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (width * height)
            
            analysis = {
                "width": width,
                "height": height,
                "mode": mode,
                "format": format_name,
                "colors": {
                    "red": {"mean": r_mean, "std": r_std},
                    "green": {"mean": g_mean, "std": g_std},
                    "blue": {"mean": b_mean, "std": b_std}
                },
                "brightness": brightness,
                "contrast": contrast,
                "edge_density": edge_density
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Image content analysis failed: {e}")
            raise create_content_management_error("image_analysis", file_path, e)
    
    async def generate_thumbnail(self, file_path: str, output_path: str, 
                               size: Tuple[int, int] = (320, 240)) -> str:
        """Generate thumbnail for content"""
        try:
            if file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                # Video thumbnail
                cap = cv2.VideoCapture(file_path)
                ret, frame = cap.read()
                cap.release()
                
                if ret:
                    # Resize frame
                    frame = cv2.resize(frame, size)
                    cv2.imwrite(output_path, frame)
                else:
                    raise ValueError("Could not read video frame")
            
            elif file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                # Image thumbnail
                image = Image.open(file_path)
                image.thumbnail(size, Image.Resampling.LANCZOS)
                image.save(output_path)
            
            else:
                raise ValueError(f"Unsupported file type for thumbnail generation: {file_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Thumbnail generation failed: {e}")
            raise create_content_management_error("thumbnail_generation", file_path, e)
    
    async def calculate_file_hash(self, file_path: str) -> str:
        """Calculate file hash"""
        try:
            hash_md5 = hashlib.md5()
            async with aiofiles.open(file_path, "rb") as f:
                while chunk := await f.read(8192):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"File hash calculation failed: {e}")
            raise create_content_management_error("hash_calculation", file_path, e)


class ContentVersionManager:
    """Content version management"""
    
    def __init__(self):
        self.versions: Dict[str, List[ContentVersion]] = {}
        self.current_versions: Dict[str, str] = {}
    
    def create_version(self, content_id: str, file_path: str, 
                      changes_description: str, created_by: str) -> ContentVersion:
        """Create new content version"""
        try:
            # Get current version number
            current_version = self.get_current_version(content_id)
            version_number = current_version.version_number + 1 if current_version else 1
            
            # Calculate file hash
            analyzer = ContentAnalyzer()
            file_hash = asyncio.run(analyzer.calculate_file_hash(file_path))
            
            # Create version
            version = ContentVersion(
                version_id=str(uuid4()),
                content_id=content_id,
                version_number=version_number,
                file_path=file_path,
                file_hash=file_hash,
                changes_description=changes_description,
                created_at=datetime.utcnow(),
                created_by=created_by,
                is_current=True
            )
            
            # Update current version
            if content_id in self.current_versions:
                old_version_id = self.current_versions[content_id]
                old_version = self.get_version(content_id, old_version_id)
                if old_version:
                    old_version.is_current = False
            
            # Add to versions
            if content_id not in self.versions:
                self.versions[content_id] = []
            
            self.versions[content_id].append(version)
            self.current_versions[content_id] = version.version_id
            
            logger.info(f"Created version {version_number} for content {content_id}")
            return version
            
        except Exception as e:
            logger.error(f"Version creation failed: {e}")
            raise create_content_management_error("version_creation", content_id, e)
    
    def get_current_version(self, content_id: str) -> Optional[ContentVersion]:
        """Get current version of content"""
        if content_id in self.current_versions:
            version_id = self.current_versions[content_id]
            return self.get_version(content_id, version_id)
        return None
    
    def get_version(self, content_id: str, version_id: str) -> Optional[ContentVersion]:
        """Get specific version"""
        if content_id in self.versions:
            for version in self.versions[content_id]:
                if version.version_id == version_id:
                    return version
        return None
    
    def get_version_history(self, content_id: str) -> List[ContentVersion]:
        """Get version history"""
        if content_id in self.versions:
            return sorted(self.versions[content_id], key=lambda v: v.version_number, reverse=True)
        return []
    
    def rollback_to_version(self, content_id: str, version_id: str) -> bool:
        """Rollback to specific version"""
        try:
            version = self.get_version(content_id, version_id)
            if not version:
                raise ValueError(f"Version {version_id} not found")
            
            # Update current version
            current_version = self.get_current_version(content_id)
            if current_version:
                current_version.is_current = False
            
            version.is_current = True
            self.current_versions[content_id] = version.version_id
            
            logger.info(f"Rolled back content {content_id} to version {version.version_number}")
            return True
            
        except Exception as e:
            logger.error(f"Version rollback failed: {e}")
            raise create_content_management_error("version_rollback", content_id, e)
    
    def delete_version(self, content_id: str, version_id: str) -> bool:
        """Delete specific version"""
        try:
            if content_id in self.versions:
                self.versions[content_id] = [
                    v for v in self.versions[content_id] 
                    if v.version_id != version_id
                ]
                
                # Update current version if deleted version was current
                if (content_id in self.current_versions and 
                    self.current_versions[content_id] == version_id):
                    
                    if self.versions[content_id]:
                        # Set latest version as current
                        latest_version = max(self.versions[content_id], 
                                           key=lambda v: v.version_number)
                        latest_version.is_current = True
                        self.current_versions[content_id] = latest_version.version_id
                    else:
                        # No versions left
                        del self.current_versions[content_id]
                
                logger.info(f"Deleted version {version_id} for content {content_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Version deletion failed: {e}")
            raise create_content_management_error("version_deletion", content_id, e)


class ContentWorkflowManager:
    """Content workflow management"""
    
    def __init__(self):
        self.workflows: Dict[str, ContentWorkflow] = {}
        self.approvals: Dict[str, List[ContentApproval]] = {}
    
    def create_workflow(self, content_id: str, stages: List[Dict[str, Any]], 
                       approvers: List[str], reviewers: List[str]) -> ContentWorkflow:
        """Create content workflow"""
        try:
            workflow = ContentWorkflow(
                workflow_id=str(uuid4()),
                content_id=content_id,
                current_stage=WorkflowStage.CREATION,
                stages=stages,
                approvers=approvers,
                reviewers=reviewers,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            self.workflows[content_id] = workflow
            logger.info(f"Created workflow for content {content_id}")
            return workflow
            
        except Exception as e:
            logger.error(f"Workflow creation failed: {e}")
            raise create_content_management_error("workflow_creation", content_id, e)
    
    def get_workflow(self, content_id: str) -> Optional[ContentWorkflow]:
        """Get content workflow"""
        return self.workflows.get(content_id)
    
    def advance_workflow(self, content_id: str, next_stage: WorkflowStage) -> bool:
        """Advance workflow to next stage"""
        try:
            workflow = self.get_workflow(content_id)
            if not workflow:
                raise ValueError(f"No workflow found for content {content_id}")
            
            workflow.current_stage = next_stage
            workflow.updated_at = datetime.utcnow()
            
            # Mark as completed if reaching final stage
            if next_stage == WorkflowStage.ARCHIVING:
                workflow.completed_at = datetime.utcnow()
            
            logger.info(f"Advanced workflow for content {content_id} to {next_stage.value}")
            return True
            
        except Exception as e:
            logger.error(f"Workflow advancement failed: {e}")
            raise create_content_management_error("workflow_advancement", content_id, e)
    
    def create_approval(self, content_id: str, approver_id: str, 
                       status: ApprovalStatus, comments: str = "") -> ContentApproval:
        """Create content approval"""
        try:
            approval = ContentApproval(
                approval_id=str(uuid4()),
                content_id=content_id,
                approver_id=approver_id,
                status=status,
                comments=comments,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            if content_id not in self.approvals:
                self.approvals[content_id] = []
            
            self.approvals[content_id].append(approval)
            logger.info(f"Created approval for content {content_id} by {approver_id}")
            return approval
            
        except Exception as e:
            logger.error(f"Approval creation failed: {e}")
            raise create_content_management_error("approval_creation", content_id, e)
    
    def get_approvals(self, content_id: str) -> List[ContentApproval]:
        """Get content approvals"""
        return self.approvals.get(content_id, [])
    
    def is_approved(self, content_id: str) -> bool:
        """Check if content is approved"""
        approvals = self.get_approvals(content_id)
        if not approvals:
            return False
        
        # Check if all required approvers have approved
        workflow = self.get_workflow(content_id)
        if not workflow:
            return False
        
        required_approvers = set(workflow.approvers)
        approved_by = set(
            approval.approver_id for approval in approvals 
            if approval.status == ApprovalStatus.APPROVED
        )
        
        return required_approvers.issubset(approved_by)


class ContentManager:
    """Main content management system"""
    
    def __init__(self):
        self.settings = get_settings()
        self.content_metadata: Dict[str, ContentMetadata] = {}
        self.content_status: Dict[str, ContentStatus] = {}
        self.analyzer = ContentAnalyzer()
        self.version_manager = ContentVersionManager()
        self.workflow_manager = ContentWorkflowManager()
        
        self._initialize_storage()
    
    def _initialize_storage(self):
        """Initialize content storage directories"""
        try:
            # Create storage directories
            storage_path = Path(self.settings.storage.local_path)
            storage_path.mkdir(parents=True, exist_ok=True)
            
            (storage_path / "content").mkdir(exist_ok=True)
            (storage_path / "thumbnails").mkdir(exist_ok=True)
            (storage_path / "versions").mkdir(exist_ok=True)
            
            logger.info("Content storage initialized")
            
        except Exception as e:
            logger.error(f"Storage initialization failed: {e}")
            raise create_content_management_error("storage_init", "content", e)
    
    async def create_content(self, file_path: str, title: str, description: str,
                           content_type: ContentType, created_by: str,
                           tags: List[str] = None, categories: List[str] = None) -> ContentMetadata:
        """Create new content"""
        try:
            content_id = str(uuid4())
            
            # Analyze content
            if content_type == ContentType.VIDEO:
                analysis = await self.analyzer.analyze_video_content(file_path)
            elif content_type == ContentType.IMAGE:
                analysis = await self.analyzer.analyze_image_content(file_path)
            else:
                analysis = {}
            
            # Generate thumbnail
            thumbnail_path = None
            if content_type in [ContentType.VIDEO, ContentType.IMAGE]:
                thumbnail_filename = f"{content_id}_thumb.jpg"
                thumbnail_path = Path(self.settings.storage.local_path) / "thumbnails" / thumbnail_filename
                await self.analyzer.generate_thumbnail(file_path, str(thumbnail_path))
            
            # Get file information
            file_size = Path(file_path).stat().st_size
            mime_type = magic.from_file(file_path, mime=True)
            
            # Create metadata
            metadata = ContentMetadata(
                content_id=content_id,
                title=title,
                description=description,
                content_type=content_type,
                file_path=file_path,
                file_size=file_size,
                mime_type=mime_type,
                duration=analysis.get("duration"),
                resolution=analysis.get("resolution"),
                fps=analysis.get("fps"),
                bitrate=analysis.get("bitrate"),
                codec=analysis.get("codec"),
                thumbnail_path=str(thumbnail_path) if thumbnail_path else None,
                tags=tags or [],
                categories=categories or [],
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                created_by=created_by,
                updated_by=created_by
            )
            
            # Store metadata
            self.content_metadata[content_id] = metadata
            self.content_status[content_id] = ContentStatus.DRAFT
            
            # Create initial version
            self.version_manager.create_version(
                content_id=content_id,
                file_path=file_path,
                changes_description="Initial version",
                created_by=created_by
            )
            
            logger.info(f"Created content: {content_id}")
            return metadata
            
        except Exception as e:
            logger.error(f"Content creation failed: {e}")
            raise create_content_management_error("content_creation", file_path, e)
    
    def get_content(self, content_id: str) -> Optional[ContentMetadata]:
        """Get content metadata"""
        return self.content_metadata.get(content_id)
    
    def update_content(self, content_id: str, updates: Dict[str, Any], updated_by: str) -> bool:
        """Update content metadata"""
        try:
            metadata = self.get_content(content_id)
            if not metadata:
                raise ValueError(f"Content {content_id} not found")
            
            # Update fields
            for key, value in updates.items():
                if hasattr(metadata, key):
                    setattr(metadata, key, value)
            
            metadata.updated_at = datetime.utcnow()
            metadata.updated_by = updated_by
            
            logger.info(f"Updated content: {content_id}")
            return True
            
        except Exception as e:
            logger.error(f"Content update failed: {e}")
            raise create_content_management_error("content_update", content_id, e)
    
    def delete_content(self, content_id: str) -> bool:
        """Delete content"""
        try:
            if content_id in self.content_metadata:
                del self.content_metadata[content_id]
            
            if content_id in self.content_status:
                del self.content_status[content_id]
            
            # Delete versions
            if content_id in self.version_manager.versions:
                del self.version_manager.versions[content_id]
            
            if content_id in self.version_manager.current_versions:
                del self.version_manager.current_versions[content_id]
            
            # Delete workflow
            if content_id in self.workflow_manager.workflows:
                del self.workflow_manager.workflows[content_id]
            
            # Delete approvals
            if content_id in self.workflow_manager.approvals:
                del self.workflow_manager.approvals[content_id]
            
            logger.info(f"Deleted content: {content_id}")
            return True
            
        except Exception as e:
            logger.error(f"Content deletion failed: {e}")
            raise create_content_management_error("content_deletion", content_id, e)
    
    def list_content(self, content_type: Optional[ContentType] = None,
                    status: Optional[ContentStatus] = None,
                    created_by: Optional[str] = None,
                    limit: int = 100) -> List[ContentMetadata]:
        """List content with filters"""
        try:
            content_list = list(self.content_metadata.values())
            
            # Apply filters
            if content_type:
                content_list = [c for c in content_list if c.content_type == content_type]
            
            if status:
                content_list = [c for c in content_list 
                              if self.content_status.get(c.content_id) == status]
            
            if created_by:
                content_list = [c for c in content_list if c.created_by == created_by]
            
            # Sort by creation date (newest first)
            content_list.sort(key=lambda x: x.created_at, reverse=True)
            
            return content_list[:limit]
            
        except Exception as e:
            logger.error(f"Content listing failed: {e}")
            return []
    
    def get_content_statistics(self) -> Dict[str, Any]:
        """Get content statistics"""
        try:
            total_content = len(self.content_metadata)
            
            # Count by type
            type_counts = {}
            for metadata in self.content_metadata.values():
                content_type = metadata.content_type.value
                type_counts[content_type] = type_counts.get(content_type, 0) + 1
            
            # Count by status
            status_counts = {}
            for content_id, status in self.content_status.items():
                status_counts[status.value] = status_counts.get(status.value, 0) + 1
            
            # Calculate total size
            total_size = sum(metadata.file_size for metadata in self.content_metadata.values())
            
            # Count versions
            total_versions = sum(len(versions) for versions in self.version_manager.versions.values())
            
            return {
                "total_content": total_content,
                "type_distribution": type_counts,
                "status_distribution": status_counts,
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "total_versions": total_versions,
                "average_versions_per_content": round(total_versions / total_content, 2) if total_content > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Content statistics failed: {e}")
            return {}
    
    def search_content(self, query: str, content_type: Optional[ContentType] = None) -> List[ContentMetadata]:
        """Search content"""
        try:
            results = []
            query_lower = query.lower()
            
            for metadata in self.content_metadata.values():
                # Check if content type matches
                if content_type and metadata.content_type != content_type:
                    continue
                
                # Search in title, description, tags
                if (query_lower in metadata.title.lower() or
                    query_lower in metadata.description.lower() or
                    any(query_lower in tag.lower() for tag in metadata.tags)):
                    results.append(metadata)
            
            # Sort by relevance (title matches first)
            results.sort(key=lambda x: (
                query_lower not in x.title.lower(),
                x.created_at
            ))
            
            return results
            
        except Exception as e:
            logger.error(f"Content search failed: {e}")
            return []


# Global content manager
content_manager = ContentManager()





























