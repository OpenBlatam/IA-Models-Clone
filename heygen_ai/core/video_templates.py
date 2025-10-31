"""
Video Templates Service for HeyGen AI
====================================

Provides comprehensive video template management, storage, and retrieval
for enterprise-grade AI video generation and customization.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import uuid
import json
import numpy as np # Added for complexity score calculation

# Core imports
from .base_service import BaseService, ServiceType, HealthCheckResult, ServiceStatus
from .error_handler import ErrorHandler, with_error_handling, with_retry
from .config_manager import ConfigurationManager
from .logging_service import LoggingService

# Video processing imports
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class VideoTemplate:
    """Video template information."""
    
    template_id: str
    name: str
    description: str
    category: str  # business, social_media, education, entertainment, etc.
    style: str = "modern"  # modern, classic, minimalist, artistic, etc.
    aspect_ratio: str = "16:9"  # 16:9, 9:16, 1:1, 4:3, etc.
    resolution: str = "1920x1080"  # 1920x1080, 1080x1920, 1080x1080, etc.
    duration_seconds: float = 30.0
    fps: int = 30
    scene_count: int = 1
    file_path: str = ""
    thumbnail_path: str = ""
    preview_path: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None
    usage_count: int = 0
    quality_rating: float = 0.0
    complexity_score: float = 0.0


@dataclass
class TemplateCategory:
    """Template category definition."""
    
    category_id: str
    name: str
    description: str
    parent_category: Optional[str] = None
    template_count: int = 0
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class TemplateStyle:
    """Template style definition."""
    
    style_id: str
    name: str
    description: str
    characteristics: List[str] = field(default_factory=list)
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class TemplateScene:
    """Template scene definition."""
    
    scene_id: str
    name: str
    description: str
    duration_seconds: float
    elements: List[Dict[str, Any]] = field(default_factory=list)
    transitions: List[Dict[str, Any]] = field(default_factory=list)
    effects: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class TemplateRequest:
    """Request for template operations."""
    
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    operation: str = ""  # create, update, delete, search, customize
    template_data: Optional[Dict[str, Any]] = None
    search_criteria: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TemplateResult:
    """Result of template operation."""
    
    request_id: str
    operation: str
    success: bool
    template_id: Optional[str] = None
    templates: Optional[List[VideoTemplate]] = None
    error_message: Optional[str] = None
    processing_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class VideoTemplateService(BaseService):
    """Service for managing video templates."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the video template service."""
        super().__init__("VideoTemplateService", ServiceType.LIBRARY, config)
        
        # Template storage
        self.templates: Dict[str, VideoTemplate] = {}
        self.categories: Dict[str, TemplateCategory] = {}
        self.styles: Dict[str, TemplateStyle] = {}
        self.scenes: Dict[str, TemplateScene] = {}
        
        # File management
        self.template_directory = Path("./video_templates")
        self.thumbnail_directory = Path("./template_thumbnails")
        self.preview_directory = Path("./template_previews")
        
        # Error handling
        self.error_handler = ErrorHandler()
        
        # Configuration manager
        self.config_manager = ConfigurationManager()
        
        # Logging service
        self.logging_service = LoggingService()
        
        # Performance tracking
        self.library_stats = {
            "total_templates": 0,
            "active_templates": 0,
            "total_categories": 0,
            "total_styles": 0,
            "total_scenes": 0,
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0
        }
        
        # Default configuration
        self.default_config = {
            "max_template_size_mb": 500,
            "supported_formats": ["mp4", "avi", "mov", "mkv", "webm"],
            "thumbnail_size": (320, 180),
            "preview_size": (640, 360),
            "max_tags_per_template": 15,
            "auto_generate_thumbnails": True,
            "auto_generate_previews": True,
            "quality_threshold": 0.7
        }

    async def _initialize_service_impl(self) -> None:
        """Initialize video template services."""
        try:
            logger.info("Initializing video template service...")
            
            # Check dependencies
            await self._check_dependencies()
            
            # Create directories
            await self._create_directories()
            
            # Load default categories and styles
            await self._load_default_categories_styles()
            
            # Load existing templates
            await self._load_existing_templates()
            
            # Validate configuration
            await self._validate_configuration()
            
            logger.info("Video template service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize video template service: {e}")
            raise

    async def _check_dependencies(self) -> None:
        """Check required dependencies."""
        missing_deps = []
        
        if not OPENCV_AVAILABLE:
            missing_deps.append("opencv-python")
        
        if not PIL_AVAILABLE:
            missing_deps.append("Pillow")
        
        if missing_deps:
            logger.warning(f"Missing dependencies: {missing_deps}")
            logger.warning("Some template features may not be available")

    async def _create_directories(self) -> None:
        """Create necessary directories."""
        try:
            self.template_directory.mkdir(exist_ok=True)
            self.thumbnail_directory.mkdir(exist_ok=True)
            self.preview_directory.mkdir(exist_ok=True)
            logger.info("Template directories created successfully")
            
        except Exception as e:
            logger.warning(f"Failed to create some directories: {e}")

    async def _load_default_categories_styles(self) -> None:
        """Load default template categories and styles."""
        try:
            # Create default categories
            default_categories = [
                TemplateCategory("cat_business", "Business", "Professional business templates"),
                TemplateCategory("cat_social_media", "Social Media", "Social media and marketing templates"),
                TemplateCategory("cat_education", "Education", "Educational and tutorial templates"),
                TemplateCategory("cat_entertainment", "Entertainment", "Entertainment and creative templates"),
                TemplateCategory("cat_presentation", "Presentation", "Presentation and slideshow templates"),
                TemplateCategory("cat_promotional", "Promotional", "Promotional and advertising templates"),
                TemplateCategory("cat_corporate", "Corporate", "Corporate and enterprise templates"),
                TemplateCategory("cat_personal", "Personal", "Personal and lifestyle templates")
            ]
            
            for category in default_categories:
                self.categories[category.category_id] = category
                self.library_stats["total_categories"] += 1
            
            # Create default styles
            default_styles = [
                TemplateStyle("style_modern", "Modern", "Contemporary and sleek designs", 
                            ["clean", "minimalist", "trendy", "professional"]),
                TemplateStyle("style_classic", "Classic", "Timeless and traditional designs", 
                            ["elegant", "sophisticated", "traditional", "refined"]),
                TemplateStyle("style_minimalist", "Minimalist", "Simple and clean designs", 
                            ["simple", "clean", "focused", "uncluttered"]),
                TemplateStyle("style_artistic", "Artistic", "Creative and expressive designs", 
                            ["creative", "expressive", "artistic", "unique"]),
                TemplateStyle("style_bold", "Bold", "Strong and impactful designs", 
                            ["strong", "impactful", "confident", "dynamic"]),
                TemplateStyle("style_playful", "Playful", "Fun and engaging designs", 
                            ["fun", "engaging", "lively", "entertaining"])
            ]
            
            for style in default_styles:
                self.styles[style.style_id] = style
                self.library_stats["total_styles"] += 1
            
            logger.info(f"Loaded {len(default_categories)} categories and {len(default_styles)} styles")
            
        except Exception as e:
            logger.warning(f"Failed to load some default categories/styles: {e}")

    async def _load_existing_templates(self) -> None:
        """Load existing templates from storage."""
        try:
            # For now, we'll create some sample templates
            # In production, this would load from database or file system
            
            sample_templates = [
                VideoTemplate(
                    template_id="template_sample_1",
                    name="Business Presentation",
                    description="Professional business presentation template",
                    category="cat_business",
                    style="style_modern",
                    aspect_ratio="16:9",
                    resolution="1920x1080",
                    duration_seconds=60.0,
                    fps=30,
                    scene_count=5,
                    tags=["business", "presentation", "professional", "corporate"],
                    file_path="./video_templates/business_presentation.mp4",
                    thumbnail_path="./template_thumbnails/business_presentation.png",
                    preview_path="./template_previews/business_presentation.mp4",
                    quality_rating=0.9,
                    complexity_score=0.7
                ),
                VideoTemplate(
                    template_id="template_sample_2",
                    name="Social Media Story",
                    description="Vertical social media story template",
                    category="cat_social_media",
                    style="style_bold",
                    aspect_ratio="9:16",
                    resolution="1080x1920",
                    duration_seconds=15.0,
                    fps=30,
                    scene_count=3,
                    tags=["social_media", "story", "vertical", "engaging"],
                    file_path="./video_templates/social_story.mp4",
                    thumbnail_path="./template_thumbnails/social_story.png",
                    preview_path="./template_previews/social_story.mp4",
                    quality_rating=0.85,
                    complexity_score=0.5
                ),
                VideoTemplate(
                    template_id="template_sample_3",
                    name="Educational Tutorial",
                    description="Educational tutorial template with animations",
                    category="cat_education",
                    style="style_playful",
                    aspect_ratio="16:9",
                    resolution="1920x1080",
                    duration_seconds=120.0,
                    fps=30,
                    scene_count=8,
                    tags=["education", "tutorial", "animated", "learning"],
                    file_path="./video_templates/educational_tutorial.mp4",
                    thumbnail_path="./template_thumbnails/educational_tutorial.png",
                    preview_path="./template_previews/educational_tutorial.mp4",
                    quality_rating=0.8,
                    complexity_score=0.8
                )
            ]
            
            for template in sample_templates:
                self.templates[template.template_id] = template
                self.library_stats["total_templates"] += 1
                self.library_stats["active_templates"] += 1
                
                # Update category count
                if template.category in self.categories:
                    self.categories[template.category].template_count += 1
            
            logger.info(f"Loaded {len(sample_templates)} sample templates")
            
        except Exception as e:
            logger.warning(f"Failed to load some existing templates: {e}")

    async def _validate_configuration(self) -> None:
        """Validate template service configuration."""
        if not self.categories:
            raise RuntimeError("No template categories configured")
        
        if not self.styles:
            raise RuntimeError("No template styles configured")

    @with_error_handling
    @with_retry(max_attempts=3)
    async def create_template(self, name: str, description: str, category: str,
                             style: str, file_path: str, **kwargs) -> str:
        """Create a new video template."""
        start_time = time.time()
        
        try:
            logger.info(f"Creating template: {name}")
            
            # Validate category and style
            if category not in self.categories:
                raise ValueError(f"Invalid category: {category}")
            
            if style not in self.styles:
                raise ValueError(f"Invalid style: {style}")
            
            # Validate file
            if not Path(file_path).exists():
                raise ValueError(f"Template file not found: {file_path}")
            
            # Generate template ID
            template_id = str(uuid.uuid4())
            
            # Generate thumbnail and preview if enabled
            thumbnail_path = ""
            preview_path = ""
            if self.default_config["auto_generate_thumbnails"] and OPENCV_AVAILABLE:
                thumbnail_path = await self._generate_thumbnail(file_path, template_id)
            
            if self.default_config["auto_generate_previews"] and OPENCV_AVAILABLE:
                preview_path = await self._generate_preview(file_path, template_id)
            
            # Calculate quality and complexity scores
            quality_rating = await self._calculate_quality_rating(file_path)
            complexity_score = await self._calculate_complexity_score(file_path)
            
            # Get video properties
            video_props = await self._get_video_properties(file_path)
            
            # Create template
            template = VideoTemplate(
                template_id=template_id,
                name=name,
                description=description,
                category=category,
                style=style,
                file_path=file_path,
                thumbnail_path=thumbnail_path,
                preview_path=preview_path,
                aspect_ratio=kwargs.get("aspect_ratio", "16:9"),
                resolution=kwargs.get("resolution", "1920x1080"),
                duration_seconds=kwargs.get("duration_seconds", video_props.get("duration", 30.0)),
                fps=kwargs.get("fps", video_props.get("fps", 30)),
                scene_count=kwargs.get("scene_count", 1),
                quality_rating=quality_rating,
                complexity_score=complexity_score,
                tags=kwargs.get("tags", []),
                metadata=kwargs.get("metadata", {})
            )
            
            # Store template
            self.templates[template_id] = template
            self.library_stats["total_templates"] += 1
            self.library_stats["active_templates"] += 1
            
            # Update category count
            if category in self.categories:
                self.categories[category].template_count += 1
            
            # Update statistics
            self._update_operation_stats(time.time() - start_time, True)
            
            logger.info(f"Template created successfully: {template_id}")
            return template_id
            
        except Exception as e:
            self._update_operation_stats(time.time() - start_time, False)
            logger.error(f"Failed to create template: {e}")
            raise

    async def _generate_thumbnail(self, file_path: str, template_id: str) -> str:
        """Generate thumbnail for template."""
        try:
            if not OPENCV_AVAILABLE:
                return ""
            
            # Open video file
            cap = cv2.VideoCapture(file_path)
            
            # Get middle frame
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            middle_frame = total_frames // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
            
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                return ""
            
            # Resize to thumbnail size
            thumbnail_size = self.default_config["thumbnail_size"]
            frame = cv2.resize(frame, thumbnail_size)
            
            # Save thumbnail
            thumbnail_path = self.thumbnail_directory / f"{template_id}_thumb.png"
            cv2.imwrite(str(thumbnail_path), frame)
            
            return str(thumbnail_path)
            
        except Exception as e:
            logger.warning(f"Failed to generate thumbnail: {e}")
            return ""

    async def _generate_preview(self, file_path: str, template_id: str) -> str:
        """Generate preview for template."""
        try:
            if not OPENCV_AVAILABLE:
                return ""
            
            # Open video file
            cap = cv2.VideoCapture(file_path)
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Create preview (first 5 seconds or full video if shorter)
            preview_duration = min(5.0, total_frames / fps)
            preview_frames = int(preview_duration * fps)
            
            # Get preview size
            preview_size = self.default_config["preview_size"]
            
            # Create video writer
            preview_path = self.preview_directory / f"{template_id}_preview.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(preview_path), fourcc, fps, preview_size)
            
            # Write preview frames
            for i in range(min(preview_frames, total_frames)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.resize(frame, preview_size)
                    out.write(frame)
            
            cap.release()
            out.release()
            
            return str(preview_path)
            
        except Exception as e:
            logger.warning(f"Failed to generate preview: {e}")
            return ""

    async def _calculate_quality_rating(self, file_path: str) -> float:
        """Calculate quality rating for template."""
        try:
            if not OPENCV_AVAILABLE:
                return 0.5  # Default rating
            
            # Open video
            cap = cv2.VideoCapture(file_path)
            
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            cap.release()
            
            # Calculate quality metrics
            resolution_score = min((width * height) / (1920 * 1080), 1.0)
            fps_score = min(fps / 60, 1.0)
            duration_score = min(total_frames / (fps * 30), 1.0)  # 30 seconds baseline
            
            # Weighted average
            quality_rating = 0.5 * resolution_score + 0.3 * fps_score + 0.2 * duration_score
            
            return max(min(quality_rating, 1.0), 0.0)
            
        except Exception as e:
            logger.warning(f"Failed to calculate quality rating: {e}")
            return 0.5

    async def _calculate_complexity_score(self, file_path: str) -> float:
        """Calculate complexity score for template."""
        try:
            if not OPENCV_AVAILABLE:
                return 0.5  # Default score
            
            # Open video
            cap = cv2.VideoCapture(file_path)
            
            # Sample frames for analysis
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            sample_frames = min(10, total_frames)
            step = total_frames // sample_frames
            
            complexity_scores = []
            
            for i in range(0, total_frames, step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    # Convert to grayscale
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    # Calculate edge density
                    edges = cv2.Canny(gray, 50, 150)
                    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
                    
                    # Calculate texture complexity
                    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                    texture_score = min(laplacian_var / 1000, 1.0)
                    
                    # Combined complexity
                    frame_complexity = 0.6 * edge_density + 0.4 * texture_score
                    complexity_scores.append(frame_complexity)
            
            cap.release()
            
            if complexity_scores:
                return np.mean(complexity_scores)
            else:
                return 0.5
                
        except Exception as e:
            logger.warning(f"Failed to calculate complexity score: {e}")
            return 0.5

    async def _get_video_properties(self, file_path: str) -> Dict[str, Any]:
        """Get video properties."""
        try:
            if not OPENCV_AVAILABLE:
                return {"duration": 30.0, "fps": 30}
            
            cap = cv2.VideoCapture(file_path)
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 30.0
            
            cap.release()
            
            return {
                "duration": duration,
                "fps": fps,
                "total_frames": total_frames
            }
            
        except Exception as e:
            logger.warning(f"Failed to get video properties: {e}")
            return {"duration": 30.0, "fps": 30}

    @with_error_handling
    async def search_templates(self, criteria: Dict[str, Any]) -> List[VideoTemplate]:
        """Search templates based on criteria."""
        try:
            logger.info(f"Searching templates with criteria: {criteria}")
            
            matching_templates = []
            
            for template in self.templates.values():
                if not template.is_active:
                    continue
                
                # Check category
                if "category" in criteria and template.category != criteria["category"]:
                    continue
                
                # Check style
                if "style" in criteria and template.style != criteria["style"]:
                    continue
                
                # Check aspect ratio
                if "aspect_ratio" in criteria and template.aspect_ratio != criteria["aspect_ratio"]:
                    continue
                
                # Check resolution
                if "resolution" in criteria and template.resolution != criteria["resolution"]:
                    continue
                
                # Check duration range
                if "min_duration" in criteria and template.duration_seconds < criteria["min_duration"]:
                    continue
                
                if "max_duration" in criteria and template.duration_seconds > criteria["max_duration"]:
                    continue
                
                # Check quality threshold
                if "min_quality" in criteria and template.quality_rating < criteria["min_quality"]:
                    continue
                
                # Check complexity range
                if "min_complexity" in criteria and template.complexity_score < criteria["min_complexity"]:
                    continue
                
                if "max_complexity" in criteria and template.complexity_score > criteria["max_complexity"]:
                    continue
                
                # Check tags
                if "tags" in criteria:
                    required_tags = criteria["tags"]
                    if not all(tag in template.tags for tag in required_tags):
                        continue
                
                # Check name/description (case-insensitive)
                if "search_text" in criteria:
                    search_text = criteria["search_text"].lower()
                    if (search_text not in template.name.lower() and 
                        search_text not in template.description.lower()):
                        continue
                
                matching_templates.append(template)
            
            # Sort by quality rating (highest first)
            matching_templates.sort(key=lambda x: x.quality_rating, reverse=True)
            
            logger.info(f"Found {len(matching_templates)} matching templates")
            return matching_templates
            
        except Exception as e:
            logger.error(f"Template search failed: {e}")
            return []

    @with_error_handling
    async def get_template(self, template_id: str) -> Optional[VideoTemplate]:
        """Get template by ID."""
        try:
            if template_id not in self.templates:
                return None
            
            template = self.templates[template_id]
            
            # Update usage statistics
            template.last_used = datetime.now()
            template.usage_count += 1
            self.templates[template_id] = template
            
            return template
            
        except Exception as e:
            logger.error(f"Failed to get template: {e}")
            return None

    @with_error_handling
    async def update_template(self, template_id: str, updates: Dict[str, Any]) -> bool:
        """Update template information."""
        try:
            logger.info(f"Updating template: {template_id}")
            
            if template_id not in self.templates:
                raise ValueError(f"Template not found: {template_id}")
            
            template = self.templates[template_id]
            
            # Update fields
            for field, value in updates.items():
                if hasattr(template, field) and field not in ["template_id", "created_at"]:
                    setattr(template, field, value)
            
            # Update storage
            self.templates[template_id] = template
            
            logger.info(f"Template updated successfully: {template_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update template: {e}")
            raise

    @with_error_handling
    async def delete_template(self, template_id: str) -> bool:
        """Delete a template."""
        try:
            logger.info(f"Deleting template: {template_id}")
            
            if template_id not in self.templates:
                raise ValueError(f"Template not found: {template_id}")
            
            template = self.templates[template_id]
            
            # Update category count
            if template.category in self.categories:
                self.categories[template.category].template_count -= 1
            
            # Remove template
            del self.templates[template_id]
            self.library_stats["total_templates"] -= 1
            self.library_stats["active_templates"] -= 1
            
            logger.info(f"Template deleted successfully: {template_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete template: {e}")
            raise

    @with_error_handling
    async def get_categories(self) -> List[TemplateCategory]:
        """Get all template categories."""
        return list(self.categories.values())

    @with_error_handling
    async def get_styles(self) -> List[TemplateStyle]:
        """Get all template styles."""
        return list(self.styles.values())

    @with_error_handling
    async def get_popular_templates(self, limit: int = 10) -> List[VideoTemplate]:
        """Get most popular templates."""
        try:
            # Sort by usage count and return top templates
            sorted_templates = sorted(
                [template for template in self.templates.values() if template.is_active],
                key=lambda x: x.usage_count,
                reverse=True
            )
            
            return sorted_templates[:limit]
            
        except Exception as e:
            logger.error(f"Failed to get popular templates: {e}")
            return []

    @with_error_handling
    async def get_templates_by_aspect_ratio(self, aspect_ratio: str) -> List[VideoTemplate]:
        """Get templates by aspect ratio."""
        try:
            return [
                template for template in self.templates.values() 
                if template.is_active and template.aspect_ratio == aspect_ratio
            ]
            
        except Exception as e:
            logger.error(f"Failed to get templates by aspect ratio: {e}")
            return []

    def _update_operation_stats(self, processing_time: float, success: bool):
        """Update operation statistics."""
        self.library_stats["total_operations"] += 1
        
        if success:
            self.library_stats["successful_operations"] += 1
        else:
            self.library_stats["failed_operations"] += 1

    async def health_check(self) -> HealthCheckResult:
        """Check the health of the video template service."""
        try:
            # Check base service health
            base_health = await super().health_check()
            
            # Check dependencies
            dependencies = {
                "opencv": OPENCV_AVAILABLE,
                "pillow": PIL_AVAILABLE
            }
            
            # Check storage
            storage_status = {
                "template_directory_exists": self.template_directory.exists(),
                "thumbnail_directory_exists": self.thumbnail_directory.exists(),
                "preview_directory_exists": self.preview_directory.exists(),
                "template_files_count": len(list(self.template_directory.glob("*.mp4")))
            }
            
            # Check library content
            library_status = {
                "total_templates": self.library_stats["total_templates"],
                "active_templates": self.library_stats["active_templates"],
                "total_categories": self.library_stats["total_categories"],
                "total_styles": self.library_stats["total_styles"],
                "total_scenes": self.library_stats["total_scenes"]
            }
            
            # Check operations
            operation_status = {
                "total_operations": self.library_stats["total_operations"],
                "successful_operations": self.library_stats["successful_operations"],
                "failed_operations": self.library_stats["failed_operations"]
            }
            
            # Update base health
            base_health.details.update({
                "dependencies": dependencies,
                "storage": storage_status,
                "library": library_status,
                "operations": operation_status,
                "library_stats": self.library_stats
            })
            
            return base_health
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return HealthCheckResult(
                status=ServiceStatus.UNHEALTHY,
                error_message=str(e)
            )

    async def cleanup_temp_files(self) -> None:
        """Clean up temporary template files."""
        try:
            temp_dir = Path("./temp")
            if temp_dir.exists():
                for template_file in temp_dir.glob("template_*"):
                    template_file.unlink()
                    logger.debug(f"Cleaned up temp file: {template_file}")
        except Exception as e:
            logger.warning(f"Failed to cleanup temp files: {e}")

    async def shutdown(self) -> None:
        """Shutdown the video template service."""
        try:
            # Clear data
            self.templates.clear()
            self.categories.clear()
            self.styles.clear()
            self.scenes.clear()
            
            logger.info("Video template service shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


