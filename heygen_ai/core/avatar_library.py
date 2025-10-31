"""
Avatar Library Service for HeyGen AI
===================================

Provides comprehensive avatar management, storage, and retrieval
for enterprise-grade AI video generation.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import uuid

# Core imports
from .base_service import BaseService, ServiceType, HealthCheckResult, ServiceStatus
from .error_handler import ErrorHandler, with_error_handling, with_retry
from .config_manager import ConfigurationManager
from .logging_service import LoggingService

# Image processing imports
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class Avatar:
    """Avatar information."""
    
    avatar_id: str
    name: str
    description: str
    category: str  # human, animal, cartoon, abstract, etc.
    gender: Optional[str] = None  # male, female, neutral
    age_group: Optional[str] = None  # child, teen, adult, senior
    ethnicity: Optional[str] = None
    style: str = "realistic"  # realistic, cartoon, anime, etc.
    file_path: str = ""
    thumbnail_path: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None
    usage_count: int = 0


@dataclass
class AvatarCategory:
    """Avatar category definition."""
    
    category_id: str
    name: str
    description: str
    parent_category: Optional[str] = None
    avatar_count: int = 0
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class AvatarStyle:
    """Avatar style definition."""
    
    style_id: str
    name: str
    description: str
    characteristics: List[str] = field(default_factory=list)
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class AvatarRequest:
    """Request for avatar operations."""
    
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    operation: str = ""  # create, update, delete, search, generate
    avatar_data: Optional[Dict[str, Any]] = None
    search_criteria: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AvatarResult:
    """Result of avatar operation."""
    
    request_id: str
    operation: str
    success: bool
    avatar_id: Optional[str] = None
    avatars: Optional[List[Avatar]] = None
    error_message: Optional[str] = None
    processing_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class AvatarLibraryService(BaseService):
    """Service for managing avatar library."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the avatar library service."""
        super().__init__("AvatarLibraryService", ServiceType.LIBRARY, config)
        
        # Avatar storage
        self.avatars: Dict[str, Avatar] = {}
        self.categories: Dict[str, AvatarCategory] = {}
        self.styles: Dict[str, AvatarStyle] = {}
        
        # File management
        self.avatar_directory = Path("./avatars")
        self.thumbnail_directory = Path("./thumbnails")
        
        # Error handling
        self.error_handler = ErrorHandler()
        
        # Configuration manager
        self.config_manager = ConfigurationManager()
        
        # Logging service
        self.logging_service = LoggingService()
        
        # Performance tracking
        self.library_stats = {
            "total_avatars": 0,
            "active_avatars": 0,
            "total_categories": 0,
            "total_styles": 0,
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0
        }
        
        # Default configuration
        self.default_config = {
            "max_avatar_size_mb": 50,
            "supported_formats": ["png", "jpg", "jpeg", "webp"],
            "thumbnail_size": (256, 256),
            "max_tags_per_avatar": 10,
            "auto_generate_thumbnails": True
        }

    async def _initialize_service_impl(self) -> None:
        """Initialize avatar library services."""
        try:
            logger.info("Initializing avatar library service...")
            
            # Check dependencies
            await self._check_dependencies()
            
            # Create directories
            await self._create_directories()
            
            # Load default categories and styles
            await self._load_default_categories_styles()
            
            # Load existing avatars
            await self._load_existing_avatars()
            
            # Validate configuration
            await self._validate_configuration()
            
            logger.info("Avatar library service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize avatar library service: {e}")
            raise

    async def _check_dependencies(self) -> None:
        """Check required dependencies."""
        missing_deps = []
        
        if not PIL_AVAILABLE:
            missing_deps.append("Pillow")
        
        if missing_deps:
            logger.warning(f"Missing dependencies: {missing_deps}")
            logger.warning("Some avatar features may not be available")

    async def _create_directories(self) -> None:
        """Create necessary directories."""
        try:
            self.avatar_directory.mkdir(exist_ok=True)
            self.thumbnail_directory.mkdir(exist_ok=True)
            logger.info("Avatar directories created successfully")
            
        except Exception as e:
            logger.warning(f"Failed to create some directories: {e}")

    async def _load_default_categories_styles(self) -> None:
        """Load default avatar categories and styles."""
        try:
            # Create default categories
            default_categories = [
                AvatarCategory("cat_human", "Human", "Human avatars and characters"),
                AvatarCategory("cat_animal", "Animal", "Animal and creature avatars"),
                AvatarCategory("cat_cartoon", "Cartoon", "Cartoon and animated characters"),
                AvatarCategory("cat_abstract", "Abstract", "Abstract and artistic avatars"),
                AvatarCategory("cat_professional", "Professional", "Business and professional avatars"),
                AvatarCategory("cat_casual", "Casual", "Casual and everyday avatars"),
                AvatarCategory("cat_fantasy", "Fantasy", "Fantasy and mythical characters"),
                AvatarCategory("cat_sci_fi", "Science Fiction", "Sci-fi and futuristic avatars")
            ]
            
            for category in default_categories:
                self.categories[category.category_id] = category
                self.library_stats["total_categories"] += 1
            
            # Create default styles
            default_styles = [
                AvatarStyle("style_realistic", "Realistic", "Photorealistic avatars", 
                           ["high_detail", "natural_lighting", "realistic_textures"]),
                AvatarStyle("style_cartoon", "Cartoon", "Cartoon-style avatars", 
                           ["simplified_features", "bold_colors", "clean_lines"]),
                AvatarStyle("style_anime", "Anime", "Anime-style avatars", 
                           ["large_eyes", "colorful_hair", "expressive_features"]),
                AvatarStyle("style_3d", "3D", "3D rendered avatars", 
                           ["depth", "shading", "dimensional"]),
                AvatarStyle("style_minimalist", "Minimalist", "Simple and clean avatars", 
                           ["simple_shapes", "limited_colors", "clean_design"])
            ]
            
            for style in default_styles:
                self.styles[style.style_id] = style
                self.library_stats["total_styles"] += 1
            
            logger.info(f"Loaded {len(default_categories)} categories and {len(default_styles)} styles")
            
        except Exception as e:
            logger.warning(f"Failed to load some default categories/styles: {e}")

    async def _load_existing_avatars(self) -> None:
        """Load existing avatars from storage."""
        try:
            # For now, we'll create some sample avatars
            # In production, this would load from database or file system
            
            sample_avatars = [
                Avatar(
                    avatar_id="avatar_sample_1",
                    name="Professional Woman",
                    description="A professional businesswoman avatar",
                    category="cat_professional",
                    gender="female",
                    age_group="adult",
                    style="style_realistic",
                    tags=["business", "professional", "woman", "adult"],
                    file_path="./avatars/sample_1.png",
                    thumbnail_path="./thumbnails/sample_1.png"
                ),
                Avatar(
                    avatar_id="avatar_sample_2",
                    name="Casual Man",
                    description="A casual young man avatar",
                    category="cat_casual",
                    gender="male",
                    age_group="adult",
                    style="style_cartoon",
                    tags=["casual", "man", "young", "friendly"],
                    file_path="./avatars/sample_2.png",
                    thumbnail_path="./thumbnails/sample_2.png"
                ),
                Avatar(
                    avatar_id="avatar_sample_3",
                    name="Fantasy Character",
                    description="A fantasy warrior avatar",
                    category="cat_fantasy",
                    gender="neutral",
                    age_group="adult",
                    style="style_3d",
                    tags=["fantasy", "warrior", "character", "adventure"],
                    file_path="./avatars/sample_3.png",
                    thumbnail_path="./thumbnails/sample_3.png"
                )
            ]
            
            for avatar in sample_avatars:
                self.avatars[avatar.avatar_id] = avatar
                self.library_stats["total_avatars"] += 1
                self.library_stats["active_avatars"] += 1
                
                # Update category count
                if avatar.category in self.categories:
                    self.categories[avatar.category].avatar_count += 1
            
            logger.info(f"Loaded {len(sample_avatars)} sample avatars")
            
        except Exception as e:
            logger.warning(f"Failed to load some existing avatars: {e}")

    async def _validate_configuration(self) -> None:
        """Validate avatar library configuration."""
        if not self.categories:
            raise RuntimeError("No avatar categories configured")
        
        if not self.styles:
            raise RuntimeError("No avatar styles configured")

    @with_error_handling
    @with_retry(max_attempts=3)
    async def create_avatar(self, name: str, description: str, category: str, 
                           style: str, file_path: str, **kwargs) -> str:
        """Create a new avatar."""
        start_time = time.time()
        
        try:
            logger.info(f"Creating avatar: {name}")
            
            # Validate category and style
            if category not in self.categories:
                raise ValueError(f"Invalid category: {category}")
            
            if style not in self.styles:
                raise ValueError(f"Invalid style: {style}")
            
            # Validate file
            if not Path(file_path).exists():
                raise ValueError(f"Avatar file not found: {file_path}")
            
            # Generate avatar ID
            avatar_id = str(uuid.uuid4())
            
            # Generate thumbnail if enabled
            thumbnail_path = ""
            if self.default_config["auto_generate_thumbnails"] and PIL_AVAILABLE:
                thumbnail_path = await self._generate_thumbnail(file_path, avatar_id)
            
            # Create avatar
            avatar = Avatar(
                avatar_id=avatar_id,
                name=name,
                description=description,
                category=category,
                style=style,
                file_path=file_path,
                thumbnail_path=thumbnail_path,
                tags=kwargs.get("tags", []),
                gender=kwargs.get("gender"),
                age_group=kwargs.get("age_group"),
                ethnicity=kwargs.get("ethnicity"),
                metadata=kwargs.get("metadata", {})
            )
            
            # Store avatar
            self.avatars[avatar_id] = avatar
            self.library_stats["total_avatars"] += 1
            self.library_stats["active_avatars"] += 1
            
            # Update category count
            if category in self.categories:
                self.categories[category].avatar_count += 1
            
            # Update statistics
            self._update_operation_stats(time.time() - start_time, True)
            
            logger.info(f"Avatar created successfully: {avatar_id}")
            return avatar_id
            
        except Exception as e:
            self._update_operation_stats(time.time() - start_time, False)
            logger.error(f"Failed to create avatar: {e}")
            raise

    async def _generate_thumbnail(self, file_path: str, avatar_id: str) -> str:
        """Generate thumbnail for avatar."""
        try:
            if not PIL_AVAILABLE:
                return ""
            
            # Open image
            with Image.open(file_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize to thumbnail size
                thumbnail_size = self.default_config["thumbnail_size"]
                img.thumbnail(thumbnail_size, Image.Resampling.LANCZOS)
                
                # Save thumbnail
                thumbnail_path = self.thumbnail_directory / f"{avatar_id}_thumb.png"
                img.save(thumbnail_path, "PNG")
                
                return str(thumbnail_path)
                
        except Exception as e:
            logger.warning(f"Failed to generate thumbnail: {e}")
            return ""

    @with_error_handling
    async def search_avatars(self, criteria: Dict[str, Any]) -> List[Avatar]:
        """Search avatars based on criteria."""
        try:
            logger.info(f"Searching avatars with criteria: {criteria}")
            
            matching_avatars = []
            
            for avatar in self.avatars.values():
                if not avatar.is_active:
                    continue
                
                # Check category
                if "category" in criteria and avatar.category != criteria["category"]:
                    continue
                
                # Check style
                if "style" in criteria and avatar.style != criteria["style"]:
                    continue
                
                # Check gender
                if "gender" in criteria and avatar.gender != criteria["gender"]:
                    continue
                
                # Check age group
                if "age_group" in criteria and avatar.age_group != criteria["age_group"]:
                    continue
                
                # Check tags
                if "tags" in criteria:
                    required_tags = criteria["tags"]
                    if not all(tag in avatar.tags for tag in required_tags):
                        continue
                
                # Check name/description (case-insensitive)
                if "search_text" in criteria:
                    search_text = criteria["search_text"].lower()
                    if (search_text not in avatar.name.lower() and 
                        search_text not in avatar.description.lower()):
                        continue
                
                matching_avatars.append(avatar)
            
            # Sort by usage count (most used first)
            matching_avatars.sort(key=lambda x: x.usage_count, reverse=True)
            
            logger.info(f"Found {len(matching_avatars)} matching avatars")
            return matching_avatars
            
        except Exception as e:
            logger.error(f"Avatar search failed: {e}")
            return []

    @with_error_handling
    async def get_avatar(self, avatar_id: str) -> Optional[Avatar]:
        """Get avatar by ID."""
        try:
            if avatar_id not in self.avatars:
                return None
            
            avatar = self.avatars[avatar_id]
            
            # Update usage statistics
            avatar.last_used = datetime.now()
            avatar.usage_count += 1
            self.avatars[avatar_id] = avatar
            
            return avatar
            
        except Exception as e:
            logger.error(f"Failed to get avatar: {e}")
            return None

    @with_error_handling
    async def update_avatar(self, avatar_id: str, updates: Dict[str, Any]) -> bool:
        """Update avatar information."""
        try:
            logger.info(f"Updating avatar: {avatar_id}")
            
            if avatar_id not in self.avatars:
                raise ValueError(f"Avatar not found: {avatar_id}")
            
            avatar = self.avatars[avatar_id]
            
            # Update fields
            for field, value in updates.items():
                if hasattr(avatar, field) and field not in ["avatar_id", "created_at"]:
                    setattr(avatar, field, value)
            
            # Update storage
            self.avatars[avatar_id] = avatar
            
            logger.info(f"Avatar updated successfully: {avatar_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update avatar: {e}")
            raise

    @with_error_handling
    async def delete_avatar(self, avatar_id: str) -> bool:
        """Delete an avatar."""
        try:
            logger.info(f"Deleting avatar: {avatar_id}")
            
            if avatar_id not in self.avatars:
                raise ValueError(f"Avatar not found: {avatar_id}")
            
            avatar = self.avatars[avatar_id]
            
            # Update category count
            if avatar.category in self.categories:
                self.categories[avatar.category].avatar_count -= 1
            
            # Remove avatar
            del self.avatars[avatar_id]
            self.library_stats["total_avatars"] -= 1
            self.library_stats["active_avatars"] -= 1
            
            logger.info(f"Avatar deleted successfully: {avatar_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete avatar: {e}")
            raise

    @with_error_handling
    async def get_categories(self) -> List[AvatarCategory]:
        """Get all avatar categories."""
        return list(self.categories.values())

    @with_error_handling
    async def get_styles(self) -> List[AvatarStyle]:
        """Get all avatar styles."""
        return list(self.styles.values())

    @with_error_handling
    async def get_popular_avatars(self, limit: int = 10) -> List[Avatar]:
        """Get most popular avatars."""
        try:
            # Sort by usage count and return top avatars
            sorted_avatars = sorted(
                [avatar for avatar in self.avatars.values() if avatar.is_active],
                key=lambda x: x.usage_count,
                reverse=True
            )
            
            return sorted_avatars[:limit]
            
        except Exception as e:
            logger.error(f"Failed to get popular avatars: {e}")
            return []

    def _update_operation_stats(self, processing_time: float, success: bool):
        """Update operation statistics."""
        self.library_stats["total_operations"] += 1
        
        if success:
            self.library_stats["successful_operations"] += 1
        else:
            self.library_stats["failed_operations"] += 1

    async def health_check(self) -> HealthCheckResult:
        """Check the health of the avatar library service."""
        try:
            # Check base service health
            base_health = await super().health_check()
            
            # Check dependencies
            dependencies = {
                "pillow": PIL_AVAILABLE
            }
            
            # Check storage
            storage_status = {
                "avatar_directory_exists": self.avatar_directory.exists(),
                "thumbnail_directory_exists": self.thumbnail_directory.exists(),
                "avatar_files_count": len(list(self.avatar_directory.glob("*.png"))) + len(list(self.avatar_directory.glob("*.jpg")))
            }
            
            # Check library content
            library_status = {
                "total_avatars": self.library_stats["total_avatars"],
                "active_avatars": self.library_stats["active_avatars"],
                "total_categories": self.library_stats["total_categories"],
                "total_styles": self.library_stats["total_styles"]
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
        """Clean up temporary avatar files."""
        try:
            temp_dir = Path("./temp")
            if temp_dir.exists():
                for avatar_file in temp_dir.glob("avatar_*"):
                    avatar_file.unlink()
                    logger.debug(f"Cleaned up temp file: {avatar_file}")
        except Exception as e:
            logger.warning(f"Failed to cleanup temp files: {e}")

    async def shutdown(self) -> None:
        """Shutdown the avatar library service."""
        try:
            # Clear data
            self.avatars.clear()
            self.categories.clear()
            self.styles.clear()
            
            logger.info("Avatar library service shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


