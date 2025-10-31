from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import asyncio
import re
import logging
from datetime import datetime, timedelta, date
from typing import Dict, List, Any, Optional, Union
from fastapi import HTTPException, UploadFile, status
from pydantic import BaseModel, Field, validator
import uuid
import os
from typing import Any, List, Dict, Optional
"""
Error-First Pattern Implementation for LinkedIn Posts System
Practical examples of handling errors and edge cases at the beginning of functions
"""


logger = logging.getLogger(__name__)

# ============================================================================
# ERROR-FIRST VALIDATION UTILITIES
# ============================================================================

class ValidationError(Exception):
    """Custom validation error with context"""
    def __init__(self, message: str, field: str = None, code: str = None):
        
    """__init__ function."""
self.message = message
        self.field = field
        self.code = code
        super().__init__(self.message)

class SecurityValidator:
    """Security validation utilities"""
    
    @staticmethod
    def validate_user_id(user_id: str) -> str:
        """Validate and sanitize user ID"""
        if not user_id or not user_id.strip():
            raise ValidationError("User ID is required", "user_id", "MISSING_USER_ID")
        
        user_id = user_id.strip()
        if not re.match(r'^[a-zA-Z0-9_-]+$', user_id):
            raise ValidationError("Invalid user ID format", "user_id", "INVALID_USER_ID_FORMAT")
        
        return user_id
    
    @staticmethod
    def validate_content_security(content: str) -> str:
        """Validate content for security threats"""
        if not content or not content.strip():
            raise ValidationError("Content cannot be empty", "content", "MISSING_CONTENT")
        
        content = content.strip()
        
        # SQL injection patterns
        sql_patterns = [
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE)\b)",
            r"(\b(UNION|OR|AND)\b\s+\d+)",
            r"(--|/\*|\*/|xp_|sp_)",
            r"(\b(WAITFOR|DELAY)\b)",
            r"(\b(SLEEP|BENCHMARK)\b)",
        ]
        
        for pattern in sql_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                logger.warning(f"SQL injection attempt detected: {content}")
                raise ValidationError("Invalid content detected", "content", "SQL_INJECTION_ATTEMPT")
        
        # XSS patterns
        xss_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",
            r"<iframe[^>]*>",
            r"<object[^>]*>",
            r"<embed[^>]*>",
        ]
        
        for pattern in xss_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                logger.warning(f"XSS attempt detected: {content}")
                raise ValidationError("Invalid content detected", "content", "XSS_ATTEMPT")
        
        return content
    
    @staticmethod
    def validate_filename(filename: str) -> str:
        """Validate filename for security"""
        if not filename:
            raise ValidationError("Filename is required", "filename", "MISSING_FILENAME")
        
        # Check for path traversal
        if '..' in filename or '/' in filename or '\\' in filename:
            raise ValidationError("Invalid filename", "filename", "PATH_TRAVERSAL_ATTEMPT")
        
        # Check for malicious patterns
        malicious_patterns = [
            r'<script',
            r'javascript:',
            r'data:text/html',
            r'vbscript:',
        ]
        
        for pattern in malicious_patterns:
            if re.search(pattern, filename, re.IGNORECASE):
                raise ValidationError("Invalid filename", "filename", "MALICIOUS_FILENAME")
        
        return filename

class ContentValidator:
    """Content validation utilities"""
    
    @staticmethod
    def validate_post_content(content: str) -> str:
        """Validate post content"""
        # Length validation
        if len(content) < 10:
            raise ValidationError("Content too short (minimum 10 characters)", "content", "CONTENT_TOO_SHORT")
        
        if len(content) > 3000:
            raise ValidationError("Content too long (maximum 3000 characters)", "content", "CONTENT_TOO_LONG")
        
        # Word count validation
        words = content.split()
        if len(words) < 5:
            raise ValidationError("Content must have at least 5 words", "content", "INSUFFICIENT_WORDS")
        
        if len(words) > 500:
            raise ValidationError("Content too long (maximum 500 words)", "content", "TOO_MANY_WORDS")
        
        # Check for excessive whitespace
        if len(content) - len(content.strip()) > len(content) * 0.1:
            raise ValidationError("Excessive whitespace detected", "content", "EXCESSIVE_WHITESPACE")
        
        return content.strip()
    
    @staticmethod
    def validate_hashtags(hashtags: List[str]) -> List[str]:
        """Validate hashtags"""
        if hashtags is None:
            return []
        
        if len(hashtags) > 30:
            raise ValidationError("Too many hashtags (maximum 30)", "hashtags", "TOO_MANY_HASHTAGS")
        
        validated_hashtags = []
        for hashtag in hashtags:
            if not hashtag.startswith('#'):
                hashtag = f"#{hashtag}"
            
            if len(hashtag) < 2:
                continue
            
            if len(hashtag) > 50:
                raise ValidationError(f"Hashtag too long: {hashtag}", "hashtags", "HASHTAG_TOO_LONG")
            
            if not re.match(r'^#[a-zA-Z0-9_]+$', hashtag):
                raise ValidationError(f"Invalid hashtag format: {hashtag}", "hashtags", "INVALID_HASHTAG_FORMAT")
            
            validated_hashtags.append(hashtag.lower())
        
        return list(set(validated_hashtags))  # Remove duplicates

class BusinessValidator:
    """Business rule validation utilities"""
    
    @staticmethod
    async def validate_user_post_limit(user_id: str, daily_limit: int = 5) -> bool:
        """Validate user's daily post limit"""
        # This would query the database in production
        daily_posts = await get_user_daily_posts(user_id)
        
        if daily_posts >= daily_limit:
            raise ValidationError(
                f"Daily post limit exceeded ({daily_limit} posts per day)",
                "user_id",
                "DAILY_LIMIT_EXCEEDED"
            )
        
        return True
    
    @staticmethod
    async def validate_duplicate_content(content: str, user_id: str, time_window: int = 3600) -> bool:
        """Validate for duplicate content"""
        # This would check the database in production
        is_duplicate = await check_duplicate_content(content, user_id, time_window)
        
        if is_duplicate:
            raise ValidationError(
                "Duplicate content detected within time window",
                "content",
                "DUPLICATE_CONTENT"
            )
        
        return True

# ============================================================================
# ERROR-FIRST FUNCTION IMPLEMENTATIONS
# ============================================================================

async def create_linkedin_post_error_first(
    user_id: str, 
    content: str, 
    hashtags: List[str] = None,
    post_type: str = "general"
) -> Dict[str, Any]:
    """
    Create LinkedIn post with comprehensive error-first validation
    """
    # ============================================================================
    # ERROR-FIRST VALIDATION (P0-P1 Priority)
    # ============================================================================
    
    try:
        # P0: Security validation (CRITICAL)
        user_id = SecurityValidator.validate_user_id(user_id)
        content = SecurityValidator.validate_content_security(content)
        
        # P1: Content validation (HIGH PRIORITY)
        content = ContentValidator.validate_post_content(content)
        validated_hashtags = ContentValidator.validate_hashtags(hashtags)
        
        # P1: Post type validation
        valid_post_types = ["general", "educational", "promotional", "personal", "industry"]
        if post_type not in valid_post_types:
            raise ValidationError(
                f"Invalid post type. Must be one of: {', '.join(valid_post_types)}",
                "post_type",
                "INVALID_POST_TYPE"
            )
        
    except ValidationError as e:
        logger.warning(f"Validation error creating post: {e.message}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": e.code,
                "message": e.message,
                "field": e.field,
                "user_id": user_id,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    # ============================================================================
    # BUSINESS LOGIC VALIDATION (P1 Priority)
    # ============================================================================
    
    try:
        # P1: Business rule validation
        await BusinessValidator.validate_user_post_limit(user_id)
        await BusinessValidator.validate_duplicate_content(content, user_id)
        
        # P1: Rate limiting
        await check_rate_limit(user_id, "post_creation")
        
    except ValidationError as e:
        logger.warning(f"Business validation error: {e.message}")
        status_code = 429 if e.code == "DAILY_LIMIT_EXCEEDED" else 400
        raise HTTPException(
            status_code=status_code,
            detail={
                "error": e.code,
                "message": e.message,
                "field": e.field,
                "user_id": user_id,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    # ============================================================================
    # BUSINESS LOGIC (After all validation passes)
    # ============================================================================
    
    try:
        # Create post in database
        post_data = await create_post_in_database(user_id, content, validated_hashtags, post_type)
        
        # Log successful post creation
        logger.info(f"Post created successfully: {post_data['id']} by user {user_id}")
        
        return {
            "status": "success",
            "post_id": post_data["id"],
            "message": "Post created successfully",
            "created_at": post_data["created_at"],
            "hashtags": validated_hashtags
        }
        
    except Exception as e:
        logger.error(f"Unexpected error creating post for user {user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "INTERNAL_ERROR",
                "message": "An unexpected error occurred while creating the post",
                "user_id": user_id,
                "timestamp": datetime.now().isoformat()
            }
        )

async def update_post_content_error_first(
    post_id: str, 
    user_id: str, 
    new_content: str
) -> Dict[str, Any]:
    """
    Update post content with error-first validation
    """
    # ============================================================================
    # ERROR-FIRST INPUT VALIDATION (P0-P1 Priority)
    # ============================================================================
    
    try:
        # P0: Security validation
        user_id = SecurityValidator.validate_user_id(user_id)
        new_content = SecurityValidator.validate_content_security(new_content)
        
        # P0: Post ID validation
        if not post_id or not post_id.strip():
            raise ValidationError("Post ID is required", "post_id", "MISSING_POST_ID")
        
        post_id = post_id.strip()
        if not re.match(r'^[a-zA-Z0-9_-]+$', post_id):
            raise ValidationError("Invalid post ID format", "post_id", "INVALID_POST_ID_FORMAT")
        
        # P1: Content validation
        new_content = ContentValidator.validate_post_content(new_content)
        
    except ValidationError as e:
        logger.warning(f"Validation error updating post: {e.message}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": e.code,
                "message": e.message,
                "field": e.field,
                "post_id": post_id,
                "user_id": user_id,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    # ============================================================================
    # ERROR-FIRST DATABASE VALIDATION (P2 Priority)
    # ============================================================================
    
    try:
        # Check if post exists
        post = await get_post_by_id(post_id)
        if not post:
            raise ValidationError("Post not found", "post_id", "POST_NOT_FOUND")
        
        # Check if user owns the post
        if post.user_id != user_id:
            raise ValidationError("Cannot update another user's post", "user_id", "UNAUTHORIZED_UPDATE")
        
        # Check if post is editable (not too old)
        post_age = datetime.now() - post.created_at
        if post_age > timedelta(hours=24):
            raise ValidationError("Post cannot be edited after 24 hours", "post_id", "POST_TOO_OLD")
        
        # Check if post is already being edited
        if post.is_being_edited:
            raise ValidationError("Post is currently being edited", "post_id", "POST_BEING_EDITED")
        
    except ValidationError as e:
        logger.warning(f"Database validation error: {e.message}")
        status_code = 404 if e.code == "POST_NOT_FOUND" else 403 if e.code == "UNAUTHORIZED_UPDATE" else 400
        raise HTTPException(
            status_code=status_code,
            detail={
                "error": e.code,
                "message": e.message,
                "field": e.field,
                "post_id": post_id,
                "user_id": user_id,
                "timestamp": datetime.now().isoformat()
            }
        )
    except Exception as e:
        logger.error(f"Database error during post validation: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "DATABASE_ERROR",
                "message": "Database validation failed",
                "post_id": post_id,
                "user_id": user_id,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    # ============================================================================
    # BUSINESS LOGIC (After all validation passes)
    # ============================================================================
    
    try:
        # Update post content
        updated_post = await update_post_in_database(post_id, new_content)
        
        # Log the update
        await log_post_update(post_id, user_id, "content_updated")
        
        return {
            "status": "success",
            "post_id": post_id,
            "message": "Post content updated successfully",
            "updated_at": updated_post.updated_at.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error updating post content: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "UPDATE_FAILED",
                "message": "Failed to update post content",
                "post_id": post_id,
                "user_id": user_id,
                "timestamp": datetime.now().isoformat()
            }
        )

async async def upload_post_image_error_first(
    user_id: str, 
    file: UploadFile, 
    post_id: str = None
) -> Dict[str, Any]:
    """
    Upload post image with error-first validation
    """
    # ============================================================================
    # ERROR-FIRST PARAMETER VALIDATION (P0 Priority)
    # ============================================================================
    
    try:
        # P0: Required parameter validation
        user_id = SecurityValidator.validate_user_id(user_id)
        
        if not file:
            raise ValidationError("File is required", "file", "MISSING_FILE")
        
        # P0: File security validation
        filename = SecurityValidator.validate_filename(file.filename)
        
    except ValidationError as e:
        logger.warning(f"Parameter validation error: {e.message}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": e.code,
                "message": e.message,
                "field": e.field,
                "user_id": user_id,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    # ============================================================================
    # ERROR-FIRST FILE VALIDATION (P0 Priority)
    # ============================================================================
    
    try:
        # Check file size
        if file.size > 5 * 1024 * 1024:  # 5MB limit
            raise ValidationError("File too large (maximum 5MB)", "file", "FILE_TOO_LARGE")
        
        # Validate file type
        allowed_types = {
            'image/jpeg', 'image/png', 'image/gif', 'image/webp'
        }
        
        if file.content_type not in allowed_types:
            raise ValidationError("Invalid file type. Only images are allowed", "file", "INVALID_FILE_TYPE")
        
        # Validate file extension
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}
        file_extension = os.path.splitext(filename)[1].lower()
        
        if file_extension not in allowed_extensions:
            raise ValidationError("Invalid file extension", "file", "INVALID_FILE_EXTENSION")
        
    except ValidationError as e:
        logger.warning(f"File validation error: {e.message}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": e.code,
                "message": e.message,
                "field": e.field,
                "user_id": user_id,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    # ============================================================================
    # ERROR-FIRST USER VALIDATION (P1 Priority)
    # ============================================================================
    
    try:
        # Check if user exists and is active
        user = await get_user_by_id(user_id)
        if not user:
            raise ValidationError("User not found", "user_id", "USER_NOT_FOUND")
        
        if not user.is_active:
            raise ValidationError("Account is deactivated", "user_id", "ACCOUNT_DEACTIVATED")
        
        # Check user's upload quota
        user_uploads = await get_user_upload_count(user_id, date.today())
        if user_uploads >= 10:  # 10 uploads per day
            raise ValidationError("Daily upload limit exceeded", "user_id", "UPLOAD_LIMIT_EXCEEDED")
        
    except ValidationError as e:
        logger.warning(f"User validation error: {e.message}")
        status_code = 404 if e.code == "USER_NOT_FOUND" else 401 if e.code == "ACCOUNT_DEACTIVATED" else 429
        raise HTTPException(
            status_code=status_code,
            detail={
                "error": e.code,
                "message": e.message,
                "field": e.field,
                "user_id": user_id,
                "timestamp": datetime.now().isoformat()
            }
        )
    except Exception as e:
        logger.error(f"User validation error: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "USER_VALIDATION_ERROR",
                "message": "User validation failed",
                "user_id": user_id,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    # ============================================================================
    # ERROR-FIRST POST VALIDATION (P1 Priority)
    # ============================================================================
    
    if post_id:
        try:
            # Check if post exists and user owns it
            post = await get_post_by_id(post_id)
            if not post:
                raise ValidationError("Post not found", "post_id", "POST_NOT_FOUND")
            
            if post.user_id != user_id:
                raise ValidationError("Cannot upload to another user's post", "post_id", "UNAUTHORIZED_UPLOAD")
            
        except ValidationError as e:
            logger.warning(f"Post validation error: {e.message}")
            status_code = 404 if e.code == "POST_NOT_FOUND" else 403
            raise HTTPException(
                status_code=status_code,
                detail={
                    "error": e.code,
                    "message": e.message,
                    "field": e.field,
                    "post_id": post_id,
                    "user_id": user_id,
                    "timestamp": datetime.now().isoformat()
                }
            )
        except Exception as e:
            logger.error(f"Post validation error: {e}")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "POST_VALIDATION_ERROR",
                    "message": "Post validation failed",
                    "post_id": post_id,
                    "user_id": user_id,
                    "timestamp": datetime.now().isoformat()
                }
            )
    
    # ============================================================================
    # BUSINESS LOGIC (After all validation passes)
    # ============================================================================
    
    try:
        # Generate safe filename
        file_extension = os.path.splitext(filename)[1].lower()
        safe_filename = f"{user_id}_{uuid.uuid4()}{file_extension}"
        
        # Upload file to storage
        file_url = await upload_file_to_storage(file, safe_filename)
        
        # Save file metadata to database
        file_record = await save_file_metadata(user_id, safe_filename, file_url, post_id)
        
        logger.info(f"File uploaded successfully: {safe_filename} by user {user_id}")
        
        return {
            "status": "success",
            "file_id": file_record.id,
            "file_url": file_url,
            "filename": safe_filename,
            "size": file.size,
            "content_type": file.content_type
        }
        
    except Exception as e:
        logger.error(f"File upload error: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "UPLOAD_FAILED",
                "message": "File upload failed",
                "user_id": user_id,
                "timestamp": datetime.now().isoformat()
            }
        )

async def get_user_posts_error_first(
    user_id: str, 
    requester_id: str, 
    page: int = 1, 
    limit: int = 20
) -> Dict[str, Any]:
    """
    Get user posts with error-first authentication and authorization
    """
    # ============================================================================
    # ERROR-FIRST AUTHENTICATION (P0 Priority)
    # ============================================================================
    
    try:
        # Validate requester authentication
        requester_id = SecurityValidator.validate_user_id(requester_id)
        
        # Validate target user
        user_id = SecurityValidator.validate_user_id(user_id)
        
        # Check if requester exists and is active
        requester = await get_user_by_id(requester_id)
        if not requester:
            raise ValidationError("Invalid authentication", "requester_id", "INVALID_AUTHENTICATION")
        
        if not requester.is_active:
            raise ValidationError("Account is deactivated", "requester_id", "ACCOUNT_DEACTIVATED")
        
    except ValidationError as e:
        logger.warning(f"Authentication error: {e.message}")
        status_code = 401
        raise HTTPException(
            status_code=status_code,
            detail={
                "error": e.code,
                "message": e.message,
                "field": e.field,
                "requester_id": requester_id,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    # ============================================================================
    # ERROR-FIRST AUTHORIZATION (P0 Priority)
    # ============================================================================
    
    try:
        # Check if user can access target user's posts
        if requester_id != user_id:
            # Check if target user exists
            target_user = await get_user_by_id(user_id)
            if not target_user:
                raise ValidationError("User not found", "user_id", "USER_NOT_FOUND")
            
            # Check if target user's posts are public
            if not target_user.posts_public:
                raise ValidationError("Posts are private", "user_id", "POSTS_PRIVATE")
            
            # Check if requester is blocked by target user
            if await is_user_blocked(target_user.id, requester_id):
                raise ValidationError("Access denied", "user_id", "ACCESS_DENIED")
        
    except ValidationError as e:
        logger.warning(f"Authorization error: {e.message}")
        status_code = 404 if e.code == "USER_NOT_FOUND" else 403
        raise HTTPException(
            status_code=status_code,
            detail={
                "error": e.code,
                "message": e.message,
                "field": e.field,
                "user_id": user_id,
                "requester_id": requester_id,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    # ============================================================================
    # ERROR-FIRST PARAMETER VALIDATION (P1 Priority)
    # ============================================================================
    
    try:
        # Validate pagination parameters
        if page < 1:
            raise ValidationError("Page number must be positive", "page", "INVALID_PAGE")
        
        if limit < 1 or limit > 100:
            raise ValidationError("Limit must be between 1 and 100", "limit", "INVALID_LIMIT")
        
    except ValidationError as e:
        logger.warning(f"Parameter validation error: {e.message}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": e.code,
                "message": e.message,
                "field": e.field,
                "user_id": user_id,
                "requester_id": requester_id,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    # ============================================================================
    # BUSINESS LOGIC (After all validation passes)
    # ============================================================================
    
    try:
        offset = (page - 1) * limit
        posts = await get_posts_from_database(user_id, offset, limit)
        total_count = await get_user_post_count(user_id)
        
        return {
            "posts": posts,
            "pagination": {
                "page": page,
                "limit": limit,
                "total": total_count,
                "pages": (total_count + limit - 1) // limit
            }
        }
        
    except Exception as e:
        logger.error(f"Error fetching posts: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "FETCH_FAILED",
                "message": "Failed to fetch posts",
                "user_id": user_id,
                "requester_id": requester_id,
                "timestamp": datetime.now().isoformat()
            }
        )

# ============================================================================
# MOCK FUNCTIONS FOR DEMONSTRATION
# ============================================================================

async def get_user_daily_posts(user_id: str) -> int:
    """Mock function to get user's daily post count"""
    return 0

async def check_duplicate_content(content: str, user_id: str, time_window: int) -> bool:
    """Mock function to check for duplicate content"""
    return False

async def check_rate_limit(user_id: str, endpoint: str) -> bool:
    """Mock function to check rate limits"""
    return True

async def create_post_in_database(user_id: str, content: str, hashtags: List[str], post_type: str) -> Dict[str, Any]:
    """Mock function to create post in database"""
    return {
        "id": f"post_{uuid.uuid4()}",
        "created_at": datetime.now()
    }

async def get_post_by_id(post_id: str) -> Dict[str, Any]:
    """Mock function to get post by ID"""
    return {
        "id": post_id,
        "user_id": "user123",
        "created_at": datetime.now() - timedelta(hours=1),
        "is_being_edited": False
    }

async def update_post_in_database(post_id: str, content: str) -> Dict[str, Any]:
    """Mock function to update post in database"""
    return {
        "id": post_id,
        "updated_at": datetime.now()
    }

async def log_post_update(post_id: str, user_id: str, action: str):
    """Mock function to log post updates"""
    pass

async def get_user_by_id(user_id: str) -> Dict[str, Any]:
    """Mock function to get user by ID"""
    return {
        "id": user_id,
        "is_active": True,
        "posts_public": True
    }

async def is_user_blocked(blocker_id: str, blocked_id: str) -> bool:
    """Mock function to check if user is blocked"""
    return False

async def get_posts_from_database(user_id: str, offset: int, limit: int) -> List[Dict[str, Any]]:
    """Mock function to get posts from database"""
    return []

async def get_user_post_count(user_id: str) -> int:
    """Mock function to get user post count"""
    return 0

async async def get_user_upload_count(user_id: str, date: date) -> int:
    """Mock function to get user upload count"""
    return 0

async async def upload_file_to_storage(file: UploadFile, filename: str) -> str:
    """Mock function to upload file to storage"""
    return f"https://storage.example.com/{filename}"

async def save_file_metadata(user_id: str, filename: str, file_url: str, post_id: str = None) -> Dict[str, Any]:
    """Mock function to save file metadata"""
    return {
        "id": f"file_{uuid.uuid4()}",
        "filename": filename,
        "url": file_url
    }

# ============================================================================
# USAGE EXAMPLE
# ============================================================================

async def example_usage():
    """Example of using error-first pattern functions"""
    
    # Example 1: Create post with error-first validation
    try:
        result = await create_linkedin_post_error_first(
            user_id="user123",
            content="This is a test post with proper content and #hashtags",
            hashtags=["#test", "#post"],
            post_type="general"
        )
        print(f"Post created: {result}")
        
    except HTTPException as e:
        print(f"Error creating post: {e.detail}")
    
    # Example 2: Update post with error-first validation
    try:
        result = await update_post_content_error_first(
            post_id="post_123",
            user_id="user123",
            new_content="Updated content with more words and proper formatting"
        )
        print(f"Post updated: {result}")
        
    except HTTPException as e:
        print(f"Error updating post: {e.detail}")
    
    # Example 3: Get posts with error-first validation
    try:
        result = await get_user_posts_error_first(
            user_id="user123",
            requester_id="user123",
            page=1,
            limit=10
        )
        print(f"Posts retrieved: {result}")
        
    except HTTPException as e:
        print(f"Error getting posts: {e.detail}")

match __name__:
    case "__main__":
    asyncio.run(example_usage()) 