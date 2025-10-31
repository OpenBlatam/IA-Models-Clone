from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

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
Early Return Pattern Implementation for LinkedIn Posts System
Practical examples of using early returns to avoid deeply nested if statements
"""


logger = logging.getLogger(__name__)

# ============================================================================
# EARLY RETURN UTILITY FUNCTIONS
# ============================================================================

def create_error_response(error_code: str, message: str, field: str = None, **kwargs) -> Dict[str, Any]:
    """Create standardized error response"""
    return {
        "error": error_code,
        "message": message,
        "field": field,
        "timestamp": datetime.now().isoformat(),
        "status": "failed",
        **kwargs
    }

def create_success_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """Create standardized success response"""
    return {
        **data,
        "status": "success",
        "timestamp": datetime.now().isoformat()
    }

# ============================================================================
# EARLY RETURN FUNCTION IMPLEMENTATIONS
# ============================================================================

async def create_post_early_return(
    user_id: str, 
    content: str, 
    hashtags: List[str] = None,
    post_type: str = "general"
) -> Dict[str, Any]:
    """
    Create LinkedIn post using early return pattern
    """
    # Early return for missing user ID
    if not user_id:
        return create_error_response("MISSING_USER_ID", "User ID is required", "user_id")
    
    # Early return for empty/whitespace user ID
    if not user_id.strip():
        return create_error_response("EMPTY_USER_ID", "User ID cannot be empty", "user_id")
    
    # Early return for invalid user ID format
    if not re.match(r'^[a-zA-Z0-9_-]+$', user_id.strip()):
        return create_error_response("INVALID_USER_ID_FORMAT", "Invalid user ID format", "user_id")
    
    # Early return for missing content
    if not content:
        return create_error_response("MISSING_CONTENT", "Content is required", "content")
    
    # Early return for empty content
    if not content.strip():
        return create_error_response("EMPTY_CONTENT", "Content cannot be empty", "content")
    
    # Early return for content too short
    content = content.strip()
    if len(content) < 10:
        return create_error_response("CONTENT_TOO_SHORT", "Content too short (minimum 10 characters)", "content")
    
    # Early return for content too long
    if len(content) > 3000:
        return create_error_response("CONTENT_TOO_LONG", "Content too long (maximum 3000 characters)", "content")
    
    # Early return for insufficient words
    words = content.split()
    if len(words) < 5:
        return create_error_response("INSUFFICIENT_WORDS", "Content must have at least 5 words", "content")
    
    # Early return for too many words
    if len(words) > 500:
        return create_error_response("TOO_MANY_WORDS", "Content too long (maximum 500 words)", "content")
    
    # Early return for too many hashtags
    if hashtags and len(hashtags) > 30:
        return create_error_response("TOO_MANY_HASHTAGS", "Too many hashtags (maximum 30)", "hashtags")
    
    # Early return for invalid post type
    valid_post_types = ["general", "educational", "promotional", "personal", "industry"]
    if post_type not in valid_post_types:
        return create_error_response(
            "INVALID_POST_TYPE", 
            f"Invalid post type. Must be one of: {', '.join(valid_post_types)}", 
            "post_type"
        )
    
    # Early return for user not found
    user = await get_user_by_id(user_id)
    if not user:
        return create_error_response("USER_NOT_FOUND", "User not found", "user_id")
    
    # Early return for deactivated account
    if not user.is_active:
        return create_error_response("ACCOUNT_DEACTIVATED", "Account is deactivated", "user_id")
    
    # Early return for daily post limit exceeded
    daily_posts = await get_user_daily_posts(user_id)
    if daily_posts >= 5:
        return create_error_response("DAILY_LIMIT_EXCEEDED", "Daily post limit exceeded (5 posts per day)", "user_id")
    
    # Early return for duplicate content
    if await is_duplicate_content(content, user_id):
        return create_error_response("DUPLICATE_CONTENT", "Duplicate content detected within time window", "content")
    
    # Early return for rate limit exceeded
    if not await check_rate_limit(user_id, "post_creation"):
        return create_error_response("RATE_LIMIT_EXCEEDED", "Rate limit exceeded. Try again later.", "user_id")
    
    # All validation passed - create post
    try:
        post_data = await create_post_in_database(user_id, content, hashtags, post_type)
        logger.info(f"Post created successfully: {post_data['id']} by user {user_id}")
        
        return create_success_response({
            "post_id": post_data["id"],
            "message": "Post created successfully",
            "created_at": post_data["created_at"],
            "hashtags": hashtags or []
        })
        
    except Exception as e:
        logger.error(f"Failed to create post for user {user_id}: {e}")
        return create_error_response("CREATION_FAILED", f"Failed to create post: {e}", "content")

async def update_post_early_return(
    post_id: str, 
    user_id: str, 
    new_content: str
) -> Dict[str, Any]:
    """
    Update post content using early return pattern
    """
    # Early return for missing post ID
    if not post_id:
        return create_error_response("MISSING_POST_ID", "Post ID is required", "post_id")
    
    # Early return for empty post ID
    if not post_id.strip():
        return create_error_response("EMPTY_POST_ID", "Post ID cannot be empty", "post_id")
    
    # Early return for invalid post ID format
    if not re.match(r'^[a-zA-Z0-9_-]+$', post_id.strip()):
        return create_error_response("INVALID_POST_ID_FORMAT", "Invalid post ID format", "post_id")
    
    # Early return for missing user ID
    if not user_id:
        return create_error_response("MISSING_USER_ID", "User ID is required", "user_id")
    
    # Early return for empty user ID
    if not user_id.strip():
        return create_error_response("EMPTY_USER_ID", "User ID cannot be empty", "user_id")
    
    # Early return for missing content
    if not new_content:
        return create_error_response("MISSING_CONTENT", "New content is required", "content")
    
    # Early return for empty content
    if not new_content.strip():
        return create_error_response("EMPTY_CONTENT", "New content cannot be empty", "content")
    
    # Early return for content too short
    new_content = new_content.strip()
    if len(new_content) < 10:
        return create_error_response("CONTENT_TOO_SHORT", "Content too short (minimum 10 characters)", "content")
    
    # Early return for content too long
    if len(new_content) > 3000:
        return create_error_response("CONTENT_TOO_LONG", "Content too long (maximum 3000 characters)", "content")
    
    # Early return for insufficient words
    words = new_content.split()
    if len(words) < 5:
        return create_error_response("INSUFFICIENT_WORDS", "Content must have at least 5 words", "content")
    
    # Early return for post not found
    post = await get_post_by_id(post_id)
    if not post:
        return create_error_response("POST_NOT_FOUND", "Post not found", "post_id")
    
    # Early return for unauthorized update
    if post.user_id != user_id:
        return create_error_response("UNAUTHORIZED_UPDATE", "Cannot update another user's post", "user_id")
    
    # Early return for post too old
    post_age = datetime.now() - post.created_at
    if post_age > timedelta(hours=24):
        return create_error_response("POST_TOO_OLD", "Post cannot be edited after 24 hours", "post_id")
    
    # Early return for post being edited
    if post.is_being_edited:
        return create_error_response("POST_BEING_EDITED", "Post is currently being edited", "post_id")
    
    # All validation passed - update post
    try:
        updated_post = await update_post_in_database(post_id, new_content)
        await log_post_update(post_id, user_id, "content_updated")
        
        return create_success_response({
            "post_id": post_id,
            "message": "Post content updated successfully",
            "updated_at": updated_post.updated_at.isoformat()
        })
        
    except Exception as e:
        logger.error(f"Failed to update post {post_id}: {e}")
        return create_error_response("UPDATE_FAILED", f"Failed to update post: {e}", "content")

async async def upload_file_early_return(
    user_id: str, 
    file: UploadFile, 
    post_id: str = None
) -> Dict[str, Any]:
    """
    Upload file using early return pattern
    """
    # Early return for missing user ID
    if not user_id:
        return create_error_response("MISSING_USER_ID", "User ID is required", "user_id")
    
    # Early return for empty user ID
    if not user_id.strip():
        return create_error_response("EMPTY_USER_ID", "User ID cannot be empty", "user_id")
    
    # Early return for missing file
    if not file:
        return create_error_response("MISSING_FILE", "File is required", "file")
    
    # Early return for missing filename
    if not file.filename:
        return create_error_response("MISSING_FILENAME", "Filename is required", "file")
    
    # Early return for file too large
    if file.size > 5 * 1024 * 1024:  # 5MB
        return create_error_response("FILE_TOO_LARGE", "File too large (maximum 5MB)", "file")
    
    # Early return for invalid file type
    allowed_types = {'image/jpeg', 'image/png', 'image/gif', 'image/webp'}
    if file.content_type not in allowed_types:
        return create_error_response("INVALID_FILE_TYPE", "Invalid file type. Only images are allowed", "file")
    
    # Early return for invalid file extension
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in allowed_extensions:
        return create_error_response("INVALID_FILE_EXTENSION", "Invalid file extension", "file")
    
    # Early return for malicious filename
    if '..' in file.filename or '/' in file.filename or '\\' in file.filename:
        return create_error_response("MALICIOUS_FILENAME", "Invalid filename", "file")
    
    # Early return for user not found
    user = await get_user_by_id(user_id)
    if not user:
        return create_error_response("USER_NOT_FOUND", "User not found", "user_id")
    
    # Early return for deactivated account
    if not user.is_active:
        return create_error_response("ACCOUNT_DEACTIVATED", "Account is deactivated", "user_id")
    
    # Early return for upload limit exceeded
    user_uploads = await get_user_upload_count(user_id, date.today())
    if user_uploads >= 10:
        return create_error_response("UPLOAD_LIMIT_EXCEEDED", "Daily upload limit exceeded (10 uploads per day)", "user_id")
    
    # Early return for post validation (if post_id provided)
    if post_id:
        # Early return for invalid post ID
        if not post_id.strip():
            return create_error_response("INVALID_POST_ID", "Invalid post ID", "post_id")
        
        # Early return for post not found
        post = await get_post_by_id(post_id)
        if not post:
            return create_error_response("POST_NOT_FOUND", "Post not found", "post_id")
        
        # Early return for unauthorized upload
        if post.user_id != user_id:
            return create_error_response("UNAUTHORIZED_UPLOAD", "Cannot upload to another user's post", "post_id")
    
    # All validation passed - upload file
    try:
        # Generate safe filename
        safe_filename = f"{user_id}_{uuid.uuid4()}{file_extension}"
        
        # Upload file to storage
        file_url = await upload_file_to_storage(file, safe_filename)
        
        # Save file metadata
        file_record = await save_file_metadata(user_id, safe_filename, file_url, post_id)
        
        logger.info(f"File uploaded successfully: {safe_filename} by user {user_id}")
        
        return create_success_response({
            "file_id": file_record.id,
            "file_url": file_url,
            "filename": safe_filename,
            "size": file.size,
            "content_type": file.content_type
        })
        
    except Exception as e:
        logger.error(f"Failed to upload file for user {user_id}: {e}")
        return create_error_response("UPLOAD_FAILED", f"File upload failed: {e}", "file")

async def get_user_posts_early_return(
    user_id: str, 
    requester_id: str, 
    page: int = 1, 
    limit: int = 20
) -> Dict[str, Any]:
    """
    Get user posts using early return pattern
    """
    # Early return for missing requester ID
    if not requester_id:
        return create_error_response("MISSING_REQUESTER_ID", "Requester ID is required", "requester_id")
    
    # Early return for empty requester ID
    if not requester_id.strip():
        return create_error_response("EMPTY_REQUESTER_ID", "Requester ID cannot be empty", "requester_id")
    
    # Early return for missing user ID
    if not user_id:
        return create_error_response("MISSING_USER_ID", "User ID is required", "user_id")
    
    # Early return for empty user ID
    if not user_id.strip():
        return create_error_response("EMPTY_USER_ID", "User ID cannot be empty", "user_id")
    
    # Early return for invalid page number
    if page < 1:
        return create_error_response("INVALID_PAGE", "Page number must be positive", "page")
    
    # Early return for invalid limit
    if limit < 1 or limit > 100:
        return create_error_response("INVALID_LIMIT", "Limit must be between 1 and 100", "limit")
    
    # Early return for invalid authentication
    requester = await get_user_by_id(requester_id)
    if not requester:
        return create_error_response("INVALID_AUTHENTICATION", "Invalid authentication", "requester_id")
    
    # Early return for deactivated account
    if not requester.is_active:
        return create_error_response("ACCOUNT_DEACTIVATED", "Account is deactivated", "requester_id")
    
    # Early return for user not found
    target_user = await get_user_by_id(user_id)
    if not target_user:
        return create_error_response("USER_NOT_FOUND", "User not found", "user_id")
    
    # Early return for private posts
    if requester_id != user_id and not target_user.posts_public:
        return create_error_response("POSTS_PRIVATE", "Posts are private", "user_id")
    
    # Early return for blocked user
    if await is_user_blocked(target_user.id, requester_id):
        return create_error_response("ACCESS_DENIED", "Access denied", "user_id")
    
    # All validation passed - get posts
    try:
        offset = (page - 1) * limit
        posts = await get_posts_from_database(user_id, offset, limit)
        total_count = await get_user_post_count(user_id)
        
        return create_success_response({
            "posts": posts,
            "pagination": {
                "page": page,
                "limit": limit,
                "total": total_count,
                "pages": (total_count + limit - 1) // limit
            }
        })
        
    except Exception as e:
        logger.error(f"Failed to fetch posts for user {user_id}: {e}")
        return create_error_response("FETCH_FAILED", f"Failed to fetch posts: {e}", "user_id")

async def process_engagement_early_return(
    post_id: str, 
    user_id: str, 
    action: str,
    comment_text: str = None
) -> Dict[str, Any]:
    """
    Process post engagement using early return pattern
    """
    # Early return for missing post ID
    if not post_id:
        return create_error_response("MISSING_POST_ID", "Post ID is required", "post_id")
    
    # Early return for missing user ID
    if not user_id:
        return create_error_response("MISSING_USER_ID", "User ID is required", "user_id")
    
    # Early return for missing action
    if not action:
        return create_error_response("MISSING_ACTION", "Action is required", "action")
    
    # Early return for invalid action
    valid_actions = ['like', 'share', 'comment']
    if action not in valid_actions:
        return create_error_response(
            "INVALID_ACTION", 
            f"Invalid action. Must be one of: {', '.join(valid_actions)}", 
            "action"
        )
    
    # Early return for missing comment text
    if action == 'comment' and not comment_text:
        return create_error_response("MISSING_COMMENT", "Comment text is required for comment action", "comment_text")
    
    # Early return for comment too long
    if action == 'comment' and comment_text and len(comment_text) > 1000:
        return create_error_response("COMMENT_TOO_LONG", "Comment too long (maximum 1000 characters)", "comment_text")
    
    # Early return for user not found
    user = await get_user_by_id(user_id)
    if not user:
        return create_error_response("USER_NOT_FOUND", "User not found", "user_id")
    
    # Early return for deactivated account
    if not user.is_active:
        return create_error_response("ACCOUNT_DEACTIVATED", "Account is deactivated", "user_id")
    
    # Early return for post not found
    post = await get_post_by_id(post_id)
    if not post:
        return create_error_response("POST_NOT_FOUND", "Post not found", "post_id")
    
    # Early return for deleted post
    if post.is_deleted:
        return create_error_response("POST_DELETED", "Post is deleted", "post_id")
    
    # Early return for engagement limit exceeded
    if not await check_user_engagement_limit(user_id):
        return create_error_response("ENGAGEMENT_LIMIT_EXCEEDED", "Engagement limit exceeded", "user_id")
    
    # Early return for already liked
    if action == 'like' and await has_user_liked_post(user_id, post_id):
        return create_error_response("ALREADY_LIKED", "Post already liked", "post_id")
    
    # All validation passed - process engagement
    try:
        if action == 'like':
            await add_like_to_post(user_id, post_id)
            return create_success_response({"action": "liked", "post_id": post_id})
        
        elif action == 'share':
            await add_share_to_post(user_id, post_id)
            return create_success_response({"action": "shared", "post_id": post_id})
        
        else:  # comment
            await add_comment_to_post(user_id, post_id, comment_text)
            return create_success_response({"action": "commented", "post_id": post_id})
        
    except Exception as e:
        logger.error(f"Failed to {action} post {post_id} by user {user_id}: {e}")
        return create_error_response("ENGAGEMENT_FAILED", f"Failed to {action} post: {e}", "action")

# ============================================================================
# GUARD CLAUSE PATTERN IMPLEMENTATION
# ============================================================================

async def create_post_with_guards(
    user_id: str, 
    content: str, 
    hashtags: List[str] = None,
    post_type: str = "general"
) -> Dict[str, Any]:
    """
    Create post using guard clause pattern
    """
    # Guard clause for missing user ID
    if not user_id or not user_id.strip():
        return create_error_response("MISSING_USER_ID", "User ID is required", "user_id")
    
    # Guard clause for missing content
    if not content or not content.strip():
        return create_error_response("MISSING_CONTENT", "Content is required", "content")
    
    # Guard clause for content length
    content = content.strip()
    if len(content) < 10:
        return create_error_response("CONTENT_TOO_SHORT", "Content too short (minimum 10 characters)", "content")
    
    if len(content) > 3000:
        return create_error_response("CONTENT_TOO_LONG", "Content too long (maximum 3000 characters)", "content")
    
    # Guard clause for user existence
    user = await get_user_by_id(user_id)
    if not user:
        return create_error_response("USER_NOT_FOUND", "User not found", "user_id")
    
    # Guard clause for user status
    if not user.is_active:
        return create_error_response("ACCOUNT_DEACTIVATED", "Account is deactivated", "user_id")
    
    # Guard clause for daily post limit
    daily_posts = await get_user_daily_posts(user_id)
    if daily_posts >= 5:
        return create_error_response("DAILY_LIMIT_EXCEEDED", "Daily post limit exceeded", "user_id")
    
    # Guard clause for duplicate content
    if await is_duplicate_content(content, user_id):
        return create_error_response("DUPLICATE_CONTENT", "Duplicate content detected", "content")
    
    # Guard clause for rate limiting
    if not await check_rate_limit(user_id, "post_creation"):
        return create_error_response("RATE_LIMIT_EXCEEDED", "Rate limit exceeded", "user_id")
    
    # All guards passed - create post
    try:
        post_data = await create_post_in_database(user_id, content, hashtags, post_type)
        return create_success_response({
            "post_id": post_data["id"],
            "message": "Post created successfully"
        })
    except Exception as e:
        return create_error_response("CREATION_FAILED", f"Failed to create post: {e}", "content")

# ============================================================================
# MOCK FUNCTIONS FOR DEMONSTRATION
# ============================================================================

async def get_user_by_id(user_id: str) -> Dict[str, Any]:
    """Mock function to get user by ID"""
    return {
        "id": user_id,
        "is_active": True,
        "posts_public": True
    }

async def get_user_daily_posts(user_id: str) -> int:
    """Mock function to get user's daily post count"""
    return 0

async def is_duplicate_content(content: str, user_id: str) -> bool:
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
        "is_being_edited": False,
        "is_deleted": False
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

async def check_user_engagement_limit(user_id: str) -> bool:
    """Mock function to check user engagement limit"""
    return True

async def has_user_liked_post(user_id: str, post_id: str) -> bool:
    """Mock function to check if user has liked post"""
    return False

async def add_like_to_post(user_id: str, post_id: str):
    """Mock function to add like to post"""
    pass

async def add_share_to_post(user_id: str, post_id: str):
    """Mock function to add share to post"""
    pass

async def add_comment_to_post(user_id: str, post_id: str, comment_text: str):
    """Mock function to add comment to post"""
    pass

# ============================================================================
# USAGE EXAMPLE
# ============================================================================

async def example_usage():
    """Example of using early return pattern functions"""
    
    print("=== Early Return Pattern Examples ===\n")
    
    # Example 1: Create post with early returns
    print("1. Creating post with early returns:")
    result = await create_post_early_return(
        user_id="user123",
        content="This is a test post with proper content and #hashtags",
        hashtags=["#test", "#post"],
        post_type="general"
    )
    print(f"Result: {result}\n")
    
    # Example 2: Create post with missing content (early return)
    print("2. Creating post with missing content (early return):")
    result = await create_post_early_return(
        user_id="user123",
        content="",  # Empty content should trigger early return
        hashtags=["#test"],
        post_type="general"
    )
    print(f"Result: {result}\n")
    
    # Example 3: Update post with early returns
    print("3. Updating post with early returns:")
    result = await update_post_early_return(
        post_id="post_123",
        user_id="user123",
        new_content="Updated content with more words and proper formatting"
    )
    print(f"Result: {result}\n")
    
    # Example 4: Get posts with early returns
    print("4. Getting posts with early returns:")
    result = await get_user_posts_early_return(
        user_id="user123",
        requester_id="user123",
        page=1,
        limit=10
    )
    print(f"Result: {result}\n")
    
    # Example 5: Process engagement with early returns
    print("5. Processing engagement with early returns:")
    result = await process_engagement_early_return(
        post_id="post_123",
        user_id="user123",
        action="like"
    )
    print(f"Result: {result}\n")
    
    # Example 6: Guard clause pattern
    print("6. Guard clause pattern:")
    result = await create_post_with_guards(
        user_id="user123",
        content="This is a test post using guard clauses",
        hashtags=["#guard", "#clause"],
        post_type="general"
    )
    print(f"Result: {result}\n")

match __name__:
    case "__main__":
    asyncio.run(example_usage()) 