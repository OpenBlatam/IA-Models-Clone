from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import asyncio
import logging
import uuid
from datetime import datetime, timedelta, date
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
            import os
from typing import Any, List, Dict, Optional
"""
Happy Path Implementation: Place the Happy Path Last

This module demonstrates how to structure functions with the happy path last
for improved readability, maintainability, and debugging.
"""


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_error_response(error_code: str, message: str, field: str = None) -> Dict[str, Any]:
    """Create standardized error response"""
    response = {
        "status": "failed",
        "error": {
            "code": error_code,
            "message": message
        }
    }
    if field:
        response["error"]["field"] = field
    return response

def create_success_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """Create standardized success response"""
    return {
        "status": "success",
        "data": data,
        "timestamp": datetime.now().isoformat()
    }

# ============================================================================
# MOCK DATABASE FUNCTIONS
# ============================================================================

async def get_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    """Mock function to get user by ID"""
    # Simulate database lookup
    await asyncio.sleep(0.01)
    if user_id == "valid_user":
        return {
            "id": user_id,
            "name": "John Doe",
            "is_active": True,
            "email": "john@example.com"
        }
    elif user_id == "inactive_user":
        return {
            "id": user_id,
            "name": "Jane Doe",
            "is_active": False,
            "email": "jane@example.com"
        }
    return None

async def get_post_by_id(post_id: str) -> Optional[Dict[str, Any]]:
    """Mock function to get post by ID"""
    await asyncio.sleep(0.01)
    if post_id == "valid_post":
        return {
            "id": post_id,
            "user_id": "valid_user",
            "content": "Original content",
            "created_at": datetime.now() - timedelta(hours=2),
            "is_deleted": False,
            "is_being_edited": False
        }
    elif post_id == "old_post":
        return {
            "id": post_id,
            "user_id": "valid_user",
            "content": "Old content",
            "created_at": datetime.now() - timedelta(days=2),
            "is_deleted": False,
            "is_being_edited": False
        }
    return None

async def get_user_daily_posts(user_id: str) -> int:
    """Mock function to get user's daily post count"""
    await asyncio.sleep(0.01)
    return 2  # Mock: user has 2 posts today

async async def get_user_upload_count(user_id: str, upload_date: date) -> int:
    """Mock function to get user's upload count for a date"""
    await asyncio.sleep(0.01)
    return 3  # Mock: user has 3 uploads today

async def has_user_liked_post(user_id: str, post_id: str) -> bool:
    """Mock function to check if user has liked a post"""
    await asyncio.sleep(0.01)
    return False  # Mock: user hasn't liked the post

async def check_rate_limit(user_id: str, action: str) -> bool:
    """Mock function to check rate limit"""
    await asyncio.sleep(0.01)
    return True  # Mock: rate limit not exceeded

async def check_user_engagement_limit(user_id: str) -> bool:
    """Mock function to check user engagement limit"""
    await asyncio.sleep(0.01)
    return True  # Mock: engagement limit not exceeded

async def is_duplicate_content(content: str, user_id: str) -> bool:
    """Mock function to check for duplicate content"""
    await asyncio.sleep(0.01)
    return False  # Mock: not duplicate

async def is_user_blocked(blocker_id: str, blocked_id: str) -> bool:
    """Mock function to check if user is blocked"""
    await asyncio.sleep(0.01)
    return False  # Mock: not blocked

# ============================================================================
# MOCK BUSINESS LOGIC FUNCTIONS
# ============================================================================

async def create_post_in_database(user_id: str, content: str, hashtags: List[str] = None) -> Dict[str, Any]:
    """Mock function to create post in database"""
    await asyncio.sleep(0.05)
    return {
        "id": f"post_{uuid.uuid4().hex[:8]}",
        "user_id": user_id,
        "content": content,
        "hashtags": hashtags or [],
        "created_at": datetime.now()
    }

async def update_post_in_database(post_id: str, new_content: str) -> Dict[str, Any]:
    """Mock function to update post in database"""
    await asyncio.sleep(0.05)
    return {
        "id": post_id,
        "content": new_content,
        "updated_at": datetime.now()
    }

async async def upload_file_to_storage(file, filename: str) -> str:
    """Mock function to upload file to storage"""
    await asyncio.sleep(0.1)
    return f"https://storage.example.com/files/{filename}"

async def save_file_metadata(user_id: str, filename: str, file_url: str, post_id: str = None) -> Dict[str, Any]:
    """Mock function to save file metadata"""
    await asyncio.sleep(0.02)
    return {
        "id": f"file_{uuid.uuid4().hex[:8]}",
        "user_id": user_id,
        "filename": filename,
        "file_url": file_url,
        "post_id": post_id
    }

async def get_posts_from_database(user_id: str) -> List[Dict[str, Any]]:
    """Mock function to get posts from database"""
    await asyncio.sleep(0.02)
    return [
        {"id": "post1", "content": "First post", "created_at": datetime.now()},
        {"id": "post2", "content": "Second post", "created_at": datetime.now()}
    ]

async def get_user_post_count(user_id: str) -> int:
    """Mock function to get user's total post count"""
    await asyncio.sleep(0.01)
    return 15

async def add_like_to_post(user_id: str, post_id: str) -> None:
    """Mock function to add like to post"""
    await asyncio.sleep(0.02)
    logger.info(f"User {user_id} liked post {post_id}")

async def add_share_to_post(user_id: str, post_id: str) -> None:
    """Mock function to add share to post"""
    await asyncio.sleep(0.02)
    logger.info(f"User {user_id} shared post {post_id}")

async def add_comment_to_post(user_id: str, post_id: str, comment_text: str) -> None:
    """Mock function to add comment to post"""
    await asyncio.sleep(0.02)
    logger.info(f"User {user_id} commented on post {post_id}: {comment_text}")

# ============================================================================
# MOCK NOTIFICATION AND ANALYTICS FUNCTIONS
# ============================================================================

async def send_notification(user_id: str, message: str) -> None:
    """Mock function to send notification"""
    await asyncio.sleep(0.01)
    logger.info(f"Notification sent to {user_id}: {message}")

async def update_user_analytics(user_id: str, action: str) -> None:
    """Mock function to update user analytics"""
    await asyncio.sleep(0.01)
    logger.info(f"Analytics updated for {user_id}: {action}")

async def update_view_analytics(viewer_id: str, post_owner_id: str) -> None:
    """Mock function to update view analytics"""
    await asyncio.sleep(0.01)
    logger.info(f"View analytics updated: {viewer_id} viewed {post_owner_id}'s posts")

async def update_post_analytics(post_id: str, action: str) -> None:
    """Mock function to update post analytics"""
    await asyncio.sleep(0.01)
    logger.info(f"Post analytics updated for {post_id}: {action}")

async def update_engagement_analytics(post_id: str, user_id: str, action: str) -> None:
    """Mock function to update engagement analytics"""
    await asyncio.sleep(0.01)
    logger.info(f"Engagement analytics updated: {user_id} {action} post {post_id}")

async def log_post_update(post_id: str, user_id: str, action: str) -> None:
    """Mock function to log post update"""
    await asyncio.sleep(0.01)
    logger.info(f"Post update logged: {post_id} by {user_id} - {action}")

async async def increment_user_upload_count(user_id: str, upload_date: date) -> None:
    """Mock function to increment user upload count"""
    await asyncio.sleep(0.01)
    logger.info(f"Upload count incremented for {user_id} on {upload_date}")

# ============================================================================
# HAPPY PATH IMPLEMENTATIONS
# ============================================================================

class PostService:
    """Service class demonstrating happy path last pattern"""
    
    @staticmethod
    async def create_post_happy_path_last(user_id: str, content: str, hashtags: List[str] = None) -> Dict[str, Any]:
        """
        Create a new post with happy path last pattern.
        
        All validations and edge cases are handled first,
        then the main business logic (happy path) is executed last.
        """
        # ============================================================================
        # VALIDATION PHASE (Handle all edge cases first)
        # ============================================================================
        
        # Input validation
        if not user_id:
            return create_error_response("MISSING_USER_ID", "User ID is required", "user_id")
        
        if not content:
            return create_error_response("MISSING_CONTENT", "Content is required", "content")
        
        # Content validation
        content = content.strip()
        if len(content) < 10:
            return create_error_response("CONTENT_TOO_SHORT", "Content too short (minimum 10 characters)", "content")
        
        if len(content) > 3000:
            return create_error_response("CONTENT_TOO_LONG", "Content too long (maximum 3000 characters)", "content")
        
        # User validation
        user = await get_user_by_id(user_id)
        if not user:
            return create_error_response("USER_NOT_FOUND", "User not found", "user_id")
        
        if not user["is_active"]:
            return create_error_response("USER_INACTIVE", "Account is deactivated", "user_id")
        
        # Business rule validation
        daily_posts = await get_user_daily_posts(user_id)
        if daily_posts >= 5:
            return create_error_response("DAILY_LIMIT_EXCEEDED", "Daily post limit exceeded (maximum 5 posts)", "user_id")
        
        if await is_duplicate_content(content, user_id):
            return create_error_response("DUPLICATE_CONTENT", "Duplicate content detected", "content")
        
        # Rate limiting validation
        if not await check_rate_limit(user_id, "post_creation"):
            return create_error_response("RATE_LIMIT_EXCEEDED", "Rate limit exceeded", "user_id")
        
        # ============================================================================
        # HAPPY PATH (Main business logic last)
        # ============================================================================
        
        try:
            # Create the post
            post_data = await create_post_in_database(user_id, content, hashtags)
            
            # Send notifications
            await send_notification(user_id, "Post created successfully")
            
            # Update analytics
            await update_user_analytics(user_id, "post_created")
            
            # Return success response
            return create_success_response({
                "post_id": post_data["id"],
                "message": "Post created successfully",
                "created_at": post_data["created_at"].isoformat(),
                "content_length": len(content)
            })
            
        except Exception as e:
            logger.error(f"Error in happy path: {e}")
            return create_error_response("CREATION_FAILED", f"Failed to create post: {str(e)}")
    
    @staticmethod
    async def get_user_posts_happy_path_last(user_id: str, requester_id: str) -> Dict[str, Any]:
        """
        Get user posts with happy path last pattern.
        
        All authentication, authorization, and validation checks first,
        then the main data retrieval logic last.
        """
        # ============================================================================
        # VALIDATION PHASE (Handle all edge cases first)
        # ============================================================================
        
        # Input validation
        if not requester_id:
            return create_error_response("MISSING_REQUESTER_ID", "Requester ID is required", "requester_id")
        
        if not user_id:
            return create_error_response("MISSING_USER_ID", "User ID is required", "user_id")
        
        # Authentication validation
        requester = await get_user_by_id(requester_id)
        if not requester:
            return create_error_response("INVALID_AUTHENTICATION", "Invalid authentication", "requester_id")
        
        if not requester["is_active"]:
            return create_error_response("ACCOUNT_DEACTIVATED", "Account is deactivated", "requester_id")
        
        # Authorization validation
        target_user = await get_user_by_id(user_id)
        if not target_user:
            return create_error_response("USER_NOT_FOUND", "User not found", "user_id")
        
        # Check if posts are private and requester is not the owner
        if requester_id != user_id:
            # Mock: assume posts are private for demonstration
            return create_error_response("POSTS_PRIVATE", "Posts are private", "user_id")
        
        if await is_user_blocked(target_user["id"], requester_id):
            return create_error_response("ACCESS_DENIED", "Access denied", "user_id")
        
        # ============================================================================
        # HAPPY PATH (Main business logic last)
        # ============================================================================
        
        try:
            # Fetch posts
            posts = await get_posts_from_database(user_id)
            
            # Get total count
            total_count = await get_user_post_count(user_id)
            
            # Update analytics
            await update_view_analytics(requester_id, user_id)
            
            # Return success response
            return create_success_response({
                "posts": posts,
                "total_count": total_count,
                "user_id": user_id,
                "posts_count": len(posts)
            })
            
        except Exception as e:
            logger.error(f"Error in happy path: {e}")
            return create_error_response("FETCH_FAILED", f"Failed to fetch posts: {str(e)}")
    
    @staticmethod
    async async def upload_file_happy_path_last(user_id: str, file, post_id: str = None) -> Dict[str, Any]:
        """
        Upload file with happy path last pattern.
        
        All file validation, user validation, and quota checks first,
        then the main upload logic last.
        """
        # ============================================================================
        # VALIDATION PHASE (Handle all edge cases first)
        # ============================================================================
        
        # Input validation
        if not user_id:
            return create_error_response("MISSING_USER_ID", "User ID is required", "user_id")
        
        if not file:
            return create_error_response("MISSING_FILE", "File is required", "file")
        
        if not hasattr(file, 'filename') or not file.filename:
            return create_error_response("MISSING_FILENAME", "Filename is required", "file")
        
        # File validation
        if hasattr(file, 'size') and file.size > 5 * 1024 * 1024:  # 5MB
            return create_error_response("FILE_TOO_LARGE", "File too large (maximum 5MB)", "file")
        
        allowed_types = {'image/jpeg', 'image/png', 'image/gif', 'image/webp'}
        if hasattr(file, 'content_type') and file.content_type not in allowed_types:
            return create_error_response("INVALID_FILE_TYPE", "Invalid file type. Only images are allowed", "file")
        
        # User validation
        user = await get_user_by_id(user_id)
        if not user:
            return create_error_response("USER_NOT_FOUND", "User not found", "user_id")
        
        if not user["is_active"]:
            return create_error_response("USER_INACTIVE", "Account is deactivated", "user_id")
        
        # Upload quota validation
        user_uploads = await get_user_upload_count(user_id, date.today())
        if user_uploads >= 10:
            return create_error_response("UPLOAD_LIMIT_EXCEEDED", "Daily upload limit exceeded (maximum 10 files)", "user_id")
        
        # Post validation (if applicable)
        if post_id:
            post = await get_post_by_id(post_id)
            if not post:
                return create_error_response("POST_NOT_FOUND", "Post not found", "post_id")
            
            if post["user_id"] != user_id:
                return create_error_response("UNAUTHORIZED_POST", "Cannot upload to another user's post", "post_id")
        
        # ============================================================================
        # HAPPY PATH (Main business logic last)
        # ============================================================================
        
        try:
            # Generate safe filename
            file_extension = os.path.splitext(file.filename)[1].lower()
            safe_filename = f"{user_id}_{uuid.uuid4()}{file_extension}"
            
            # Upload file to storage
            file_url = await upload_file_to_storage(file, safe_filename)
            
            # Save file metadata
            file_record = await save_file_metadata(user_id, safe_filename, file_url, post_id)
            
            # Update user upload count
            await increment_user_upload_count(user_id, date.today())
            
            # Send notification
            await send_notification(user_id, "File uploaded successfully")
            
            # Return success response
            return create_success_response({
                "file_id": file_record["id"],
                "file_url": file_url,
                "filename": safe_filename,
                "size": getattr(file, 'size', 0),
                "content_type": getattr(file, 'content_type', 'unknown')
            })
            
        except Exception as e:
            logger.error(f"Error in happy path: {e}")
            return create_error_response("UPLOAD_FAILED", f"File upload failed: {str(e)}")
    
    @staticmethod
    async def update_post_happy_path_last(post_id: str, user_id: str, new_content: str) -> Dict[str, Any]:
        """
        Update post with happy path last pattern.
        
        All validation, authorization, and business rule checks first,
        then the main update logic last.
        """
        # ============================================================================
        # VALIDATION PHASE (Handle all edge cases first)
        # ============================================================================
        
        # Input validation
        if not post_id:
            return create_error_response("MISSING_POST_ID", "Post ID is required", "post_id")
        
        if not user_id:
            return create_error_response("MISSING_USER_ID", "User ID is required", "user_id")
        
        if not new_content:
            return create_error_response("MISSING_CONTENT", "New content is required", "new_content")
        
        # Content validation
        new_content = new_content.strip()
        if len(new_content) < 10:
            return create_error_response("CONTENT_TOO_SHORT", "Content too short (minimum 10 characters)", "new_content")
        
        if len(new_content) > 3000:
            return create_error_response("CONTENT_TOO_LONG", "Content too long (maximum 3000 characters)", "new_content")
        
        # Database validation
        post = await get_post_by_id(post_id)
        if not post:
            return create_error_response("POST_NOT_FOUND", "Post not found", "post_id")
        
        if post["user_id"] != user_id:
            return create_error_response("UNAUTHORIZED_UPDATE", "Cannot update another user's post", "post_id")
        
        # Business rule validation
        post_age = datetime.now() - post["created_at"]
        if post_age > timedelta(hours=24):
            return create_error_response("EDIT_TIME_EXPIRED", "Post cannot be edited after 24 hours", "post_id")
        
        if post["is_being_edited"]:
            return create_error_response("POST_BEING_EDITED", "Post is currently being edited", "post_id")
        
        # ============================================================================
        # HAPPY PATH (Main business logic last)
        # ============================================================================
        
        try:
            # Update post content
            updated_post = await update_post_in_database(post_id, new_content)
            
            # Log the update
            await log_post_update(post_id, user_id, "content_updated")
            
            # Update analytics
            await update_post_analytics(post_id, "content_updated")
            
            # Send notification
            await send_notification(user_id, "Post updated successfully")
            
            # Return success response
            return create_success_response({
                "post_id": post_id,
                "message": "Post content updated successfully",
                "updated_at": updated_post["updated_at"].isoformat(),
                "content_length": len(new_content)
            })
            
        except Exception as e:
            logger.error(f"Error in happy path: {e}")
            return create_error_response("UPDATE_FAILED", f"Failed to update post: {str(e)}")
    
    @staticmethod
    async def process_engagement_happy_path_last(post_id: str, user_id: str, action: str, comment_text: str = None) -> Dict[str, Any]:
        """
        Process engagement with happy path last pattern.
        
        All validation, authorization, and business rule checks first,
        then the main engagement logic last.
        """
        # ============================================================================
        # VALIDATION PHASE (Handle all edge cases first)
        # ============================================================================
        
        # Input validation
        if not post_id:
            return create_error_response("MISSING_POST_ID", "Post ID is required", "post_id")
        
        if not user_id:
            return create_error_response("MISSING_USER_ID", "User ID is required", "user_id")
        
        if not action:
            return create_error_response("MISSING_ACTION", "Action is required", "action")
        
        # Action validation
        valid_actions = ['like', 'share', 'comment']
        if action not in valid_actions:
            return create_error_response("INVALID_ACTION", f"Invalid action. Must be one of: {', '.join(valid_actions)}", "action")
        
        # Comment validation
        if action == 'comment' and not comment_text:
            return create_error_response("MISSING_COMMENT", "Comment text is required for comment action", "comment_text")
        
        if action == 'comment' and comment_text and len(comment_text) > 1000:
            return create_error_response("COMMENT_TOO_LONG", "Comment too long (maximum 1000 characters)", "comment_text")
        
        # User validation
        user = await get_user_by_id(user_id)
        if not user:
            return create_error_response("USER_NOT_FOUND", "User not found", "user_id")
        
        if not user["is_active"]:
            return create_error_response("USER_INACTIVE", "Account is deactivated", "user_id")
        
        # Post validation
        post = await get_post_by_id(post_id)
        if not post:
            return create_error_response("POST_NOT_FOUND", "Post not found", "post_id")
        
        if post["is_deleted"]:
            return create_error_response("POST_DELETED", "Post is deleted", "post_id")
        
        # Engagement limit validation
        if not await check_user_engagement_limit(user_id):
            return create_error_response("ENGAGEMENT_LIMIT_EXCEEDED", "Engagement limit exceeded", "user_id")
        
        # Duplicate like validation
        if action == 'like' and await has_user_liked_post(user_id, post_id):
            return create_error_response("ALREADY_LIKED", "Post already liked", "post_id")
        
        # ============================================================================
        # HAPPY PATH (Main business logic last)
        # ============================================================================
        
        try:
            # Process the engagement
            if action == 'like':
                await add_like_to_post(user_id, post_id)
                engagement_type = "liked"
            elif action == 'share':
                await add_share_to_post(user_id, post_id)
                engagement_type = "shared"
            else:  # comment
                await add_comment_to_post(user_id, post_id, comment_text)
                engagement_type = "commented"
            
            # Update analytics
            await update_engagement_analytics(post_id, user_id, action)
            
            # Send notification to post owner
            if post["user_id"] != user_id:
                await send_notification(post["user_id"], f"Your post was {engagement_type}")
            
            # Return success response
            return create_success_response({
                "action": engagement_type,
                "post_id": post_id,
                "user_id": user_id,
                "message": f"Post {engagement_type} successfully"
            })
            
        except Exception as e:
            logger.error(f"Error in happy path: {e}")
            return create_error_response("ENGAGEMENT_FAILED", f"Failed to {action} post: {str(e)}")

# ============================================================================
# USAGE EXAMPLES
# ============================================================================

async def demonstrate_happy_path_pattern():
    """Demonstrate the happy path last pattern with various scenarios"""
    
    post_service = PostService()
    
    print("=" * 80)
    print("HAPPY PATH LAST PATTERN DEMONSTRATION")
    print("=" * 80)
    
    # Test 1: Successful post creation
    print("\n1. SUCCESSFUL POST CREATION:")
    result = await post_service.create_post_happy_path_last(
        user_id="valid_user",
        content="This is a test post with sufficient content length to pass validation.",
        hashtags=["test", "demo"]
    )
    print(f"Result: {result}")
    
    # Test 2: Failed post creation (validation error)
    print("\n2. FAILED POST CREATION (Content too short):")
    result = await post_service.create_post_happy_path_last(
        user_id="valid_user",
        content="Short",
        hashtags=["test"]
    )
    print(f"Result: {result}")
    
    # Test 3: Successful post update
    print("\n3. SUCCESSFUL POST UPDATE:")
    result = await post_service.update_post_happy_path_last(
        post_id="valid_post",
        user_id="valid_user",
        new_content="This is the updated content with sufficient length to pass all validations."
    )
    print(f"Result: {result}")
    
    # Test 4: Failed post update (unauthorized)
    print("\n4. FAILED POST UPDATE (Unauthorized):")
    result = await post_service.update_post_happy_path_last(
        post_id="valid_post",
        user_id="inactive_user",
        new_content="This should fail due to unauthorized access."
    )
    print(f"Result: {result}")
    
    # Test 5: Successful engagement
    print("\n5. SUCCESSFUL ENGAGEMENT (Like):")
    result = await post_service.process_engagement_happy_path_last(
        post_id="valid_post",
        user_id="valid_user",
        action="like"
    )
    print(f"Result: {result}")
    
    # Test 6: Successful engagement (Comment)
    print("\n6. SUCCESSFUL ENGAGEMENT (Comment):")
    result = await post_service.process_engagement_happy_path_last(
        post_id="valid_post",
        user_id="valid_user",
        action="comment",
        comment_text="This is a test comment with sufficient content."
    )
    print(f"Result: {result}")
    
    # Test 7: Failed engagement (Invalid action)
    print("\n7. FAILED ENGAGEMENT (Invalid action):")
    result = await post_service.process_engagement_happy_path_last(
        post_id="valid_post",
        user_id="valid_user",
        action="invalid_action"
    )
    print(f"Result: {result}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(demonstrate_happy_path_pattern()) 