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
from typing import Any, List, Dict, Optional
"""
Guard Clauses Implementation: Handle Preconditions and Invalid States Early

This module demonstrates how to use guard clauses at the beginning of functions
to handle preconditions, invalid states, and edge cases early for fail-fast behavior.
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
    await asyncio.sleep(0.01)
    if user_id == "valid_user":
        return {
            "id": user_id,
            "name": "John Doe",
            "is_active": True,
            "email": "john@example.com",
            "role": "user",
            "posts_public": True
        }
    elif user_id == "inactive_user":
        return {
            "id": user_id,
            "name": "Jane Doe",
            "is_active": False,
            "email": "jane@example.com",
            "role": "user",
            "posts_public": True
        }
    elif user_id == "private_user":
        return {
            "id": user_id,
            "name": "Private User",
            "is_active": True,
            "email": "private@example.com",
            "role": "user",
            "posts_public": False
        }
    elif user_id == "admin_user":
        return {
            "id": user_id,
            "name": "Admin User",
            "is_active": True,
            "email": "admin@example.com",
            "role": "admin",
            "posts_public": True
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
            "is_public": True,
            "is_being_edited": False,
            "status": "published"
        }
    elif post_id == "private_post":
        return {
            "id": post_id,
            "user_id": "private_user",
            "content": "Private content",
            "created_at": datetime.now() - timedelta(hours=1),
            "is_deleted": False,
            "is_public": False,
            "is_being_edited": False,
            "status": "published"
        }
    elif post_id == "deleted_post":
        return {
            "id": post_id,
            "user_id": "valid_user",
            "content": "Deleted content",
            "created_at": datetime.now() - timedelta(days=1),
            "is_deleted": True,
            "is_public": True,
            "is_being_edited": False,
            "status": "deleted"
        }
    elif post_id == "old_post":
        return {
            "id": post_id,
            "user_id": "valid_user",
            "content": "Old content",
            "created_at": datetime.now() - timedelta(days=2),
            "is_deleted": False,
            "is_public": True,
            "is_being_edited": False,
            "status": "published"
        }
    elif post_id == "editing_post":
        return {
            "id": post_id,
            "user_id": "valid_user",
            "content": "Content being edited",
            "created_at": datetime.now() - timedelta(hours=1),
            "is_deleted": False,
            "is_public": True,
            "is_being_edited": True,
            "status": "published"
        }
    return None

async def get_order_by_id(order_id: str) -> Optional[Dict[str, Any]]:
    """Mock function to get order by ID"""
    await asyncio.sleep(0.01)
    if order_id == "pending_order":
        return {
            "id": order_id,
            "user_id": "valid_user",
            "status": "pending",
            "total": 99.99,
            "created_at": datetime.now() - timedelta(hours=1)
        }
    elif order_id == "confirmed_order":
        return {
            "id": order_id,
            "user_id": "valid_user",
            "status": "confirmed",
            "total": 149.99,
            "created_at": datetime.now() - timedelta(days=1)
        }
    elif order_id == "shipped_order":
        return {
            "id": order_id,
            "user_id": "valid_user",
            "status": "shipped",
            "total": 199.99,
            "created_at": datetime.now() - timedelta(days=2)
        }
    return None

async def get_user_daily_posts(user_id: str) -> int:
    """Mock function to get user's daily post count"""
    await asyncio.sleep(0.01)
    return 2  # Mock: user has 2 posts today

async def has_user_liked_post(user_id: str, post_id: str) -> bool:
    """Mock function to check if user has liked a post"""
    await asyncio.sleep(0.01)
    return False  # Mock: user hasn't liked the post

async def is_duplicate_content(content: str, user_id: str) -> bool:
    """Mock function to check for duplicate content"""
    await asyncio.sleep(0.01)
    return False  # Mock: not duplicate

async def check_rate_limit(user_id: str, action: str) -> bool:
    """Mock function to check rate limit"""
    await asyncio.sleep(0.01)
    return True  # Mock: rate limit not exceeded

async def check_payment(order_id: str) -> bool:
    """Mock function to check payment status"""
    await asyncio.sleep(0.01)
    return order_id == "pending_order"  # Mock: only pending order has payment

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
        "created_at": datetime.now(),
        "status": "published"
    }

async def update_post_in_database(post_id: str, new_content: str) -> Dict[str, Any]:
    """Mock function to update post in database"""
    await asyncio.sleep(0.05)
    return {
        "id": post_id,
        "content": new_content,
        "updated_at": datetime.now()
    }

async def update_order_status(order_id: str, status: str) -> None:
    """Mock function to update order status"""
    await asyncio.sleep(0.02)
    logger.info(f"Order {order_id} status updated to {status}")

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

async async def upload_file_to_storage(file) -> str:
    """Mock function to upload file to storage"""
    await asyncio.sleep(0.1)
    return f"https://storage.example.com/files/{uuid.uuid4().hex[:8]}"

async def get_posts_from_database(user_id: str) -> List[Dict[str, Any]]:
    """Mock function to get posts from database"""
    await asyncio.sleep(0.02)
    return [
        {"id": "post1", "content": "First post", "created_at": datetime.now()},
        {"id": "post2", "content": "Second post", "created_at": datetime.now()}
    ]

# ============================================================================
# GUARD CLAUSES IMPLEMENTATIONS
# ============================================================================

class PostService:
    """Service class demonstrating guard clauses pattern"""
    
    @staticmethod
    async def create_post_with_guard_clauses(user_id: str, content: str, hashtags: List[str] = None) -> Dict[str, Any]:
        """
        Create a new post using guard clauses.
        
        All preconditions and invalid states are handled early with guard clauses.
        """
        # ============================================================================
        # GUARD CLAUSES (Handle preconditions and invalid states early)
        # ============================================================================
        
        # Precondition: Required parameters
        if not user_id:
            return create_error_response("MISSING_USER_ID", "User ID is required", "user_id")
        
        if not content:
            return create_error_response("MISSING_CONTENT", "Content is required", "content")
        
        # Guard: Content validation
        content = content.strip()
        if len(content) < 10:
            return create_error_response("CONTENT_TOO_SHORT", "Content too short (minimum 10 characters)", "content")
        
        if len(content) > 3000:
            return create_error_response("CONTENT_TOO_LONG", "Content too long (maximum 3000 characters)", "content")
        
        # Guard: User validation
        user = await get_user_by_id(user_id)
        if not user:
            return create_error_response("USER_NOT_FOUND", "User not found", "user_id")
        
        if not user["is_active"]:
            return create_error_response("USER_INACTIVE", "Account is deactivated", "user_id")
        
        # Guard: Business rule validation
        if await is_duplicate_content(content, user_id):
            return create_error_response("DUPLICATE_CONTENT", "Duplicate content detected", "content")
        
        # Guard: Rate limiting
        if not await check_rate_limit(user_id, "post_creation"):
            return create_error_response("RATE_LIMIT_EXCEEDED", "Rate limit exceeded", "user_id")
        
        # ============================================================================
        # MAIN BUSINESS LOGIC (Clean and focused)
        # ============================================================================
        
        try:
            post_data = await create_post_in_database(user_id, content, hashtags)
            return create_success_response({
                "post_id": post_data["id"],
                "message": "Post created successfully",
                "created_at": post_data["created_at"].isoformat()
            })
        except Exception as e:
            logger.error(f"Error creating post: {e}")
            return create_error_response("CREATION_FAILED", f"Failed to create post: {str(e)}")
    
    @staticmethod
    async def get_user_posts_with_guard_clauses(user_id: str, requester_id: str) -> Dict[str, Any]:
        """
        Get user posts using guard clauses.
        
        Authentication and authorization are handled early with guard clauses.
        """
        # ============================================================================
        # GUARD CLAUSES (Handle authentication and authorization early)
        # ============================================================================
        
        # Precondition: Required parameters
        if not requester_id:
            return create_error_response("MISSING_REQUESTER_ID", "Requester ID is required", "requester_id")
        
        if not user_id:
            return create_error_response("MISSING_USER_ID", "User ID is required", "user_id")
        
        # Guard: Authentication
        requester = await get_user_by_id(requester_id)
        if not requester:
            return create_error_response("INVALID_AUTHENTICATION", "Invalid authentication", "requester_id")
        
        if not requester["is_active"]:
            return create_error_response("ACCOUNT_DEACTIVATED", "Account is deactivated", "requester_id")
        
        # Guard: Target user validation
        target_user = await get_user_by_id(user_id)
        if not target_user:
            return create_error_response("USER_NOT_FOUND", "User not found", "user_id")
        
        # Guard: Authorization
        if requester_id != user_id and not target_user["posts_public"]:
            return create_error_response("POSTS_PRIVATE", "Posts are private", "user_id")
        
        # Guard: Blocking check
        if await is_user_blocked(target_user["id"], requester_id):
            return create_error_response("ACCESS_DENIED", "Access denied", "user_id")
        
        # ============================================================================
        # MAIN BUSINESS LOGIC (Clean and focused)
        # ============================================================================
        
        try:
            posts = await get_posts_from_database(user_id)
            return create_success_response({
                "posts": posts,
                "user_id": user_id,
                "posts_count": len(posts)
            })
        except Exception as e:
            logger.error(f"Error fetching posts: {e}")
            return create_error_response("FETCH_FAILED", f"Failed to fetch posts: {str(e)}")
    
    @staticmethod
    async def update_post_with_guard_clauses(post_id: str, user_id: str, new_content: str) -> Dict[str, Any]:
        """
        Update post using guard clauses.
        
        All validation and state checks are handled early with guard clauses.
        """
        # ============================================================================
        # GUARD CLAUSES (Handle validation and state checks early)
        # ============================================================================
        
        # Precondition: Required parameters
        if not post_id:
            return create_error_response("MISSING_POST_ID", "Post ID is required", "post_id")
        
        if not user_id:
            return create_error_response("MISSING_USER_ID", "User ID is required", "user_id")
        
        if not new_content:
            return create_error_response("MISSING_CONTENT", "New content is required", "new_content")
        
        # Guard: Content validation
        new_content = new_content.strip()
        if len(new_content) < 10:
            return create_error_response("CONTENT_TOO_SHORT", "Content too short (minimum 10 characters)", "new_content")
        
        if len(new_content) > 3000:
            return create_error_response("CONTENT_TOO_LONG", "Content too long (maximum 3000 characters)", "new_content")
        
        # Guard: Post existence
        post = await get_post_by_id(post_id)
        if not post:
            return create_error_response("POST_NOT_FOUND", "Post not found", "post_id")
        
        # Guard: Authorization
        if post["user_id"] != user_id:
            return create_error_response("UNAUTHORIZED_UPDATE", "Cannot update another user's post", "post_id")
        
        # Guard: Post state validation
        if post["is_deleted"]:
            return create_error_response("POST_DELETED", "Cannot update deleted post", "post_id")
        
        if post["is_being_edited"]:
            return create_error_response("POST_BEING_EDITED", "Post is currently being edited", "post_id")
        
        # Guard: Time-based validation
        post_age = datetime.now() - post["created_at"]
        if post_age > timedelta(hours=24):
            return create_error_response("EDIT_TIME_EXPIRED", "Post cannot be edited after 24 hours", "post_id")
        
        # ============================================================================
        # MAIN BUSINESS LOGIC (Clean and focused)
        # ============================================================================
        
        try:
            updated_post = await update_post_in_database(post_id, new_content)
            return create_success_response({
                "post_id": post_id,
                "message": "Post updated successfully",
                "updated_at": updated_post["updated_at"].isoformat()
            })
        except Exception as e:
            logger.error(f"Error updating post: {e}")
            return create_error_response("UPDATE_FAILED", f"Failed to update post: {str(e)}")
    
    @staticmethod
    async def process_engagement_with_guard_clauses(post_id: str, user_id: str, action: str, comment_text: str = None) -> Dict[str, Any]:
        """
        Process engagement using guard clauses.
        
        All validation and business rules are handled early with guard clauses.
        """
        # ============================================================================
        # GUARD CLAUSES (Handle all validation and business rules early)
        # ============================================================================
        
        # Precondition: Required parameters
        if not post_id:
            return create_error_response("MISSING_POST_ID", "Post ID is required", "post_id")
        
        if not user_id:
            return create_error_response("MISSING_USER_ID", "User ID is required", "user_id")
        
        if not action:
            return create_error_response("MISSING_ACTION", "Action is required", "action")
        
        # Guard: Action validation
        valid_actions = ['like', 'share', 'comment']
        if action not in valid_actions:
            return create_error_response("INVALID_ACTION", f"Invalid action. Must be one of: {', '.join(valid_actions)}", "action")
        
        # Guard: Comment validation
        if action == 'comment' and not comment_text:
            return create_error_response("MISSING_COMMENT", "Comment text is required for comment action", "comment_text")
        
        if action == 'comment' and comment_text and len(comment_text) > 1000:
            return create_error_response("COMMENT_TOO_LONG", "Comment too long (maximum 1000 characters)", "comment_text")
        
        # Guard: User validation
        user = await get_user_by_id(user_id)
        if not user:
            return create_error_response("USER_NOT_FOUND", "User not found", "user_id")
        
        if not user["is_active"]:
            return create_error_response("USER_INACTIVE", "Account is deactivated", "user_id")
        
        # Guard: Post validation
        post = await get_post_by_id(post_id)
        if not post:
            return create_error_response("POST_NOT_FOUND", "Post not found", "post_id")
        
        if post["is_deleted"]:
            return create_error_response("POST_DELETED", "Post is deleted", "post_id")
        
        # Guard: Access validation
        if post["user_id"] != user_id and not post["is_public"]:
            return create_error_response("ACCESS_DENIED", "Post is private", "post_id")
        
        # Guard: Duplicate like validation
        if action == 'like' and await has_user_liked_post(user_id, post_id):
            return create_error_response("ALREADY_LIKED", "Post already liked", "post_id")
        
        # ============================================================================
        # MAIN BUSINESS LOGIC (Clean and focused)
        # ============================================================================
        
        try:
            if action == 'like':
                await add_like_to_post(user_id, post_id)
                engagement_type = "liked"
            elif action == 'share':
                await add_share_to_post(user_id, post_id)
                engagement_type = "shared"
            else:  # comment
                await add_comment_to_post(user_id, post_id, comment_text)
                engagement_type = "commented"
            
            return create_success_response({
                "action": engagement_type,
                "post_id": post_id,
                "user_id": user_id,
                "message": f"Post {engagement_type} successfully"
            })
        except Exception as e:
            logger.error(f"Error processing engagement: {e}")
            return create_error_response("ENGAGEMENT_FAILED", f"Failed to {action} post: {str(e)}")

class OrderService:
    """Service class demonstrating guard clauses for order processing"""
    
    @staticmethod
    async def process_order_with_guard_clauses(order_id: str, user_id: str, action: str) -> Dict[str, Any]:
        """
        Process order using guard clauses.
        
        All business rules and state validation are handled early with guard clauses.
        """
        # ============================================================================
        # GUARD CLAUSES (Handle business rules and state validation early)
        # ============================================================================
        
        # Precondition: Required parameters
        if not order_id:
            return create_error_response("MISSING_ORDER_ID", "Order ID is required", "order_id")
        
        if not user_id:
            return create_error_response("MISSING_USER_ID", "User ID is required", "user_id")
        
        if not action:
            return create_error_response("MISSING_ACTION", "Action is required", "action")
        
        # Guard: Order validation
        order = await get_order_by_id(order_id)
        if not order:
            return create_error_response("ORDER_NOT_FOUND", "Order not found", "order_id")
        
        # Guard: Authorization
        if order["user_id"] != user_id:
            return create_error_response("UNAUTHORIZED_ORDER", "Cannot modify another user's order", "order_id")
        
        # Guard: Order state validation
        if order["status"] not in ['pending', 'confirmed']:
            return create_error_response("ORDER_IMMUTABLE", "Order cannot be modified", "order_id")
        
        # Guard: Action validation based on order status
        if order["status"] == 'pending':
            if action not in ['confirm', 'cancel']:
                return create_error_response("INVALID_ACTION_PENDING", "Invalid action for pending order", "action")
        elif order["status"] == 'confirmed':
            if action != 'ship':
                return create_error_response("INVALID_ACTION_CONFIRMED", "Invalid action for confirmed order", "action")
        
        # ============================================================================
        # MAIN BUSINESS LOGIC (Clean and focused)
        # ============================================================================
        
        try:
            if order["status"] == 'pending':
                if action == 'confirm':
                    if not await check_payment(order_id):
                        return create_error_response("PAYMENT_INCOMPLETE", "Payment not completed", "order_id")
                    await update_order_status(order_id, 'confirmed')
                    return create_success_response({
                        "action": "order_confirmed",
                        "message": "Order confirmed successfully"
                    })
                else:  # cancel
                    await update_order_status(order_id, 'cancelled')
                    return create_success_response({
                        "action": "order_cancelled",
                        "message": "Order cancelled successfully"
                    })
            else:  # confirmed
                await update_order_status(order_id, 'shipped')
                return create_success_response({
                    "action": "order_shipped",
                    "message": "Order shipped successfully"
                })
        except Exception as e:
            logger.error(f"Error processing order: {e}")
            return create_error_response("ORDER_PROCESSING_FAILED", f"Failed to process order: {str(e)}")

class FileService:
    """Service class demonstrating guard clauses for file operations"""
    
    @staticmethod
    async async def upload_file_with_guard_clauses(user_id: str, file) -> Dict[str, Any]:
        """
        Upload file using guard clauses.
        
        All file validation and user validation are handled early with guard clauses.
        """
        # ============================================================================
        # GUARD CLAUSES (Handle file validation and user validation early)
        # ============================================================================
        
        # Precondition: File existence
        if not file:
            return create_error_response("MISSING_FILE", "File is required", "file")
        
        if not hasattr(file, 'filename') or not file.filename:
            return create_error_response("MISSING_FILENAME", "Filename is required", "file")
        
        # Guard: File size validation
        if hasattr(file, 'size') and file.size > 5 * 1024 * 1024:  # 5MB
            return create_error_response("FILE_TOO_LARGE", "File too large (maximum 5MB)", "file")
        
        # Guard: File type validation
        allowed_types = {'image/jpeg', 'image/png', 'image/gif', 'image/webp'}
        if hasattr(file, 'content_type') and file.content_type not in allowed_types:
            return create_error_response("INVALID_FILE_TYPE", "Invalid file type. Only images are allowed", "file")
        
        # Guard: User validation
        user = await get_user_by_id(user_id)
        if not user:
            return create_error_response("USER_NOT_FOUND", "User not found", "user_id")
        
        if not user["is_active"]:
            return create_error_response("USER_INACTIVE", "Account is deactivated", "user_id")
        
        # ============================================================================
        # MAIN BUSINESS LOGIC (Clean and focused)
        # ============================================================================
        
        try:
            file_url = await upload_file_to_storage(file)
            return create_success_response({
                "file_url": file_url,
                "filename": file.filename,
                "size": getattr(file, 'size', 0),
                "content_type": getattr(file, 'content_type', 'unknown')
            })
        except Exception as e:
            logger.error(f"Error uploading file: {e}")
            return create_error_response("UPLOAD_FAILED", f"File upload failed: {str(e)}")

# ============================================================================
# COMPARISON EXAMPLES: BAD vs GOOD
# ============================================================================

class ComparisonExamples:
    """Examples showing bad vs good patterns"""
    
    @staticmethod
    async def create_post_bad(user_id: str, content: str) -> Dict[str, Any]:
        """❌ Bad: No guard clauses - validation mixed with logic"""
        if user_id and content:
            if len(content) >= 10:
                if len(content) <= 3000:
                    user = await get_user_by_id(user_id)
                    if user:
                        if user["is_active"]:
                            post = await create_post_in_database(user_id, content)
                            return {"status": "success", "post_id": post["id"]}
                        else:
                            return {"error": "Account is deactivated"}
                    else:
                        return {"error": "User not found"}
                else:
                    return {"error": "Content too long"}
            else:
                return {"error": "Content too short"}
        else:
            return {"error": "Missing required parameters"}
    
    @staticmethod
    async def create_post_good(user_id: str, content: str) -> Dict[str, Any]:
        """✅ Good: Guard clauses - validation handled early"""
        # Guard: Required parameters
        if not user_id:
            return {"error": "User ID is required"}
        
        if not content:
            return {"error": "Content is required"}
        
        # Guard: Content validation
        content = content.strip()
        if len(content) < 10:
            return {"error": "Content too short"}
        
        if len(content) > 3000:
            return {"error": "Content too long"}
        
        # Guard: User validation
        user = await get_user_by_id(user_id)
        if not user:
            return {"error": "User not found"}
        
        if not user["is_active"]:
            return {"error": "Account is deactivated"}
        
        # Main business logic
        post = await create_post_in_database(user_id, content)
        return {"status": "success", "post_id": post["id"]}

# ============================================================================
# USAGE EXAMPLES
# ============================================================================

async def demonstrate_guard_clauses():
    """Demonstrate the guard clauses pattern with various scenarios"""
    
    post_service = PostService()
    order_service = OrderService()
    file_service = FileService()
    comparison = ComparisonExamples()
    
    print("=" * 80)
    print("GUARD CLAUSES PATTERN DEMONSTRATION")
    print("=" * 80)
    
    # Test 1: Successful post creation
    print("\n1. SUCCESSFUL POST CREATION (Guard Clauses):")
    result = await post_service.create_post_with_guard_clauses(
        user_id="valid_user",
        content="This is a test post with sufficient content length to pass validation.",
        hashtags=["test", "demo"]
    )
    print(f"Result: {result}")
    
    # Test 2: Failed post creation (validation error)
    print("\n2. FAILED POST CREATION (Content too short):")
    result = await post_service.create_post_with_guard_clauses(
        user_id="valid_user",
        content="Short",
        hashtags=["test"]
    )
    print(f"Result: {result}")
    
    # Test 3: Failed post creation (inactive user)
    print("\n3. FAILED POST CREATION (Inactive user):")
    result = await post_service.create_post_with_guard_clauses(
        user_id="inactive_user",
        content="This is a test post with sufficient content length.",
        hashtags=["test"]
    )
    print(f"Result: {result}")
    
    # Test 4: Successful post update
    print("\n4. SUCCESSFUL POST UPDATE (Guard Clauses):")
    result = await post_service.update_post_with_guard_clauses(
        post_id="valid_post",
        user_id="valid_user",
        new_content="This is the updated content with sufficient length to pass all validations."
    )
    print(f"Result: {result}")
    
    # Test 5: Failed post update (post being edited)
    print("\n5. FAILED POST UPDATE (Post being edited):")
    result = await post_service.update_post_with_guard_clauses(
        post_id="editing_post",
        user_id="valid_user",
        new_content="This should fail because the post is being edited."
    )
    print(f"Result: {result}")
    
    # Test 6: Failed post update (old post)
    print("\n6. FAILED POST UPDATE (Old post):")
    result = await post_service.update_post_with_guard_clauses(
        post_id="old_post",
        user_id="valid_user",
        new_content="This should fail because the post is too old to edit."
    )
    print(f"Result: {result}")
    
    # Test 7: Successful engagement
    print("\n7. SUCCESSFUL ENGAGEMENT (Like):")
    result = await post_service.process_engagement_with_guard_clauses(
        post_id="valid_post",
        user_id="valid_user",
        action="like"
    )
    print(f"Result: {result}")
    
    # Test 8: Failed engagement (private post)
    print("\n8. FAILED ENGAGEMENT (Private post):")
    result = await post_service.process_engagement_with_guard_clauses(
        post_id="private_post",
        user_id="valid_user",
        action="like"
    )
    print(f"Result: {result}")
    
    # Test 9: Order processing
    print("\n9. ORDER PROCESSING (Confirm order):")
    result = await order_service.process_order_with_guard_clauses(
        order_id="pending_order",
        user_id="valid_user",
        action="confirm"
    )
    print(f"Result: {result}")
    
    # Test 10: Comparison - Bad vs Good
    print("\n10. COMPARISON - BAD vs GOOD PATTERNS:")
    
    print("\n   Bad Pattern (No guard clauses):")
    result_bad = await comparison.create_post_bad("valid_user", "Valid content with sufficient length")
    print(f"   Result: {result_bad}")
    
    print("\n   Good Pattern (Guard clauses):")
    result_good = await comparison.create_post_good("valid_user", "Valid content with sufficient length")
    print(f"   Result: {result_good}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(demonstrate_guard_clauses()) 