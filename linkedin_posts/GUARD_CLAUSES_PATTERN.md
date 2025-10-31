# Guard Clauses Pattern: Handle Preconditions and Invalid States Early

## Core Principle: Fail Fast with Guard Clauses

Use guard clauses at the beginning of functions to handle preconditions, invalid states, and edge cases early. This creates:
- **Fail-fast behavior** that catches issues immediately
- **Cleaner main logic** without nested conditionals
- **Better error handling** with specific validation messages
- **Improved readability** with early exits for invalid states

## 1. Basic Guard Clause Pattern

### ❌ **No Guard Clauses (Bad)**
```python
async def create_post_bad(user_id: str, content: str, hashtags: List[str] = None) -> Dict[str, Any]:
    # Main logic mixed with validation
    if user_id and content:
        if len(content) >= 10:
            if len(content) <= 3000:
                user = await get_user_by_id(user_id)
                if user:
                    if user.is_active:
                        # Main business logic buried deep
                        post = await create_post_in_database(user_id, content, hashtags)
                        return {"status": "success", "post_id": post.id}
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
```

### ✅ **Guard Clauses (Good)**
```python
async def create_post_good(user_id: str, content: str, hashtags: List[str] = None) -> Dict[str, Any]:
    # ============================================================================
    # GUARD CLAUSES (Handle preconditions and invalid states early)
    # ============================================================================
    
    # Precondition: Required parameters
    if not user_id:
        return {"error": "User ID is required", "status": "failed"}
    
    if not content:
        return {"error": "Content is required", "status": "failed"}
    
    # Precondition: Content validation
    content = content.strip()
    if len(content) < 10:
        return {"error": "Content too short (minimum 10 characters)", "status": "failed"}
    
    if len(content) > 3000:
        return {"error": "Content too long (maximum 3000 characters)", "status": "failed"}
    
    # Precondition: User validation
    user = await get_user_by_id(user_id)
    if not user:
        return {"error": "User not found", "status": "failed"}
    
    if not user.is_active:
        return {"error": "Account is deactivated", "status": "failed"}
    
    # ============================================================================
    # MAIN BUSINESS LOGIC (Clean and focused)
    # ============================================================================
    
    try:
        post = await create_post_in_database(user_id, content, hashtags)
        return {"status": "success", "post_id": post.id}
    except Exception as e:
        return {"error": f"Failed to create post: {e}", "status": "failed"}
```

## 2. Authentication Guard Clauses

### ❌ **Authentication Mixed with Logic (Bad)**
```python
async def get_user_posts_bad(user_id: str, requester_id: str) -> Dict[str, Any]:
    # Authentication logic mixed with business logic
    if requester_id:
        requester = await get_user_by_id(requester_id)
        if requester:
            if requester.is_active:
                if user_id:
                    target_user = await get_user_by_id(user_id)
                    if target_user:
                        if requester_id == user_id or target_user.posts_public:
                            posts = await get_posts_from_database(user_id)
                            return {"posts": posts, "status": "success"}
                        else:
                            return {"error": "Posts are private"}
                    else:
                        return {"error": "User not found"}
                else:
                    return {"error": "User ID required"}
            else:
                return {"error": "Account is deactivated"}
        else:
            return {"error": "Invalid authentication"}
    else:
        return {"error": "Requester ID required"}
```

### ✅ **Authentication Guard Clauses (Good)**
```python
async def get_user_posts_good(user_id: str, requester_id: str) -> Dict[str, Any]:
    # ============================================================================
    # GUARD CLAUSES (Handle authentication and authorization early)
    # ============================================================================
    
    # Precondition: Required parameters
    if not requester_id:
        return {"error": "Requester ID is required", "status": "failed"}
    
    if not user_id:
        return {"error": "User ID is required", "status": "failed"}
    
    # Guard: Authentication
    requester = await get_user_by_id(requester_id)
    if not requester:
        return {"error": "Invalid authentication", "status": "failed"}
    
    if not requester.is_active:
        return {"error": "Account is deactivated", "status": "failed"}
    
    # Guard: Target user validation
    target_user = await get_user_by_id(user_id)
    if not target_user:
        return {"error": "User not found", "status": "failed"}
    
    # Guard: Authorization
    if requester_id != user_id and not target_user.posts_public:
        return {"error": "Posts are private", "status": "failed"}
    
    # ============================================================================
    # MAIN BUSINESS LOGIC (Clean and focused)
    # ============================================================================
    
    try:
        posts = await get_posts_from_database(user_id)
        return {"posts": posts, "status": "success"}
    except Exception as e:
        return {"error": f"Failed to fetch posts: {e}", "status": "failed"}
```

## 3. Business Rule Guard Clauses

### ❌ **Business Rules Mixed with Logic (Bad)**
```python
async def process_order_bad(order_id: str, user_id: str, action: str) -> Dict[str, Any]:
    # Business rules mixed with processing logic
    order = await get_order_by_id(order_id)
    if order:
        if order.user_id == user_id:
            if order.status == 'pending':
                if action == 'confirm':
                    if await check_payment(order_id):
                        await update_order_status(order_id, 'confirmed')
                        return {"status": "success", "message": "Order confirmed"}
                    else:
                        return {"error": "Payment not completed"}
                else:
                    if action == 'cancel':
                        await update_order_status(order_id, 'cancelled')
                        return {"status": "success", "message": "Order cancelled"}
                    else:
                        return {"error": "Invalid action for pending order"}
            else:
                if order.status == 'confirmed':
                    if action == 'ship':
                        await update_order_status(order_id, 'shipped')
                        return {"status": "success", "message": "Order shipped"}
                    else:
                        return {"error": "Invalid action for confirmed order"}
                else:
                    return {"error": "Order cannot be modified"}
        else:
            return {"error": "Cannot modify another user's order"}
    else:
        return {"error": "Order not found"}
```

### ✅ **Business Rule Guard Clauses (Good)**
```python
async def process_order_good(order_id: str, user_id: str, action: str) -> Dict[str, Any]:
    # ============================================================================
    # GUARD CLAUSES (Handle business rules and preconditions early)
    # ============================================================================
    
    # Precondition: Required parameters
    if not order_id:
        return {"error": "Order ID is required", "status": "failed"}
    
    if not user_id:
        return {"error": "User ID is required", "status": "failed"}
    
    if not action:
        return {"error": "Action is required", "status": "failed"}
    
    # Guard: Order validation
    order = await get_order_by_id(order_id)
    if not order:
        return {"error": "Order not found", "status": "failed"}
    
    # Guard: Authorization
    if order.user_id != user_id:
        return {"error": "Cannot modify another user's order", "status": "failed"}
    
    # Guard: Order state validation
    if order.status not in ['pending', 'confirmed']:
        return {"error": "Order cannot be modified", "status": "failed"}
    
    # Guard: Action validation based on order status
    if order.status == 'pending':
        if action not in ['confirm', 'cancel']:
            return {"error": "Invalid action for pending order", "status": "failed"}
    elif order.status == 'confirmed':
        if action != 'ship':
            return {"error": "Invalid action for confirmed order", "status": "failed"}
    
    # ============================================================================
    # MAIN BUSINESS LOGIC (Clean and focused)
    # ============================================================================
    
    try:
        if order.status == 'pending':
            if action == 'confirm':
                if not await check_payment(order_id):
                    return {"error": "Payment not completed", "status": "failed"}
                await update_order_status(order_id, 'confirmed')
                return {"status": "success", "message": "Order confirmed"}
            else:  # cancel
                await update_order_status(order_id, 'cancelled')
                return {"status": "success", "message": "Order cancelled"}
        else:  # confirmed
            await update_order_status(order_id, 'shipped')
            return {"status": "success", "message": "Order shipped"}
    except Exception as e:
        return {"error": f"Failed to process order: {e}", "status": "failed"}
```

## 4. File Upload Guard Clauses

### ❌ **File Validation Mixed with Logic (Bad)**
```python
async def upload_file_bad(user_id: str, file: UploadFile) -> Dict[str, Any]:
    # File validation mixed with upload logic
    if file:
        if file.size <= 5 * 1024 * 1024:  # 5MB
            if file.content_type in ['image/jpeg', 'image/png']:
                user = await get_user_by_id(user_id)
                if user:
                    if user.is_active:
                        file_url = await upload_file_to_storage(file)
                        return {"status": "success", "file_url": file_url}
                    else:
                        return {"error": "Account is deactivated"}
                else:
                    return {"error": "User not found"}
            else:
                return {"error": "Invalid file type"}
        else:
            return {"error": "File too large"}
    else:
        return {"error": "No file provided"}
```

### ✅ **File Upload Guard Clauses (Good)**
```python
async def upload_file_good(user_id: str, file: UploadFile) -> Dict[str, Any]:
    # ============================================================================
    # GUARD CLAUSES (Handle file validation and user validation early)
    # ============================================================================
    
    # Precondition: File existence
    if not file:
        return {"error": "No file provided", "status": "failed"}
    
    if not file.filename:
        return {"error": "Filename is required", "status": "failed"}
    
    # Guard: File size validation
    if file.size > 5 * 1024 * 1024:  # 5MB
        return {"error": "File too large (maximum 5MB)", "status": "failed"}
    
    # Guard: File type validation
    allowed_types = {'image/jpeg', 'image/png', 'image/gif', 'image/webp'}
    if file.content_type not in allowed_types:
        return {"error": "Invalid file type. Only images are allowed", "status": "failed"}
    
    # Guard: User validation
    user = await get_user_by_id(user_id)
    if not user:
        return {"error": "User not found", "status": "failed"}
    
    if not user.is_active:
        return {"error": "Account is deactivated", "status": "failed"}
    
    # ============================================================================
    # MAIN BUSINESS LOGIC (Clean and focused)
    # ============================================================================
    
    try:
        file_url = await upload_file_to_storage(file)
        return {"status": "success", "file_url": file_url}
    except Exception as e:
        return {"error": f"File upload failed: {e}", "status": "failed"}
```

## 5. Rate Limiting Guard Clauses

### ❌ **Rate Limiting Mixed with Logic (Bad)**
```python
async def create_post_with_rate_limit_bad(user_id: str, content: str) -> Dict[str, Any]:
    # Rate limiting mixed with post creation logic
    daily_posts = await get_user_daily_posts(user_id)
    if daily_posts < 5:
        if await check_rate_limit(user_id, "post_creation"):
            user = await get_user_by_id(user_id)
            if user:
                if user.is_active:
                    if content and len(content) >= 10:
                        post = await create_post_in_database(user_id, content)
                        return {"status": "success", "post_id": post.id}
                    else:
                        return {"error": "Invalid content"}
                else:
                    return {"error": "Account is deactivated"}
            else:
                return {"error": "User not found"}
        else:
            return {"error": "Rate limit exceeded"}
    else:
        return {"error": "Daily post limit exceeded"}
```

### ✅ **Rate Limiting Guard Clauses (Good)**
```python
async def create_post_with_rate_limit_good(user_id: str, content: str) -> Dict[str, Any]:
    # ============================================================================
    # GUARD CLAUSES (Handle rate limiting and validation early)
    # ============================================================================
    
    # Precondition: Required parameters
    if not user_id:
        return {"error": "User ID is required", "status": "failed"}
    
    if not content:
        return {"error": "Content is required", "status": "failed"}
    
    # Guard: Content validation
    content = content.strip()
    if len(content) < 10:
        return {"error": "Content too short (minimum 10 characters)", "status": "failed"}
    
    # Guard: User validation
    user = await get_user_by_id(user_id)
    if not user:
        return {"error": "User not found", "status": "failed"}
    
    if not user.is_active:
        return {"error": "Account is deactivated", "status": "failed"}
    
    # Guard: Rate limiting
    if not await check_rate_limit(user_id, "post_creation"):
        return {"error": "Rate limit exceeded", "status": "failed"}
    
    # Guard: Daily limit
    daily_posts = await get_user_daily_posts(user_id)
    if daily_posts >= 5:
        return {"error": "Daily post limit exceeded (maximum 5 posts)", "status": "failed"}
    
    # ============================================================================
    # MAIN BUSINESS LOGIC (Clean and focused)
    # ============================================================================
    
    try:
        post = await create_post_in_database(user_id, content)
        return {"status": "success", "post_id": post.id}
    except Exception as e:
        return {"error": f"Failed to create post: {e}", "status": "failed"}
```

## 6. Database State Guard Clauses

### ❌ **Database State Mixed with Logic (Bad)**
```python
async def update_post_bad(post_id: str, user_id: str, new_content: str) -> Dict[str, Any]:
    # Database state validation mixed with update logic
    post = await get_post_by_id(post_id)
    if post:
        if post.user_id == user_id:
            if not post.is_deleted:
                if not post.is_being_edited:
                    post_age = datetime.now() - post.created_at
                    if post_age <= timedelta(hours=24):
                        if len(new_content) >= 10:
                            if len(new_content) <= 3000:
                                updated_post = await update_post_in_database(post_id, new_content)
                                return {"status": "success", "post": updated_post}
                            else:
                                return {"error": "Content too long"}
                        else:
                            return {"error": "Content too short"}
                    else:
                        return {"error": "Post cannot be edited after 24 hours"}
                else:
                    return {"error": "Post is currently being edited"}
            else:
                return {"error": "Cannot update deleted post"}
        else:
            return {"error": "Cannot update another user's post"}
    else:
        return {"error": "Post not found"}
```

### ✅ **Database State Guard Clauses (Good)**
```python
async def update_post_good(post_id: str, user_id: str, new_content: str) -> Dict[str, Any]:
    # ============================================================================
    # GUARD CLAUSES (Handle database state validation early)
    # ============================================================================
    
    # Precondition: Required parameters
    if not post_id:
        return {"error": "Post ID is required", "status": "failed"}
    
    if not user_id:
        return {"error": "User ID is required", "status": "failed"}
    
    if not new_content:
        return {"error": "New content is required", "status": "failed"}
    
    # Guard: Content validation
    new_content = new_content.strip()
    if len(new_content) < 10:
        return {"error": "Content too short (minimum 10 characters)", "status": "failed"}
    
    if len(new_content) > 3000:
        return {"error": "Content too long (maximum 3000 characters)", "status": "failed"}
    
    # Guard: Post existence
    post = await get_post_by_id(post_id)
    if not post:
        return {"error": "Post not found", "status": "failed"}
    
    # Guard: Authorization
    if post.user_id != user_id:
        return {"error": "Cannot update another user's post", "status": "failed"}
    
    # Guard: Post state validation
    if post.is_deleted:
        return {"error": "Cannot update deleted post", "status": "failed"}
    
    if post.is_being_edited:
        return {"error": "Post is currently being edited", "status": "failed"}
    
    # Guard: Time-based validation
    post_age = datetime.now() - post.created_at
    if post_age > timedelta(hours=24):
        return {"error": "Post cannot be edited after 24 hours", "status": "failed"}
    
    # ============================================================================
    # MAIN BUSINESS LOGIC (Clean and focused)
    # ============================================================================
    
    try:
        updated_post = await update_post_in_database(post_id, new_content)
        return {"status": "success", "post": updated_post}
    except Exception as e:
        return {"error": f"Failed to update post: {e}", "status": "failed"}
```

## 7. Complex Business Logic Guard Clauses

### ❌ **Complex Logic Mixed (Bad)**
```python
async def process_engagement_bad(post_id: str, user_id: str, action: str, comment_text: str = None) -> Dict[str, Any]:
    # Complex validation mixed with engagement logic
    if post_id and user_id and action:
        if action in ['like', 'share', 'comment']:
            if action != 'comment' or comment_text:
                user = await get_user_by_id(user_id)
                if user:
                    if user.is_active:
                        post = await get_post_by_id(post_id)
                        if post:
                            if not post.is_deleted:
                                if post.user_id == user_id or post.is_public:
                                    if action == 'like':
                                        if not await has_user_liked_post(user_id, post_id):
                                            await add_like_to_post(user_id, post_id)
                                            return {"status": "success", "action": "liked"}
                                        else:
                                            return {"error": "Post already liked"}
                                    else:
                                        if action == 'share':
                                            await add_share_to_post(user_id, post_id)
                                            return {"status": "success", "action": "shared"}
                                        else:
                                            await add_comment_to_post(user_id, post_id, comment_text)
                                            return {"status": "success", "action": "commented"}
                                else:
                                    return {"error": "Access denied"}
                            else:
                                return {"error": "Post is deleted"}
                        else:
                            return {"error": "Post not found"}
                    else:
                        return {"error": "Account is deactivated"}
                else:
                    return {"error": "User not found"}
            else:
                return {"error": "Comment text required"}
        else:
            return {"error": "Invalid action"}
    else:
        return {"error": "Missing required parameters"}
```

### ✅ **Complex Logic Guard Clauses (Good)**
```python
async def process_engagement_good(post_id: str, user_id: str, action: str, comment_text: str = None) -> Dict[str, Any]:
    # ============================================================================
    # GUARD CLAUSES (Handle all preconditions and invalid states early)
    # ============================================================================
    
    # Precondition: Required parameters
    if not post_id:
        return {"error": "Post ID is required", "status": "failed"}
    
    if not user_id:
        return {"error": "User ID is required", "status": "failed"}
    
    if not action:
        return {"error": "Action is required", "status": "failed"}
    
    # Guard: Action validation
    valid_actions = ['like', 'share', 'comment']
    if action not in valid_actions:
        return {"error": f"Invalid action. Must be one of: {', '.join(valid_actions)}", "status": "failed"}
    
    # Guard: Comment validation
    if action == 'comment' and not comment_text:
        return {"error": "Comment text is required for comment action", "status": "failed"}
    
    if action == 'comment' and comment_text and len(comment_text) > 1000:
        return {"error": "Comment too long (maximum 1000 characters)", "status": "failed"}
    
    # Guard: User validation
    user = await get_user_by_id(user_id)
    if not user:
        return {"error": "User not found", "status": "failed"}
    
    if not user.is_active:
        return {"error": "Account is deactivated", "status": "failed"}
    
    # Guard: Post validation
    post = await get_post_by_id(post_id)
    if not post:
        return {"error": "Post not found", "status": "failed"}
    
    if post.is_deleted:
        return {"error": "Post is deleted", "status": "failed"}
    
    # Guard: Access validation
    if post.user_id != user_id and not post.is_public:
        return {"error": "Access denied", "status": "failed"}
    
    # Guard: Duplicate like validation
    if action == 'like' and await has_user_liked_post(user_id, post_id):
        return {"error": "Post already liked", "status": "failed"}
    
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
        
        return {"status": "success", "action": engagement_type}
    except Exception as e:
        return {"error": f"Failed to {action} post: {e}", "status": "failed"}
```

## 8. Function Structure Template

### **Standard Guard Clause Structure**
```python
async def function_with_guard_clauses(param1: str, param2: str) -> Dict[str, Any]:
    # ============================================================================
    # GUARD CLAUSES (Handle preconditions and invalid states early)
    # ============================================================================
    
    # 1. Input validation guards
    if not param1:
        return create_error_response("MISSING_PARAM1", "Parameter 1 is required")
    
    if not param2:
        return create_error_response("MISSING_PARAM2", "Parameter 2 is required")
    
    # 2. Format validation guards
    if not is_valid_format(param1):
        return create_error_response("INVALID_FORMAT", "Invalid parameter 1 format")
    
    # 3. Business rule guards
    if not await check_business_rule(param1):
        return create_error_response("BUSINESS_RULE_VIOLATION", "Business rule violated")
    
    # 4. State validation guards
    if not await check_state_validity(param1, param2):
        return create_error_response("INVALID_STATE", "Invalid state for operation")
    
    # 5. Authorization guards
    if not await check_authorization(param1, param2):
        return create_error_response("UNAUTHORIZED", "Operation not authorized")
    
    # ============================================================================
    # MAIN BUSINESS LOGIC (Clean and focused)
    # ============================================================================
    
    try:
        # Perform main operation
        result = await perform_main_operation(param1, param2)
        
        # Return success response
        return create_success_response({
            "result": result,
            "message": "Operation completed successfully"
        })
        
    except Exception as e:
        logger.error(f"Error in main logic: {e}")
        return create_error_response("OPERATION_FAILED", f"Operation failed: {e}")
```

## 9. Benefits of Guard Clauses

### **Fail-Fast Behavior**
- **Immediate validation**: Issues are caught at the function entry point
- **Early exits**: Invalid states don't proceed to expensive operations
- **Clear error messages**: Each guard provides specific validation feedback

### **Cleaner Code**
- **Reduced nesting**: No deep conditional structures
- **Linear flow**: Code reads from top to bottom
- **Focused logic**: Main business logic is isolated and clean

### **Better Maintainability**
- **Easy to add guards**: New validations can be added at the top
- **Easy to modify logic**: Changes don't affect nested structures
- **Easy to test**: Each guard can be tested independently

### **Improved Performance**
- **Early exits**: Expensive operations only happen after all guards pass
- **Resource efficiency**: No unnecessary processing for invalid states
- **Better error handling**: Specific errors help with debugging

## 10. When to Use Guard Clauses

### **✅ Use Guard Clauses For:**
- Functions with multiple validation steps
- Functions that need to fail fast
- Functions with complex business rules
- Functions that interact with external services
- Functions that perform expensive operations
- Functions with multiple error conditions

### **❌ Avoid Guard Clauses For:**
- Simple utility functions with single validations
- Functions where the logic is truly sequential
- Functions that need to perform cleanup in all cases
- Functions where the main logic is very short

## 11. Common Guard Clause Patterns

### **Input Validation Guards**
```python
# Required parameters
if not param:
    return error_response("MISSING_PARAM", "Parameter is required")

# Format validation
if not is_valid_format(param):
    return error_response("INVALID_FORMAT", "Invalid format")

# Range validation
if param < min_value or param > max_value:
    return error_response("OUT_OF_RANGE", "Value out of range")
```

### **State Validation Guards**
```python
# Entity existence
entity = await get_entity_by_id(entity_id)
if not entity:
    return error_response("NOT_FOUND", "Entity not found")

# Entity state
if entity.status != expected_status:
    return error_response("INVALID_STATE", "Entity in invalid state")

# Entity ownership
if entity.user_id != user_id:
    return error_response("UNAUTHORIZED", "Not authorized")
```

### **Business Rule Guards**
```python
# Rate limiting
if not await check_rate_limit(user_id, action):
    return error_response("RATE_LIMIT", "Rate limit exceeded")

# Business constraints
if await violates_business_rule(data):
    return error_response("BUSINESS_RULE", "Business rule violated")

# Duplicate prevention
if await is_duplicate(data):
    return error_response("DUPLICATE", "Duplicate detected")
```

Guard clauses transform your functions into robust, maintainable code by handling all preconditions and invalid states early, ensuring that only valid requests proceed to the main business logic. 