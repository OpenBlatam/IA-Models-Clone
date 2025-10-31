# If-Return Pattern: Avoid Unnecessary Else Statements

## Core Principle: Use If-Return Instead of If-Else

Replace unnecessary `else` statements with early `return` statements after `if` conditions. This creates:
- **Cleaner code flow** with early exits
- **Reduced nesting** and indentation levels
- **Better readability** with linear execution
- **Easier maintenance** with fewer code paths

## 1. Basic If-Return Pattern

### ❌ **Unnecessary Else (Bad)**
```python
async def validate_user_bad(user_id: str) -> Dict[str, Any]:
    if not user_id:
        return {"error": "User ID is required"}
    else:
        user = await get_user_by_id(user_id)
        if user:
            if user.is_active:
                return {"status": "valid", "user": user}
            else:
                return {"error": "Account is deactivated"}
        else:
            return {"error": "User not found"}
```

### ✅ **If-Return Pattern (Good)**
```python
async def validate_user_good(user_id: str) -> Dict[str, Any]:
    if not user_id:
        return {"error": "User ID is required"}
    
    user = await get_user_by_id(user_id)
    if not user:
        return {"error": "User not found"}
    
    if not user.is_active:
        return {"error": "Account is deactivated"}
    
    return {"status": "valid", "user": user}
```

## 2. Nested Conditionals Pattern

### ❌ **Nested If-Else (Bad)**
```python
async def process_post_bad(post_id: str, user_id: str, action: str) -> Dict[str, Any]:
    if post_id:
        if user_id:
            if action in ['like', 'share', 'comment']:
                post = await get_post_by_id(post_id)
                if post:
                    if post.user_id == user_id or post.is_public:
                        if action == 'like':
                            await add_like(post_id, user_id)
                            return {"status": "success", "action": "liked"}
                        else:
                            if action == 'share':
                                await add_share(post_id, user_id)
                                return {"status": "success", "action": "shared"}
                            else:
                                await add_comment(post_id, user_id)
                                return {"status": "success", "action": "commented"}
                    else:
                        return {"error": "Access denied"}
                else:
                    return {"error": "Post not found"}
            else:
                return {"error": "Invalid action"}
        else:
            return {"error": "User ID required"}
    else:
        return {"error": "Post ID required"}
```

### ✅ **If-Return Pattern (Good)**
```python
async def process_post_good(post_id: str, user_id: str, action: str) -> Dict[str, Any]:
    if not post_id:
        return {"error": "Post ID required"}
    
    if not user_id:
        return {"error": "User ID required"}
    
    if action not in ['like', 'share', 'comment']:
        return {"error": "Invalid action"}
    
    post = await get_post_by_id(post_id)
    if not post:
        return {"error": "Post not found"}
    
    if post.user_id != user_id and not post.is_public:
        return {"error": "Access denied"}
    
    # Process the action
    if action == 'like':
        await add_like(post_id, user_id)
        return {"status": "success", "action": "liked"}
    
    if action == 'share':
        await add_share(post_id, user_id)
        return {"status": "success", "action": "shared"}
    
    # Default case: comment
    await add_comment(post_id, user_id)
    return {"status": "success", "action": "commented"}
```

## 3. Validation Chain Pattern

### ❌ **Validation with Else (Bad)**
```python
async def create_post_bad(user_id: str, content: str, hashtags: List[str] = None) -> Dict[str, Any]:
    if not user_id:
        return {"error": "User ID is required"}
    else:
        if not content:
            return {"error": "Content is required"}
        else:
            if len(content) < 10:
                return {"error": "Content too short"}
            else:
                if len(content) > 3000:
                    return {"error": "Content too long"}
                else:
                    user = await get_user_by_id(user_id)
                    if user:
                        if user.is_active:
                            post = await create_post_in_database(user_id, content, hashtags)
                            return {"status": "success", "post_id": post.id}
                        else:
                            return {"error": "Account is deactivated"}
                    else:
                        return {"error": "User not found"}
```

### ✅ **Validation Chain with If-Return (Good)**
```python
async def create_post_good(user_id: str, content: str, hashtags: List[str] = None) -> Dict[str, Any]:
    # Input validation chain
    if not user_id:
        return {"error": "User ID is required"}
    
    if not content:
        return {"error": "Content is required"}
    
    if len(content) < 10:
        return {"error": "Content too short"}
    
    if len(content) > 3000:
        return {"error": "Content too long"}
    
    # User validation
    user = await get_user_by_id(user_id)
    if not user:
        return {"error": "User not found"}
    
    if not user.is_active:
        return {"error": "Account is deactivated"}
    
    # Create post (happy path)
    post = await create_post_in_database(user_id, content, hashtags)
    return {"status": "success", "post_id": post.id}
```

## 4. Business Logic Pattern

### ❌ **Business Logic with Else (Bad)**
```python
async def handle_user_action_bad(user_id: str, action: str, data: Dict[str, Any]) -> Dict[str, Any]:
    user = await get_user_by_id(user_id)
    if user:
        if user.is_active:
            if action == 'create_post':
                if data.get('content'):
                    post = await create_post(user_id, data['content'])
                    return {"status": "success", "post_id": post.id}
                else:
                    return {"error": "Content required"}
            else:
                if action == 'update_profile':
                    if data.get('name'):
                        await update_user_profile(user_id, data)
                        return {"status": "success", "message": "Profile updated"}
                    else:
                        return {"error": "Name required"}
                else:
                    if action == 'delete_account':
                        await delete_user_account(user_id)
                        return {"status": "success", "message": "Account deleted"}
                    else:
                        return {"error": "Invalid action"}
        else:
            return {"error": "Account is deactivated"}
    else:
        return {"error": "User not found"}
```

### ✅ **Business Logic with If-Return (Good)**
```python
async def handle_user_action_good(user_id: str, action: str, data: Dict[str, Any]) -> Dict[str, Any]:
    # User validation
    user = await get_user_by_id(user_id)
    if not user:
        return {"error": "User not found"}
    
    if not user.is_active:
        return {"error": "Account is deactivated"}
    
    # Action handling
    if action == 'create_post':
        if not data.get('content'):
            return {"error": "Content required"}
        
        post = await create_post(user_id, data['content'])
        return {"status": "success", "post_id": post.id}
    
    if action == 'update_profile':
        if not data.get('name'):
            return {"error": "Name required"}
        
        await update_user_profile(user_id, data)
        return {"status": "success", "message": "Profile updated"}
    
    if action == 'delete_account':
        await delete_user_account(user_id)
        return {"status": "success", "message": "Account deleted"}
    
    return {"error": "Invalid action"}
```

## 5. Error Handling Pattern

### ❌ **Error Handling with Else (Bad)**
```python
async def process_file_upload_bad(user_id: str, file: UploadFile) -> Dict[str, Any]:
    if file:
        if file.size <= 5 * 1024 * 1024:  # 5MB
            if file.content_type in ['image/jpeg', 'image/png']:
                user = await get_user_by_id(user_id)
                if user:
                    if user.is_active:
                        file_url = await upload_file(file)
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

### ✅ **Error Handling with If-Return (Good)**
```python
async def process_file_upload_good(user_id: str, file: UploadFile) -> Dict[str, Any]:
    # File validation
    if not file:
        return {"error": "No file provided"}
    
    if file.size > 5 * 1024 * 1024:  # 5MB
        return {"error": "File too large"}
    
    if file.content_type not in ['image/jpeg', 'image/png']:
        return {"error": "Invalid file type"}
    
    # User validation
    user = await get_user_by_id(user_id)
    if not user:
        return {"error": "User not found"}
    
    if not user.is_active:
        return {"error": "Account is deactivated"}
    
    # Upload file (happy path)
    file_url = await upload_file(file)
    return {"status": "success", "file_url": file_url}
```

## 6. Database Operation Pattern

### ❌ **Database Operations with Else (Bad)**
```python
async def update_post_bad(post_id: str, user_id: str, new_content: str) -> Dict[str, Any]:
    post = await get_post_by_id(post_id)
    if post:
        if post.user_id == user_id:
            if len(new_content) >= 10:
                if len(new_content) <= 3000:
                    updated_post = await update_post_in_database(post_id, new_content)
                    return {"status": "success", "post": updated_post}
                else:
                    return {"error": "Content too long"}
            else:
                return {"error": "Content too short"}
        else:
            return {"error": "Cannot update another user's post"}
    else:
        return {"error": "Post not found"}
```

### ✅ **Database Operations with If-Return (Good)**
```python
async def update_post_good(post_id: str, user_id: str, new_content: str) -> Dict[str, Any]:
    # Content validation
    if len(new_content) < 10:
        return {"error": "Content too short"}
    
    if len(new_content) > 3000:
        return {"error": "Content too long"}
    
    # Post validation
    post = await get_post_by_id(post_id)
    if not post:
        return {"error": "Post not found"}
    
    if post.user_id != user_id:
        return {"error": "Cannot update another user's post"}
    
    # Update post (happy path)
    updated_post = await update_post_in_database(post_id, new_content)
    return {"status": "success", "post": updated_post}
```

## 7. Authentication Pattern

### ❌ **Authentication with Else (Bad)**
```python
async def authenticate_user_bad(token: str) -> Dict[str, Any]:
    if token:
        decoded_token = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
        if decoded_token:
            user_id = decoded_token.get('user_id')
            if user_id:
                user = await get_user_by_id(user_id)
                if user:
                    if user.is_active:
                        return {"status": "authenticated", "user": user}
                    else:
                        return {"error": "Account is deactivated"}
                else:
                    return {"error": "User not found"}
            else:
                return {"error": "Invalid token format"}
        else:
            return {"error": "Invalid token"}
    else:
        return {"error": "Token required"}
```

### ✅ **Authentication with If-Return (Good)**
```python
async def authenticate_user_good(token: str) -> Dict[str, Any]:
    # Token validation
    if not token:
        return {"error": "Token required"}
    
    try:
        decoded_token = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
    except jwt.InvalidTokenError:
        return {"error": "Invalid token"}
    
    user_id = decoded_token.get('user_id')
    if not user_id:
        return {"error": "Invalid token format"}
    
    # User validation
    user = await get_user_by_id(user_id)
    if not user:
        return {"error": "User not found"}
    
    if not user.is_active:
        return {"error": "Account is deactivated"}
    
    # Authentication successful (happy path)
    return {"status": "authenticated", "user": user}
```

## 8. Complex Business Logic Pattern

### ❌ **Complex Logic with Else (Bad)**
```python
async def process_order_bad(order_id: str, user_id: str, action: str) -> Dict[str, Any]:
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

### ✅ **Complex Logic with If-Return (Good)**
```python
async def process_order_good(order_id: str, user_id: str, action: str) -> Dict[str, Any]:
    # Order validation
    order = await get_order_by_id(order_id)
    if not order:
        return {"error": "Order not found"}
    
    if order.user_id != user_id:
        return {"error": "Cannot modify another user's order"}
    
    # Action validation based on order status
    if order.status == 'pending':
        if action == 'confirm':
            if not await check_payment(order_id):
                return {"error": "Payment not completed"}
            
            await update_order_status(order_id, 'confirmed')
            return {"status": "success", "message": "Order confirmed"}
        
        if action == 'cancel':
            await update_order_status(order_id, 'cancelled')
            return {"status": "success", "message": "Order cancelled"}
        
        return {"error": "Invalid action for pending order"}
    
    if order.status == 'confirmed':
        if action == 'ship':
            await update_order_status(order_id, 'shipped')
            return {"status": "success", "message": "Order shipped"}
        
        return {"error": "Invalid action for confirmed order"}
    
    return {"error": "Order cannot be modified"}
```

## 9. Function Structure Template

### **Standard If-Return Structure**
```python
async def function_with_if_return_pattern(param1: str, param2: str) -> Dict[str, Any]:
    # ============================================================================
    # VALIDATION PHASE (Early returns for invalid inputs)
    # ============================================================================
    
    # Input validation
    if not param1:
        return create_error_response("MISSING_PARAM1", "Parameter 1 is required")
    
    if not param2:
        return create_error_response("MISSING_PARAM2", "Parameter 2 is required")
    
    # Format validation
    if not is_valid_format(param1):
        return create_error_response("INVALID_FORMAT", "Invalid parameter 1 format")
    
    # Business rule validation
    if not await check_business_rule(param1):
        return create_error_response("BUSINESS_RULE_VIOLATION", "Business rule violated")
    
    # Database validation
    if not await check_database_constraints(param1, param2):
        return create_error_response("DATABASE_VIOLATION", "Database constraint violated")
    
    # ============================================================================
    # HAPPY PATH (Main business logic)
    # ============================================================================
    
    try:
        # Perform main operation
        result = await perform_main_operation(param1, param2)
        
        # Update related data
        await update_related_data(result)
        
        # Return success response
        return create_success_response({
            "result": result,
            "message": "Operation completed successfully"
        })
        
    except Exception as e:
        logger.error(f"Error in happy path: {e}")
        return create_error_response("OPERATION_FAILED", f"Operation failed: {e}")
```

## 10. Benefits of If-Return Pattern

### **Readability**
- **Linear flow**: Code reads from top to bottom without nesting
- **Clear exits**: Each validation has a clear exit point
- **Reduced complexity**: Fewer code paths to follow

### **Maintainability**
- **Easy to add validations**: New checks can be added at the top
- **Easy to modify logic**: Changes don't affect nested structures
- **Easy to test**: Each validation can be tested independently

### **Performance**
- **Early exits**: Invalid requests are rejected immediately
- **No unnecessary processing**: Business logic only runs after all validations pass
- **Better resource usage**: Expensive operations only happen in happy path

### **Debugging**
- **Clear error location**: Each validation has its own return statement
- **Isolated logic**: Business logic is separate from validation
- **Better error messages**: Each validation can have specific error messages

## 11. When to Use If-Return Pattern

### **✅ Use If-Return For:**
- Functions with multiple validation steps
- Functions with complex conditional logic
- Functions that need early exits for performance
- Functions with multiple error conditions
- Functions that return different types of responses

### **❌ Avoid If-Return For:**
- Simple utility functions with single conditions
- Functions where the logic is truly mutually exclusive
- Functions that need to perform cleanup in all cases
- Functions where the else logic is significantly different

## 12. Common Anti-Patterns to Avoid

### **❌ Don't Do This:**
```python
# Anti-pattern: Unnecessary else after return
if condition:
    return result
else:
    return other_result

# Anti-pattern: Nested if-else chains
if condition1:
    if condition2:
        if condition3:
            return result
        else:
            return error3
    else:
        return error2
else:
    return error1
```

### **✅ Do This Instead:**
```python
# Good: Early returns
if condition:
    return result
return other_result

# Good: Flat if-return chain
if not condition1:
    return error1

if not condition2:
    return error2

if not condition3:
    return error3

return result
```

The if-return pattern transforms your code into a clean, linear flow that's easy to read, maintain, and debug by eliminating unnecessary nesting and making the execution path clear and predictable. 