# Error-First Pattern: Handle Errors and Edge Cases at the Beginning

## Core Principle: Fail Fast, Fail Early

Handle all potential errors and edge cases at the beginning of functions before any business logic. This ensures:
- **Early validation** prevents unnecessary processing
- **Clear error boundaries** make debugging easier
- **Better performance** by avoiding wasted computation
- **Defensive programming** reduces unexpected failures

## 1. Input Validation Pattern

```python
async def create_linkedin_post(user_id: str, content: str, hashtags: List[str] = None) -> Dict[str, Any]:
    """
    Create a LinkedIn post with comprehensive error-first validation
    """
    # ============================================================================
    # ERROR-FIRST VALIDATION (P0-P1 Priority)
    # ============================================================================
    
    # P0: Security validation (CRITICAL)
    if not user_id or not user_id.strip():
        raise HTTPException(status_code=400, detail="User ID is required")
    
    # Sanitize and validate user input
    user_id = user_id.strip()
    if not re.match(r'^[a-zA-Z0-9_-]+$', user_id):
        raise HTTPException(status_code=400, detail="Invalid user ID format")
    
    # P0: Content security validation
    if not content or not content.strip():
        raise HTTPException(status_code=400, detail="Content cannot be empty")
    
    # Check for SQL injection attempts
    sql_patterns = [r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE)\b)"]
    for pattern in sql_patterns:
        if re.search(pattern, content, re.IGNORECASE):
            logger.warning(f"SQL injection attempt detected: {content}")
            raise HTTPException(status_code=400, detail="Invalid content detected")
    
    # Check for XSS attempts
    xss_patterns = [r"<script[^>]*>.*?</script>", r"javascript:", r"on\w+\s*="]
    for pattern in xss_patterns:
        if re.search(pattern, content, re.IGNORECASE):
            logger.warning(f"XSS attempt detected: {content}")
            raise HTTPException(status_code=400, detail="Invalid content detected")
    
    # P1: Content validation (HIGH PRIORITY)
    content = content.strip()
    if len(content) < 10:
        raise HTTPException(status_code=400, detail="Content too short (minimum 10 characters)")
    
    if len(content) > 3000:
        raise HTTPException(status_code=400, detail="Content too long (maximum 3000 characters)")
    
    # Word count validation
    words = content.split()
    if len(words) < 5:
        raise HTTPException(status_code=400, detail="Content must have at least 5 words")
    
    if len(words) > 500:
        raise HTTPException(status_code=400, detail="Content too long (maximum 500 words)")
    
    # P1: Hashtag validation
    if hashtags is None:
        hashtags = []
    
    if len(hashtags) > 30:
        raise HTTPException(status_code=400, detail="Too many hashtags (maximum 30)")
    
    validated_hashtags = []
    for hashtag in hashtags:
        if not hashtag.startswith('#'):
            hashtag = f"#{hashtag}"
        
        if len(hashtag) < 2:
            continue
        
        if len(hashtag) > 50:
            raise HTTPException(status_code=400, detail=f"Hashtag too long: {hashtag}")
        
        if not re.match(r'^#[a-zA-Z0-9_]+$', hashtag):
            raise HTTPException(status_code=400, detail=f"Invalid hashtag format: {hashtag}")
        
        validated_hashtags.append(hashtag.lower())
    
    # Remove duplicates
    validated_hashtags = list(set(validated_hashtags))
    
    # ============================================================================
    # BUSINESS LOGIC (After all validation passes)
    # ============================================================================
    
    try:
        # P1: Business rule validation
        await validate_user_post_limit(user_id)
        await validate_duplicate_content(content, user_id)
        
        # P1: Rate limiting
        await check_rate_limit(user_id, "post_creation")
        
        # P2: Database operation
        post_data = await create_post_in_database(user_id, content, validated_hashtags)
        
        return {
            "status": "success",
            "post_id": post_data["id"],
            "message": "Post created successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error creating post: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
```

## 2. Authentication and Authorization Pattern

```python
async def get_user_posts(user_id: str, requester_id: str, page: int = 1, limit: int = 20) -> Dict[str, Any]:
    """
    Get user posts with error-first authentication and authorization
    """
    # ============================================================================
    # ERROR-FIRST AUTHENTICATION (P0 Priority)
    # ============================================================================
    
    # Validate requester authentication
    if not requester_id or not requester_id.strip():
        raise HTTPException(status_code=401, detail="Authentication required")
    
    # Validate target user
    if not user_id or not user_id.strip():
        raise HTTPException(status_code=400, detail="User ID is required")
    
    # Check if requester exists and is active
    requester = await get_user_by_id(requester_id)
    if not requester:
        raise HTTPException(status_code=401, detail="Invalid authentication")
    
    if not requester.is_active:
        raise HTTPException(status_code=401, detail="Account is deactivated")
    
    # ============================================================================
    # ERROR-FIRST AUTHORIZATION (P0 Priority)
    # ============================================================================
    
    # Check if user can access target user's posts
    if requester_id != user_id:
        # Check if target user exists
        target_user = await get_user_by_id(user_id)
        if not target_user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Check if target user's posts are public
        if not target_user.posts_public:
            raise HTTPException(status_code=403, detail="Posts are private")
        
        # Check if requester is blocked by target user
        if await is_user_blocked(target_user.id, requester_id):
            raise HTTPException(status_code=403, detail="Access denied")
    
    # ============================================================================
    # ERROR-FIRST PARAMETER VALIDATION (P1 Priority)
    # ============================================================================
    
    # Validate pagination parameters
    if page < 1:
        raise HTTPException(status_code=400, detail="Page number must be positive")
    
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=400, detail="Limit must be between 1 and 100")
    
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
        raise HTTPException(status_code=500, detail="Failed to fetch posts")
```

## 3. Database Operation Pattern

```python
async def update_post_content(post_id: str, user_id: str, new_content: str) -> Dict[str, Any]:
    """
    Update post content with error-first database validation
    """
    # ============================================================================
    # ERROR-FIRST INPUT VALIDATION (P0-P1 Priority)
    # ============================================================================
    
    # Validate required parameters
    if not post_id or not post_id.strip():
        raise HTTPException(status_code=400, detail="Post ID is required")
    
    if not user_id or not user_id.strip():
        raise HTTPException(status_code=400, detail="User ID is required")
    
    if not new_content or not new_content.strip():
        raise HTTPException(status_code=400, detail="New content is required")
    
    # Sanitize inputs
    post_id = post_id.strip()
    user_id = user_id.strip()
    new_content = new_content.strip()
    
    # P0: Security validation
    if not re.match(r'^[a-zA-Z0-9_-]+$', post_id):
        raise HTTPException(status_code=400, detail="Invalid post ID format")
    
    # Check for malicious content
    malicious_patterns = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE)\b)"
    ]
    
    for pattern in malicious_patterns:
        if re.search(pattern, new_content, re.IGNORECASE):
            logger.warning(f"Malicious content detected in post update: {new_content}")
            raise HTTPException(status_code=400, detail="Invalid content detected")
    
    # P1: Content validation
    if len(new_content) < 10:
        raise HTTPException(status_code=400, detail="Content too short (minimum 10 characters)")
    
    if len(new_content) > 3000:
        raise HTTPException(status_code=400, detail="Content too long (maximum 3000 characters)")
    
    words = new_content.split()
    if len(words) < 5:
        raise HTTPException(status_code=400, detail="Content must have at least 5 words")
    
    # ============================================================================
    # ERROR-FIRST DATABASE VALIDATION (P2 Priority)
    # ============================================================================
    
    try:
        # Check if post exists
        post = await get_post_by_id(post_id)
        if not post:
            raise HTTPException(status_code=404, detail="Post not found")
        
        # Check if user owns the post
        if post.user_id != user_id:
            raise HTTPException(status_code=403, detail="Cannot update another user's post")
        
        # Check if post is editable (not too old)
        post_age = datetime.now() - post.created_at
        if post_age > timedelta(hours=24):
            raise HTTPException(status_code=400, detail="Post cannot be edited after 24 hours")
        
        # Check if post is already being edited
        if post.is_being_edited:
            raise HTTPException(status_code=409, detail="Post is currently being edited")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Database validation error: {e}")
        raise HTTPException(status_code=500, detail="Database validation failed")
    
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
        raise HTTPException(status_code=500, detail="Failed to update post content")
```

## 4. External Service Pattern

```python
async def analyze_post_sentiment(content: str, user_id: str) -> Dict[str, Any]:
    """
    Analyze post sentiment with error-first external service handling
    """
    # ============================================================================
    # ERROR-FIRST INPUT VALIDATION (P0-P1 Priority)
    # ============================================================================
    
    # Validate input parameters
    if not content or not content.strip():
        raise HTTPException(status_code=400, detail="Content is required for sentiment analysis")
    
    if not user_id or not user_id.strip():
        raise HTTPException(status_code=400, detail="User ID is required")
    
    content = content.strip()
    user_id = user_id.strip()
    
    # P0: Security validation
    if len(content) > 10000:  # Limit content size for external API
        raise HTTPException(status_code=400, detail="Content too large for analysis")
    
    # Check for sensitive information
    sensitive_patterns = [
        r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
        r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',  # Credit card
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
    ]
    
    for pattern in sensitive_patterns:
        if re.search(pattern, content):
            logger.warning(f"Sensitive information detected in sentiment analysis request")
            raise HTTPException(status_code=400, detail="Content contains sensitive information")
    
    # ============================================================================
    # ERROR-FIRST RATE LIMITING (P1 Priority)
    # ============================================================================
    
    try:
        # Check rate limit for sentiment analysis
        await check_rate_limit(user_id, "sentiment_analysis", limit=50, window=3600)
    except HTTPException as e:
        if e.status_code == 429:
            raise HTTPException(status_code=429, detail="Sentiment analysis rate limit exceeded")
        raise
    
    # ============================================================================
    # ERROR-FIRST EXTERNAL SERVICE VALIDATION (P2 Priority)
    # ============================================================================
    
    # Check if sentiment analysis service is available
    if not await is_sentiment_service_available():
        logger.warning("Sentiment analysis service unavailable")
        return {
            "sentiment": "neutral",
            "confidence": 0.5,
            "message": "Sentiment analysis service temporarily unavailable"
        }
    
    # ============================================================================
    # BUSINESS LOGIC (After all validation passes)
    # ============================================================================
    
    try:
        # Call external sentiment analysis service
        sentiment_result = await call_sentiment_analysis_service(content)
        
        # Validate response from external service
        if not sentiment_result or "sentiment" not in sentiment_result:
            logger.error("Invalid response from sentiment analysis service")
            raise HTTPException(status_code=502, detail="Invalid response from analysis service")
        
        return {
            "sentiment": sentiment_result["sentiment"],
            "confidence": sentiment_result.get("confidence", 0.5),
            "analysis_time": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Sentiment analysis error: {e}")
        raise HTTPException(status_code=500, detail="Sentiment analysis failed")
```

## 5. File Upload Pattern

```python
async def upload_post_image(user_id: str, file: UploadFile, post_id: str = None) -> Dict[str, Any]:
    """
    Upload post image with error-first file validation
    """
    # ============================================================================
    # ERROR-FIRST PARAMETER VALIDATION (P0 Priority)
    # ============================================================================
    
    # Validate required parameters
    if not user_id or not user_id.strip():
        raise HTTPException(status_code=400, detail="User ID is required")
    
    if not file:
        raise HTTPException(status_code=400, detail="File is required")
    
    user_id = user_id.strip()
    
    # ============================================================================
    # ERROR-FIRST FILE VALIDATION (P0 Priority)
    # ============================================================================
    
    # Check file size
    if file.size > 5 * 1024 * 1024:  # 5MB limit
        raise HTTPException(status_code=413, detail="File too large (maximum 5MB)")
    
    # Validate file type
    allowed_types = {
        'image/jpeg', 'image/png', 'image/gif', 'image/webp'
    }
    
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Invalid file type. Only images are allowed")
    
    # Validate file extension
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Invalid file extension")
    
    # Check for malicious file names
    malicious_patterns = [
        r'\.\./',  # Path traversal
        r'<script',  # XSS in filename
        r'javascript:',  # JavaScript in filename
    ]
    
    for pattern in malicious_patterns:
        if re.search(pattern, file.filename, re.IGNORECASE):
            logger.warning(f"Malicious filename detected: {file.filename}")
            raise HTTPException(status_code=400, detail="Invalid filename")
    
    # ============================================================================
    # ERROR-FIRST USER VALIDATION (P1 Priority)
    # ============================================================================
    
    try:
        # Check if user exists and is active
        user = await get_user_by_id(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        if not user.is_active:
            raise HTTPException(status_code=401, detail="Account is deactivated")
        
        # Check user's upload quota
        user_uploads = await get_user_upload_count(user_id, date.today())
        if user_uploads >= 10:  # 10 uploads per day
            raise HTTPException(status_code=429, detail="Daily upload limit exceeded")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"User validation error: {e}")
        raise HTTPException(status_code=500, detail="User validation failed")
    
    # ============================================================================
    # ERROR-FIRST POST VALIDATION (P1 Priority)
    # ============================================================================
    
    if post_id:
        try:
            # Check if post exists and user owns it
            post = await get_post_by_id(post_id)
            if not post:
                raise HTTPException(status_code=404, detail="Post not found")
            
            if post.user_id != user_id:
                raise HTTPException(status_code=403, detail="Cannot upload to another user's post")
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Post validation error: {e}")
            raise HTTPException(status_code=500, detail="Post validation failed")
    
    # ============================================================================
    # BUSINESS LOGIC (After all validation passes)
    # ============================================================================
    
    try:
        # Generate safe filename
        safe_filename = f"{user_id}_{uuid.uuid4()}{file_extension}"
        
        # Upload file to storage
        file_url = await upload_file_to_storage(file, safe_filename)
        
        # Save file metadata to database
        file_record = await save_file_metadata(user_id, safe_filename, file_url, post_id)
        
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
        raise HTTPException(status_code=500, detail="File upload failed")
```

## 6. Error-First Pattern Best Practices

### **1. Validation Order (Priority-Based)**
```python
def process_user_request(user_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
    # P0: Security validation (CRITICAL - Do first)
    validate_security_parameters(user_id, data)
    
    # P1: Input validation (HIGH - Do second)
    validate_input_parameters(data)
    
    # P2: Business validation (MEDIUM - Do third)
    validate_business_rules(data)
    
    # P3: Performance validation (LOW - Do last)
    validate_performance_limits(data)
    
    # Business logic (After all validation passes)
    return process_validated_request(user_id, data)
```

### **2. Early Returns for Invalid States**
```python
async def process_post_engagement(post_id: str, user_id: str, action: str) -> Dict[str, Any]:
    # Early return for invalid post ID
    if not post_id or not post_id.strip():
        return {"error": "Invalid post ID", "status": "failed"}
    
    # Early return for invalid user ID
    if not user_id or not user_id.strip():
        return {"error": "Invalid user ID", "status": "failed"}
    
    # Early return for invalid action
    if action not in ["like", "share", "comment"]:
        return {"error": "Invalid action", "status": "failed"}
    
    # Process valid request
    return await perform_engagement_action(post_id, user_id, action)
```

### **3. Defensive Programming with Guards**
```python
async def get_user_analytics(user_id: str, date_range: str) -> Dict[str, Any]:
    # Guard clause for missing user ID
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID is required")
    
    # Guard clause for invalid date range
    if date_range not in ["day", "week", "month", "year"]:
        raise HTTPException(status_code=400, detail="Invalid date range")
    
    # Guard clause for user existence
    user = await get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Guard clause for user permissions
    if not user.has_analytics_access:
        raise HTTPException(status_code=403, detail="Analytics access denied")
    
    # Process analytics (all guards passed)
    return await calculate_user_analytics(user_id, date_range)
```

### **4. Comprehensive Error Context**
```python
async def validate_and_process_post(post_data: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    try:
        # Validation with detailed error context
        if not post_data.get("content"):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "missing_content",
                    "field": "content",
                    "message": "Post content is required",
                    "user_id": user_id,
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        # Process valid post
        return await create_post(post_data, user_id)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing post for user {user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "internal_error",
                "message": "An unexpected error occurred",
                "user_id": user_id,
                "timestamp": datetime.now().isoformat()
            }
        )
```

## 7. Error-First Pattern Checklist

### **Before Business Logic, Always Validate:**

✅ **P0 (Critical) - Security:**
- [ ] Input sanitization
- [ ] Authentication validation
- [ ] Authorization checks
- [ ] SQL injection prevention
- [ ] XSS attack detection

✅ **P1 (High) - Data Integrity:**
- [ ] Required field validation
- [ ] Data type validation
- [ ] Length and format checks
- [ ] Business rule validation
- [ ] Rate limiting checks

✅ **P2 (Medium) - Infrastructure:**
- [ ] Database connection status
- [ ] External service availability
- [ ] Resource availability
- [ ] Concurrent access checks

✅ **P3 (Low) - Performance:**
- [ ] Payload size limits
- [ ] Processing time estimates
- [ ] Memory usage checks
- [ ] Cache availability

### **Error Handling Best Practices:**

✅ **Fail Fast:**
- [ ] Validate inputs immediately
- [ ] Return errors before processing
- [ ] Use guard clauses
- [ ] Avoid deep nesting

✅ **Clear Error Messages:**
- [ ] Provide specific error details
- [ ] Include field names in errors
- [ ] Add context information
- [ ] Use consistent error format

✅ **Proper Logging:**
- [ ] Log all validation failures
- [ ] Include user context
- [ ] Use appropriate log levels
- [ ] Add request correlation IDs

✅ **Graceful Degradation:**
- [ ] Provide fallback responses
- [ ] Handle partial failures
- [ ] Return meaningful defaults
- [ ] Maintain system stability

This error-first pattern ensures that your LinkedIn posts system is robust, secure, and maintainable by catching and handling all potential issues before they can cause problems in the business logic. 