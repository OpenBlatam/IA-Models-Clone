# 📚 API Documentation - Improved Blog System

Complete API reference for the improved blog system built with FastAPI.

## 🌐 Base URL

```
Production: https://yourdomain.com/api/v1
Development: http://localhost:8000/api/v1
```

## 🔐 Authentication

The API uses JWT (JSON Web Token) for authentication. Include the token in the Authorization header:

```
Authorization: Bearer <your-jwt-token>
```

### Getting a Token

```bash
# Register a new user
POST /api/v1/users/
{
  "email": "user@example.com",
  "username": "username",
  "password": "SecurePassword123!",
  "full_name": "Full Name"
}

# Login (if login endpoint is implemented)
POST /api/v1/auth/login
{
  "email": "user@example.com",
  "password": "SecurePassword123!"
}
```

## 📝 Blog Posts API

### Create Blog Post

**POST** `/api/v1/blog-posts/`

Create a new blog post.

**Request Body:**
```json
{
  "title": "Getting Started with FastAPI",
  "content": "FastAPI is a modern, fast web framework for building APIs with Python...",
  "excerpt": "Learn how to build modern APIs with FastAPI",
  "category": "technology",
  "tags": ["python", "fastapi", "web-development"],
  "seo_title": "FastAPI Tutorial - Build Modern APIs",
  "seo_description": "Complete guide to building APIs with FastAPI",
  "seo_keywords": ["fastapi", "python", "api", "tutorial"],
  "featured_image_url": "https://example.com/image.jpg",
  "scheduled_at": "2024-01-15T10:00:00Z"
}
```

**Response:**
```json
{
  "id": 1,
  "uuid": "123e4567-e89b-12d3-a456-426614174000",
  "title": "Getting Started with FastAPI",
  "slug": "getting-started-with-fastapi",
  "content": "FastAPI is a modern, fast web framework...",
  "excerpt": "Learn how to build modern APIs with FastAPI",
  "author_id": "user-uuid",
  "status": "draft",
  "category": "technology",
  "tags": ["python", "fastapi", "web-development"],
  "view_count": 0,
  "like_count": 0,
  "comment_count": 0,
  "word_count": 150,
  "reading_time_minutes": 1,
  "created_at": "2024-01-15T09:00:00Z",
  "updated_at": "2024-01-15T09:00:00Z",
  "published_at": null,
  "scheduled_at": "2024-01-15T10:00:00Z",
  "sentiment_score": 0.8,
  "readability_score": 75.5,
  "topic_tags": ["programming", "web-development"]
}
```

### Get Blog Post

**GET** `/api/v1/blog-posts/{post_id}`

Get a specific blog post by ID.

**Response:** Same as create response

### Get Blog Post by Slug

**GET** `/api/v1/blog-posts/slug/{slug}`

Get a blog post by its slug.

**Response:** Same as create response

### List Blog Posts

**GET** `/api/v1/blog-posts/`

List blog posts with pagination and filtering.

**Query Parameters:**
- `page` (int): Page number (default: 1)
- `size` (int): Page size (default: 20, max: 100)
- `status` (string): Filter by status (draft, published, archived)
- `category` (string): Filter by category
- `author_id` (string): Filter by author
- `include_drafts` (bool): Include draft posts (default: false)

**Response:**
```json
{
  "items": [
    {
      "id": 1,
      "uuid": "123e4567-e89b-12d3-a456-426614174000",
      "title": "Getting Started with FastAPI",
      "slug": "getting-started-with-fastapi",
      "excerpt": "Learn how to build modern APIs with FastAPI",
      "author_id": "user-uuid",
      "status": "published",
      "category": "technology",
      "tags": ["python", "fastapi"],
      "view_count": 150,
      "like_count": 25,
      "comment_count": 8,
      "reading_time_minutes": 5,
      "featured_image_url": "https://example.com/image.jpg",
      "created_at": "2024-01-15T09:00:00Z",
      "published_at": "2024-01-15T10:00:00Z"
    }
  ],
  "total": 1,
  "page": 1,
  "size": 20,
  "pages": 1,
  "has_next": false,
  "has_prev": false
}
```

### Search Blog Posts

**GET** `/api/v1/blog-posts/search`

Search blog posts with advanced filtering.

**Query Parameters:**
- `query` (string): Search query (required)
- `page` (int): Page number (default: 1)
- `size` (int): Page size (default: 20)
- `category` (string): Filter by category
- `tags` (string): Comma-separated tags
- `author_id` (string): Filter by author
- `status` (string): Filter by status
- `sort_by` (string): Sort field (default: created_at)
- `sort_order` (string): Sort order (asc/desc, default: desc)

**Response:** Same as list response

### Update Blog Post

**PUT** `/api/v1/blog-posts/{post_id}`

Update an existing blog post.

**Request Body:**
```json
{
  "title": "Updated Title",
  "content": "Updated content...",
  "status": "published",
  "category": "technology",
  "tags": ["updated", "tags"]
}
```

**Response:** Same as create response

### Delete Blog Post

**DELETE** `/api/v1/blog-posts/{post_id}`

Delete a blog post.

**Response:** 204 No Content

### Increment View Count

**POST** `/api/v1/blog-posts/{post_id}/view`

Increment the view count for a blog post.

**Response:**
```json
{
  "message": "View count incremented"
}
```

## 👥 Users API

### Create User

**POST** `/api/v1/users/`

Create a new user account.

**Request Body:**
```json
{
  "email": "user@example.com",
  "username": "username",
  "password": "SecurePassword123!",
  "full_name": "Full Name",
  "bio": "User bio",
  "website_url": "https://userwebsite.com"
}
```

**Response:**
```json
{
  "id": "user-uuid",
  "email": "user@example.com",
  "username": "username",
  "full_name": "Full Name",
  "bio": "User bio",
  "website_url": "https://userwebsite.com",
  "is_active": true,
  "is_verified": false,
  "roles": ["user"],
  "avatar_url": null,
  "created_at": "2024-01-15T09:00:00Z",
  "updated_at": "2024-01-15T09:00:00Z",
  "last_login_at": null
}
```

### Get Current User

**GET** `/api/v1/users/me`

Get the current authenticated user's profile.

**Response:** Same as create user response

### Get User

**GET** `/api/v1/users/{user_id}`

Get a specific user by ID.

**Response:** Same as create user response

### Update Current User

**PUT** `/api/v1/users/me`

Update the current user's profile.

**Request Body:**
```json
{
  "full_name": "Updated Name",
  "bio": "Updated bio",
  "website_url": "https://newwebsite.com",
  "avatar_url": "https://example.com/avatar.jpg"
}
```

**Response:** Same as create user response

## 💬 Comments API

### Create Comment

**POST** `/api/v1/comments/`

Create a new comment on a blog post.

**Query Parameters:**
- `post_id` (int): Blog post ID (required)

**Request Body:**
```json
{
  "content": "This is a great post! Thanks for sharing.",
  "parent_id": null
}
```

**Response:**
```json
{
  "id": 1,
  "uuid": "comment-uuid",
  "post_id": 1,
  "author_id": "user-uuid",
  "content": "This is a great post! Thanks for sharing.",
  "parent_id": null,
  "is_approved": false,
  "created_at": "2024-01-15T09:00:00Z",
  "updated_at": "2024-01-15T09:00:00Z"
}
```

### List Comments

**GET** `/api/v1/comments/`

List comments for a blog post.

**Query Parameters:**
- `post_id` (int): Blog post ID (required)
- `page` (int): Page number (default: 1)
- `size` (int): Page size (default: 20)
- `approved_only` (bool): Show only approved comments (default: true)

**Response:**
```json
{
  "items": [
    {
      "id": 1,
      "uuid": "comment-uuid",
      "post_id": 1,
      "author_id": "user-uuid",
      "content": "This is a great post!",
      "parent_id": null,
      "is_approved": true,
      "created_at": "2024-01-15T09:00:00Z",
      "updated_at": "2024-01-15T09:00:00Z"
    }
  ],
  "total": 1,
  "page": 1,
  "size": 20,
  "pages": 1,
  "has_next": false,
  "has_prev": false
}
```

### Get Comment

**GET** `/api/v1/comments/{comment_id}`

Get a specific comment by ID.

**Response:** Same as create comment response

### Approve Comment

**PUT** `/api/v1/comments/{comment_id}/approve`

Approve a comment (admin/moderator only).

**Response:** Same as create comment response

### Reject Comment

**PUT** `/api/v1/comments/{comment_id}/reject`

Reject a comment (admin/moderator only).

**Response:** Same as create comment response

### Delete Comment

**DELETE** `/api/v1/comments/{comment_id}`

Delete a comment.

**Response:** 204 No Content

## 📁 File Management API

### Upload File

**POST** `/api/v1/files/upload`

Upload a file.

**Request:** Multipart form data with file

**Response:**
```json
{
  "id": 1,
  "uuid": "file-uuid",
  "filename": "unique-filename.jpg",
  "original_filename": "image.jpg",
  "file_size": 1024000,
  "mime_type": "image/jpeg",
  "url": "/files/file-uuid",
  "created_at": "2024-01-15T09:00:00Z"
}
```

### Get File Info

**GET** `/api/v1/files/{file_uuid}`

Get file information by UUID.

**Response:** Same as upload response

### Download File

**GET** `/api/v1/files/{file_uuid}/download`

Download a file.

**Response:** File download

### List User Files

**GET** `/api/v1/files/`

List files uploaded by the current user.

**Query Parameters:**
- `page` (int): Page number (default: 1)
- `size` (int): Page size (default: 20)

**Response:**
```json
{
  "items": [
    {
      "id": 1,
      "uuid": "file-uuid",
      "filename": "unique-filename.jpg",
      "original_filename": "image.jpg",
      "file_size": 1024000,
      "mime_type": "image/jpeg",
      "url": "/files/file-uuid",
      "created_at": "2024-01-15T09:00:00Z"
    }
  ],
  "total": 1,
  "page": 1,
  "size": 20,
  "pages": 1,
  "has_next": false,
  "has_prev": false
}
```

### Delete File

**DELETE** `/api/v1/files/{file_uuid}`

Delete a file.

**Response:** 204 No Content

### Get File Stats

**GET** `/api/v1/files/stats/summary`

Get file upload statistics for the current user.

**Response:**
```json
{
  "total_files": 10,
  "total_size": 52428800,
  "files_by_type": {
    "image/jpeg": 5,
    "image/png": 3,
    "application/pdf": 2
  }
}
```

## 🏥 Health Check API

### Health Check

**GET** `/api/v1/health/`

Comprehensive health check.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T09:00:00Z",
  "services": {
    "database": true,
    "cache": true
  },
  "version": "1.0.0"
}
```

### Simple Health Check

**GET** `/api/v1/health/simple`

Simple health check.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T09:00:00Z",
  "version": "1.0.0"
}
```

## 📊 Status Codes

| Code | Description |
|------|-------------|
| 200 | OK - Request successful |
| 201 | Created - Resource created successfully |
| 204 | No Content - Request successful, no content returned |
| 400 | Bad Request - Invalid request data |
| 401 | Unauthorized - Authentication required |
| 403 | Forbidden - Insufficient permissions |
| 404 | Not Found - Resource not found |
| 409 | Conflict - Resource already exists |
| 422 | Unprocessable Entity - Validation error |
| 429 | Too Many Requests - Rate limit exceeded |
| 500 | Internal Server Error - Server error |

## 🔍 Error Responses

All error responses follow this format:

```json
{
  "error": "Error message",
  "error_code": "ERROR_CODE",
  "detail": "Detailed error information",
  "timestamp": "2024-01-15T09:00:00Z",
  "path": "/api/v1/blog-posts/",
  "request_id": "request-uuid"
}
```

### Common Error Codes

- `VALIDATION_ERROR`: Input validation failed
- `NOT_FOUND`: Resource not found
- `CONFLICT`: Resource conflict
- `AUTHENTICATION_ERROR`: Authentication failed
- `AUTHORIZATION_ERROR`: Insufficient permissions
- `RATE_LIMIT_EXCEEDED`: Rate limit exceeded
- `FILE_UPLOAD_ERROR`: File upload failed

## 🔧 Rate Limiting

The API implements rate limiting to prevent abuse:

- **Default**: 100 requests per minute per IP
- **Headers**: Rate limit information is included in response headers
- **Exceeded**: Returns 429 status code with retry information

## 📝 Pagination

List endpoints support pagination:

- **page**: Page number (starts from 1)
- **size**: Number of items per page (max 100)
- **Response**: Includes pagination metadata

## 🔍 Filtering and Sorting

Many endpoints support filtering and sorting:

- **Filtering**: Use query parameters to filter results
- **Sorting**: Use `sort_by` and `sort_order` parameters
- **Search**: Full-text search with `query` parameter

## 📱 SDKs and Examples

### Python Example

```python
import requests

# Set base URL and token
BASE_URL = "https://yourdomain.com/api/v1"
TOKEN = "your-jwt-token"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}

# Create a blog post
post_data = {
    "title": "My First Post",
    "content": "This is my first blog post!",
    "category": "technology",
    "tags": ["python", "fastapi"]
}

response = requests.post(
    f"{BASE_URL}/blog-posts/",
    json=post_data,
    headers=HEADERS
)

if response.status_code == 201:
    post = response.json()
    print(f"Created post: {post['title']}")
```

### JavaScript Example

```javascript
const BASE_URL = 'https://yourdomain.com/api/v1';
const TOKEN = 'your-jwt-token';

// Create a blog post
const postData = {
  title: 'My First Post',
  content: 'This is my first blog post!',
  category: 'technology',
  tags: ['javascript', 'nodejs']
};

fetch(`${BASE_URL}/blog-posts/`, {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${TOKEN}`,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify(postData)
})
.then(response => response.json())
.then(post => console.log('Created post:', post.title));
```

## 🔗 Interactive Documentation

- **Swagger UI**: `https://yourdomain.com/docs`
- **ReDoc**: `https://yourdomain.com/redoc`

## 📞 Support

For API support and questions:

1. Check the interactive documentation
2. Review error messages and status codes
3. Verify authentication and permissions
4. Check rate limiting and quotas

This API documentation provides comprehensive information for integrating with the improved blog system. All endpoints are designed to be RESTful and follow modern API best practices.






























