# Bulk TruthGPT API Documentation

## Overview

Bulk TruthGPT is an ultra-advanced Flask application that provides continuous document generation based on a single query. The system is built with Flask best practices, functional programming, and advanced optimization techniques.

## Base URL

```
https://bulk-truthgpt.example.com/api/v1
```

## Authentication

The API uses JWT (JSON Web Token) authentication. Include the token in the Authorization header:

```
Authorization: Bearer <your-jwt-token>
```

## Endpoints

### Health Check

#### GET /health

Check the health status of the application.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00Z",
  "version": "1.0.0",
  "uptime": 3600
}
```

### Authentication

#### POST /auth/login

Authenticate a user and receive JWT tokens.

**Request Body:**
```json
{
  "username": "string",
  "password": "string"
}
```

**Response:**
```json
{
  "access_token": "string",
  "refresh_token": "string",
  "user": {
    "id": "string",
    "username": "string",
    "email": "string",
    "is_active": true,
    "is_admin": false,
    "created_at": "2024-01-01T00:00:00Z",
    "last_login": "2024-01-01T00:00:00Z"
  }
}
```

#### POST /auth/register

Register a new user.

**Request Body:**
```json
{
  "username": "string",
  "email": "string",
  "password": "string",
  "confirm_password": "string"
}
```

**Response:**
```json
{
  "message": "User registered successfully",
  "user": {
    "id": "string",
    "username": "string",
    "email": "string",
    "is_active": true,
    "is_admin": false,
    "created_at": "2024-01-01T00:00:00Z"
  }
}
```

#### POST /auth/refresh

Refresh the access token using the refresh token.

**Response:**
```json
{
  "access_token": "string"
}
```

#### POST /auth/logout

Logout the current user.

**Response:**
```json
{
  "message": "Logged out successfully"
}
```

### User Management

#### GET /user/profile

Get the current user's profile.

**Headers:**
```
Authorization: Bearer <token>
```

**Response:**
```json
{
  "user": {
    "id": "string",
    "username": "string",
    "email": "string",
    "is_active": true,
    "is_admin": false,
    "created_at": "2024-01-01T00:00:00Z",
    "last_login": "2024-01-01T00:00:00Z"
  }
}
```

#### POST /user/change-password

Change the user's password.

**Headers:**
```
Authorization: Bearer <token>
```

**Request Body:**
```json
{
  "current_password": "string",
  "new_password": "string",
  "confirm_password": "string"
}
```

**Response:**
```json
{
  "message": "Password changed successfully"
}
```

### Document Generation

#### POST /documents/generate

Generate documents based on a query.

**Headers:**
```
Authorization: Bearer <token>
```

**Request Body:**
```json
{
  "query": "string",
  "count": 10,
  "format": "markdown",
  "options": {
    "include_metadata": true,
    "include_sources": true,
    "language": "en"
  }
}
```

**Response:**
```json
{
  "task_id": "string",
  "status": "processing",
  "estimated_completion": "2024-01-01T00:05:00Z",
  "documents": []
}
```

#### GET /documents/status/{task_id}

Get the status of a document generation task.

**Headers:**
```
Authorization: Bearer <token>
```

**Response:**
```json
{
  "task_id": "string",
  "status": "completed",
  "progress": 100,
  "documents": [
    {
      "id": "string",
      "title": "string",
      "content": "string",
      "format": "markdown",
      "created_at": "2024-01-01T00:00:00Z",
      "metadata": {
        "word_count": 1000,
        "reading_time": 5,
        "language": "en"
      }
    }
  ]
}
```

#### GET /documents/{document_id}

Get a specific document.

**Headers:**
```
Authorization: Bearer <token>
```

**Response:**
```json
{
  "id": "string",
  "title": "string",
  "content": "string",
  "format": "markdown",
  "created_at": "2024-01-01T00:00:00Z",
  "metadata": {
    "word_count": 1000,
    "reading_time": 5,
    "language": "en"
  }
}
```

#### DELETE /documents/{document_id}

Delete a specific document.

**Headers:**
```
Authorization: Bearer <token>
```

**Response:**
```json
{
  "message": "Document deleted successfully"
}
```

### Optimization

#### GET /optimization/status

Get the current optimization status.

**Headers:**
```
Authorization: Bearer <token>
```

**Response:**
```json
{
  "status": "active",
  "algorithms": [
    "genetic",
    "bayesian",
    "gradient",
    "simulated_annealing"
  ],
  "performance_metrics": {
    "response_time": 0.5,
    "memory_usage": 512,
    "cpu_usage": 25.5
  },
  "optimization_history": [
    {
      "timestamp": "2024-01-01T00:00:00Z",
      "algorithm": "genetic",
      "improvement": 0.15
    }
  ]
}
```

#### POST /optimization/optimize

Trigger manual optimization.

**Headers:**
```
Authorization: Bearer <token>
```

**Request Body:**
```json
{
  "algorithm": "genetic",
  "parameters": {
    "population_size": 100,
    "generations": 50,
    "mutation_rate": 0.1
  }
}
```

**Response:**
```json
{
  "optimization_id": "string",
  "status": "started",
  "estimated_completion": "2024-01-01T00:05:00Z"
}
```

#### GET /optimization/results/{optimization_id}

Get optimization results.

**Headers:**
```
Authorization: Bearer <token>
```

**Response:**
```json
{
  "optimization_id": "string",
  "status": "completed",
  "results": {
    "original_parameters": {},
    "optimized_parameters": {},
    "improvement": 0.25
  }
}
```

### Performance Monitoring

#### GET /monitoring/metrics

Get current performance metrics.

**Headers:**
```
Authorization: Bearer <token>
```

**Response:**
```json
{
  "timestamp": "2024-01-01T00:00:00Z",
  "system_metrics": {
    "cpu_percent": 25.5,
    "memory_percent": 60.2,
    "disk_percent": 45.8
  },
  "application_metrics": {
    "response_time": 0.5,
    "throughput": 100,
    "error_rate": 0.01
  },
  "custom_metrics": {
    "api_calls": 1000,
    "database_queries": 500
  }
}
```

#### GET /monitoring/alerts

Get current alerts.

**Headers:**
```
Authorization: Bearer <token>
```

**Response:**
```json
{
  "alerts": [
    {
      "id": "string",
      "type": "high_cpu_usage",
      "severity": "high",
      "message": "CPU usage is above 90%",
      "timestamp": "2024-01-01T00:00:00Z",
      "status": "active"
    }
  ]
}
```

#### GET /monitoring/dashboard

Get dashboard data.

**Headers:**
```
Authorization: Bearer <token>
```

**Response:**
```json
{
  "timestamp": "2024-01-01T00:00:00Z",
  "system": {
    "cpu_percent": 25.5,
    "memory_percent": 60.2,
    "disk_percent": 45.8
  },
  "application": {
    "memory_usage": 512,
    "cpu_percent": 25.5
  },
  "alerts": [],
  "metrics": {
    "real_time": {},
    "performance": {},
    "resource": {},
    "custom": {}
  }
}
```

### Analytics

#### GET /analytics/performance

Get performance analytics.

**Headers:**
```
Authorization: Bearer <token>
```

**Response:**
```json
{
  "cpu": {
    "mean": 25.5,
    "std": 5.2,
    "max": 90.0,
    "min": 10.0,
    "trend": 0.1
  },
  "memory": {
    "mean": 60.2,
    "std": 10.5,
    "max": 95.0,
    "min": 30.0,
    "trend": 0.2
  },
  "insights": [
    {
      "type": "cpu_trend",
      "message": "CPU usage is increasing",
      "severity": "medium"
    }
  ]
}
```

#### GET /analytics/usage

Get usage analytics.

**Headers:**
```
Authorization: Bearer <token>
```

**Response:**
```json
{
  "total_requests": 10000,
  "unique_users": 1000,
  "average_response_time": 0.5,
  "error_rate": 0.01,
  "top_endpoints": [
    {
      "endpoint": "/documents/generate",
      "count": 5000
    }
  ]
}
```

## Error Responses

All endpoints may return the following error responses:

### 400 Bad Request
```json
{
  "error": "Validation error",
  "details": {
    "field": "error message"
  }
}
```

### 401 Unauthorized
```json
{
  "error": "Authentication required"
}
```

### 403 Forbidden
```json
{
  "error": "Insufficient permissions"
}
```

### 404 Not Found
```json
{
  "error": "Resource not found"
}
```

### 429 Too Many Requests
```json
{
  "error": "Rate limit exceeded",
  "message": "Too many requests. Limit: 100 per hour",
  "retry_after": 60
}
```

### 500 Internal Server Error
```json
{
  "error": "Internal server error",
  "message": "An unexpected error occurred"
}
```

## Rate Limiting

The API implements rate limiting to ensure fair usage:

- **Authentication endpoints**: 5 requests per minute
- **Document generation**: 10 requests per hour
- **General API**: 100 requests per hour

Rate limit headers are included in responses:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1640995200
```

## Pagination

List endpoints support pagination:

```
GET /documents?page=1&limit=10&sort=created_at&order=desc
```

**Response:**
```json
{
  "data": [],
  "pagination": {
    "page": 1,
    "limit": 10,
    "total": 100,
    "pages": 10,
    "has_next": true,
    "has_prev": false
  }
}
```

## Filtering and Sorting

Many endpoints support filtering and sorting:

```
GET /documents?status=completed&created_after=2024-01-01&sort=created_at&order=desc
```

## Webhooks

The API supports webhooks for real-time notifications:

### Webhook Events

- `document.generated` - When a document is generated
- `document.failed` - When document generation fails
- `optimization.completed` - When optimization is completed
- `alert.triggered` - When an alert is triggered

### Webhook Payload

```json
{
  "event": "document.generated",
  "timestamp": "2024-01-01T00:00:00Z",
  "data": {
    "document_id": "string",
    "user_id": "string",
    "status": "completed"
  }
}
```

## SDKs

Official SDKs are available for:

- **Python**: `pip install bulk-truthgpt-sdk`
- **JavaScript**: `npm install bulk-truthgpt-sdk`
- **Go**: `go get github.com/bulk-truthgpt/sdk-go`

## Examples

### Python Example

```python
from bulk_truthgpt import BulkTruthGPT

client = BulkTruthGPT(
    api_key="your-api-key",
    base_url="https://bulk-truthgpt.example.com/api/v1"
)

# Generate documents
task = client.documents.generate(
    query="Artificial Intelligence",
    count=10,
    format="markdown"
)

# Check status
status = client.documents.status(task.task_id)
print(f"Status: {status.status}")
print(f"Progress: {status.progress}%")

# Get documents
if status.status == "completed":
    for document in status.documents:
        print(f"Title: {document.title}")
        print(f"Content: {document.content[:100]}...")
```

### JavaScript Example

```javascript
const BulkTruthGPT = require('bulk-truthgpt-sdk');

const client = new BulkTruthGPT({
  apiKey: 'your-api-key',
  baseURL: 'https://bulk-truthgpt.example.com/api/v1'
});

// Generate documents
const task = await client.documents.generate({
  query: 'Artificial Intelligence',
  count: 10,
  format: 'markdown'
});

// Check status
const status = await client.documents.status(task.taskId);
console.log(`Status: ${status.status}`);
console.log(`Progress: ${status.progress}%`);

// Get documents
if (status.status === 'completed') {
  for (const document of status.documents) {
    console.log(`Title: ${document.title}`);
    console.log(`Content: ${document.content.substring(0, 100)}...`);
  }
}
```

## Support

For support, please contact:

- **Email**: support@bulk-truthgpt.com
- **Documentation**: https://docs.bulk-truthgpt.com
- **GitHub**: https://github.com/bulk-truthgpt/bulk-truthgpt
- **Discord**: https://discord.gg/bulk-truthgpt









