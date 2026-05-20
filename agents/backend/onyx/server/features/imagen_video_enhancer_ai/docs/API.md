# API Documentation - Imagen Video Enhancer AI

## Base URL

```
http://localhost:8000
```

## Authentication

Currently no authentication required. For production, add API key authentication.

## Endpoints

### File Upload

#### POST /upload-image
Upload and enhance an image.

**Request:**
- `file` (multipart/form-data): Image file
- `enhancement_type` (form): Type of enhancement (default: "general")
- `priority` (form): Task priority (default: 0)
- `options` (form, optional): JSON string with additional options

**Response:**
```json
{
  "success": true,
  "task_id": "uuid",
  "file_path": "/path/to/file",
  "image_info": {
    "width": 1920,
    "height": 1080,
    "file_size_mb": 2.5
  },
  "estimated_time_seconds": 5.0,
  "message": "Image uploaded and enhancement task created"
}
```

#### POST /upload-video
Upload and enhance a video.

**Request:**
- `file` (multipart/form-data): Video file
- `enhancement_type` (form): Type of enhancement
- `priority` (form): Task priority
- `options` (form, optional): JSON string with additional options

**Response:**
```json
{
  "success": true,
  "task_id": "uuid",
  "file_path": "/path/to/file",
  "video_analysis": {
    "fps": 30.0,
    "resolution": "1920x1080",
    "duration_seconds": 60.0
  },
  "estimated_time_seconds": 120.0,
  "message": "Video uploaded and enhancement task created"
}
```

### Enhancement Services

#### POST /enhance-image
Enhance an existing image.

**Request Body:**
```json
{
  "file_path": "/path/to/image.jpg",
  "enhancement_type": "general",
  "options": {},
  "priority": 0
}
```

#### POST /enhance-video
Enhance an existing video.

#### POST /upscale
Upscale an image or video.

**Request Body:**
```json
{
  "file_path": "/path/to/image.jpg",
  "scale_factor": 2,
  "options": {},
  "priority": 0
}
```

#### POST /denoise
Reduce noise in an image or video.

#### POST /restore
Restore a damaged image.

#### POST /color-correction
Apply color correction.

### Analysis

#### POST /analyze
Analyze a file (image or video).

**Request Body:**
```json
{
  "file_path": "/path/to/file",
  "file_type": "image"  // optional, auto-detected
}
```

**Response:**
```json
{
  "file_type": "image",
  "info": {
    "width": 1920,
    "height": 1080,
    "file_size_mb": 2.5
  }
}
```

### Batch Processing

#### POST /batch-process
Process multiple files in batch.

**Request Body:**
```json
{
  "items": [
    {
      "file_path": "/path/to/image1.jpg",
      "service_type": "enhance_image",
      "enhancement_type": "general",
      "options": {},
      "priority": 0
    },
    {
      "file_path": "/path/to/image2.jpg",
      "service_type": "upscale",
      "options": {"scale_factor": 2},
      "priority": 0
    }
  ]
}
```

**Response:**
```json
{
  "total_items": 2,
  "completed": 2,
  "failed": 0,
  "success_rate": 1.0,
  "duration": 15.5,
  "results": [...],
  "errors": []
}
```

### Task Management

#### GET /task/{task_id}/status
Get task status.

**Response:**
```json
{
  "id": "task-id",
  "status": "completed",
  "created_at": "2024-01-01T00:00:00",
  "started_at": "2024-01-01T00:00:01",
  "completed_at": "2024-01-01T00:00:05"
}
```

#### GET /task/{task_id}/result
Get task result.

**Response:**
```json
{
  "enhancement_guide": "Detailed enhancement guide...",
  "tokens_used": 1500,
  "model": "anthropic/claude-3.5-sonnet"
}
```

### Webhooks

#### POST /webhooks/register
Register a webhook.

**Request Body:**
```json
{
  "url": "https://example.com/webhook",
  "events": ["task.completed", "task.failed"],
  "secret": "your-secret-key",
  "timeout": 10.0,
  "retries": 3,
  "enabled": true
}
```

#### DELETE /webhooks/unregister
Unregister a webhook.

**Query Parameters:**
- `url`: Webhook URL to unregister

### Statistics

#### GET /stats
Get agent statistics.

**Response:**
```json
{
  "executor_stats": {
    "total_tasks": 100,
    "completed_tasks": 95,
    "failed_tasks": 5
  },
  "cache_stats": {
    "hits": 50,
    "misses": 50,
    "hit_rate": 0.5
  },
  "webhook_stats": {
    "total_sent": 200,
    "successful": 195,
    "failed": 5
  },
  "performance_stats": {
    "enhance_image": {
      "count": 50,
      "average": 2.5,
      "min": 1.0,
      "max": 5.0
    }
  }
}
```

### Health

#### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "agent_initialized": true
}
```

## Error Responses

All errors follow this format:

```json
{
  "detail": "Error message"
}
```

**Status Codes:**
- `400`: Bad Request (validation errors)
- `404`: Not Found
- `429`: Too Many Requests (rate limit)
- `500`: Internal Server Error
- `503`: Service Unavailable (agent not initialized)

## Rate Limiting

Default rate limit: 10 requests per second per IP, with burst of 20.

When rate limited, response includes:
- Status: `429`
- Header: `Retry-After: <seconds>`




