# Lovable Community API

> Part of the [Blatam Academy Integrated Platform](../README.md)

Lovable-style community system where users can publish their chats, remix other users' chats, and the best chats appear at the top based on a ranking algorithm.

## Features

### Core Functionality
- **Publish chats**: Users can publish their conversations to the community
- **Remix**: Users can create remixes (modified versions) of existing chats
- **Voting System**: Upvote/downvote to rank content
- **Smart Ranking**: Algorithm combining votes, remixes, views, and recency
- **Advanced Search**: Search by text, tags, user, etc.
- **Top Chats**: View the most popular chats sorted by score

### Advanced Functionality
- **Update Chats**: Users can update their published chats
- **Delete Chats**: Users can delete their own chats
- **Feature Chats**: System to mark chats as featured
- **User Profiles**: View statistics and profile of any user
- **Trending Chats**: View trending chats in different periods (hour, day, week, month)
- **Analytics**: Aggregated community statistics
- **Batch Operations**: Perform operations on multiple chats simultaneously
- **Detailed Statistics**: Advanced metrics including upvotes, downvotes, and engagement rate

## Structure

```
lovable_community/
├── __init__.py
├── models.py          # Database models (SQLAlchemy)
├── schemas.py         # Pydantic schemas for validation
├── services.py        # Business logic and services
├── helpers.py         # Helper functions and utilities
├── validators.py      # Reusable validators
├── exceptions.py      # Custom exceptions
├── dependencies.py    # FastAPI dependencies
├── config.py          # Application configuration
├── api/
│   ├── __init__.py
│   ├── routes.py      # API Endpoints
│   └── router.py      # Main router
├── main.py            # Main FastAPI application
└── README.md
```

## Database Models

### PublishedChat
- `id`: Unique chat ID
- `user_id`: ID of the user who published
- `title`: Chat title
- `description`: Optional description
- `chat_content`: Chat content (JSON or text)
- `tags`: Comma-separated tags
- `vote_count`: Number of votes
- `remix_count`: Number of remixes
- `view_count`: Number of views
- `score`: Calculated ranking score
- `is_public`: Whether it is public
- `is_featured`: Whether it is featured
- `original_chat_id`: ID of the original chat (if remix)

### ChatRemix
- `id`: Unique remix ID
- `original_chat_id`: Original chat ID
- `remix_chat_id`: Remix chat ID
- `user_id`: ID of the user who created the remix

### ChatVote
- `id`: Unique vote ID
- `chat_id`: Voted chat ID
- `user_id`: ID of the user who voted
- `vote_type`: "upvote" or "downvote"

### ChatView
- `id`: Unique view ID
- `chat_id`: Viewed chat ID
- `user_id`: User ID (optional)

## Ranking Algorithm

The score is calculated using the following formula:

```
score = (votes * 2 + remixes * 3 + views * 0.1) / time_decay
```

Where `time_decay` increases with time to prioritize recent content.

## Endpoints

### POST `/lovable/community/publish`
Publishes a new chat to the community.

**Request:**
```json
{
  "title": "My amazing chat",
  "description": "A conversation about AI",
  "chat_content": "{...}",
  "tags": ["ai", "chat", "conversation"],
  "is_public": true
}
```

### GET `/lovable/community/chats`
Lists chats with pagination.

**Query Parameters:**
- `page`: Page number (default: 1)
- `page_size`: Page size (default: 20, max: 100)
- `sort_by`: Sort by: `score`, `created_at`, `vote_count`, `remix_count` (default: `score`)
- `order`: `asc` or `desc` (default: `desc`)
- `user_id`: Filter by user (optional)

### GET `/lovable/community/chats/{chat_id}`
Gets details of a specific chat.

### POST `/lovable/community/chats/{chat_id}/remix`
Creates a remix of an existing chat.

**Request:**
```json
{
  "original_chat_id": "chat-id",
  "title": "My remix",
  "description": "Improved version",
  "chat_content": "{...}",
  "tags": ["remix", "improved"]
}
```

### POST `/lovable/community/chats/{chat_id}/vote`
Votes for a chat (upvote or downvote).

**Request:**
```json
{
  "chat_id": "chat-id",
  "vote_type": "upvote"
}
```

### GET `/lovable/community/chats/{chat_id}/remixes`
Gets all remixes of a chat.

**Query Parameters:**
- `limit`: Result limit (default: 20, max: 100)

### GET `/lovable/community/search`
Searches chats by text, tags, user, etc.

**Query Parameters:**
- `query`: Search text (optional)
- `tags`: Comma-separated tags (optional)
- `user_id`: Filter by user (optional)
- `sort_by`: Sort by (default: `score`)
- `order`: `asc` or `desc` (default: `desc`)
- `page`: Page number (default: 1)
- `page_size`: Page size (default: 20)

### GET `/lovable/community/top`
Gets the most popular chats sorted by score.

**Query Parameters:**
- `limit`: Result limit (default: 20, max: 100)

### GET `/lovable/community/chats/{chat_id}/stats`
Gets engagement statistics for a chat.

### PUT `/lovable/community/chats/{chat_id}`
Updates an existing chat. Only the owner can update their chat.

**Request:**
```json
{
  "title": "Updated title",
  "description": "New description",
  "tags": ["new", "tag"],
  "is_public": true
}
```

### DELETE `/lovable/community/chats/{chat_id}`
Deletes a chat. Only the owner can delete their chat.

### POST `/lovable/community/chats/{chat_id}/feature`
Features or unfeatures a chat. Requires admin permissions.

**Query Parameters:**
- `featured`: `true` to feature, `false` to unfeature

### GET `/lovable/community/users/{user_id}/profile`
Gets a user's profile and statistics.

**Response:**
```json
{
  "user_id": "user-123",
  "total_chats": 10,
  "total_remixes": 5,
  "total_votes": 25,
  "average_score": 8.5,
  "top_chat_id": "chat-456"
}
```

### GET `/lovable/community/trending`
Gets trending chats in different time periods.

**Query Parameters:**
- `period`: `hour`, `day`, `week`, or `month` (default: `day`)
- `limit`: Result limit (default: 20, max: 100)

### GET `/lovable/community/analytics`
Gets aggregated statistics for the entire community.

**Query Parameters:**
- `period_days`: Number of days to filter (optional)

**Response:**
```json
{
  "total_chats": 1000,
  "total_users": 150,
  "total_votes": 5000,
  "total_remixes": 200,
  "total_views": 50000,
  "average_score": 7.5,
  "top_tags": [
    {"tag": "ai", "count": 150},
    {"tag": "chat", "count": 120}
  ],
  "period": "all time"
}
```

### POST `/lovable/community/bulk`
Performs a batch operation on multiple chats (max 100).

**Request:**
```json
{
  "chat_ids": ["chat1", "chat2", "chat3"],
  "operation": "feature"
}
```

**Available Operations:**
- `delete`: Delete chats (requires user_id)
- `feature`: Feature chats
- `unfeature`: Unfeature chats
- `make_public`: Make public
- `make_private`: Make private

### GET `/lovable/community/chats/{chat_id}/stats/detailed`
Gets detailed statistics including upvotes, downvotes, and engagement rate.

**Response:**
```json
{
  "chat_id": "chat-123",
  "vote_count": 50,
  "remix_count": 10,
  "view_count": 500,
  "score": 8.5,
  "rank": 5,
  "upvote_count": 45,
  "downvote_count": 5,
  "engagement_rate": 10.0
}
```

## Installation

1. Install dependencies:
```bash
pip install fastapi uvicorn sqlalchemy pydantic
```

2. Run the application:
```bash
python -m features.lovable_community.main
```

Or using uvicorn directly:
```bash
uvicorn features.lovable_community.main:app --host 0.0.0.0 --port 8007
```

## Database

By default uses SQLite (`lovable_community.db`). To switch to PostgreSQL or another database, modify the URL in `api/routes.py` and `main.py`.

## Authentication

Currently the system uses a simplified `user_id`. For production, you should:
1. Implement JWT authentication
2. Get `user_id` from token
3. Validate permissions before sensitive operations

## Architecture and Implemented Improvements

### Validation and Sanitization
- ✅ Exhaustive input validation with Pydantic
- ✅ Automatic data sanitization
- ✅ Reusable validators in `validators.py`
- ✅ Conversion and formatting helpers in `helpers.py`

### Error Handling
- ✅ Custom exceptions with descriptive messages
- ✅ Consistent error handling across all endpoints
- ✅ Detailed logging for debugging

### Optimizations
- ✅ Optimized database indexes
- ✅ Efficient queries with SQLAlchemy
- ✅ Optimized pagination
- ✅ Optimized score calculation

### New Features
- ✅ Chat update and deletion
- ✅ Featured system
- ✅ User profiles with statistics
- ✅ Trending chats by period
- ✅ Aggregated analytics
- ✅ Batch operations
- ✅ Detailed statistics

## Future Improvements

- [ ] Full JWT authentication
- [ ] Comment system (schemas already created)
- [ ] Notifications when someone remixes your chat
- [ ] Reporting and moderation system
- [ ] Rate limiting implemented
- [ ] Chat export
- [ ] Integration with other systems
- [ ] Cache for frequent queries
- [ ] WebSockets for real-time updates

---

[← Back to Main README](../README.md)
