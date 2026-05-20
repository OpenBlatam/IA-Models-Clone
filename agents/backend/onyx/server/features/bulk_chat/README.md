# Bulk Chat - Proactive Continuous Chat System

> Part of the [Blatam Academy Integrated Platform](../README.md)

**Bulk Chat** is a proactive, continuous chat system similar to ChatGPT, but with the unique feature that **it does not stop automatically**. The chat continuously generates responses until the user explicitly pauses it.

## 🚀 Features

### Core Capabilities
- ✅ **Continuous Chat**: Automatically generates responses without stopping
- ✅ **Pause Control**: User can pause/resume the chat at any time
- ✅ **Real-Time Streaming**: Support for Server-Sent Events (SSE)
- ✅ **Multiple Sessions**: Support for multiple simultaneous chat sessions
- ✅ **LLM Integration**: Support for OpenAI, Anthropic, and other providers

### Advanced Functionality
- ✅ **REST API**: Complete endpoints for chat control
- ✅ **Session Persistence**: Automatic saving to JSON or Redis
- ✅ **Metrics System**: Comprehensive performance and usage monitoring
- ✅ **Rate Limiting**: Request rate control to prevent abuse
- ✅ **Retry Logic**: Automatic retries with exponential backoff
- ✅ **Auto-save**: Automatic saving every 30 seconds (configurable)
- ✅ **WebSockets**: Bi-directional real-time streaming
- ✅ **Response Cache**: Reduces duplicate LLM calls
- ✅ **Plugin System**: Extensible with custom plugins
- ✅ **Conversation Analysis**: Advanced insights and statistics
- ✅ **Multi-format Export**: JSON, Markdown, CSV, HTML, TXT
- ✅ **Template System**: Predefined messages with variables
- ✅ **Webhooks**: Real-time notifications to external systems
- ✅ **JWT Authentication**: Complete auth system with roles
- ✅ **Automatic Backups**: Scheduled backups and restoration
- ✅ **Web Dashboard**: Interactive web interface for monitoring
- ✅ **Testing Framework**: Comprehensive tests with pytest
- ✅ **Performance Optimizer**: P50/P95/P99 metrics and bottleneck detection

## 📋 Requirements

- Python 3.8+
- FastAPI
- An LLM provider (OpenAI, Anthropic, etc.)

## 🔧 Installation

### Option 1: Automatic Installation (Recommended)

```bash
# Automatic installation
python install.py

# Verify installation
python verify_setup.py
```

### Option 2: Manual Installation

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Verify setup
python verify_setup.py

# 3. Configure environment variables (optional)
# Copy .env.example to .env and edit with your API keys
```

**Note**: If you don't have an API key, you can use `mock` mode:
```bash
python -m bulk_chat.main --llm-provider mock
```

### Quick Start Scripts

**Windows:**
```bash
start.bat
start.bat openai
```

**Linux/Mac:**
```bash
chmod +x start.sh
./start.sh
./start.sh openai
```

**Python (Cross-platform):**
```bash
python run.py server
python run.py server --provider openai
```

See [COMMANDS.md](COMMANDS.md) for more useful commands.

## 🚀 Quick Usage

### Start the Server

```bash
# Basic usage
python -m bulk_chat.main

# With custom options
python -m bulk_chat.main --host 0.0.0.0 --port 8006 --llm-provider openai --llm-model gpt-4
```

### Example Usage with cURL

```bash
# 1. Create a chat session
curl -X POST "http://localhost:8006/api/v1/chat/sessions" \
  -H "Content-Type: application/json" \
  -d '{
    "initial_message": "Hello, I want you to explain artificial intelligence to me",
    "auto_continue": true
  }'

# Response:
# {
#   "session_id": "abc123...",
#   "state": "active",
#   "is_paused": false,
#   "message_count": 1,
#   "auto_continue": true
# }

# 2. The chat will start generating responses automatically
# You can see messages in real-time:

curl "http://localhost:8006/api/v1/chat/sessions/{session_id}/messages"

# 3. Pause the chat
curl -X POST "http://localhost:8006/api/v1/chat/sessions/{session_id}/pause" \
  -H "Content-Type: application/json" \
  -d '{"action": "pause", "reason": "User paused"}'

# 4. Resume the chat
curl -X POST "http://localhost:8006/api/v1/chat/sessions/{session_id}/resume"

# 5. Stop the chat completely
curl -X POST "http://localhost:8006/api/v1/chat/sessions/{session_id}/stop"
```

### Python Example

```python
import asyncio
from bulk_chat.core.chat_engine import ContinuousChatEngine
from bulk_chat.config.chat_config import ChatConfig

async def main():
    # Create configuration
    config = ChatConfig()
    config.llm_provider = "openai"
    config.llm_model = "gpt-4"
    config.auto_continue = True
    config.response_interval = 2.0
    
    # Create chat engine
    engine = ContinuousChatEngine(
        llm_provider=config.get_llm_provider(),
        auto_continue=config.auto_continue,
        response_interval=config.response_interval,
    )
    
    # Create session
    session = await engine.create_session(
        user_id="user123",
        initial_message="Hello, explain machine learning to me",
        auto_continue=True,
    )
    
    # Start continuous chat
    await engine.start_continuous_chat(session.session_id)
    
    # The chat will now generate responses automatically
    # Wait a bit to see responses
    await asyncio.sleep(10)
    
    # View messages
    print(f"Generated messages: {len(session.messages)}")
    for msg in session.messages:
        print(f"{msg.role}: {msg.content[:100]}...")
    
    # Pause chat
    await engine.pause_session(session.session_id, "Paused by user")
    
    # Resume
    await engine.resume_session(session.session_id)
    
    # Stop
    await engine.stop_session(session.session_id)

if __name__ == "__main__":
    asyncio.run(main())
```

## 📡 API Endpoints

### Sessions
- `POST /api/v1/chat/sessions` - Create new session
- `GET /api/v1/chat/sessions` - List all sessions
- `GET /api/v1/chat/sessions/{session_id}` - Get session info
- `DELETE /api/v1/chat/sessions/{session_id}` - Delete session

### Messages
- `POST /api/v1/chat/sessions/{session_id}/messages` - Send message
- `GET /api/v1/chat/sessions/{session_id}/messages` - Get messages

### Chat Control
- `POST /api/v1/chat/sessions/{session_id}/start` - Start continuous chat
- `POST /api/v1/chat/sessions/{session_id}/pause` - Pause chat
- `POST /api/v1/chat/sessions/{session_id}/resume` - Resume chat
- `POST /api/v1/chat/sessions/{session_id}/stop` - Stop chat

### Streaming
- `GET /api/v1/chat/sessions/{session_id}/stream` - Response stream (SSE)
- `WS /ws/chat/{session_id}` - WebSocket for bi-directional streaming

### Metrics
- `GET /api/v1/chat/sessions/{session_id}/metrics` - Session metrics
- `GET /api/v1/chat/metrics` - Global system metrics
- `GET /api/v1/chat/rate-limit/{identifier}` - Rate limiting statistics

### Advanced Features
- **Cache**: `GET /api/v1/chat/cache/stats`, `POST /api/v1/chat/cache/clear`
- **Analysis**: `GET /api/v1/chat/sessions/{session_id}/analyze`, `GET /api/v1/chat/sessions/{session_id}/summary`
- **Export**: `GET /api/v1/chat/sessions/{session_id}/export/{format}`
- **Templates**: `GET /api/v1/chat/templates`
- **Webhooks**: `POST /api/v1/chat/webhooks`
- **Authentication**: `POST /api/v1/auth/register`, `POST /api/v1/auth/login`
- **Backups**: `POST /api/v1/chat/backup/create`

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Open Pull Request

## 📄 License

Proprietary - Blatam Academy

---

[← Back to Main README](../README.md)
