# GitHub Autonomous Agent AI

> Part of the [Blatam Academy Integrated Platform](../README.md)

Autonomous agent that connects to any GitHub repository and executes instructions continuously, even with the computer off, until the user stops it.

## 🚀 Features

- ✅ **Connect to any GitHub repository** - Supports any public or private repository
- ✅ **Receive instructions from frontend** - Web interface to send commands
- ✅ **Continuous task execution** - Agent works non-stop until stopped
- ✅ **Service/Daemon operation** - Can run in background
- ✅ **Start/Stop control** - Full control from frontend or API
- ✅ **Task queue system** - Efficient management of multiple tasks
- ✅ **Data persistence** - Tasks are saved in SQLite database

## 📋 Requirements

- Python 3.8+
- GitHub Token (optional, for private repositories)
- FastAPI and Uvicorn

## 🔧 Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment variables (automatic)
python setup_env.py

# Or manually, copy env.example to .env
# cp env.example .env
```

### GitHub OAuth Configuration

GitHub OAuth credentials are pre-configured:
- **Client ID**: `Ov23liSy9XyIj3dD0dQc`
- **Client Secret**: `6ed948f00e7662bbba0eacd356e60747dd12f08f`

**⚠️ IMPORTANT**: Ensure that in your GitHub OAuth App the **Authorization callback URL** is exactly:
```
http://localhost:8025/api/github/auth/callback
```

See `SETUP_GITHUB_OAUTH.md` for more details.

## 🎯 Usage

### Service Mode (Daemon)

Run the agent as a persistent service:

```bash
python main.py --mode service
```

### API Mode

Run the API server:

```bash
python main.py --mode api --port 8025
```

## 📡 API Endpoints

### Agent

- `GET /api/agent/status` - Get agent status
- `POST /api/agent/start` - Start the agent
- `POST /api/agent/stop` - Stop the agent
- `POST /api/agent/pause` - Pause the agent
- `POST /api/agent/resume` - Resume the agent

### Tasks

- `POST /api/tasks/` - Create a new task
- `GET /api/tasks/` - List tasks
- `GET /api/tasks/{task_id}` - Get a specific task

## 📝 Usage Example

### Create a task

```bash
curl -X POST "http://localhost:8025/api/tasks/" \
  -H "Content-Type: application/json" \
  -d '{
    "repository": "owner/repo",
    "instruction": "Create a README.md file with project information",
    "metadata": {
      "file_path": "README.md",
      "file_content": "# My Project\n\nProject description"
    }
  }'
```

### Start the agent

```bash
curl -X POST "http://localhost:8025/api/agent/start"
```

### Check status

```bash
curl "http://localhost:8025/api/agent/status"
```

## 🏗️ Architecture

```
github_autonomous_agent_ai/
├── api/                    # FastAPI API
│   ├── routes/            # API Routes
│   └── models/            # Pydantic Models
├── core/                   # Core Logic
│   ├── agent.py           # Main Agent
│   ├── service.py         # Persistent Service
│   ├── github_client.py   # GitHub Client
│   ├── task_queue.py      # Task Queue
│   └── task_executor.py   # Task Executor
├── config/                 # Configuration
│   ├── settings.py        # App Settings
├── frontend/               # Frontend (coming soon)
├── main.py                 # Entry Point
└── requirements.txt        # Dependencies
```

## 🔐 Configuration

Environment variables:

- `GITHUB_TOKEN` - GitHub Token (optional)
- `API_PORT` - API Server Port (default: 8025)
- `AGENT_POLL_INTERVAL` - Polling interval in seconds (default: 5)
- `AGENT_MAX_CONCURRENT_TASKS` - Maximum concurrent tasks (default: 3)
- `STORAGE_PATH` - Storage path (default: ./data)

## 🚧 Upcoming Improvements

- [ ] Complete frontend with React/Next.js
- [ ] Support for more instruction types
- [ ] Integration with AI models to process natural instructions
- [ ] Notification system
- [ ] Monitoring dashboard
- [ ] Support for multiple simultaneous repositories

## 📄 License

MIT

---

[← Back to Main README](../README.md)
