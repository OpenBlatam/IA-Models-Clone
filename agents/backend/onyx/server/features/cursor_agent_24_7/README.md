# Cursor Agent 24/7 🤖

> Part of the [Blatam Academy Integrated Platform](../README.md)

Persistent agent that listens for commands from the Cursor window and executes them continuously, even when the computer is off (as a service).

## 🚀 Features

### Core Features
- ✅ **Listens to Cursor commands**: Automatically activates when you type something in Cursor
- ✅ **Continuous execution**: Executes tasks non-stop, 24/7
- ✅ **Simple control**: Easy start/stop button
- ✅ **Persistent service**: Can run in background as system service
- ✅ **REST API**: Full control via API

### Advanced Features
- ✅ **Serverless Ready**: Support for AWS Lambda, Azure Functions, GCP Functions
- ✅ **API Gateway**: Integration with Kong and AWS API Gateway
- ✅ **High Performance**: JSON parsing 3-10x faster, advanced compression
- ✅ **Circuit Breakers**: Automatic resilience with circuit breakers
- ✅ **Rate Limiting**: Distributed rate limiting with Redis
- ✅ **Service Discovery**: Consul, etcd, Kubernetes DNS
- ✅ **Elasticsearch**: Integrated advanced search
- ✅ **Webhooks**: Complete webhooks system
- ✅ **Bulk Operations**: Optimized batch operations
- ✅ **Web interface**: Simple and modern control panel
- ✅ **Persistent state**: Saves state even after restart
- ✅ **WebSocket**: Real-time communication
- ✅ **Notifications**: Advanced notification system
- ✅ **Metrics**: Metrics collection and analysis
- ✅ **Health Checks**: System health monitoring
- ✅ **Rate Limiting**: Request rate control
- ✅ **Data export**: Export tasks and metrics
- ✅ **Task scheduling**: Execute tasks at specific times
- ✅ **Backups**: Automatic backup system
- ✅ **Plugins**: Extensible plugin system
- ✅ **Authentication**: Authentication and authorization system
- ✅ **Cache**: Command result cache
- ✅ **Templates**: Reusable command templates
- ✅ **Validation**: Command validation and sanitization
- ✅ **Event Bus**: Pub/sub event system
- ✅ **Clustering**: Basic support for multiple instances
- ✅ **Advanced Logging**: Professional logging system
- ✅ **Middleware**: Middleware for security and logging
- ✅ **Maintenance scripts**: Automated maintenance tools
- ✅ **Real-time monitor**: Visual agent status monitor
- ✅ **AI Processing**: Intelligent command processing with LLMs
- ✅ **Semantic search**: Command search using embeddings
- ✅ **Pattern learning**: Learns from successful commands to improve
- ✅ **Code generation**: Automatically generates code from descriptions
- ✅ **Auto summary**: Summarizes long results automatically
- ✅ **LLM Pipeline**: Professional pipeline with PyTorch and Transformers
- ✅ **Fine-tuning**: Support for model fine-tuning
- ✅ **Gradio Interface**: Interactive web interface with Gradio
- ✅ **Code Completion**: Auto-completes code
- ✅ **Explain Code**: Explains code in natural language
- ✅ **Fix Code**: Fixes code with errors
- ✅ **GPU Support**: Automatic GPU detection and usage
- ✅ **Mixed Precision**: Support for training with float16
- ✅ **Modular Architecture**: Clear separation of models, data, training and evaluation
- ✅ **YAML Configurations**: Hyperparameter management in YAML files
- ✅ **Experiment Tracking**: Ready for WandB/TensorBoard
- ✅ **Callbacks System**: Callbacks system for training
- ✅ **Checkpointing**: Automatic model saving
- ✅ **🤖 Devin Personality**: Intelligent personality system with reasoning
- ✅ **🔒 Enhanced Security**: Secret detection and protection
- ✅ **📚 Code Understanding**: LSP-like system to understand codebase
- ✅ **💬 Intelligent Communication**: Strategic communication with user
- ✅ **⚠️ Issue Reporting**: Automatic environment issue detection and reporting
- ✅ **🔧 Devin Command System**: Structured Devin-style commands
- ✅ **📋 Planning System**: Work plan management with steps and dependencies
- ✅ **🛠️ Tool Manager**: Automatic system tool detection
- ✅ **📐 Convention Analyzer**: Detects and follows project code conventions
- ✅ **✅ Change Verifier**: Verifies all changes are complete before reporting
- ✅ **📚 Library Verification**: Verifies libraries in requirements.txt before use
- ✅ **🧪 Test Runner**: Automatically runs tests before reporting changes
- ✅ **🔗 Reference Tracker**: Tracks and verifies references to modified code
- ✅ **⚡ Parallel Executor**: Executes commands in parallel when no dependencies
- ✅ **🔍 Context Analyzer**: Analyzes code context before making changes
- ✅ **✅ Completeness Verifier**: Verifies all requirements are met
- ✅ **🔄 Iteration Manager**: Manages iterations until changes are correct
- ✅ **🔍 Critical Verifier**: Critical verification before reporting to user
- ✅ **🧠 Reasoning Trigger System**: Activates automatic reasoning on critical decisions
- ✅ **🎯 Intent Verifier**: Verifies user intent fulfillment
- ✅ **🔄 Automatic Integration**: Automatic integration into task flow
- ✅ **🛡️ Test Protector**: Prevents test modification unless explicitly requested
- ✅ **🔗 CI Integration**: Uses CI for testing when environment issues exist
- ✅ **🔀 Git Manager**: Git management following specific Devin rules
- ✅ **📍 Multiple Location Verifier**: Verifies all locations were edited
- ✅ **🌐 Browser Integration**: Inspects web pages without assuming content
- ✅ **📋 Planning Verifier**: Verifies all information is available before suggesting plan

## 📦 Installation

### 🐳 Docker (Recommended - Easiest)

```bash
# Clone or navigate to directory
cd agents/backend/onyx/server/features/cursor_agent_24_7

# Start with Docker Compose
docker-compose up

# Or use Makefile
make quick-start
```

See [DOCKER.md](DOCKER.md) for complete Docker documentation.

### Local Installation (No Docker)

```bash
cd agents/backend/onyx/server/features/cursor_agent_24_7
pip install -r requirements.txt
```

### Minimal Installation (Essentials Only)

```bash
pip install -r requirements-minimal.txt
```

### Development Installation

```bash
pip install -r requirements-dev.txt
```

### Fast Installation with UV (Faster than pip)

```bash
# Install UV
pip install uv

# Install dependencies
uv pip install -r requirements.txt
```

### Installation Notes

- **Python 3.10+** required (3.12+ recommended)
- Some libraries require compilation (orjson, etc.)
- For best performance: `pip install --no-cache-dir -r requirements.txt`
- On Windows, some libraries may require Visual C++ Build Tools

## 🎯 Quick Usage

### 🐳 Docker (Recommended)

```bash
# Quick start with Docker Compose
docker-compose up

# Or with Makefile
make quick-start

# Full stack (API + Workers + Redis)
make dev

# Production
make prod
```

See [DOCKER.md](DOCKER.md) for more Docker options.

### ⚡ Simple Command (No Docker)

```bash
# Start API on port 8024 (default)
python run.py

# Or with options
python run.py --port 8080
python run.py --aws
python run.py --mode service
```

Then open your browser at: `http://localhost:8024`

### API Mode (Alternative)

```bash
python main.py --mode api --port 8024
```

### Service Mode

```bash
python run.py --mode service
# Or
python main.py --mode service
```

> 💡 **Tip**: Use Docker for the best experience. See [DOCKER.md](DOCKER.md) for more information.

## 🖥️ Web Interface

When starting in API mode, you will have access to a simple web interface with:

- **Start Button**: Starts the agent
- **Pause Button**: Temporarily pauses the agent
- **Stop Button**: Completely stops the agent
- **Command Field**: Type commands and press Enter to add them
- **Task List**: View all executed tasks

## 📡 API Endpoints

### Agent Control

- `POST /api/start` - Start the agent
- `POST /api/stop` - Stop the agent
- `POST /api/pause` - Pause the agent
- `POST /api/resume` - Resume the agent
- `GET /api/status` - Get agent status

### Tasks

- `POST /api/tasks` - Add a new task
  ```json
  {
    "command": "your command here"
  }
  ```
- `GET /api/tasks?limit=50` - Get task list

## 🔧 Configuration

You can configure the agent by editing `AgentConfig`:

```python
from cursor_agent_24_7.core.agent import AgentConfig, CursorAgent

config = AgentConfig(
    check_interval=1.0,  # Seconds between checks
    max_concurrent_tasks=5,  # Maximum concurrent tasks
    task_timeout=300.0,  # Timeout per task (seconds)
    auto_restart=True,  # Automatically restart on error
    persistent_storage=True,  # Save state to disk
    storage_path="./data/agent_state.json"  # State file path
)

agent = CursorAgent(config)
```

## 🛠️ Run as System Service

### Windows

Use **NSSM** (Non-Sucking Service Manager):

```bash
# Install NSSM
# Download from: https://nssm.cc/download

# Create service
nssm install CursorAgent24_7 "C:\Python\python.exe" "C:\path\to\main.py" --mode service

# Start service
nssm start CursorAgent24_7
```

Or use Windows **Task Scheduler**.

### Linux (systemd)

Create file `/etc/systemd/system/cursor-agent-24-7.service`:

```ini
[Unit]
Description=Cursor Agent 24/7
After=network.target

[Service]
Type=simple
User=your_user
WorkingDirectory=/path/to/cursor_agent_24_7
ExecStart=/usr/bin/python3 /path/to/cursor_agent_24_7/main.py --mode service
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Then:

```bash
sudo systemctl daemon-reload
sudo systemctl enable cursor-agent-24-7
sudo systemctl start cursor-agent-24-7
```

### macOS (launchd)

Create file `~/Library/LaunchAgents/com.cursor.agent24-7.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.cursor.agent24-7</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/bin/python3</string>
        <string>/path/to/cursor_agent_24_7/main.py</string>
        <string>--mode</string>
        <string>service</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
</dict>
</plist>
```

Then:

```bash
launchctl load ~/Library/LaunchAgents/com.cursor.agent24-7.plist
```

## 🔌 Cursor API Integration

**TODO**: Currently the agent is prepared to integrate with the Cursor API, but the actual integration needs to be implemented.

To integrate:

1. Obtain access to the Cursor API
2. Implement `CommandListener._listen_loop()` to receive real commands
3. Implement `TaskExecutor._run_command()` to execute real commands

## 📊 Persistent State

The agent saves its state to `./data/agent_state.json` by default. This includes:

- Current agent status
- All tasks (pending, running, completed, failed)
- Results and errors

Upon restart, the agent automatically loads the saved state.

## 🛠️ Utility Scripts

### Real-Time Monitor

Monitor the agent state in real-time:

```bash
python scripts/monitor.py
```

Or with custom interval:

```bash
python scripts/monitor.py --interval 1.0
```

### Maintenance

Execute maintenance tasks:

```bash
# Cleanup old tasks
python scripts/maintenance.py cleanup --days 30

# Check health
python scripts/maintenance.py health

# Generate report
python scripts/maintenance.py report

# Run everything
python scripts/maintenance.py all
```

### Install as Service

Install the agent as a system service:

```bash
python scripts/install_service.py
```

### Gradio Interface

Launch interactive Gradio web interface:

```bash
python scripts/launch_gradio.py
```

With public link:

```bash
python scripts/launch_gradio.py --share
```

The interface will be available at `http://localhost:7860`

## 🤖 Devin Improvements

The agent now includes a complete Devin personality system making it smarter, safer, and more professional:

- **Devin Personality System**: Devin-like behavior with internal reasoning
- **Strategic Communication**: Communicates with user when necessary
- **Environment Issue Reporting**: Automatically detects and reports issues
- **Operation Modes**: Planning mode and standard mode
- **Code Understanding**: LSP-like system to understand the codebase
- **Enhanced Security**: Secret detection and protection

See **[DEVIN_IMPROVEMENTS.md](DEVIN_IMPROVEMENTS.md)** for more details.

## 📚 Additional Documentation

### 🚀 Deployment and Orchestration
- **[ORCHESTRATION.md](ORCHESTRATION.md)**: Complete orchestration guide (Docker, K8s, ECS)
- **[ORCHESTRATION_QUICK_START.md](ORCHESTRATION_QUICK_START.md)**: Orchestration quick start
- **[DOCKER.md](DOCKER.md)**: Complete Docker Compose
- **[AWS_DEPLOYMENT.md](aws/AWS_DEPLOYMENT.md)**: AWS Deployment
- **[DEPLOYMENT.md](DEPLOYMENT.md)**: General deployment guide

### 🏗️ Architecture and Features
- **[ARCHITECTURE.md](ARCHITECTURE.md)**: Complete system architecture
- **[ADVANCED_FEATURES.md](ADVANCED_FEATURES.md)**: Advanced features
- **[MORE_IMPROVEMENTS.md](MORE_IMPROVEMENTS.md)**: Additional improvements
- **[FINAL_IMPROVEMENTS.md](FINAL_IMPROVEMENTS.md)**: Final improvements (Service Discovery, Elasticsearch)
- **[API_IMPROVEMENTS.md](API_IMPROVEMENTS.md)**: API improvements (Webhooks, Bulk, Cache)
- **[LIBRARIES_IMPROVEMENTS.md](LIBRARIES_IMPROVEMENTS.md)**: Library improvements (Serverless, Performance)
- **[MODULAR_ARCHITECTURE.md](MODULAR_ARCHITECTURE.md)**: Modular architecture

### 🤖 Capabilities
- **[DEVIN_IMPROVEMENTS.md](DEVIN_IMPROVEMENTS.md)**: Devin-based improvements
- **[FEATURES.md](FEATURES.md)**: Full list of features
- **[AI_FEATURES.md](AI_FEATURES.md)**: AI and Machine Learning features
- **[ADVANCED_AI.md](ADVANCED_AI.md)**: Advanced Deep Learning features

### 📖 References
- **[EXAMPLES.md](EXAMPLES.md)**: Usage examples
- **[API_REFERENCE.md](API_REFERENCE.md)**: Complete API reference
- **[COMMANDS.md](COMMANDS.md)**: Available commands
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)**: Troubleshooting
- **[LIBRARIES.md](LIBRARIES.md)**: Documentation of used libraries
- **[QUICK_START.md](QUICK_START.md)**: Quick start guide

## 🐛 Troubleshooting

For detailed troubleshooting, consult [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

### Common Issues

**Agent doesn't start:**
- Verify port 8024 is available
- Check logs: `tail -f logs/agent.log`
- Ensure all dependencies are installed

**Tasks not executing:**
- Verify agent is in "running" state: `curl http://localhost:8024/api/status`
- Check logs for execution errors
- Verify health: `curl http://localhost:8024/api/health`

**Service persistence failing:**
- Verify write permissions in data directory
- Check system service configuration
- Ensure `persistent_storage=True` in configuration

## 📝 License

Part of the Blatam Academy project.

## 🤝 Contributing

Contributions are welcome. Please:

1. Fork the project
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📞 Support

For support, open an issue in the repository or contact the Blatam Academy team.

---

[← Back to Main README](../README.md)
