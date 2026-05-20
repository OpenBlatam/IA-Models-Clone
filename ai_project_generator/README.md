# AI Project Generator 🚀

> Part of the [Blatam Academy Integrated Platform](../README.md)

Automatic AI project generator that creates complete backend and frontend structures based on user descriptions. Works continuously to process project requests.

## ✨ Key Features

- ✅ **Smart Automatic Generation**
  - Complete backend (FastAPI) with modular structure
  - Complete frontend (React + TypeScript + Vite + Tailwind)
  - Automatic detection of AI type (chat, vision, audio, NLP, etc.)
  - Detection of necessary features (auth, database, websocket, etc.)

- ✅ **Continuous Generation**
  - Processes projects automatically without stopping
  - Priority queue system
  - State persistence
  - Real-time monitoring

- ✅ **Complete REST API**
  - Endpoints to generate, list, and monitor projects
  - Statistics and metrics
  - Queue management

- ✅ **Smart Code**
  - Generates code according to the detected AI type
  - Includes WebSocket if necessary
  - File upload support
  - Cache, queue, database configuration as needed
  - Automatic dependencies based on features

- ✅ **Production Ready**
  - Docker and docker-compose included
  - Automatic tests
  - Generated documentation
  - Professional structure

## 📦 Installation

```bash
cd ai_project_generator
pip install -r requirements.txt
```

## 🚀 CI/CD Pipeline

This project includes a complete CI/CD pipeline with GitHub Actions:

- **Continuous Integration**: Automated testing, linting, and security scanning
- **Continuous Deployment**: Automated deployment to AWS EC2
- **Release Management**: Automated release creation
- **Security Scanning**: Comprehensive security checks

See [CI_CD_README.md](CI_CD_README.md) for complete CI/CD documentation.

### Quick CI/CD Setup

1. **Configure GitHub Secrets**:
   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`

2. **Enable Workflows**: Workflows are automatically enabled

3. **Install Pre-commit Hooks**:
   ```bash
   pip install pre-commit
   pre-commit install
   ```

4. **Test Locally**:
   ```bash
   ./scripts/ci_cd/test.sh all
   ./scripts/ci_cd/build.sh docker
   ```

## 🏃 Usage

### Start the Server

```bash
python main.py
```

Server will be available at `http://localhost:8020`

### Generate a Project

```bash
curl -X POST "http://localhost:8020/api/v1/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "description": "An AI chat system answering programming questions",
    "project_name": "programming_chat_ai",
    "author": "Blatam Academy"
  }'
```

### View Generator Status

```bash
curl "http://localhost:8020/api/v1/status"
```

### View Queue

```bash
curl "http://localhost:8020/api/v1/queue"
```

## 📚 API Endpoints

### Generation
- `POST /api/v1/generate` — Generate a new project
  - Body: `{description, project_name?, author?, version?, priority?, backend_framework?, frontend_framework?, generate_tests?, include_docker?, include_docs?}`

### Status and Monitoring
- `GET /api/v1/status` — Continuous generator status
- `GET /api/v1/project/{project_id}` — Specific project status
- `GET /api/v1/queue` — Pending projects queue
- `GET /api/v1/stats` — Generator statistics
- `GET /api/v1/projects` — List generated projects (with filters)

### Control
- `POST /api/v1/start` — Start continuous generator
- `POST /api/v1/stop` — Stop continuous generator
- `DELETE /api/v1/project/{project_id}` — Remove project from queue

### Export and Validation
- `POST /api/v1/export/zip` — Export project to ZIP
- `POST /api/v1/export/tar` — Export project to TAR
- `POST /api/v1/validate` — Validate a generated project

### Deployment
- `POST /api/v1/deploy/generate` — Generate deployment configurations (Vercel, Netlify, Railway, Heroku)

### Cloning and Templates
- `POST /api/v1/clone` — Clone an existing project
- `POST /api/v1/templates/save` — Save a custom template
- `GET /api/v1/templates/list` — List all templates
- `GET /api/v1/templates/{name}` — Get a specific template
- `DELETE /api/v1/templates/{name}` — Delete a template

### Advanced Search
- `GET /api/v1/search` — Search projects with advanced filters
- `GET /api/v1/search/stats` — Search statistics

### Webhooks
- `POST /api/v1/webhooks/register` — Register a webhook
- `GET /api/v1/webhooks` — List registered webhooks
- `DELETE /api/v1/webhooks/{id}` — Unregister a webhook

### Cache
- `POST /api/v1/cache/clear` — Clear cache
- `GET /api/v1/cache/stats` — Cache statistics

### Rate Limiting
- `GET /api/v1/rate-limit` — Rate limit information

### Authentication
- `POST /api/v1/auth/register` — Register user
- `POST /api/v1/auth/login` — Authenticate user
- `POST /api/v1/auth/api-key` — Create API key

### Metrics
- `GET /api/v1/metrics` — System metrics
- `GET /api/v1/metrics/prometheus` — Metrics in Prometheus format

### Backup and Restore
- `POST /api/v1/backup/create` — Create full backup
- `GET /api/v1/backup/list` — List backups
- `POST /api/v1/backup/restore` — Restore backup
- `DELETE /api/v1/backup/{name}` — Delete backup

### Dashboard and UI
- `GET /dashboard` — Interactive web dashboard
- `POST /api/v1/dashboard/generate` — Generate dashboard

### Health and Versions
- `GET /health` — Basic health check
- `GET /health/detailed` — Detailed health check
- `GET /api/version` — API version information

### Notifications
- `POST /api/v1/notifications/register` — Register notification channel
- `GET /api/v1/notifications/channels` — List channels

### Plugins
- `POST /api/v1/plugins/register` — Register plugin
- `GET /api/v1/plugins` — List plugins
- `POST /api/v1/plugins/{name}/enable` — Enable plugin
- `POST /api/v1/plugins/{name}/disable` — Disable plugin

### Events
- `GET /api/v1/events/history` — Event history
- `GET /api/v1/events/stats` — Event statistics

### Logs
- `GET /api/v1/logs/stats` — Log statistics

### WebSocket
- `WS /ws` — WebSocket for real-time updates
- `WS /ws/project/{project_id}` — WebSocket to subscribe to a specific project

### Batch Generation
- `POST /api/v1/generate/batch` — Generate multiple projects in batch

### Performance
- `GET /api/v1/performance/stats` — Performance statistics
- `GET /api/v1/performance/optimize` — Optimization suggestions

### Real-time Streaming
- `GET /api/v1/stream/events` — Streaming events
- `GET /api/v1/stream/stats` — Streaming statistics

### Analytics
- `GET /api/v1/analytics/trends` — Analytics trends
- `GET /api/v1/analytics/top-ai-types` — Most popular AI types
- `GET /api/v1/analytics/performance` — Performance report
- `GET /api/v1/analytics/frameworks` — Framework usage
- `GET /api/v1/analytics/authors` — Stats by author
- `GET /api/v1/analytics/report` — Full report

### Recommendations
- `GET /api/v1/recommendations` — Smart recommendations
- `GET /api/v1/recommendations/features` — Recommended features
- `GET /api/v1/recommendations/framework` — Recommended framework

### Documentation
- `POST /api/v1/documentation/generate` — Automatic documentation generation

### Alerts
- `POST /api/v1/alerts/rule` — Create alert rule
- `POST /api/v1/alerts/trigger` — Trigger alert
- `GET /api/v1/alerts` — Get active alerts
- `POST /api/v1/alerts/{id}/acknowledge` — Acknowledge alert
- `GET /api/v1/alerts/stats` — Alert statistics

### Scheduling
- `POST /api/v1/scheduler/task` — Schedule task
- `GET /api/v1/scheduler/tasks` — List tasks
- `GET /api/v1/scheduler/task/{id}` — Task status
- `POST /api/v1/scheduler/task/{id}/enable` — Enable task
- `POST /api/v1/scheduler/task/{id}/disable` — Disable task

### Advanced Import/Export
- `POST /api/v1/export/advanced` — Advanced project export
- `POST /api/v1/import` — Import project

### Machine Learning
- `POST /api/v1/ml/predict/time` — Predict generation time
- `POST /api/v1/ml/predict/success` — Predict success probability
- `POST /api/v1/ml/train` — Train ML model
- `GET /api/v1/ml/stats` — Model statistics

### Automatic Optimization
- `POST /api/v1/optimize/analyze` — Analyze project
- `POST /api/v1/optimize/config` — Optimize configuration
- `POST /api/v1/optimize/recommendations` — Optimization recommendations

### Advanced Testing
- `POST /api/v1/testing/run` — Run project tests

### Automatic Deployment
- `POST /api/v1/deploy/vercel` — Deploy to Vercel
- `POST /api/v1/deploy/netlify` — Deploy to Netlify
- `POST /api/v1/deploy/railway` — Deploy to Railway
- `GET /api/v1/deploy/history` — Deployment history

### Performance Analysis
- `GET /api/v1/performance/analyze` — Analyze performance
- `GET /api/v1/performance/predict/{operation}` — Predict time
- `GET /api/v1/resources/stats` — Resource statistics

### Advanced Reports
- `POST /api/v1/reports/generate/project` — Generate project report
- `POST /api/v1/reports/generate/system` — Generate system report
- `GET /api/v1/reports` — List reports

### Real-time Monitoring
- `POST /api/v1/monitoring/start` — Start monitoring
- `POST /api/v1/monitoring/stop` — Stop monitoring
- `GET /api/v1/monitoring/metrics` — Current metrics
- `GET /api/v1/monitoring/history` — Metrics history
- `GET /api/v1/monitoring/alerts` — Recent alerts

### Automation
- `POST /api/v1/automation/create` — Create automation
- `GET /api/v1/automation/list` — List automations
- `GET /api/v1/automation/history` — Execution history

### Advanced Security
- `POST /api/v1/security/api-key/generate` — Generate API key
- `POST /api/v1/security/api-key/validate` — Validate API key
- `GET /api/v1/security/stats` — Security statistics

### Code Quality Analysis
- `POST /api/v1/quality/analyze/file` — Analyze file quality
- `POST /api/v1/quality/analyze/project` — Analyze project quality

### Smart Suggestions
- `POST /api/v1/suggestions/generate` — Generate suggestions
- `POST /api/v1/suggestions/feedback` — Submit feedback
- `GET /api/v1/suggestions/stats` — Suggestion statistics

### Benchmarking
- `POST /api/v1/benchmark/record` — Record benchmark
- `POST /api/v1/benchmark/compare` — Compare projects
- `GET /api/v1/benchmark/leaderboard` — Benchmark leaderboard

### Advanced Metrics
- `POST /api/v1/metrics/record` — Record metric
- `GET /api/v1/metrics/stats/{metric_name}` — Metric statistics
- `POST /api/v1/metrics/custom/create` — Create custom metric
- `GET /api/v1/metrics/summary` — Metrics summary

### System
- `GET /api/v1/system/info` — System information

## 🎯 Usage Example

```python
import requests

# Generate project
response = requests.post(
    "http://localhost:8020/api/v1/generate",
    json={
        "description": "An image analyzer with AI that detects objects",
        "project_name": "image_analyzer",
        "author": "Your Name"
    }
)

project_id = response.json()["project_id"]
print(f"Project in queue: {project_id}")

# View status
status = requests.get(f"http://localhost:8020/api/v1/project/{project_id}")
print(status.json())
```

## 📁 Generated Structure

Each generated project includes:

```
project/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── core/
│   │   ├── models/
│   │   ├── services/
│   │   └── utils/
│   ├── main.py
│   ├── requirements.txt
│   ├── Dockerfile
│   └── README.md
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── services/
│   │   └── utils/
│   ├── package.json
│   ├── vite.config.ts
│   ├── Dockerfile
│   └── README.md
├── docker-compose.yml
├── README.md
└── project_info.json
```

## 🔄 Continuous Generation

The system works continuously:
1. Receives project descriptions (via API)
2. Analyses and extracts features automatically
3. Adds them to a priority queue
4. Processes them automatically one by one
5. Generates full backend and frontend with smart code
6. Ready to use immediately

## 🧠 Smart Detection

The system automatically detects:

### AI Types
- **Chat**: Conversational, assistant, chatbot
- **Vision**: Images, detection, recognition, OCR
- **Audio**: Music, voice, transcription, speech
- **NLP**: Text, translation, sentiment analysis
- **Video**: Streaming, video processing
- **Recommendation**: Recommendation systems
- **Analytics**: Data analysis, reporting
- **Generation**: Content generation
- **Classification**: Classification, categorization
- **QA**: Questions and Answers

### Features
- **Auth**: User authentication
- **Database**: Database (PostgreSQL, MySQL, MongoDB)
- **WebSocket**: Real-time communication
- **File Upload**: File uploads
- **Cache**: Redis, Memcached
- **Queue**: Background tasks
- **Streaming**: Real-time processing

### Model Providers
- OpenAI (GPT)
- Anthropic (Claude)
- Google (Gemini)
- HuggingFace (Transformers)
- Local Models (Llama, Mistral)

## 📝 Notes

- Projects are generated in `generated_projects/` directory
- Queue is saved in `project_queue.json`
- System automatically detects AI type and generates appropriate code
- Projects are automatically validated after generation
- Metadata is automatically exported for each project

---

[← Back to Main README](../README.md)
