# Manuales Hogar AI 🏠🔧

> Part of the [Blatam Academy Integrated Platform](../README.md)

AI system to generate LEGO-style step-by-step manuals for popular trades. Allows processing internal images (photos of problems) or text descriptions and generates visual and detailed guides on how to solve problems in the home, work, and trades.

**✨ Refactored for microservices, serverless, and cloud-native** - See [REFACTORING.md](REFACTORING.md) and [ARCHITECTURE.md](ARCHITECTURE.md)

**🛡️ Improved for maximum stability** - See [STABILITY.md](STABILITY.md)

**☁️ Ready for any EC2** - See [EC2_DEPLOYMENT.md](EC2_DEPLOYMENT.md)

## ⚡ Quick Start

```bash
# Single command to start everything
./start.sh        # Linux/Mac
.\start.ps1       # Windows
python run.py     # Any platform
```

See [QUICKSTART.md](QUICKSTART.md) for more details.

## 🛠️ Utility Scripts

The project includes **25+ utility scripts** for management and maintenance:

### Basic Management
```bash
./status.sh              # Service status
./scripts/setup.sh       # Initial setup
./scripts/clean.sh       # Cleanup
```

### Monitoring and Diagnostics
```bash
./scripts/monitor.sh      # Real-time monitoring
./scripts/health-monitor.sh # Continuous monitoring with alerts
./scripts/diagnostics.sh  # Complete diagnostics
./scripts/watch.sh        # Auto-restart on changes
```

### Testing and Performance
```bash
./scripts/test-api.sh     # API tests
./scripts/quick-test.sh   # Quick validation
./scripts/performance-test.sh # Performance tests
```

### Backup and Security
```bash
./scripts/backup.sh       # Database backup
./scripts/restore.sh      # Restore from backup
./scripts/security-check.sh # Security validation
```

### Optimization and Maintenance
```bash
./scripts/optimize.sh     # Optimize Docker
./scripts/update.sh       # Update application
./scripts/export-logs.sh  # Export logs
```

See [scripts/README.md](scripts/README.md) for the full list and [FEATURES_COMPLETE.md](FEATURES_COMPLETE.md) for all features.

## 🎯 Features

- ✅ **Support for multiple AI models** via OpenRouter
- ✅ **Image processing** with vision models
- ✅ **Support for multiple images** (up to 5 images per request)
- ✅ **Text processing** for problem descriptions
- ✅ **Automatic category detection** with intelligent algorithm
- ✅ **Cache system** (memory + persistent in DB)
- ✅ **Automatic image validation and optimization**
- ✅ **Database persistence** with Alembic
- ✅ **Manual history** with advanced search
- ✅ **Automatic usage statistics**
- ✅ **LEGO-style step-by-step manuals** with visual format
- ✅ **Multiple trade categories**:
  - Plumbing
  - Roofing and repairs
  - Carpentry
  - Electricity
  - Masonry
  - Painting
  - Blacksmithing
  - Gardening
  - General

## 📋 Requirements

- Python 3.8+ (or Docker)
- PostgreSQL 12+ (for database)
- OpenRouter API Key (configure in `OPENROUTER_API_KEY` environment variable)

## 🚀 Installation and Usage

### Quick Start (Single Command)

**Linux/Mac:**
```bash
./start.sh
```

**Windows (PowerShell):**
```powershell
.\start.ps1
```

**Python (any platform):**
```bash
python run.py
```

**Production:**
```bash
./start.sh prod
# or
python run.py prod
```

The script automatically:
- ✅ Checks if Docker is running
- ✅ Creates `.env` file if it doesn't exist
- ✅ Starts all services (app, PostgreSQL, Redis)
- ✅ Waits for the service to be ready
- ✅ Shows the API URL

### Option 2: Manual Docker Compose

**Development:**
```bash
docker-compose up -d
```

**Production:**
```bash
docker-compose -f docker-compose.prod.yml up -d
```

See [DOCKER.md](DOCKER.md) for more details.

### Option 3: Local Installation

```bash
pip install -r requirements.txt
```

## ⚙️ Configuration

### Environment Variables

Configure environment variables or create a `.env` file (see `.env.example`):

```bash
# OpenRouter API Key (required)
OPENROUTER_API_KEY=your-api-key-here

# Database (optional, for persistence)
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/manuales_hogar
# Or individually:
DB_HOST=localhost
DB_PORT=5432
DB_USER=postgres
DB_PASSWORD=your_password
DB_NAME=manuales_hogar
```

### Initialize Database

If you want to use data persistence, initialize the database:

```bash
# Create initial migration
alembic revision --autogenerate -m "Initial migration"

# Apply migrations
alembic upgrade head
```

See [MIGRATIONS.md](MIGRATIONS.md) for more details on migrations.

## 📖 Usage

### API Endpoints

#### 1. Health Check
```bash
GET /api/v1/health
```

#### 2. List Available Models
```bash
GET /api/v1/models
```

#### 3. Generate Manual from Text
```bash
POST /api/v1/generate-from-text
Content-Type: application/json

{
  "problem_description": "I have a water leak in the kitchen faucet",
  "category": "plumbing",
  "include_safety": true,
  "include_tools": true,
  "include_materials": true
}
```

#### 4. Generate Manual from Image
```bash
POST /api/v1/generate-from-image
Content-Type: multipart/form-data

file: [image.jpg]
problem_description: "Additional description (optional)"
category: "plumbing" (optional, automatically detected)
```

#### 5. Generate Combined Manual (Text + Image)
```bash
POST /api/v1/generate-combined
Content-Type: multipart/form-data

problem_description: "I have a problem with..."
file: [image.jpg] (optional)
category: "general"
```

#### 6. Generate Manual from Multiple Images
```bash
POST /api/v1/generate-from-multiple-images
Content-Type: multipart/form-data

files: [image1.jpg, image2.jpg, image3.jpg] (max 5)
problem_description: "Additional description" (optional)
category: "plumbing" (optional, automatically detected)
```

#### 7. Get Supported Categories
```bash
GET /api/v1/categories
```

#### 8. Cache Statistics
```bash
GET /api/v1/cache/stats
```

#### 9. Clear Cache (Memory)
```bash
DELETE /api/v1/cache/clear
```

#### 10. List Manuals (History)
```bash
GET /api/v1/manuals?category=plumbing&limit=20&offset=0
```

#### 11. Get Manual by ID
```bash
GET /api/v1/manuals/{manual_id}
```

#### 12. Recent Manuals
```bash
GET /api/v1/manuals/recent?limit=10
```

#### 13. Usage Statistics
```bash
GET /api/v1/statistics?days=30
```

#### 14. Persistent Cache Statistics
```bash
GET /api/v1/cache/stats-db
```

#### 15. Clear Persistent Cache
```bash
DELETE /api/v1/cache/clear-db
```

#### 16. Clear Expired Cache
```bash
POST /api/v1/cache/cleanup-expired
```

### Python Usage Example

```python
from manuales_hogar_ai import ManualGenerator
from manuales_hogar_ai.infrastructure import OpenRouterClient

# Create client and generator
client = OpenRouterClient()
generator = ManualGenerator(openrouter_client=client)

# Generate manual from text
result = await generator.generate_manual_from_text(
    problem_description="Water leak in faucet",
    category="plumbing"
)

print(result["manual"])

# Generate manual from image
result = await generator.generate_manual_from_image(
    image_path="problem.jpg",
    problem_description="Visible leak at connection",
    category="plumbing"
)

print(result["manual"])
```

## 🏗️ Project Structure

```
manuales_hogar_ai/
├── __init__.py
├── alembic/                    # Database migrations
│   ├── versions/               # Migration files
│   ├── env.py                  # Alembic configuration
│   └── script.py.mako         # Migration template
├── alembic.ini                 # Alembic configuration
├── api/
│   ├── __init__.py
│   └── routes/
│       ├── __init__.py
│       └── manuales.py
├── config/
│   ├── __init__.py
│   └── settings.py
├── core/
│   ├── __init__.py
│   └── manual_generator.py
├── database/                   # DB models and sessions
│   ├── __init__.py
│   ├── models.py               # SQLAlchemy models
│   └── session.py              # Session management
├── infrastructure/
│   ├── __init__.py
│   └── openrouter_client.py
├── scripts/                    # Utility scripts
│   ├── __init__.py
│   └── init_db.py              # Initialize DB
├── services/
│   └── __init__.py
├── utils/
│   ├── __init__.py
│   ├── cache_manager.py
│   ├── category_detector.py
│   └── image_validator.py
├── requirements.txt
├── README.md
└── MIGRATIONS.md               # Migrations guide
```

## 🎨 Manual Format

The generated manuals follow a LEGO-style format with:

1. **Manual Title**
2. **Diagnosis** of the problem
3. **Safety Warnings** ⚠️
4. **Required Tools** 🔧
5. **Required Materials** 📦
6. **Repair Steps** (LEGO format):
   - Step number
   - Clear description
   - Verbal illustration
   - Estimated time
   - Difficulty (Easy/Medium/Hard)
   - Precautions
7. **Verification**
8. **Preventive Maintenance**
9. **When to Call a Professional**

## 🤖 Supported Models

The system can use any model available on OpenRouter, including:

- `anthropic/claude-3.5-sonnet` (default)
- `openai/gpt-4o`
- `openai/gpt-4-turbo`
- `google/gemini-pro-1.5`
- `meta-llama/llama-3.1-70b-instruct`
- `anthropic/claude-3-opus`

For models with vision, it is recommended to use:
- `anthropic/claude-3.5-sonnet`
- `openai/gpt-4o`
- `google/gemini-pro-1.5`

## 📝 Supported Categories

- `plumbing` - Plumbing
- `roofing` - Roofing Repair
- `carpentry` - Carpentry
- `electricity` - Electricity
- `masonry` - Masonry
- `painting` - Painting
- `blacksmithing` - Blacksmithing
- `gardening` - Gardening
- `general` - General Repair

## 🔒 Security

- Images are temporarily processed and deleted after processing
- Maximum image size: 10MB
- File types are validated before processing
- Safety warnings included in generated manuals
- Input validation on all endpoints
- Secure error handling without exposing sensitive information

## 📊 Database and Persistence

The system includes full persistence with:

- **Manual History**: All generated manuals are saved automatically
- **Persistent Cache**: Database cache with automatic expiration
- **Statistics**: Automatic metrics for usage, tokens, models, etc.
- **Advanced Search**: Search by category, term, date, etc.

See [MIGRATIONS.md](MIGRATIONS.md) for database configuration and [FEATURES.md](FEATURES.md) for feature details.

## 🐛 Troubleshooting

### Error: "OpenRouter API key not configured"
- Make sure to configure the `OPENROUTER_API_KEY` environment variable

### Error: "The image is too large"
- The maximum size is 10MB. Reduce the image size before uploading.

### Error: "Category not supported"
- Verify that the category is in the list of supported categories using `GET /api/v1/categories`

## 📄 License

Proprietary - Blatam Academy

## 👥 Author

Blatam Academy

## 🔄 Version

1.0.0

---

[← Back to Main README](../README.md)
