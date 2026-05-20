# Physical Store Designer AI

> Part of the [Blatam Academy Integrated Platform](../README.md)

Complete AI system for designing physical stores, including visual design, marketing plan, and decoration strategy.

## 🏪 Features

- **Interactive Chat**: AI conversation to gather client information
- **Visual Design**: Generation of store visualizations (exterior, interior, layout)
- **Marketing Plan**: Complete marketing and sales strategies
- **Decoration Plan**: Detailed decoration recommendations, furniture, and materials
- **Multiple Styles**: Support for different design styles (modern, classic, minimalist, industrial, etc.)
- **REST API**: Complete API with FastAPI for integration

## 📋 Requirements

- Python 3.10+
- OpenAI API key (optional, for advanced generation with LLM)

## 🚀 Installation

### Full Installation (Recommended)
```bash
pip install -r requirements.txt
```

### Minimal Installation (Core Only)
```bash
pip install -r requirements-minimal.txt
```

### Installation for Development
```bash
pip install -r requirements-dev.txt
```

For more details on dependencies, see [DEPENDENCIES.md](DEPENDENCIES.md).

2. **Configure environment variables** (optional):
```bash
# Create .env file
OPENAI_API_KEY=your_api_key
```

## 🎯 Usage

### Start API Server

```bash
python main.py
# Or using uvicorn directly
uvicorn physical_store_designer_ai.api.main:app --host 0.0.0.0 --port 8030
```

### Use Interactive Chat

```python
import requests

# Create chat session
response = requests.post("http://localhost:8030/api/v1/chat/session")
session = response.json()
session_id = session["session_id"]

# Send message
message = {
    "role": "user",
    "content": "I want to open a modern coffee shop downtown"
}
response = requests.post(
    f"http://localhost:8030/api/v1/chat/{session_id}/message",
    json=message
)
print(response.json())
```

### Generate Complete Design

```python
from physical_store_designer_ai.core.models import (
    StoreDesignRequest,
    StoreType,
    DesignStyle
)

# Create request
request = StoreDesignRequest(
    store_name="Modern Coffee",
    store_type=StoreType.CAFE,
    style_preference=DesignStyle.MODERN,
    budget_range="medium",
    location="City center",
    target_audience="Young professionals and students",
    dimensions={"width": 8.0, "length": 12.0, "height": 3.0}
)

# Send request to API
import requests
response = requests.post(
    "http://localhost:8030/api/v1/design/generate",
    json=request.dict()
)
design = response.json()
```

### Generate Design from Chat

```python
# After chatting, generate design
response = requests.post(
    f"http://localhost:8030/api/v1/design/from-chat/{session_id}"
)
design = response.json()
```

## 📊 Design Structure

The generated design includes:

### 1. Store Layout
- Dimensions
- Store zones
- Furniture placement
- Traffic flow
- Accessibility

### 2. Visualizations
- Exterior view
- Interior view
- Floor plan

### 3. Marketing Plan
- Target audience
- Marketing strategies
- Sales tactics
- Pricing strategy
- Promotion ideas
- Social media plan
- Opening strategy

### 4. Decoration Plan
- Color scheme
- Lighting plan
- Furniture recommendations
- Decorative elements
- Recommended materials
- Budget estimation

## 🎨 Available Styles

- **Modern**: Clean lines, minimalist
- **Classic**: Elegant, traditional
- **Minimalist**: Open spaces, few elements
- **Industrial**: Rustic materials, metallic finishes
- **Rustic**: Natural materials, cozy atmosphere
- **Luxury**: Premium materials, refined finishes
- **Eco-friendly**: Sustainable materials, plants
- **Vintage**: Retro elements, nostalgia

## 🏪 Store Types

- Restaurant
- Cafe
- Boutique
- Retail
- Supermarket
- Pharmacy
- Electronics
- Clothing
- Furniture
- Others

## 📡 API Endpoints

### Chat
- `POST /api/v1/chat/session` - Create chat session
- `POST /api/v1/chat/{session_id}/message` - Send message
- `GET /api/v1/chat/{session_id}` - Get session

### Designs
- `POST /api/v1/design/generate` - Generate design
- `GET /api/v1/design/{store_id}` - Get design
- `GET /api/v1/designs` - List designs
- `POST /api/v1/design/from-chat/{session_id}` - Generate from chat
- `GET /api/v1/design/{store_id}/export?format=json|markdown|html` - Export design
- `DELETE /api/v1/design/{store_id}` - Delete design

### Analysis
- `GET /api/v1/analysis/competitor/{store_id}` - Competitor analysis
- `GET /api/v1/analysis/financial/{store_id}` - Financial analysis
- `GET /api/v1/analysis/inventory/{store_id}` - Inventory recommendations
- `GET /api/v1/analysis/kpis/{store_id}` - KPIs and metrics
- `GET /api/v1/analysis/full/{store_id}` - Full analysis

### Advanced Features
- `POST /api/v1/compare/designs` - Compare multiple designs
- `GET /api/v1/technical-plans/{store_id}` - Detailed technical plans
- `POST /api/v1/feedback/{store_id}` - Add feedback
- `GET /api/v1/feedback/{store_id}` - Get feedback and suggestions
- `GET /api/v1/recommendations/{store_id}` - Smart recommendations
- `GET /api/v1/location/analyze` - Analyze location
- `POST /api/v1/versions/{store_id}` - Create design version
- `GET /api/v1/versions/{store_id}` - Version history
- `GET /api/v1/versions/{store_id}/compare` - Compare versions
- `POST /api/v1/versions/{store_id}/approve` - Approve version

### Premium Features
- `GET /api/v1/reports/{store_id}` - Full report
- `GET /api/v1/reports/{store_id}/pdf` - PDF report
- `GET /api/v1/reports/{store_id}/excel` - Excel report
- `POST /api/v1/share/{store_id}` - Share design
- `GET /api/v1/share/{share_id}` - Get shared design
- `POST /api/v1/share/{share_id}/revoke` - Revoke sharing
- `POST /api/v1/comments/{store_id}` - Add comment
- `GET /api/v1/comments/{store_id}` - Get comments
- `GET /api/v1/dashboard` - Full dashboard
- `GET /api/v1/templates` - List templates
- `GET /api/v1/templates/{template_id}` - Get template
- `POST /api/v1/templates/{template_id}/apply` - Apply template
- `GET /api/v1/trends` - Trend analysis
- `GET /api/v1/notifications/{user_id}` - Get notifications
- `POST /api/v1/notifications/{user_id}/read/{notification_id}` - Mark as read

### Enterprise Features
- `POST /api/v1/auth/register` - User registration
- `POST /api/v1/auth/login` - Login
- `GET /api/v1/auth/me` - Current user
- `POST /api/v1/optimize/budget/{store_id}` - Optimize budget
- `GET /api/v1/optimize/layout/{store_id}` - Optimize layout
- `POST /api/v1/optimize/marketing/{store_id}` - Optimize marketing
- `GET /api/v1/predict/success/{store_id}` - Predict success
- `GET /api/v1/predict/revenue/{store_id}` - Predict revenue
- `GET /api/v1/predict/traffic/{store_id}` - Predict traffic
- `GET /api/v1/monitoring/health/{store_id}` - Design health
- `GET /api/v1/monitoring/alerts/{store_id}` - Get alerts
- `POST /api/v1/monitoring/alerts/{store_id}/acknowledge/{alert_id}` - Acknowledge alert
- `GET /api/v1/monitoring/metrics/{store_id}` - Get metrics

## 🔧 Configuration

The service can be configured via environment variables (with `PSD_` prefix):

### Main Variables

- `PSD_OPENAI_API_KEY`: OpenAI API key (optional)
- `PSD_HOST`: Server host (default: 0.0.0.0)
- `PSD_PORT`: Server port (default: 8030)

### Security & Performance

- `PSD_CORS_ORIGINS`: Allowed CORS origins (default: "*", comma-separated)
- `PSD_CORS_ALLOW_CREDENTIALS`: Allow credentials in CORS (default: true)
- `PSD_RATE_LIMIT_PER_MINUTE`: Request limit per minute (default: 60)
- `PSD_API_KEY_HEADER`: Optional API key header

### Logging

- `PSD_LOG_LEVEL`: Logging level - DEBUG, INFO, WARNING, ERROR, CRITICAL (default: INFO)
- `PSD_LOG_FORMAT`: Log format - "json" or "text" (default: json)

### Storage

- `PSD_STORAGE_PATH`: Base path for storage (default: storage)
- `PSD_DESIGNS_PATH`: Path to store designs (default: storage/designs)

### Performance

- `PSD_MAX_WORKERS`: Max number of workers (default: 4)
- `PSD_REQUEST_TIMEOUT`: Request timeout in seconds (default: 300)

### Feature Flags

- `PSD_ENABLE_ML_FEATURES`: Enable advanced ML features (default: false)
- `PSD_ENABLE_DEEP_LEARNING`: Enable deep learning (default: false)

### Example .env file

```bash
PSD_OPENAI_API_KEY=your_api_key_here
PSD_HOST=0.0.0.0
PSD_PORT=8030
PSD_LOG_LEVEL=INFO
PSD_LOG_FORMAT=json
PSD_RATE_LIMIT_PER_MINUTE=60
PSD_CORS_ORIGINS=http://localhost:3000,https://app.example.com
```

---

[← Back to Main README](../README.md)
