# Dermatology AI — Skin Analysis & Skincare System

> Part of the [Blatam Academy Integrated Platform](../README.md)

## 🚀 Description

Advanced AI system for skin quality analysis via photos or video with sensors. The system determines skin quality and provides personalized skincare recommendations based on the analysis.

## ✨ Key Features

### Skin Analysis
- **Quality Metrics** — Comprehensive analysis of texture, hydration, elasticity, pigmentation, pores, wrinkles, redness, and dark spots
- **Condition Detection** — Identification of acne, rosacea, eczema, hyperpigmentation, dryness, and sensitivity
- **Image Analysis** — Processing and analysis of skin photographs
- **Video Analysis** — Aggregated analysis of video sequences for greater accuracy

### Skincare Recommendations
- **Personalized Routines** — Morning, evening, and weekly treatment routines
- **Product Recommendations** — Specific recommendations based on skin type and conditions
- **Personalized Tips** — Advice based on individual analysis
- **Prioritization** — Identification of priority areas for improvement

## 📁 Project Structure (Reorganized)

```
dermatology_ai/
├── ml/                     # 🧠 Machine Learning Core
│   ├── models/            # Model architectures
│   ├── training/          # Training components
│   ├── data/              # Data processing
│   ├── experiments/       # Experiment management
│   ├── inference/         # Inference engines
│   └── visualization/     # Demos & visualization
│
├── core/                   # 💼 Business Logic
│   ├── application/       # Use cases (hexagonal)
│   ├── domain/            # Domain entities
│   ├── infrastructure/    # Infrastructure adapters
│   └── ...                # Core components
│
├── api/                    # 🌐 API Layer
│   ├── controllers/       # Request handlers
│   ├── routers/           # API routes
│   └── middleware/        # API middleware
│
├── services/               # 🔧 Business Services
├── utils/                  # 🛠️ Utilities & Optimizations
├── config/                 # ⚙️ Configuration
├── examples/               # 📚 Examples
├── docs/                   # 📖 Documentation
├── tests/                  # 🧪 Tests
├── scripts/                # 🔨 Utility scripts
├── main.py                 # Main server
└── requirements.txt        # Dependencies
```

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for the full structure and [ORGANIZATION_GUIDE.md](ORGANIZATION_GUIDE.md) for organization guide.

## 🔧 Installation

### Prerequisites

- Python 3.8+
- pip

### Quick Install

```bash
cd dermatology_ai
pip install -r requirements.txt
```

## 🚀 Usage

### Start the Server

```bash
python main.py
```

The server will be available at `http://localhost:8006`

### API Documentation

Once the server is running:
- **Swagger UI**: `http://localhost:8006/docs`
- **ReDoc**: `http://localhost:8006/redoc`

## 📖 Main Endpoints

### 1. Analyze Image

```bash
POST /dermatology/analyze-image
Content-Type: multipart/form-data

file: [image]
enhance: true
```

**Response:**
```json
{
  "success": true,
  "analysis": {
    "quality_scores": {
      "overall_score": 75.5,
      "texture_score": 80.0,
      "hydration_score": 70.0
    },
    "conditions": [
      {
        "name": "acne",
        "confidence": 0.65,
        "severity": "moderate"
      }
    ],
    "skin_type": "combination",
    "recommendations_priority": ["hydration", "texture"]
  }
}
```

### 2. Analyze Video

```bash
POST /dermatology/analyze-video
Content-Type: multipart/form-data

file: [video]
max_frames: 30
```

### 3. Get Recommendations

```bash
POST /dermatology/get-recommendations
Content-Type: multipart/form-data

file: [image]
include_routine: true
```

**Response:**
```json
{
  "success": true,
  "analysis": {},
  "recommendations": {
    "routine": {
      "morning": [
        {
          "name": "Gentle Cleanser",
          "category": "cleanser",
          "description": "...",
          "key_ingredients": ["Glycerin", "..."],
          "usage_frequency": "Twice daily",
          "priority": 1
        }
      ],
      "evening": [],
      "weekly": []
    },
    "specific_recommendations": [],
    "tips": []
  }
}
```

## 💻 Programmatic Usage

### Image Analysis (Legacy)

```python
from dermatology_ai import SkinAnalyzer, ImageProcessor
from PIL import Image
import numpy as np

# Initialize
analyzer = SkinAnalyzer()
processor = ImageProcessor()

# Load image
image = Image.open("skin_photo.jpg")
img_array = np.array(image)

# Analyze
result = analyzer.analyze_image(img_array)
print(f"Overall score: {result['quality_scores']['overall_score']}")
```

### Using the ML Module (Recommended)

```python
from ml import ViTSkinAnalyzer, Trainer, SkinDataset, get_train_transforms
from ml.inference import FastInferenceEngine
from ml.experiments import ExperimentTracker

# Create model
model = ViTSkinAnalyzer(num_conditions=6, num_metrics=8)

# Optimized inference
engine = FastInferenceEngine(model, use_compile=True)
output = engine.predict(input_tensor)
```

### Get Recommendations

```python
from dermatology_ai import SkincareRecommender

recommender = SkincareRecommender()
recommendations = recommender.generate_recommendations(analysis_result)

print("Morning routine:")
for product in recommendations["routine"]["morning"]:
    print(f"- {product['name']}")
```

See [examples/](examples/) for complete examples.

## 🔬 Quality Metrics

The system analyzes the following metrics (0–100, where 100 is best):

| Metric | Description |
|--------|-------------|
| **Overall Score** | Overall skin quality score |
| **Texture Score** | Smoothness and uniformity of texture |
| **Hydration Score** | Hydration level |
| **Elasticity Score** | Elasticity and firmness |
| **Pigmentation Score** | Pigmentation uniformity |
| **Pore Size Score** | Pore size (100 = very small pores) |
| **Wrinkles Score** | Wrinkle presence (100 = no wrinkles) |
| **Redness Score** | Redness (100 = no redness) |
| **Dark Spots Score** | Dark spots (100 = no spots) |

## 🎯 Detected Conditions

| Condition | Description |
|-----------|-------------|
| **Acne** | Pimples and bumps |
| **Rosacea** | Chronic redness |
| **Eczema** | Dermatitis |
| **Hyperpigmentation** | Dark spots |
| **Dryness** | Dehydrated skin |
| **Sensitivity** | Sensitive or irritated skin |

## 🧴 Recommended Product Types

Cleanser · Moisturizer · Serum · Sunscreen · Toner · Exfoliant · Mask · Eye Cream

## 🔧 Configuration

### Environment Variables

Create a `.env` file:

```env
# Server configuration
HOST=0.0.0.0
PORT=8006

# Processing configuration
IMAGE_TARGET_SIZE=512,512
VIDEO_MAX_FRAMES=30
VIDEO_TARGET_FPS=1
```

## 🧪 Testing

```bash
# Run tests
pytest tests/

# With coverage
pytest --cov=dermatology_ai tests/
```

## 🆕 Improvements in v1.1.0

- ✅ **Advanced Analysis** — Enhanced computer vision techniques
- ✅ **Logging System** — Complete logging with files and metrics
- ✅ **Cache System** — In-memory and disk caching for better performance
- ✅ **Error Handling** — Custom exceptions and improved error management
- ✅ **Detailed Metrics** — Deeper analysis with additional metrics

See [IMPROVEMENTS.md](IMPROVEMENTS.md) for full details.

## 📊 Future Improvements

- [ ] Advanced ML integration (CNNs, Vision Transformers)
- [ ] Temporal progress analysis (before/after comparison)
- [ ] Real product database
- [ ] Specialized sensor integration
- [ ] Multi-body area analysis
- [ ] Multi-language support
- [ ] Interactive web dashboard
- [ ] History and progress tracking

## 🤝 Contributing

Contributions are welcome:

1. Fork the project
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is part of Blatam Academy.

## 📞 Support

For support, contact the Blatam Academy team.

---

[← Back to Main README](../README.md)
