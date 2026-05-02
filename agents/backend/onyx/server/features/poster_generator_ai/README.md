# Poster Generator AI

> Part of the [Blatam Academy Integrated Platform](../README.md)

## 📋 Description

AI-based poster generation system for marketing and events.

## 🚀 Features

- **AI-Powered Design**: Generates poster designs automatically
- **Customizable Templates**: Uses templates adapted to your needs
- **Text and Image Integration**: Integrates text and images seamlessly
- **Export Formats**: Supports multiple export formats (PDF, PNG, JPG)

## 📁 Structure

```
poster_generator_ai/
├── generator.py           # Generation logic
├── templates/             # Design templates
├── assets/                # Graphic resources
└── api.py                 # API Endpoints
```

## 🔧 Installation

```bash
pip install -r requirements.txt
```

## 💻 Usage

```python
from poster_generator_ai.generator import PosterGenerator

# Initialize generator
generator = PosterGenerator()

# Generate poster
poster = generator.create(
    title="Event Name",
    subtitle="Description",
    date="2023-12-01",
    image_path="image.jpg",
    template="modern"
)
```

---

[← Back to Main README](../README.md)
