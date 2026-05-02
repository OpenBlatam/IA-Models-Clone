# Key Messages Management

> Part of the [Blatam Academy Integrated Platform](../README.md)

## 📋 Description

Complete system for key messages management with ML capabilities, cybersecurity, and functional architecture.

## 🚀 Key Features

- **Key Messages Management**: Complete system for key messages
- **Machine Learning**: ML integration for analysis
- **Cybersecurity**: Advanced security system
- **Functional Architecture**: Functional and modular design
- **Reporting**: Integrated reporting system
- **Routers**: Routing system
- **Types**: Well-defined types

## 📁 Structure

```
key_messages/
├── ml/                    # Machine Learning
├── routers/               # API Routers
├── types/                 # Types and schemas
├── attackers/             # Security system
├── reporting/             # Reporting system
├── utils/                 # Utilities
└── docs/                  # Documentation
```

## 🔧 Installation

```bash
# Minimal installation
pip install -r requirements-minimal.txt

# For development
pip install -r requirements-dev.txt

# For production
pip install -r requirements-prod.txt

# With cybersecurity
pip install -r requirements-cyber.txt
```

## 💻 Basic Usage

```python
from key_messages.service import KeyMessagesService
from key_messages.config import Config

# Initialize service
service = KeyMessagesService(Config())

# Create key message
message = service.create_key_message(
    content="Important message",
    category="marketing"
)
```

## 📚 Documentation

- [Project Definition](PROJECT_DEFINITION.md)
- [Migration Summary](MIGRATION_SUMMARY.md)
- [Dependencies](dependencies.md)

## 🔗 Integration

This module integrates with:
- **Integration System**: For orchestration
- **Business Agents**: For automation
- **Export IA**: For message export
- **Security Systems**: For security

---

[← Back to Main README](../README.md)
