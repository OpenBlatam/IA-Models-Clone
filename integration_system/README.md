# Integration System

> Part of the [Blatam Academy Integrated Platform](../README.md)

## 📋 Description

Main integration system and API Gateway for Blatam Academy. Orchestrates all services and acts as a single entry point.

## 🚀 Key Features

- **API Gateway** — Gateway for all services
- **Orchestration** — Orchestrates all system services
- **Routing** — Intelligent request routing
- **Authentication** — Centralized authentication
- **Monitoring** — Complete system monitoring

## 📁 Structure

```
integration_system/
├── api/                    # API Endpoints
├── config/                 # Configurations
├── core/                   # Core logic
└── main.py                 # Entry point
```

## 🔧 Installation

This module is installed with the main system using `start_system.py`.

## 💻 Usage

The system starts automatically with:

```bash
python start_system.py start
```

## 📊 Port

- Default Port: **8000**
- Health endpoint: `http://localhost:8000/health`
- API Docs: `http://localhost:8000/docs`

## 🔗 Integration

This system orchestrates:
- All Blatam Academy ecosystem services
- **[Content Redundancy Detector](../content_redundancy_detector/README.md)** (Port 8001)
- **[BUL](../bulk/README.md)** (Port 8002)
- **Gamma App** (Port 8003)
- **[Business Agents](../business_agents/README.md)** (Port 8004)
- **[Export IA](../export_ia/README.md)** (Port 8005)

---

[← Back to Main README](../README.md)
