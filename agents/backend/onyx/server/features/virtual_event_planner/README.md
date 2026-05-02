# Virtual Event Planner

<div align="center">

![Status](https://img.shields.io/badge/status-active-success.svg)
![Version](https://img.shields.io/badge/version-1.8-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Events](https://img.shields.io/badge/Event-Management-orange.svg)

**A comprehensive AI-driven platform for orchestrating virtual conferences, webinars, and interactive online experiences.**

[Overview](#-overview) •
[Features](#-key-features) •
[Architecture](#-architecture) •
[Installation](#-installation) •
[Usage](#-usage) •
[API Reference](#-api-reference) •
[Contributing](#-contributing)

</div>

---

## 📋 Overview

**Virtual Event Planner** automates the complex logistics of hosting online events. From dynamic agenda management to attendee engagement tracking, it provides a centralized command center for event organizers.

It integrates seamlessly with major streaming platforms (Zoom, Webex, YouTube) and uses AI to optimize scheduling, matchmake attendees for networking, and generate post-event analytics.

### Why Virtual Event Planner?

- **Smart Scheduling**: AI algorithms avoid conflict and optimize time slots for global audiences.
- **Engagement Analytics**: Real-time tracking of attendee participation and sentiment.
- **Automated Logistics**: Automatic handling of registrations, reminders, and follow-ups.

## 🚀 Key Features

| Feature | Description |
|---------|-------------|
| **Dynamic Agenda** | Drag-and-drop interface for building multi-track schedules with timezone support. |
| **Attendee Management** | robust CRM for handling registrations, ticket tiers, and access control. |
| **Networking AI** | Intelligent matchmaking algorithms to suggest connections between attendees. |
| **Hybrid Support** | Tools for bridging physical and virtual event components. |
| **Sponsor Portals** | Dedicated spaces for sponsors to showcase products and track leads. |
| **Real-Time Analytics** | Dashboards for monitoring attendance, retention, and interaction rates. |

## 🏗 Architecture

The system is built on a scalable microservices architecture to handle high-concurrency events.

```mermaid
graph TD
    A[Attendee Portal] --> B(API Gateway)
    B --> C{Event Core}
    
    subgraph "Management Layer"
    C --> D[Agenda Service]
    C --> E[User Service]
    C --> F[Analytics Engine]
    end
    
    subgraph "Integration Layer"
    D --> G[Zoom/Webex API]
    E --> H[CRM (HubSpot/Salesforce)]
    F --> I[(Data Warehouse)]
    end
    
    subgraph "Real-Time"
    J[WebSocket Server] <--> A
    J <--> C
    end
```

## 💻 Installation

### Prerequisites

- Python 3.10+
- PostgreSQL
- Redis (for real-time features)

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/blatam-academy/virtual_event_planner.git
   cd virtual_event_planner
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Initialize Database**
   ```bash
   python manage.py db init
   python manage.py db upgrade
   ```

## ⚡ Usage

### Python SDK

```python
from virtual_event_planner import EventManager, Session

# Initialize manager
manager = EventManager()

# Create Event
event = manager.create_event(
    name="Global AI Summit 2026",
    date="2026-06-15",
    timezone="UTC"
)

# Add Session
keynote = Session(
    title="The Future of AGIs",
    speaker="Jane Doe",
    duration=60,
    type="keynote"
)
event.add_session(keynote)

# Publish
manager.publish_event(event.id)
print(f"Event published at: {event.url}")
```

### API Endpoints

**POST /api/v1/events**
```json
{
  "name": "TechConf 2026",
  "start_date": "2026-09-01T09:00:00Z",
  "end_date": "2026-09-03T18:00:00Z",
  "tracks": ["Development", "Design", "Product"]
}
```

**GET /api/v1/analytics/engagement**
```json
{
  "total_attendees": 1500,
  "avg_session_duration": 45,
  "top_session": "Keynote: AI Ethics"
}
```

## 🔧 Configuration

Configure the platform via `.env`:

```ini
DATABASE_URL=postgresql://user:pass@localhost/events_db
REDIS_URL=redis://localhost:6379/0
ZOOM_API_KEY=your_zoom_key
SMTP_SERVER=smtp.sendgrid.net
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <b>Built with ❤️ by Blatam Academy</b><br>
  Part of the Onyx Server Architecture<br>
  <a href="../README.md">← Back to Main README</a>
</div>
