# Artist Manager AI

> Part of the [Blatam Academy Integrated Platform](../README.md)

AI system for comprehensive artist management that helps fulfill routines, calendars, protocols, and provides wardrobe recommendations.

## 🚀 Features

- 📅 **Calendar Management**: Complete organization of events and commitments
- 🔄 **Routine Management**: Tracking of daily and weekly routines
- 📋 **Behavior Protocols**: Verification of protocol compliance
- 👔 **Wardrobe Management**: Intelligent outfit recommendations based on events
- 🤖 **AI with OpenRouter**: Intelligent generation of recommendations and summaries
- 💾 **Data Persistence**: SQLite database for storing information
- 🔔 **Notification System**: Automatic reminders for events and routines
- 📊 **Analytics and Metrics**: Tracking of statistics and performance
- ⚡ **Smart Cache**: Performance optimization with cache system
- ✅ **Robust Validation**: Data validation and improved error handling
- 🧠 **Machine Learning**: Intelligent predictions of duration and completion
- 🔍 **Advanced Search**: Fuzzy search with relevance scoring
- 🚨 **Alert System**: Automatic detection of conflicts and issues
- 🔄 **Automatic Synchronization**: Integration with Google Calendar and Outlook
- 📡 **Webhooks**: Real-time external notifications
- 📦 **Backup and Restore**: Complete backup system
- 📝 **Templates**: Quick creation from predefined templates
- 📊 **Reports**: Generation of activity and compliance reports

## 🛠️ Installation

```bash
pip install -r requirements.txt
```

## ⚙️ Configuration

Create `.env` file:

```env
OPENROUTER_API_KEY=your_api_key_here
OPENROUTER_MODEL=anthropic/claude-3-haiku
LOG_LEVEL=INFO
```

## 🏗️ Project Structure

```
artist_manager_ai/
├── core/                    # Core modules
│   ├── artist_manager.py    # Main manager
│   ├── calendar_manager.py  # Calendar management
│   ├── routine_manager.py   # Routine management
│   ├── protocol_manager.py  # Protocol management
│   └── wardrobe_manager.py  # Wardrobe management
├── infrastructure/          # Infrastructure
│   └── openrouter_client.py # OpenRouter client
├── api/                     # REST API
│   └── routes/              # API routes
├── mcp_server/              # MCP Server
├── config/                  # Configuration
└── requirements.txt         # Dependencies
```

## 📖 Usage

### REST API

```python
from fastapi import FastAPI
from artist_manager_ai.api.routes import router

app = FastAPI()
app.include_router(router)
```

### Direct Usage

```python
from artist_manager_ai import ArtistManager
from artist_manager_ai.core.calendar_manager import CalendarEvent, EventType
from datetime import datetime, timedelta

async with ArtistManager(
    artist_id="artist_123",
    openrouter_api_key="...",
    enable_persistence=True,  # Enable DB persistence
    enable_notifications=True,  # Enable notifications
    enable_analytics=True  # Enable analytics
) as manager:
    # Create event with automatic reminders
    event = CalendarEvent(
        id="event_001",
        title="Concert",
        description="Main concert",
        event_type=EventType.CONCERT,
        start_time=datetime.now() + timedelta(days=3),
        end_time=datetime.now() + timedelta(days=3, hours=3)
    )
    manager.create_event_with_reminders(event, reminder_minutes=[60, 30, 15])
    
    # Get dashboard
    dashboard = manager.get_dashboard_data()
    
    # Generate daily summary (cached)
    summary = await manager.generate_daily_summary()
    
    # Get wardrobe recommendation (AI enhanced)
    recommendation = await manager.generate_wardrobe_recommendation(event_id="event_001")
    
    # Check protocol compliance (AI enhanced)
    compliance = await manager.check_protocol_compliance(event_id="event_001")
    
    # Get statistics
    stats = manager.get_statistics(days=30)
    
    # Data is automatically saved to DB on close
```

## 📡 API Endpoints

### Dashboard
- `GET /artist-manager/dashboard/{artist_id}` - Get dashboard
- `GET /artist-manager/dashboard/{artist_id}/daily-summary` - Daily summary with AI

### Calendar
- `POST /artist-manager/calendar/{artist_id}/events` - Create event
- `GET /artist-manager/calendar/{artist_id}/events` - List events
- `GET /artist-manager/calendar/{artist_id}/events/{event_id}` - Get event
- `PUT /artist-manager/calendar/{artist_id}/events/{event_id}` - Update event
- `DELETE /artist-manager/calendar/{artist_id}/events/{event_id}` - Delete event
- `GET /artist-manager/calendar/{artist_id}/events/{event_id}/wardrobe-recommendation` - Wardrobe recommendation

### Routines
- `POST /artist-manager/routines/{artist_id}/tasks` - Create routine
- `GET /artist-manager/routines/{artist_id}/tasks` - List routines
- `POST /artist-manager/routines/{artist_id}/tasks/{task_id}/complete` - Complete routine
- `GET /artist-manager/routines/{artist_id}/pending` - Pending routines

### Protocols
- `POST /artist-manager/protocols/{artist_id}` - Create protocol
- `GET /artist-manager/protocols/{artist_id}` - List protocols
- `POST /artist-manager/protocols/{artist_id}/events/{event_id}/check-compliance` - Check compliance

### Wardrobe
- `POST /artist-manager/wardrobe/{artist_id}/items` - Add item
- `GET /artist-manager/wardrobe/{artist_id}/items` - List items
- `POST /artist-manager/wardrobe/{artist_id}/outfits` - Create outfit
- `GET /artist-manager/wardrobe/{artist_id}/outfits` - List outfits

## 🤖 AI Features

### Daily Summary
Generates an intelligent summary of the day with:
- Summary of scheduled events
- Important routine reminders
- General recommendations
- Positive motivation
- **Automatic cache** to optimize performance

### Wardrobe Recommendations
Analyzes the event and generates recommendations based on:
- Event type
- Applicable protocols
- Items available in the wardrobe
- Weather considerations
- **Enhanced prompts** for better results
- **Robust parsing** of AI responses

### Protocol Verification
Automatically verifies compliance with protocols using AI:
- Analysis of applicable protocols
- Detection of violations
- Improvement recommendations
- **Detailed audit** per protocol

## ✨ New Features

### Notification System
- Automatic event reminders
- Pending routine alerts
- Protocol notifications
- Priority system (low, normal, high, urgent)

### Data Persistence
- Integrated SQLite database
- Automatic saving of events and routines
- Automatic loading on initialization
- Optimized indices for performance

### Analytics and Metrics
- Tracking of custom metrics
- Statistics by artist
- Averages and sums of metrics
- Trend analysis

### Smart Cache
- Automatic cache of daily summaries
- Configurable TTL
- Automatic cleaning of expired entries
- Cache usage statistics

### Improved Validation
- Artist ID validation
- Time range validation
- Priority validation
- URL and email validation

## 📊 Data Models

### CalendarEvent
- `id`: Unique ID
- `title`: Event title
- `description`: Description
- `event_type`: Type (concert, interview, photoshoot, etc.)
- `start_time`: Start time
- `end_time`: End time
- `location`: Location
- `protocol_requirements`: Protocol requirements
- `wardrobe_requirements`: Wardrobe requirements

### RoutineTask
- `id`: Unique ID
- `title`: Routine title
- `description`: Description
- `routine_type`: Type (morning, afternoon, evening, etc.)
- `scheduled_time`: Scheduled time
- `duration_minutes`: Duration
- `priority`: Priority (1-10)
- `days_of_week`: Days of the week

### Protocol
- `id`: Unique ID
- `title`: Protocol title
- `description`: Description
- `category`: Category (social_media, interview, etc.)
- `priority`: Priority (critical, high, medium, low)
- `rules`: List of rules
- `do_s`: Things to do
- `dont_s`: Things to avoid

### WardrobeItem
- `id`: Unique ID
- `name`: Item name
- `category`: Category (shirt, pants, shoes, etc.)
- `color`: Color
- `dress_codes`: Applicable dress codes
- `season`: Season

## 📄 License

Proprietary - Blatam Academy

## 👥 Author

Blatam Academy

---

[← Back to Main README](../README.md)
