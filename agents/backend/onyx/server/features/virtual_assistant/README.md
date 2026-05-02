# Virtual Assistant

<div align="center">

![Status](https://img.shields.io/badge/status-active-success.svg)
![Version](https://img.shields.io/badge/version-2.2-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![AI](https://img.shields.io/badge/AI-Personal%20Assistant-purple.svg)

**A next-generation intelligent assistant for autonomous task management, scheduling, and information retrieval.**

[Overview](#-overview) •
[Features](#-key-features) •
[Architecture](#-architecture) •
[Installation](#-installation) •
[Usage](#-usage) •
[Integrations](#-integrations) •
[Contributing](#-contributing)

</div>

---

## 📋 Overview

**Virtual Assistant** is designed to be more than just a chatbot; it's a proactive agent that helps manage your digital life. Built on advanced LLM technology (OpenAI/Anthropic) and integrated with a robust tool-use framework, it can execute complex multi-step workflows—from scheduling meetings based on your calendar availability to researching topics and summarizing emails.

It features a modular "Skills" system, allowing developers to easily extend its capabilities with new integrations and logic.

### Why Virtual Assistant?

- **Proactive**: Can be configured to alert you about deadlines or important emails without being asked.
- **Context Aware**: Remembers past conversations and preferences (Memory Graph).
- **Secure**: Runs locally or in a private cloud, ensuring your personal data stays yours.

## 🚀 Key Features

| Feature | Description |
|---------|-------------|
| **Smart Scheduling** | Connects to Google Calendar/Outlook to find free slots and book meetings naturally. |
| **Email Triage** | Drafts replies, summarizes long threads, and highlights urgent messages. |
| **Research Mode** | Browses the web to gather information and compiles it into a structured report. |
| **Device Control** | Integrates with Home Assistant to control IoT devices via voice or text. |
| **Voice Interface** | Bi-directional voice communication using Whisper (STT) and ElevenLabs (TTS). |

## 🏗 Architecture

The assistant uses a "Brain-Body-Tools" architecture.

```mermaid
graph TD
    A[User Input (Text/Voice)] --> B(Context Manager)
    B --> C{LLM Brain}
    
    subgraph "Capabilities"
    C --> D[Skill Router]
    D --> E[Calendar Skill]
    D --> F[Email Skill]
    D --> G[Web Search Skill]
    end
    
    subgraph "Memory"
    H[(Vector DB)] <--> B
    I[(User Profile)] <--> B
    end
    
    E --> J[Action Executor]
    F --> J
    G --> J
    
    J --> K[Response Generator]
    K --> L[User Output]
```

## 💻 Installation

### Prerequisites

- Python 3.10+
- OpenAI API Key (or local LLM server endpoint)
- Docker (optional, for self-hosting)

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/blatam-academy/virtual_assistant.git
   cd virtual_assistant
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

## ⚡ Usage

### CLI Interaction

```bash
python main.py chat
# > User: "Schedule a meeting with John for next Tuesday at 2pm"
# > Assistant: "Checking your calendar... Tuesday at 2pm is free. Sending invite to John."
```

### Python SDK

```python
from virtual_assistant import Assistant
from virtual_assistant.skills import CalendarSkill, EmailSkill

# Initialize
bot = Assistant(name="Jarvis")

# Add Skills
bot.add_skill(CalendarSkill(provider="google"))
bot.add_skill(EmailSkill(provider="gmail"))

# Execute Task
response = bot.process("Find the email from AWS about the bill and summarize it")
print(response)
```

## 🔌 Integrations

The Virtual Assistant supports a wide range of external services:

- **Productivity**: Google Workspace, Microsoft 365, Notion, Slack
- **Communication**: WhatsApp, Telegram, Discord
- **Development**: GitHub, Jira, Linear
- **IoT**: Home Assistant, Philips Hue

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
