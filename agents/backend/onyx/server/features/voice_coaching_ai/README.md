# Voice Coaching AI

## 📋 Descripción

Sistema de coaching de voz con IA con múltiples versiones mejoradas y optimizaciones avanzadas.

## 🚀 Características Principales

- **Coaching de Voz**: Sistema completo para coaching de voz con IA
- **Múltiples Versiones Mejoradas**: Varias versiones optimizadas
- **Engines Especializados**: Motores especializados para procesamiento de voz
- **Factories**: Sistema de factories para creación de objetos
- **Services**: Servicios de negocio

## 📁 Estructura

```
voice_coaching_ai/
├── core/                   # Lógica central
├── engines/                # Motores especializados
├── factories/              # Factories
├── services/               # Servicios
└── utils/                  # Utilidades
```

## 🔧 Instalación

```bash
pip install -r requirements.txt
```

## 💻 Uso

```python
from voice_coaching_ai.example_usage import VoiceCoachingAI

# Inicializar sistema
coaching = VoiceCoachingAI()

# Analizar voz
analysis = coaching.analyze_voice(audio_file="voice.mp3")

# Proporcionar feedback
feedback = coaching.provide_feedback(analysis)
```

## 🔗 Integración

Este módulo se integra con:
- **Integration System**: Para orquestación
- **Blatam AI**: Motor de IA
- **Export IA**: Para exportación de resultados
