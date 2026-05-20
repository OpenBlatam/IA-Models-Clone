"""
API Documentation
=================
"""

API_DESCRIPTION = """
# Dog Training Coaching AI API

AI-powered dog training coaching assistant using OpenRouter.

## Features

- Expert coaching advice
- Personalized training plans
- Behavior analysis
- Conversational chat
- Progress tracking
- Training assessment
- Educational resources
- Trend analysis

## Authentication

Currently no authentication required. Rate limiting is applied per IP.

## Rate Limits

- `/coach`: 10 requests/minute
- `/training-plan`: 5 requests/minute
- `/analyze-behavior`: 10 requests/minute
- `/chat`: 20 requests/minute
- Other endpoints: 100 requests/hour

## OpenRouter

This service uses OpenRouter to access multiple AI models.
Configure your API key in environment variables.
"""

