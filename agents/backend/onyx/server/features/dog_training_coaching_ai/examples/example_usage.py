"""
Example Usage - Dog Training Coaching AI
========================================
"""

import asyncio
import httpx


async def example_coaching():
    """Ejemplo de uso del endpoint de coaching."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8030/api/v1/coach",
            json={
                "question": "How do I teach my puppy to sit?",
                "dog_breed": "Golden Retriever",
                "dog_age": "3 months",
                "training_goal": "obedience",
                "experience_level": "beginner"
            }
        )
        print(response.json())


async def example_training_plan():
    """Ejemplo de creación de plan de entrenamiento."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8030/api/v1/training-plan",
            json={
                "dog_breed": "German Shepherd",
                "dog_age": "1 year",
                "training_goals": ["obedience", "agility"],
                "time_available": "30 minutes per day"
            }
        )
        print(response.json())


if __name__ == "__main__":
    asyncio.run(example_coaching())

