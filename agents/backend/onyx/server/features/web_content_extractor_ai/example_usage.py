"""
Ejemplo de uso del Web Content Extractor AI
"""

import asyncio
import httpx
import json


async def example_extract():
    """Ejemplo de extracción de contenido"""
    
    url = "http://localhost:8000/api/v1/extract"
    
    payload = {
        "url": "https://example.com",
        "model": "anthropic/claude-3.5-sonnet",
        "max_tokens": 4000
    }
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(url, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Extracción exitosa")
            print(f"URL: {result['url']}")
            print(f"Título: {result['raw_data']['title']}")
            print(f"Método: {result['raw_data']['extraction_method']}")
            print(f"Tokens usados: {result['processing_metadata']['tokens_used']}")
            print("\nInformación extraída:")
            print(result['extracted_info'][:500] + "...")
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)


async def example_cache_stats():
    """Ejemplo de consulta de estadísticas de cache"""
    
    url = "http://localhost:8000/api/v1/extract/cache/stats"
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        
        if response.status_code == 200:
            stats = response.json()
            print("📊 Estadísticas de cache:")
            print(json.dumps(stats, indent=2))
        else:
            print(f"❌ Error: {response.status_code}")


if __name__ == "__main__":
    print("Ejecutando ejemplos...\n")
    asyncio.run(example_extract())
    print("\n" + "="*50 + "\n")
    asyncio.run(example_cache_stats())








