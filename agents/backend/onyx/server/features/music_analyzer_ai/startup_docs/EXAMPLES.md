# Ejemplos de Uso - Music Analyzer AI

## 🎵 Ejemplos Básicos

### 1. Buscar Canciones

#### Python (requests)

```python
import requests

BASE_URL = "http://localhost:8010"

# Buscar canciones
response = requests.post(
    f"{BASE_URL}/music/search",
    json={
        "query": "Bohemian Rhapsody Queen",
        "limit": 5
    }
)

results = response.json()
print(f"Encontradas {results['total']} canciones")

for track in results['results']:
    print(f"- {track['name']} by {', '.join(track['artists'])}")
```

#### Python (httpx async)

```python
import httpx
import asyncio

BASE_URL = "http://localhost:8010"

async def search_tracks(query: str, limit: int = 5):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/music/search",
            json={"query": query, "limit": limit}
        )
        return response.json()

# Uso
results = asyncio.run(search_tracks("Bohemian Rhapsody"))
```

#### cURL

```bash
curl -X POST http://localhost:8010/music/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Bohemian Rhapsody Queen",
    "limit": 5
  }'
```

#### JavaScript (fetch)

```javascript
async function searchTracks(query, limit = 5) {
  const response = await fetch('http://localhost:8010/music/search', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ query, limit }),
  });
  
  return await response.json();
}

// Uso
searchTracks('Bohemian Rhapsody', 5)
  .then(results => {
    console.log(`Encontradas ${results.total} canciones`);
    results.results.forEach(track => {
      console.log(`- ${track.name} by ${track.artists.join(', ')}`);
    });
  });
```

### 2. Analizar Canción

#### Python (requests)

```python
import requests

BASE_URL = "http://localhost:8010"

# Analizar canción
response = requests.post(
    f"{BASE_URL}/music/analyze",
    json={
        "track_id": "4uLU6hMCjMI75M1A2tKUQC",
        "include_coaching": True
    }
)

analysis = response.json()

print(f"Canción: {analysis['track_name']}")
print(f"Artistas: {', '.join(analysis['artists'])}")
print(f"Tonalidad: {analysis['analysis']['key']} {analysis['analysis']['mode']}")
print(f"Tempo: {analysis['analysis']['tempo']} BPM")
```

#### Python (httpx async)

```python
import httpx
import asyncio

BASE_URL = "http://localhost:8010"

async def analyze_track(track_id: str, include_coaching: bool = False):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/music/analyze",
            json={
                "track_id": track_id,
                "include_coaching": include_coaching
            }
        )
        return response.json()

# Uso
analysis = asyncio.run(
    analyze_track("4uLU6hMCjMI75M1A2tKUQC", include_coaching=True)
)
```

### 3. Obtener Recomendaciones

#### Python

```python
import requests

BASE_URL = "http://localhost:8010"

# Obtener recomendaciones
response = requests.get(
    f"{BASE_URL}/music/recommendations/4uLU6hMCjMI75M1A2tKUQC",
    params={"limit": 10}
)

recommendations = response.json()

print(f"Recomendaciones para: {recommendations['track_name']}")
for rec in recommendations['recommendations']:
    print(f"- {rec['name']} by {', '.join(rec['artists'])}")
```

## 🔄 Ejemplos Avanzados

### 4. Análisis en Lote

```python
import httpx
import asyncio

BASE_URL = "http://localhost:8010"

async def analyze_multiple_tracks(track_ids: list):
    async with httpx.AsyncClient() as client:
        tasks = [
            client.post(
                f"{BASE_URL}/music/analyze",
                json={"track_id": track_id, "include_coaching": False}
            )
            for track_id in track_ids
        ]
        
        responses = await asyncio.gather(*tasks)
        return [r.json() for r in responses]

# Uso
track_ids = [
    "4uLU6hMCjMI75M1A2tKUQC",  # Bohemian Rhapsody
    "3n3Ppam7vgaVa1iaRUc9Lp",  # Another track
    "4PTG3Z6ehGkBFwjyzjWFsX"   # Another track
]

analyses = asyncio.run(analyze_multiple_tracks(track_ids))
```

### 5. Búsqueda y Análisis Automático

```python
import requests
import time

BASE_URL = "http://localhost:8010"

def search_and_analyze(query: str):
    # 1. Buscar
    search_response = requests.post(
        f"{BASE_URL}/music/search",
        json={"query": query, "limit": 1}
    )
    search_results = search_response.json()
    
    if not search_results['results']:
        print("No se encontraron canciones")
        return None
    
    # 2. Obtener primer resultado
    track = search_results['results'][0]
    track_id = track['id']
    
    print(f"Analizando: {track['name']} by {', '.join(track['artists'])}")
    
    # 3. Analizar
    analysis_response = requests.post(
        f"{BASE_URL}/music/analyze",
        json={"track_id": track_id, "include_coaching": True}
    )
    analysis = analysis_response.json()
    
    return analysis

# Uso
result = search_and_analyze("Bohemian Rhapsody")
if result:
    print(f"Tonalidad: {result['analysis']['key']}")
    print(f"Tempo: {result['analysis']['tempo']} BPM")
```

### 6. Clase Wrapper para la API

```python
import requests
from typing import List, Dict, Optional

class MusicAnalyzerClient:
    def __init__(self, base_url: str = "http://localhost:8010"):
        self.base_url = base_url
    
    def search(self, query: str, limit: int = 5) -> Dict:
        """Buscar canciones"""
        response = requests.post(
            f"{self.base_url}/music/search",
            json={"query": query, "limit": limit}
        )
        response.raise_for_status()
        return response.json()
    
    def analyze(self, track_id: str, include_coaching: bool = False) -> Dict:
        """Analizar canción"""
        response = requests.post(
            f"{self.base_url}/music/analyze",
            json={"track_id": track_id, "include_coaching": include_coaching}
        )
        response.raise_for_status()
        return response.json()
    
    def recommendations(self, track_id: str, limit: int = 10) -> Dict:
        """Obtener recomendaciones"""
        response = requests.get(
            f"{self.base_url}/music/recommendations/{track_id}",
            params={"limit": limit}
        )
        response.raise_for_status()
        return response.json()
    
    def search_and_analyze(self, query: str) -> Optional[Dict]:
        """Buscar y analizar automáticamente"""
        search_results = self.search(query, limit=1)
        
        if not search_results['results']:
            return None
        
        track_id = search_results['results'][0]['id']
        return self.analyze(track_id, include_coaching=True)

# Uso
client = MusicAnalyzerClient()

# Buscar
results = client.search("Bohemian Rhapsody")

# Analizar
analysis = client.analyze("4uLU6hMCjMI75M1A2tKUQC", include_coaching=True)

# Recomendaciones
recommendations = client.recommendations("4uLU6hMCjMI75M1A2tKUQC")

# Búsqueda y análisis automático
result = client.search_and_analyze("Bohemian Rhapsody")
```

## 🎓 Ejemplos de Coaching

### 7. Obtener Coaching Musical

```python
import requests

BASE_URL = "http://localhost:8010"

# Obtener coaching
response = requests.post(
    f"{BASE_URL}/music/coaching",
    json={
        "track_id": "4uLU6hMCjMI75M1A2tKUQC",
        "skill_level": "intermediate",
        "instrument": "piano"
    }
)

coaching = response.json()

print("Ruta de Aprendizaje:")
for step in coaching['coaching']['learning_path']:
    print(f"{step['step']}. {step['title']}")
    print(f"   {step['description']}")

print("\nEjercicios:")
for exercise in coaching['coaching']['exercises']:
    print(f"- {exercise['name']}: {exercise['description']}")
```

## 🔧 Ejemplos de Integración

### 8. Integración con Flask

```python
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)
MUSIC_API_URL = "http://localhost:8010"

@app.route('/api/search', methods=['POST'])
def search():
    query = request.json.get('query')
    limit = request.json.get('limit', 5)
    
    response = requests.post(
        f"{MUSIC_API_URL}/music/search",
        json={"query": query, "limit": limit}
    )
    
    return jsonify(response.json())

@app.route('/api/analyze', methods=['POST'])
def analyze():
    track_id = request.json.get('track_id')
    
    response = requests.post(
        f"{MUSIC_API_URL}/music/analyze",
        json={"track_id": track_id, "include_coaching": True}
    )
    
    return jsonify(response.json())

if __name__ == '__main__':
    app.run(port=5000)
```

### 9. Integración con Django

```python
# views.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import requests
import json

MUSIC_API_URL = "http://localhost:8010"

@csrf_exempt
def search_tracks(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        query = data.get('query')
        limit = data.get('limit', 5)
        
        response = requests.post(
            f"{MUSIC_API_URL}/music/search",
            json={"query": query, "limit": limit}
        )
        
        return JsonResponse(response.json())
    
    return JsonResponse({"error": "Method not allowed"}, status=405)
```

## 📚 Más Ejemplos

Para más ejemplos, consulta:

- **Documentación de API**: http://localhost:8010/docs
- **Ejemplos en el código**: [../examples/](../examples/)
- **Tests**: [../tests/](../tests/)

---

**Última actualización**: 2025  
**Versión**: 2.21.0






