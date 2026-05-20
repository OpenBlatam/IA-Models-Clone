# 🔌 Guía de Integraciones - Music Analyzer AI

Esta guía cubre cómo integrar Music Analyzer AI con otros sistemas y servicios.

## 🎵 Integraciones Musicales

### Spotify

#### Integración Básica

```python
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Configurar cliente
client_credentials_manager = SpotifyClientCredentials(
    client_id=os.getenv("SPOTIFY_CLIENT_ID"),
    client_secret=os.getenv("SPOTIFY_CLIENT_SECRET")
)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Usar
track = sp.track("4uLU6hMCjMI75M1A2tKUQC")
```

#### Webhooks de Spotify

```python
@app.post("/webhooks/spotify")
async def spotify_webhook(request: Request):
    data = await request.json()
    # Procesar webhook
    return {"status": "ok"}
```

### YouTube Music

```python
from ytmusicapi import YTMusic

ytmusic = YTMusic()

# Buscar
results = ytmusic.search("Bohemian Rhapsody", filter="songs")

# Obtener detalles
song = ytmusic.get_song("song_id")
```

### Apple Music

```python
import requests

APPLE_MUSIC_TOKEN = os.getenv("APPLE_MUSIC_TOKEN")

headers = {
    "Authorization": f"Bearer {APPLE_MUSIC_TOKEN}",
    "Music-User-Token": os.getenv("APPLE_MUSIC_USER_TOKEN")
}

response = requests.get(
    "https://api.music.apple.com/v1/catalog/us/songs/123456",
    headers=headers
)
```

## 🔗 Integraciones de APIs

### REST API

#### Cliente Python

```python
import requests

class MusicAnalyzerClient:
    def __init__(self, base_url: str = "http://localhost:8010"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def search(self, query: str, limit: int = 5):
        response = self.session.post(
            f"{self.base_url}/music/search",
            json={"query": query, "limit": limit}
        )
        return response.json()
    
    def analyze(self, track_id: str, include_coaching: bool = False):
        response = self.session.post(
            f"{self.base_url}/music/analyze",
            json={"track_id": track_id, "include_coaching": include_coaching}
        )
        return response.json()

# Uso
client = MusicAnalyzerClient()
results = client.search("Bohemian Rhapsody")
```

#### Cliente JavaScript

```javascript
class MusicAnalyzerClient {
  constructor(baseUrl = 'http://localhost:8010') {
    this.baseUrl = baseUrl;
  }

  async search(query, limit = 5) {
    const response = await fetch(`${this.baseUrl}/music/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, limit })
    });
    return await response.json();
  }

  async analyze(trackId, includeCoaching = false) {
    const response = await fetch(`${this.baseUrl}/music/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ track_id: trackId, include_coaching: includeCoaching })
    });
    return await response.json();
  }
}

// Uso
const client = new MusicAnalyzerClient();
const results = await client.search('Bohemian Rhapsody');
```

### GraphQL (Futuro)

```graphql
query {
  track(id: "4uLU6hMCjMI75M1A2tKUQC") {
    name
    artists
    analysis {
      key
      tempo
      energy
    }
    coaching {
      learningPath {
        title
        description
      }
    }
  }
}
```

## 🗄️ Integraciones de Base de Datos

### PostgreSQL

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/tracks/save")
async def save_track_analysis(
    analysis: AnalysisResponse,
    db: Session = Depends(get_db)
):
    db_analysis = TrackAnalysis(**analysis.dict())
    db.add(db_analysis)
    db.commit()
    return {"status": "saved"}
```

### MongoDB

```python
from pymongo import MongoClient

client = MongoClient(MONGODB_URL)
db = client.music_analyzer

def save_analysis(analysis: dict):
    db.analyses.insert_one(analysis)

def get_analysis(track_id: str):
    return db.analyses.find_one({"track_id": track_id})
```

### Redis

```python
import redis
import json

redis_client = redis.Redis.from_url(REDIS_URL)

def cache_analysis(track_id: str, analysis: dict, ttl=3600):
    redis_client.setex(
        f"analysis:{track_id}",
        ttl,
        json.dumps(analysis)
    )

def get_cached_analysis(track_id: str):
    cached = redis_client.get(f"analysis:{track_id}")
    if cached:
        return json.loads(cached)
    return None
```

## 📊 Integraciones de Analytics

### Google Analytics

```python
from google.analytics.data_v1beta import BetaAnalyticsDataClient

client = BetaAnalyticsDataClient()

def track_event(event_name: str, parameters: dict):
    # Enviar evento a Google Analytics
    pass
```

### Mixpanel

```python
from mixpanel import Mixpanel

mp = Mixpanel(MIXPANEL_TOKEN)

def track_analysis(track_id: str, user_id: str):
    mp.track(user_id, "track_analyzed", {
        "track_id": track_id,
        "timestamp": datetime.utcnow().isoformat()
    })
```

### Segment

```python
from segment import analytics

analytics.write_key = SEGMENT_WRITE_KEY

def track_event(user_id: str, event: str, properties: dict):
    analytics.track(user_id, event, properties)
```

## 🔔 Integraciones de Notificaciones

### Email (SendGrid)

```python
import sendgrid
from sendgrid.helpers.mail import Mail

sg = sendgrid.SendGridAPIClient(SENDGRID_API_KEY)

def send_analysis_email(to_email: str, analysis: dict):
    message = Mail(
        from_email="noreply@musicanalyzer.ai",
        to_emails=to_email,
        subject="Your Music Analysis is Ready",
        html_content=format_analysis_html(analysis)
    )
    sg.send(message)
```

### Slack

```python
from slack_sdk import WebClient

slack = WebClient(SLACK_TOKEN)

def send_slack_notification(channel: str, message: str):
    slack.chat_postMessage(
        channel=channel,
        text=message
    )
```

### Discord

```python
import discord
from discord import Webhook, RequestsWebhookAdapter

webhook = Webhook.from_url(DISCORD_WEBHOOK_URL, adapter=RequestsWebhookAdapter())

def send_discord_notification(message: str):
    webhook.send(message)
```

## ☁️ Integraciones Cloud

### AWS

#### S3 para Almacenamiento

```python
import boto3

s3_client = boto3.client('s3')

def save_analysis_to_s3(track_id: str, analysis: dict):
    s3_client.put_object(
        Bucket='music-analyzer-analyses',
        Key=f"analyses/{track_id}.json",
        Body=json.dumps(analysis)
    )
```

#### SQS para Colas

```python
sqs = boto3.client('sqs')

def queue_analysis(track_id: str):
    sqs.send_message(
        QueueUrl=ANALYSIS_QUEUE_URL,
        MessageBody=json.dumps({"track_id": track_id})
    )
```

### Google Cloud

#### Cloud Storage

```python
from google.cloud import storage

storage_client = storage.Client()
bucket = storage_client.bucket('music-analyzer-analyses')

def save_to_gcs(track_id: str, analysis: dict):
    blob = bucket.blob(f"analyses/{track_id}.json")
    blob.upload_from_string(json.dumps(analysis))
```

### Azure

#### Blob Storage

```python
from azure.storage.blob import BlobServiceClient

blob_service = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)

def save_to_azure(track_id: str, analysis: dict):
    blob_client = blob_service.get_blob_client(
        container="analyses",
        blob=f"{track_id}.json"
    )
    blob_client.upload_blob(json.dumps(analysis))
```

## 🔐 Integraciones de Autenticación

### OAuth 2.0

```python
from authlib.integrations.starlette_client import OAuth

oauth = OAuth()

oauth.register(
    name='spotify',
    client_id=SPOTIFY_CLIENT_ID,
    client_secret=SPOTIFY_CLIENT_SECRET,
    server_metadata_url='https://accounts.spotify.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'user-read-private user-read-email'}
)

@app.get("/auth/spotify")
async def auth_spotify(request: Request):
    redirect_uri = request.url_for('auth_callback')
    return await oauth.spotify.authorize_redirect(request, redirect_uri)
```

### JWT

```python
import jwt
from datetime import datetime, timedelta

def create_jwt_token(user_id: str) -> str:
    payload = {
        "user_id": user_id,
        "exp": datetime.utcnow() + timedelta(hours=24)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")
```

## 📱 Integraciones Móviles

### React Native

```javascript
import axios from 'axios';

const API_BASE = 'http://localhost:8010';

export const MusicAnalyzerAPI = {
  search: async (query) => {
    const response = await axios.post(`${API_BASE}/music/search`, {
      query,
      limit: 5
    });
    return response.data;
  },
  
  analyze: async (trackId) => {
    const response = await axios.post(`${API_BASE}/music/analyze`, {
      track_id: trackId,
      include_coaching: true
    });
    return response.data;
  }
};
```

### Flutter

```dart
import 'package:http/http.dart' as http;
import 'dart:convert';

class MusicAnalyzerAPI {
  static const String baseUrl = 'http://localhost:8010';
  
  static Future<Map<String, dynamic>> search(String query) async {
    final response = await http.post(
      Uri.parse('$baseUrl/music/search'),
      headers: {'Content-Type': 'application/json'},
      body: json.encode({'query': query, 'limit': 5}),
    );
    return json.decode(response.body);
  }
}
```

## 🔄 Webhooks

### Recibir Webhooks

```python
@app.post("/webhooks/analysis-complete")
async def analysis_webhook(request: Request):
    data = await request.json()
    # Procesar webhook
    return {"status": "received"}
```

### Enviar Webhooks

```python
import httpx

async def send_webhook(url: str, payload: dict):
    async with httpx.AsyncClient() as client:
        await client.post(url, json=payload)
```

## 📚 Ejemplos de Integración Completa

### Flask Integration

```python
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)
MUSIC_API_URL = "http://localhost:8010"

@app.route('/analyze', methods=['POST'])
def analyze():
    track_id = request.json.get('track_id')
    
    response = requests.post(
        f"{MUSIC_API_URL}/music/analyze",
        json={"track_id": track_id, "include_coaching": True}
    )
    
    return jsonify(response.json())
```

### Django Integration

```python
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import requests
import json

MUSIC_API_URL = "http://localhost:8010"

@csrf_exempt
def analyze_track(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        track_id = data.get('track_id')
        
        response = requests.post(
            f"{MUSIC_API_URL}/music/analyze",
            json={"track_id": track_id}
        )
        
        return JsonResponse(response.json())
```

---

**Última actualización**: 2025  
**Versión**: 2.21.0






