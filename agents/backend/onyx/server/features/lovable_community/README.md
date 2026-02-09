# Lovable Community API

Sistema de comunidad estilo Lovable donde los usuarios pueden publicar sus chats, remixar chats de otros usuarios, y los mejores chats aparecen en la parte superior según un algoritmo de ranking.

## Características

### Funcionalidades Principales
- **Publicar chats**: Los usuarios pueden publicar sus conversaciones en la comunidad
- **Remixar**: Los usuarios pueden crear remixes (versiones modificadas) de chats existentes
- **Sistema de votación**: Upvote/downvote para rankear contenido
- **Ranking inteligente**: Algoritmo que combina votos, remixes, vistas y recencia
- **Búsqueda avanzada**: Buscar por texto, tags, usuario, etc.
- **Top chats**: Ver los chats más populares ordenados por score

### Funcionalidades Avanzadas
- **Actualizar chats**: Los usuarios pueden actualizar sus chats publicados
- **Eliminar chats**: Los usuarios pueden eliminar sus propios chats
- **Destacar chats**: Sistema para marcar chats como destacados (featured)
- **Perfiles de usuario**: Ver estadísticas y perfil de cualquier usuario
- **Chats trending**: Ver chats trending en diferentes períodos (hora, día, semana, mes)
- **Analytics**: Estadísticas agregadas de toda la comunidad
- **Operaciones en lote**: Realizar operaciones sobre múltiples chats simultáneamente
- **Estadísticas detalladas**: Métricas avanzadas incluyendo upvotes, downvotes y engagement rate

## Estructura

```
lovable_community/
├── __init__.py
├── models.py          # Modelos de base de datos (SQLAlchemy)
├── schemas.py          # Schemas Pydantic para validación
├── services.py         # Lógica de negocio y servicios
├── helpers.py          # Funciones auxiliares y utilidades
├── validators.py       # Validadores reutilizables
├── exceptions.py       # Excepciones personalizadas
├── dependencies.py     # Dependencias de FastAPI
├── config.py           # Configuración de la aplicación
├── api/
│   ├── __init__.py
│   ├── routes.py       # Endpoints de la API
│   └── router.py       # Router principal
├── main.py             # Aplicación FastAPI principal
└── README.md
```

## Modelos de Base de Datos

### PublishedChat
- `id`: ID único del chat
- `user_id`: ID del usuario que publicó
- `title`: Título del chat
- `description`: Descripción opcional
- `chat_content`: Contenido del chat (JSON o texto)
- `tags`: Tags separados por coma
- `vote_count`: Número de votos
- `remix_count`: Número de remixes
- `view_count`: Número de vistas
- `score`: Score calculado para ranking
- `is_public`: Si es público
- `is_featured`: Si está destacado
- `original_chat_id`: ID del chat original (si es remix)

### ChatRemix
- `id`: ID único del remix
- `original_chat_id`: ID del chat original
- `remix_chat_id`: ID del chat remix
- `user_id`: ID del usuario que creó el remix

### ChatVote
- `id`: ID único del voto
- `chat_id`: ID del chat votado
- `user_id`: ID del usuario que votó
- `vote_type`: "upvote" o "downvote"

### ChatView
- `id`: ID único de la vista
- `chat_id`: ID del chat visto
- `user_id`: ID del usuario (opcional)

## Algoritmo de Ranking

El score se calcula usando la siguiente fórmula:

```
score = (votes * 2 + remixes * 3 + views * 0.1) / time_decay
```

Donde `time_decay` aumenta con el tiempo para dar prioridad a contenido reciente.

## Endpoints

### POST `/lovable/community/publish`
Publica un nuevo chat en la comunidad.

**Request:**
```json
{
  "title": "Mi chat increíble",
  "description": "Una conversación sobre IA",
  "chat_content": "{...}",
  "tags": ["ai", "chat", "conversation"],
  "is_public": true
}
```

### GET `/lovable/community/chats`
Lista chats con paginación.

**Query Parameters:**
- `page`: Número de página (default: 1)
- `page_size`: Tamaño de página (default: 20, max: 100)
- `sort_by`: Ordenar por: `score`, `created_at`, `vote_count`, `remix_count` (default: `score`)
- `order`: `asc` o `desc` (default: `desc`)
- `user_id`: Filtrar por usuario (opcional)

### GET `/lovable/community/chats/{chat_id}`
Obtiene los detalles de un chat específico.

### POST `/lovable/community/chats/{chat_id}/remix`
Crea un remix de un chat existente.

**Request:**
```json
{
  "original_chat_id": "chat-id",
  "title": "Mi remix",
  "description": "Versión mejorada",
  "chat_content": "{...}",
  "tags": ["remix", "improved"]
}
```

### POST `/lovable/community/chats/{chat_id}/vote`
Vota un chat (upvote o downvote).

**Request:**
```json
{
  "chat_id": "chat-id",
  "vote_type": "upvote"
}
```

### GET `/lovable/community/chats/{chat_id}/remixes`
Obtiene todos los remixes de un chat.

**Query Parameters:**
- `limit`: Límite de resultados (default: 20, max: 100)

### GET `/lovable/community/search`
Busca chats por texto, tags, usuario, etc.

**Query Parameters:**
- `query`: Texto de búsqueda (opcional)
- `tags`: Tags separados por coma (opcional)
- `user_id`: Filtrar por usuario (opcional)
- `sort_by`: Ordenar por (default: `score`)
- `order`: `asc` o `desc` (default: `desc`)
- `page`: Número de página (default: 1)
- `page_size`: Tamaño de página (default: 20)

### GET `/lovable/community/top`
Obtiene los chats más populares ordenados por score.

**Query Parameters:**
- `limit`: Límite de resultados (default: 20, max: 100)

### GET `/lovable/community/chats/{chat_id}/stats`
Obtiene las estadísticas de engagement de un chat.

### PUT `/lovable/community/chats/{chat_id}`
Actualiza un chat existente. Solo el propietario puede actualizar su chat.

**Request:**
```json
{
  "title": "Título actualizado",
  "description": "Nueva descripción",
  "tags": ["nuevo", "tag"],
  "is_public": true
}
```

### DELETE `/lovable/community/chats/{chat_id}`
Elimina un chat. Solo el propietario puede eliminar su chat.

### POST `/lovable/community/chats/{chat_id}/feature`
Destaca o quita el destacado de un chat. Requiere permisos de administrador.

**Query Parameters:**
- `featured`: `true` para destacar, `false` para quitar destacado

### GET `/lovable/community/users/{user_id}/profile`
Obtiene el perfil y estadísticas de un usuario.

**Response:**
```json
{
  "user_id": "user-123",
  "total_chats": 10,
  "total_remixes": 5,
  "total_votes": 25,
  "average_score": 8.5,
  "top_chat_id": "chat-456"
}
```

### GET `/lovable/community/trending`
Obtiene los chats trending en diferentes períodos de tiempo.

**Query Parameters:**
- `period`: `hour`, `day`, `week`, o `month` (default: `day`)
- `limit`: Límite de resultados (default: 20, max: 100)

### GET `/lovable/community/analytics`
Obtiene estadísticas agregadas de toda la comunidad.

**Query Parameters:**
- `period_days`: Número de días para filtrar (opcional)

**Response:**
```json
{
  "total_chats": 1000,
  "total_users": 150,
  "total_votes": 5000,
  "total_remixes": 200,
  "total_views": 50000,
  "average_score": 7.5,
  "top_tags": [
    {"tag": "ai", "count": 150},
    {"tag": "chat", "count": 120}
  ],
  "period": "all time"
}
```

### POST `/lovable/community/bulk`
Realiza una operación en lote sobre múltiples chats (máximo 100).

**Request:**
```json
{
  "chat_ids": ["chat1", "chat2", "chat3"],
  "operation": "feature"
}
```

**Operations disponibles:**
- `delete`: Eliminar chats (requiere user_id)
- `feature`: Destacar chats
- `unfeature`: Quitar destacado
- `make_public`: Hacer públicos
- `make_private`: Hacer privados

### GET `/lovable/community/chats/{chat_id}/stats/detailed`
Obtiene estadísticas detalladas incluyendo upvotes, downvotes y engagement rate.

**Response:**
```json
{
  "chat_id": "chat-123",
  "vote_count": 50,
  "remix_count": 10,
  "view_count": 500,
  "score": 8.5,
  "rank": 5,
  "upvote_count": 45,
  "downvote_count": 5,
  "engagement_rate": 10.0
}
```

## Instalación

1. Instalar dependencias:
```bash
pip install fastapi uvicorn sqlalchemy pydantic
```

2. Ejecutar la aplicación:
```bash
python -m features.lovable_community.main
```

O usando uvicorn directamente:
```bash
uvicorn features.lovable_community.main:app --host 0.0.0.0 --port 8007
```

## Base de Datos

Por defecto usa SQLite (`lovable_community.db`). Para cambiar a PostgreSQL u otra base de datos, modifica la URL en `api/routes.py` y `main.py`.

## Autenticación

Actualmente el sistema usa un `user_id` simplificado. Para producción, deberías:
1. Implementar autenticación JWT
2. Obtener `user_id` del token
3. Validar permisos antes de operaciones sensibles

## Arquitectura y Mejoras Implementadas

### Validación y Sanitización
- ✅ Validación exhaustiva de inputs con Pydantic
- ✅ Sanitización automática de datos
- ✅ Validadores reutilizables en `validators.py`
- ✅ Helpers para conversión y formateo en `helpers.py`

### Manejo de Errores
- ✅ Excepciones personalizadas con mensajes descriptivos
- ✅ Manejo consistente de errores en todos los endpoints
- ✅ Logging detallado para debugging

### Optimizaciones
- ✅ Índices de base de datos optimizados
- ✅ Consultas eficientes con SQLAlchemy
- ✅ Paginación optimizada
- ✅ Cálculo de scores optimizado

### Nuevas Funcionalidades
- ✅ Actualización y eliminación de chats
- ✅ Sistema de destacados (featured)
- ✅ Perfiles de usuario con estadísticas
- ✅ Chats trending por período
- ✅ Analytics agregados
- ✅ Operaciones en lote
- ✅ Estadísticas detalladas

## Mejoras Futuras

- [ ] Autenticación JWT completa
- [ ] Sistema de comentarios (schemas ya creados)
- [ ] Notificaciones cuando alguien remixa tu chat
- [ ] Sistema de reportes y moderación
- [ ] Rate limiting implementado
- [ ] Exportación de chats
- [ ] Integración con otros sistemas
- [ ] Cache para consultas frecuentes
- [ ] WebSockets para actualizaciones en tiempo real

