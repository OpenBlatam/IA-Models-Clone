# PDF Variantes - Arquitectura Mejorada

## 🏗️ Estructura de Capas

```
pdf_variantes/
├── core/                    # Núcleo de la aplicación
│   ├── base_service.py      # Clase base para servicios
│   ├── events.py            # Sistema de eventos pub/sub
│   └── repository.py        # Interfaces de repositorio
│
├── api/                     # Capa de API
│   ├── main.py             # Aplicación FastAPI
│   ├── lifecycle.py        # Ciclo de vida
│   ├── dependencies.py     # Inyección de dependencias
│   ├── middleware.py        # Middleware HTTP
│   ├── config.py           # Configuración
│   ├── routers.py          # Registro de routers
│   ├── routes.py           # Endpoints
│   ├── exceptions.py      # Excepciones personalizadas
│   └── validators.py       # Validadores de requests
│
├── services/                # Capa de servicios
│   ├── pdf_service.py
│   ├── cache_service.py
│   ├── security_service.py
│   ├── performance_service.py
│   ├── monitoring_service.py
│   └── collaboration_service.py
│
├── models/                  # Modelos de dominio
│   └── models.py
│
└── utils/                   # Utilidades
    ├── config.py
    └── logging_config.py
```

## 📋 Principios Arquitectónicos

### 1. Separación de Responsabilidades (SoC)
- **API Layer**: Manejo de HTTP, validación, serialización
- **Service Layer**: Lógica de negocio
- **Core Layer**: Interfaces y abstracciones base
- **Models Layer**: Entidades de dominio

### 2. Inversión de Dependencias (DIP)
- Servicios dependen de interfaces (repositorios, eventos)
- API depende de servicios, no de implementaciones concretas
- Uso de dependency injection de FastAPI

### 3. Single Responsibility Principle (SRP)
- Cada servicio tiene una responsabilidad única
- Cada módulo tiene un propósito claro
- Validadores separados de la lógica de negocio

## 🔄 Flujo de Datos

```
Request → Middleware → Router → Validator → Service → Repository → Data
         ↓
      Events
         ↓
      Handlers
```

## 🎯 Patrones Implementados

### 1. Repository Pattern
```python
# Abstracción para acceso a datos
class DocumentRepository(BaseRepository):
    async def get_by_user(self, user_id: str) -> List[Document]
    async def search(self, query: str) -> List[Document]
```

### 2. Service Layer Pattern
```python
# Lógica de negocio encapsulada
class PDFVariantesService(BaseService):
    async def upload_pdf(self, file: File) -> PDFDocument
    async def generate_variants(self, document_id: str) -> List[Variant]
```

### 3. Event-Driven Architecture
```python
# Comunicación desacoplada
event_bus.emit(EventType.PDF_UPLOADED, payload, source="pdf_service")
```

### 4. Dependency Injection
```python
# Inyección automática de servicios
@app.get("/documents")
async def list_documents(pdf_service: PDFService = Depends(get_pdf_service)):
    return await pdf_service.list_documents()
```

## 📦 Componentes Principales

### BaseService
- Clase base para todos los servicios
- Manejo de inicialización/cleanup
- Health checks
- Logging integrado

### EventBus
- Sistema pub/sub centralizado
- Historial de eventos
- Manejadores async
- Desacoplamiento de componentes

### Repository Interfaces
- Abstracción de acceso a datos
- Facilita testing (mocks)
- Permite cambiar implementación

### Exception Handling
- Excepciones tipadas
- Códigos de error consistentes
- Metadata para debugging
- Formato frontend-friendly

### Validators
- Validación centralizada
- Reutilizables
- Mensajes de error claros
- Validación temprana

## 🔌 Integración de Servicios

Los servicios pueden:
1. **Heredar de BaseService**: Inicialización automática, cleanup, status
2. **Usar EventBus**: Publicar y subscribirse a eventos
3. **Implementar Repository**: Acceso a datos consistente
4. **Inyectarse vía DI**: Disponibles en todos los endpoints

## 📊 Ejemplo de Flujo Completo

```python
# 1. Request llega
POST /api/v1/pdf/upload

# 2. Middleware intercepta
- Rate limiting
- Logging
- Security check

# 3. Router recibe
@router.post("/upload")
async def upload_pdf(file: UploadFile, ...)

# 4. Validator valida
FileValidator.validate_upload_file(file)

# 5. Service ejecuta lógica
pdf_service.upload_pdf(file)

# 6. Event se publica
event_bus.emit(EventType.PDF_UPLOADED, {...})

# 7. Handlers reaccionan
- Analytics tracking
- Notification sending
- Cache invalidation

# 8. Response se retorna
{
  "success": true,
  "data": {...}
}
```

## ✅ Beneficios

1. **Testabilidad**: Interfaces mockeables, servicios aislados
2. **Mantenibilidad**: Código organizado, responsabilidades claras
3. **Escalabilidad**: Fácil agregar servicios, handlers, validadores
4. **Reutilización**: Componentes base reutilizables
5. **Consistencia**: Excepciones, validación, respuestas uniformes
6. **Desacoplamiento**: Eventos permiten cambios sin afectar otros componentes

## 🚀 Próximos Pasos

1. Implementar repositorios concretos (database, cache)
2. Agregar más eventos según necesidad
3. Implementar sagas para transacciones complejas
4. Agregar CQRS si se necesita separación read/write
5. Implementar caching strategy en repositorios






