# Guía de Refactorización - Addiction Recovery AI

## Resumen

Este documento describe la refactorización del código siguiendo las mejores prácticas de FastAPI y Python.

## Principios Aplicados

### 1. **Schemas Pydantic para Validación**
- ✅ Todos los request/response ahora usan modelos Pydantic
- ✅ Validación automática de tipos y valores
- ✅ Documentación automática en OpenAPI/Swagger
- ✅ Ubicación: `schemas/`

### 2. **Dependency Injection**
- ✅ Servicios inyectados usando el sistema de dependencias de FastAPI
- ✅ Instancias singleton con `@lru_cache()`
- ✅ Mejor testabilidad y mantenibilidad
- ✅ Ubicación: `dependencies.py`

### 3. **Manejo de Errores Mejorado**
- ✅ Guard clauses al inicio de funciones
- ✅ Early returns para condiciones de error
- ✅ Validación temprana de inputs
- ✅ Mensajes de error descriptivos
- ✅ Uso de `HTTPException` con códigos de estado apropiados

### 4. **Rutas Modulares**
- ✅ Rutas divididas por dominio funcional
- ✅ Cada módulo de rutas en su propio archivo
- ✅ Prefijos y tags organizados
- ✅ Ubicación: `api/routes/`

### 5. **Patrón RORO (Receive an Object, Return an Object)**
- ✅ Funciones reciben objetos Pydantic
- ✅ Funciones retornan objetos Pydantic
- ✅ Sin manipulación directa de diccionarios en endpoints

### 6. **Async/Await**
- ✅ Todas las operaciones I/O son asíncronas
- ✅ Uso correcto de `async def` para endpoints

## Estructura de Archivos

```
addiction_recovery_ai/
├── schemas/                    # Modelos Pydantic
│   ├── __init__.py
│   ├── common.py              # Schemas comunes (ErrorResponse, SuccessResponse)
│   ├── assessment.py          # Schemas de evaluación
│   ├── recovery_plan.py       # Schemas de planes de recuperación
│   ├── progress.py            # Schemas de progreso
│   ├── relapse.py             # Schemas de prevención de recaídas
│   └── support.py             # Schemas de soporte y motivación
├── api/
│   ├── routes/                # Rutas modulares
│   │   ├── __init__.py
│   │   ├── assessment.py     # Rutas de evaluación
│   │   ├── progress.py       # Rutas de progreso
│   │   ├── relapse.py        # Rutas de prevención de recaídas
│   │   └── support.py        # Rutas de soporte
│   └── recovery_api.py       # Router principal (legacy, en proceso de migración)
├── dependencies.py            # Dependency injection
└── main.py                    # Aplicación FastAPI principal
```

## Ejemplo de Uso

### Antes (Código Legacy)

```python
@router.post("/assess")
async def assess_addiction(request: AssessmentRequest):
    try:
        if not AddictionTypeValidator.validate_type(request.addiction_type):
            raise HTTPException(status_code=400, detail="...")
        
        assessment_data = request.dict()
        analysis = analyzer.assess_addiction(assessment_data)
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
```

### Después (Código Refactorizado)

```python
@router.post(
    "/assess",
    response_model=AssessmentResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request data"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def assess_addiction(
    request: AssessmentRequest,
    analyzer: AddictionAnalyzerDep
) -> AssessmentResponse:
    # Guard clause: Validate early
    if not AddictionTypeValidator.validate_type(request.addiction_type):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Tipo de adicción no válido..."
        )
    
    # Process assessment
    try:
        assessment_data = request.model_dump()
        analysis = analyzer.assess_addiction(assessment_data)
        
        # Transform to response model
        return AssessmentResponse(
            assessment_id=analysis.get("assessment_id", ""),
            addiction_type=request.addiction_type,
            severity_score=analysis.get("severity_score", 0.0),
            risk_level=analysis.get("risk_level", "unknown"),
            recommendations=analysis.get("recommendations", []),
            next_steps=analysis.get("next_steps", [])
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error en evaluación: {str(e)}"
        )
```

## Beneficios

1. **Validación Automática**: Pydantic valida automáticamente todos los inputs
2. **Type Safety**: Type hints en todas las funciones
3. **Documentación Automática**: OpenAPI/Swagger se genera automáticamente
4. **Mejor Testabilidad**: Dependency injection facilita testing
5. **Código Más Limpio**: Guard clauses y early returns mejoran legibilidad
6. **Mantenibilidad**: Rutas modulares son más fáciles de mantener
7. **Escalabilidad**: Estructura preparada para crecimiento

## Migración Gradual

El código legacy (`recovery_api.py`) sigue funcionando. La migración puede hacerse gradualmente:

1. ✅ Crear schemas para nuevos endpoints
2. ✅ Crear rutas modulares para nuevos endpoints
3. ⏳ Migrar endpoints existentes uno por uno
4. ⏳ Deprecar código legacy cuando todo esté migrado

## Próximos Pasos

1. Continuar migrando más endpoints a la nueva estructura
2. Refactorizar servicios a funciones puras donde sea posible
3. Agregar más tests usando la nueva estructura
4. Documentar todos los endpoints con ejemplos

