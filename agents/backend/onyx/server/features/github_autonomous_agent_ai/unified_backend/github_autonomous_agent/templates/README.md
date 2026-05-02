# Templates de Código - GitHub Autonomous Agent

Colección de templates para acelerar el desarrollo de nuevas funcionalidades.

---

## 📋 Templates Disponibles

### 1. `api_route_template.py`
Template para crear nuevos endpoints de API FastAPI.

**Características:**
- Router configurado
- Schemas Pydantic (Create, Update, Response)
- CRUD completo (Create, Read, Update, Delete)
- Manejo de errores
- Documentación con docstrings
- Autenticación opcional

**Uso:**
```bash
cp templates/api_route_template.py api/routes/mi_ruta.py
# Editar y personalizar
```

**Incluye:**
- GET `/` - Listar todos
- GET `/{id}` - Obtener por ID
- POST `/` - Crear nuevo
- PUT `/{id}` - Actualizar
- DELETE `/{id}` - Eliminar

---

### 2. `service_template.py`
Template para crear servicios de negocio.

**Características:**
- Lógica de negocio encapsulada
- Validación de datos
- Procesamiento de datos
- Manejo de errores
- Logging integrado
- Métodos privados de ayuda

**Uso:**
```bash
cp templates/service_template.py core/mi_servicio.py
# Editar y personalizar
```

**Incluye:**
- `create()` - Crear
- `get_by_id()` - Obtener por ID
- `list_all()` - Listar con paginación
- `update()` - Actualizar
- `delete()` - Eliminar
- Métodos privados de ayuda

---

### 3. `test_template.py`
Template para crear tests.

**Características:**
- Tests unitarios
- Tests de integración
- Tests de API
- Fixtures
- Mocks
- Async support

**Uso:**
```bash
cp templates/test_template.py tests/test_mi_modulo.py
# Editar y personalizar
```

**Incluye:**
- Fixtures para datos y mocks
- Tests unitarios del servicio
- Tests de integración
- Tests de endpoints API
- Ejemplos de uso de pytest

---

## 🚀 Flujo de Desarrollo con Templates

### 1. Crear Nueva Funcionalidad

```bash
# 1. Crear servicio
cp templates/service_template.py core/mi_servicio.py
# Editar mi_servicio.py

# 2. Crear endpoint API
cp templates/api_route_template.py api/routes/mi_ruta.py
# Editar mi_ruta.py
# Registrar router en main.py

# 3. Crear tests
cp templates/test_template.py tests/test_mi_servicio.py
# Editar test_mi_servicio.py
```

### 2. Personalizar

1. **Reemplazar nombres:**
   - `Template` → `TuNombre`
   - `template` → `tu_nombre`

2. **Implementar lógica:**
   - Completar TODOs
   - Agregar validaciones
   - Conectar con repositorios

3. **Agregar tests:**
   - Implementar tests unitarios
   - Agregar tests de integración
   - Probar endpoints

---

## 📝 Convenciones en Templates

### Nombres
- **Clases**: `PascalCase` (ej: `TemplateService`)
- **Funciones**: `snake_case` (ej: `create_template`)
- **Variables**: `snake_case` (ej: `template_id`)

### Docstrings
- Usar formato Google style
- Documentar Args, Returns, Raises

### Type Hints
- Siempre usar type hints
- Usar `Optional` para valores opcionales
- Usar `List`, `Dict` de `typing`

### Errores
- Usar excepciones personalizadas
- HTTPException para API
- CustomException para servicios

---

## 🔧 Personalización

### Agregar Autenticación

En `api_route_template.py`:
```python
from api.dependencies import get_current_user

@router.get("/")
async def list_templates(
    current_user = Depends(get_current_user),  # Descomentar
):
    ...
```

### Agregar Validación Personalizada

En `service_template.py`:
```python
def _validate_data(self, data: Dict[str, Any]) -> None:
    # Agregar validaciones personalizadas
    if 'email' in data:
        if not self._is_valid_email(data['email']):
            raise CustomException("Email inválido")
```

### Agregar Filtros

En `service_template.py`:
```python
def _build_filters(self, filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    # Agregar lógica de filtros
    if 'status' in filters:
        processed_filters['status'] = filters['status']
    return processed_filters
```

---

## ✅ Checklist de Uso

Al usar un template:

- [ ] Copiado el template
- [ ] Reemplazados todos los nombres
- [ ] Implementada la lógica (TODOs completados)
- [ ] Agregadas validaciones
- [ ] Conectado con repositorios/servicios
- [ ] Agregados tests
- [ ] Documentación actualizada
- [ ] Registrado en main.py (si es router)
- [ ] Verificado que funciona

---

## 📚 Ejemplos de Uso

### Ejemplo 1: Crear Endpoint de Usuarios

```bash
# 1. Crear servicio
cp templates/service_template.py core/user_service.py
# Editar: TemplateService → UserService

# 2. Crear endpoint
cp templates/api_route_template.py api/routes/user_routes.py
# Editar: template → user, Template → User

# 3. Registrar en main.py
# app.include_router(user_routes.router, prefix="/api/v1/users")

# 4. Crear tests
cp templates/test_template.py tests/test_user_service.py
```

### Ejemplo 2: Crear Endpoint de Tareas

```bash
# Similar proceso pero para "task"
cp templates/service_template.py core/task_service.py
cp templates/api_route_template.py api/routes/task_routes.py
cp templates/test_template.py tests/test_task_service.py
```

---

## 🎯 Mejores Prácticas

1. **Siempre usar templates** - Ahorra tiempo y mantiene consistencia
2. **Completar TODOs** - No dejar código incompleto
3. **Agregar tests** - Cobertura mínima del 80%
4. **Documentar** - Docstrings claros y completos
5. **Seguir convenciones** - Mantener estilo consistente

---

## 🔄 Actualizar Templates

Si mejoras un template:

1. Actualiza el template base
2. Documenta los cambios
3. Notifica al equipo
4. Considera migrar código existente

---

**Última actualización:** 2024




