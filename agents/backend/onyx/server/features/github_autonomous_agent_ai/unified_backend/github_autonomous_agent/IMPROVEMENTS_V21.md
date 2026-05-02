# Mejoras V21 - Componentes UI, Manejo de Errores y Utilidades de Desarrollo

## Resumen Ejecutivo

Esta versión introduce mejoras significativas en el frontend con componentes UI reutilizables, manejo centralizado de errores, y utilidades de desarrollo en el backend para debugging y profiling.

## 🎯 Mejoras Implementadas

### 1. Manejo Centralizado de Errores

**Archivo**: `github_autonomous_agent_ai/frontend/app/hooks/useErrorHandler.ts`

- **Manejo Unificado**: Hook centralizado para manejar todos los errores
- **Integración con Notificaciones**: Errores automáticamente notificados
- **Historial de Errores**: Mantener historial de errores recientes
- **Logging Automático**: Logging automático a consola
- **Callbacks Personalizados**: Callbacks para manejo personalizado

**Características**:
- Manejo de Error, string, o unknown
- Contexto opcional para errores
- Historial de últimos 50 errores
- Integración con toast notifications
- Callback personalizado

**Ejemplo de Uso**:
```typescript
import { useErrorHandler } from '@/hooks/useErrorHandler';

function MyComponent() {
  const { handleError, handleAsyncError, errors, lastError } = useErrorHandler({
    showToast: true,
    logToConsole: true,
    onError: (error) => {
      // Manejo personalizado
      console.log('Error personalizado:', error);
    }
  });
  
  const processData = async () => {
    const result = await handleAsyncError(
      async () => {
        return await fetchData();
      },
      'processData'
    );
    
    if (result) {
      // Procesar resultado
    }
  };
  
  return (
    <div>
      {lastError && (
        <div className="error">
          {lastError.message}
        </div>
      )}
    </div>
  );
}
```

### 2. Componentes UI Reutilizables

#### Button Component

**Archivo**: `github_autonomous_agent_ai/frontend/app/components/ui/Button.tsx`

- **Variantes**: primary, secondary, danger, ghost, outline
- **Tamaños**: sm, md, lg
- **Estados**: loading, disabled
- **Iconos**: leftIcon, rightIcon
- **Full Width**: Opción para ancho completo

**Ejemplo de Uso**:
```typescript
import { Button } from '@/components/ui/Button';

<Button
  variant="primary"
  size="md"
  isLoading={isLoading}
  leftIcon={<Icon />}
  onClick={handleClick}
>
  Enviar
</Button>
```

#### Input Component

**Archivo**: `github_autonomous_agent_ai/frontend/app/components/ui/Input.tsx`

- **Label y Helper Text**: Soporte para labels y texto de ayuda
- **Estados de Error**: Visualización de errores
- **Iconos**: leftIcon, rightIcon
- **Full Width**: Opción para ancho completo

**Ejemplo de Uso**:
```typescript
import { Input } from '@/components/ui/Input';

<Input
  label="Email"
  type="email"
  error={errors.email}
  helperText="Ingresa tu email"
  leftIcon={<MailIcon />}
  fullWidth
/>
```

#### Modal Component

**Archivo**: `github_autonomous_agent_ai/frontend/app/components/ui/Modal.tsx`

- **Tamaños**: sm, md, lg, xl, full
- **Cerrar Automático**: Con overlay click o Escape
- **Prevenir Scroll**: Bloquea scroll del body cuando está abierto
- **Header Personalizable**: Título y botón de cerrar opcionales

**Ejemplo de Uso**:
```typescript
import { Modal } from '@/components/ui/Modal';

<Modal
  isOpen={isOpen}
  onClose={() => setIsOpen(false)}
  title="Confirmar acción"
  size="md"
  closeOnOverlayClick={true}
  closeOnEscape={true}
>
  <p>¿Estás seguro de realizar esta acción?</p>
  <Button onClick={handleConfirm}>Confirmar</Button>
</Modal>
```

#### Select Component

**Archivo**: `github_autonomous_agent_ai/frontend/app/components/ui/Select.tsx`

- **Opciones Tipadas**: Interface para opciones
- **Placeholder**: Soporte para placeholder
- **Estados de Error**: Visualización de errores
- **Helper Text**: Texto de ayuda opcional

**Ejemplo de Uso**:
```typescript
import { Select } from '@/components/ui/Select';

<Select
  label="Status"
  options={[
    { value: 'pending', label: 'Pendiente' },
    { value: 'completed', label: 'Completada' }
  ]}
  placeholder="Selecciona un status"
  error={errors.status}
/>
```

### 3. Utilidad CN Mejorada

**Archivo**: `github_autonomous_agent_ai/frontend/app/utils/cn.ts`

- **Tailwind Merge**: Integración con tailwind-merge para resolver conflictos
- **Clsx**: Combinación de clases condicionales
- **Manejo de Conflictos**: Resuelve automáticamente conflictos de clases de Tailwind

**Ejemplo de Uso**:
```typescript
import { cn } from '@/utils/cn';

<div className={cn(
  'p-4',
  isActive && 'bg-blue-500',
  isDisabled && 'opacity-50',
  className
)}>
  Content
</div>
```

### 4. Servicio de Validación

**Archivo**: `core/services/validation_service.py`

- **Reglas de Validación**: Sistema flexible de reglas
- **Sanitización**: Sanitización automática de datos
- **Validadores Comunes**: Email, URL, GitHub repo
- **Estadísticas**: Tracking de validaciones
- **Mensajes de Error**: Mensajes descriptivos

**Validadores Incluidos**:
- `is_required`: Campo requerido
- `is_min_length`: Longitud mínima
- `is_max_length`: Longitud máxima
- `is_in_range`: Rango numérico
- `validate_email`: Validación de email
- `validate_url`: Validación de URL
- `validate_github_repo`: Validación de repositorio GitHub

**Ejemplo de Uso**:
```python
from core.services import ValidationService
from config.di_setup import get_service
from core.services.validation_service import is_required, is_min_length

validation: ValidationService = get_service("validation_service")

# Agregar regla
validation.add_rule(
    "instruction",
    is_required,
    "La instrucción es requerida",
    code="INSTRUCTION_REQUIRED"
)

# Validar datos
try:
    validated = validation.validate({
        "instruction": "create file",
        "repository": "owner/repo"
    })
    print("Datos válidos:", validated)
except ValidationError as e:
    print(f"Error: {e.message} (campo: {e.field})")
```

### 5. Utilidades de Desarrollo

**Archivo**: `core/utils/dev_tools.py`

- **Timing Decorators**: Medir tiempo de ejecución
- **Logging de Funciones**: Log automático de llamadas
- **Performance Monitor**: Monitor de rendimiento
- **Memory Usage**: Medición de memoria
- **Debug Helpers**: Utilidades para debugging

**Decoradores Disponibles**:
- `@timing_decorator`: Medir tiempo de funciones síncronas
- `@async_timing_decorator`: Medir tiempo de funciones async
- `@log_function_call`: Log automático de llamadas

**Ejemplo de Uso**:
```python
from core.utils.dev_tools import (
    timing_decorator,
    async_timing_decorator,
    measure_time,
    performance_monitor
)

# Decorador para timing
@timing_decorator
def process_data(data):
    # Procesar datos
    return result

# Context manager para timing
with measure_time("database_query"):
    result = db.query(...)

# Performance monitor
performance_monitor.record("api_call", 0.5)
stats = performance_monitor.get_stats("api_call")
print(f"Promedio: {stats['avg']}s")
```

### 6. Rutas de API para Validación

**Archivo**: `api/routes/validation_routes.py`

**Endpoints**:
- `POST /api/v1/validation/validate` - Validar datos
- `POST /api/v1/validation/email` - Validar email
- `POST /api/v1/validation/url` - Validar URL
- `POST /api/v1/validation/github-repo` - Validar repositorio GitHub
- `GET /api/v1/validation/stats` - Estadísticas de validación

## 📊 Impacto y Beneficios

### Frontend
- **Componentes Reutilizables**: Consistencia en toda la aplicación
- **Manejo de Errores**: Errores manejados de forma centralizada
- **Mejor UX**: Componentes accesibles y bien diseñados
- **Desarrollo Rápido**: Componentes listos para usar

### Backend
- **Validación Robusta**: Validación y sanitización automática
- **Debugging Mejorado**: Herramientas para desarrollo y profiling
- **Performance Monitoring**: Monitoreo de rendimiento integrado
- **Calidad de Código**: Validación consistente en toda la aplicación

## 🔄 Integración

### Frontend - Componentes UI

```typescript
// Usar componentes en cualquier parte
import { Button, Input, Modal, Select } from '@/components/ui';
import { useErrorHandler } from '@/hooks/useErrorHandler';

function TaskForm() {
  const { handleError, handleAsyncError } = useErrorHandler();
  const [isOpen, setIsOpen] = useState(false);
  
  const handleSubmit = async (data) => {
    const result = await handleAsyncError(
      async () => await createTask(data),
      'createTask'
    );
    
    if (result) {
      setIsOpen(false);
    }
  };
  
  return (
    <>
      <Modal
        isOpen={isOpen}
        onClose={() => setIsOpen(false)}
        title="Crear Tarea"
      >
        <form onSubmit={handleSubmit}>
          <Input
            label="Repositorio"
            placeholder="owner/repo"
            error={errors.repository}
          />
          <Input
            label="Instrucción"
            placeholder="Instrucción para el agente"
            error={errors.instruction}
          />
          <Select
            label="Prioridad"
            options={priorityOptions}
          />
          <Button type="submit" variant="primary">
            Crear
          </Button>
        </form>
      </Modal>
    </>
  );
}
```

### Backend - Validación

```python
# En cualquier endpoint
from core.services import ValidationService
from config.di_setup import get_service

@router.post("/tasks")
async def create_task(request: CreateTaskRequest):
    validation: ValidationService = get_service("validation_service")
    
    try:
        validated = validation.validate({
            "instruction": request.instruction,
            "repository": request.repository
        })
        
        # Procesar con datos validados
        task = await create_task_internal(validated)
        return task
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=e.message)
```

## 📝 Ejemplos de Uso

### Frontend - Manejo de Errores Completo

```typescript
import { useErrorHandler } from '@/hooks/useErrorHandler';
import { Button } from '@/components/ui/Button';

function DataProcessor() {
  const {
    handleError,
    handleAsyncError,
    errors,
    lastError,
    clearErrors
  } = useErrorHandler({
    showToast: true,
    onError: (error) => {
      // Enviar a servicio de tracking
      trackError(error);
    }
  });
  
  const processData = async () => {
    const result = await handleAsyncError(
      async () => {
        const data = await fetchData();
        if (!data) {
          throw new Error('No se pudo obtener datos');
        }
        return processComplexData(data);
      },
      'processData'
    );
    
    if (result) {
      // Continuar con resultado
    }
  };
  
  return (
    <div>
      {lastError && (
        <div className="error-banner">
          <p>{lastError.message}</p>
          <Button onClick={clearErrors}>Cerrar</Button>
        </div>
      )}
      <Button onClick={processData}>Procesar</Button>
    </div>
  );
}
```

### Backend - Performance Monitoring

```python
from core.utils.dev_tools import (
    performance_monitor,
    measure_time,
    timing_decorator
)

# Usar decorador
@timing_decorator
def expensive_operation():
    # Operación costosa
    pass

# Usar context manager
def process_batch(items):
    with measure_time("batch_processing"):
        for item in items:
            with measure_time(f"process_item_{item.id}"):
                process_item(item)
                performance_monitor.record("item_processed", duration)

# Obtener estadísticas
stats = performance_monitor.get_stats("item_processed")
print(f"Promedio: {stats['avg']:.4f}s")
print(f"P95: {stats['p95']:.4f}s")
```

## 🧪 Testing

### Tests Recomendados

1. **Frontend Hooks**:
   - useErrorHandler: Manejo de diferentes tipos de errores
   - Verificar integración con notificaciones
   - Verificar historial de errores

2. **Frontend Components**:
   - Button: Variantes, estados, iconos
   - Input: Validación, errores, iconos
   - Modal: Apertura/cierre, tamaños
   - Select: Opciones, placeholder, errores

3. **Backend Services**:
   - ValidationService: Reglas, validación, sanitización
   - Dev tools: Timing, logging, performance monitor

## 📚 Documentación Relacionada

- `IMPROVEMENTS_V20.md` - Enhanced Task Store y Analytics
- `components/ui/` - Componentes UI
- `hooks/useErrorHandler.ts` - Manejo de errores
- `core/utils/dev_tools.py` - Utilidades de desarrollo

## 🚀 Próximos Pasos

Posibles mejoras futuras:
- [ ] Más componentes UI (Card, Badge, Tooltip, etc.)
- [ ] Storybook para componentes
- [ ] Tests visuales para componentes
- [ ] Validación en frontend con Zod/Yup
- [ ] Error boundaries en React
- [ ] Performance profiling en frontend
- [ ] Más utilidades de desarrollo

## ✅ Checklist de Implementación

- [x] Hook de manejo de errores
- [x] Componentes UI (Button, Input, Modal, Select)
- [x] Utilidad CN mejorada
- [x] Servicio de validación
- [x] Utilidades de desarrollo
- [x] Rutas de API para validación
- [x] Integración en DI container
- [x] Documentación

---

**Versión**: 21.0  
**Fecha**: 2024-01-01  
**Autor**: GitHub Autonomous Agent Team
