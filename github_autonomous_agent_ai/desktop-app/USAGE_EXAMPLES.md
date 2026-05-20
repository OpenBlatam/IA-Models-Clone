# Ejemplos de Uso - Desktop App

## 🚀 Inicio Rápido

### 1. Instalación y Configuración Inicial

```bash
# 1. Instalar dependencias
cd desktop-app
npm install

# 2. Iniciar en modo desarrollo
npm run dev
```

### 2. Configuración del Backend

1. **Abrir la aplicación**
2. **Ir a "Control" en el sidebar**
3. **Configurar API Key:**
   - Ingresa tu API key del backend
   - Haz clic en "Guardar API Key"
4. **Verificar conexión:**
   - El indicador debe mostrar "Conectado" en verde

### 3. Autenticación con GitHub

1. **Ir a "Agente Continuo"**
2. **En el panel de GitHub Auth:**
   - Opción 1: Usar token predefinido (conexión rápida)
   - Opción 2: Ingresar tu propio token
   - Opción 3: Usar OAuth (abre en navegador)
3. **Una vez autenticado**, verás tu información de usuario

## 📋 Casos de Uso

### Caso 1: Crear un Nuevo Agente

```typescript
// 1. Ir a "Agente Continuo"
// 2. Hacer clic en "+ Crear Agente"
// 3. Completar el formulario:
{
  name: "Mi Agente de Code Review",
  description: "Agente para revisar código automáticamente",
  config: {
    taskType: "code_review",
    frequency: 60, // segundos
    goal: "Revisar todos los PRs automáticamente"
  }
}
// 4. Hacer clic en "Crear Agente"
```

### Caso 2: Gestionar Tareas en Kanban

```typescript
// 1. Ir a "Kanban"
// 2. Ver tareas organizadas por estado:
//    - Pendiente
//    - En Progreso
//    - Completado
//    - Fallido
// 3. Cada tarjeta muestra:
//    - Título
//    - Descripción
//    - Estado (badge de color)
//    - Fecha de creación
//    - Botón para eliminar
```

### Caso 3: Seleccionar Modelo de IA

```typescript
// 1. Ir a "Control"
// 2. En "Configuración de IA":
//    - Seleccionar modelo de la lista
//    - Opciones disponibles:
//      * DeepSeek Chat
//      * Gemini 2.0 Flash
//      * Claude Sonnet 4.5
//      * Grok 4.1
//      * ChatGPT 5
// 3. El modelo se guarda automáticamente
```

### Caso 4: Seleccionar Repositorio GitHub

```typescript
// 1. Autenticarse con GitHub primero
// 2. Usar RepositorySelector component
// 3. Funcionalidades:
//    - Búsqueda en tiempo real
//    - Filtros: Todos, Públicos, Privados
//    - Ordenamiento: Por nombre o más recientes
//    - Muestra: lenguaje, estrellas, forks, tipo
```

## 🔧 Ejemplos de Código

### Usar Hooks Personalizados

```typescript
import { useContinuousAgents } from './hooks/useContinuousAgents';
import { useTasks } from './hooks/useAPI';

// En un componente
function MyComponent() {
  // Hook para agentes
  const {
    agents,
    loading,
    error,
    createAgent,
    updateAgent,
    deleteAgent,
    toggleAgent,
  } = useContinuousAgents({
    autoRefresh: true,
    refreshInterval: 5000,
  });

  // Hook para tareas
  const {
    tasks,
    loading: tasksLoading,
    createTask,
    updateTask,
    deleteTask,
  } = useTasks();

  // Usar las funciones...
}
```

### Usar Componentes UI

```typescript
import {
  Button,
  Card,
  CardHeader,
  CardContent,
  Input,
  Modal,
  Badge,
  StatusBadge,
} from './components';

function MyComponent() {
  return (
    <Card>
      <CardHeader title="Mi Título" />
      <CardContent>
        <Input label="Nombre" placeholder="Ingresa nombre" />
        <Button variant="primary" onClick={handleClick}>
          Enviar
        </Button>
        <StatusBadge status="completed" />
      </CardContent>
    </Card>
  );
}
```

### Usar Utilidades

```typescript
import { formatDate, formatRelativeTime, truncate } from './utils';
import { isValidEmail, isValidUrl } from './utils/validation';

// Formatear fechas
const fecha = formatDate(new Date()); // "15 ene 2024, 14:30"
const relativa = formatRelativeTime(date); // "hace 2 horas"

// Truncar texto
const texto = truncate("Texto muy largo...", 50);

// Validar
const esEmail = isValidEmail("user@example.com");
const esUrl = isValidUrl("https://example.com");
```

### Usar Servicios API

```typescript
import { agentService } from './services/agentService';
import { getAPIClient } from './lib/api-client';

// Crear agente
const nuevoAgente = await agentService.createAgent({
  name: "Mi Agente",
  description: "Descripción",
  config: {
    taskType: "code_review",
    frequency: 60,
    parameters: {},
    goal: "Objetivo del agente",
  },
});

// Obtener tareas
const apiClient = getAPIClient();
const tareas = await apiClient.getTasks();
```

## 🎨 Personalización

### Cambiar Colores y Estilos

Los estilos están en `src/renderer/styles/globals.css` y usan Tailwind CSS.

```css
/* Personalizar colores principales */
:root {
  --primary-color: #3b82f6;
  --success-color: #10b981;
  --danger-color: #ef4444;
}
```

### Agregar Nuevos Componentes

1. Crear componente en `src/renderer/components/`
2. Exportar en `src/renderer/components/index.ts`
3. Usar en las páginas

```typescript
// src/renderer/components/MyComponent.tsx
export const MyComponent = () => {
  return <div>Mi Componente</div>;
};

// src/renderer/components/index.ts
export { MyComponent } from './MyComponent';
```

## 🔍 Debugging

### Ver Logs en Desarrollo

```typescript
// La aplicación abre DevTools automáticamente en desarrollo
// Ver logs en la consola del navegador
console.log('Mi log');
```

### Verificar Estado de la Aplicación

```typescript
// Verificar localStorage
localStorage.getItem('api_key');
localStorage.getItem('selected_ai_model');
localStorage.getItem('github_access_token');
```

## 📱 Atajos de Teclado

- `Esc` - Cerrar modales
- `Ctrl/Cmd + R` - Recargar aplicación (en desarrollo)

## 🐛 Solución de Problemas Comunes

### La aplicación no se conecta al backend

1. Verificar que el backend esté corriendo en `http://localhost:8030`
2. Verificar que la API key esté configurada correctamente
3. Revisar la consola para errores

### Los agentes no se cargan

1. Verificar autenticación de GitHub
2. Verificar que el backend tenga los endpoints correctos
3. Revisar la consola para errores de red

### Los estilos no se aplican

1. Verificar que Tailwind esté configurado correctamente
2. Verificar que `globals.css` esté importado
3. Limpiar cache: `npm run build`

## 📚 Recursos Adicionales

- [Documentación de Electron](https://www.electronjs.org/docs)
- [Documentación de React](https://react.dev)
- [Documentación de Tailwind CSS](https://tailwindcss.com/docs)
- [Documentación de TypeScript](https://www.typescriptlang.org/docs)


