# ChatInterface - Componente Mejorado

## Mejoras Implementadas

### 🎨 Interfaz y UX
- ✅ **Modo oscuro/claro** con detección automática del sistema
- ✅ **Auto-scroll inteligente** al recibir nuevos mensajes
- ✅ **Búsqueda en mensajes** con resaltado
- ✅ **Exportar conversación** a archivo de texto
- ✅ **Copiar mensajes** individuales
- ✅ **Limpiar conversación** con confirmación
- ✅ **Contador de mensajes** en el header

### 💬 Funcionalidades de Chat
- ✅ **Soporte para Markdown** con renderizado avanzado
- ✅ **Indicadores de escritura** animados
- ✅ **Manejo de errores** mejorado con mensajes claros
- ✅ **Límite de mensajes** configurable
- ✅ **Historial de timestamps** en cada mensaje

### ⌨️ Mejoras de Input
- ✅ **Textarea auto-expandible** que crece con el contenido
- ✅ **Enter para enviar**, Shift+Enter para nueva línea
- ✅ **Placeholder personalizable**
- ✅ **Deshabilitar botón** cuando está cargando o vacío

### 🎯 Características Avanzadas
- ✅ **Filtrado de mensajes** por búsqueda
- ✅ **Tema configurable** (light/dark/auto)
- ✅ **Transiciones suaves** en cambios de tema
- ✅ **Responsive design** optimizado

## Uso

```tsx
import { ChatInterface } from './components/ChatInterface';

function App() {
  const handleSendMessage = async (message: string) => {
    // Tu lógica para enviar mensaje
    const response = await fetch('/api/chat', {
      method: 'POST',
      body: JSON.stringify({ message }),
    });
    const data = await response.json();
    return data.response;
  };

  return (
    <ChatInterface
      onSendMessage={handleSendMessage}
      title="TruthGPT Chat"
      placeholder="Escribe tu pregunta..."
      enableMarkdown={true}
      enableSearch={true}
      enableExport={true}
      theme="auto"
      maxMessages={1000}
    />
  );
}
```

## Props

| Prop | Tipo | Default | Descripción |
|------|------|---------|-------------|
| `onSendMessage` | `(message: string) => Promise<string>` | - | Función para enviar mensajes |
| `initialMessages` | `Message[]` | `[]` | Mensajes iniciales |
| `placeholder` | `string` | `"Escribe tu mensaje..."` | Placeholder del input |
| `title` | `string` | `"Chat"` | Título del chat |
| `maxMessages` | `number` | `1000` | Límite de mensajes |
| `enableMarkdown` | `boolean` | `true` | Habilitar renderizado Markdown |
| `enableSearch` | `boolean` | `true` | Habilitar búsqueda |
| `enableExport` | `boolean` | `true` | Habilitar exportar |
| `theme` | `'light' \| 'dark' \| 'auto'` | `'auto'` | Tema del componente |

## Instalación de Dependencias

```bash
npm install react-markdown remark-gfm lucide-react
```

## Características Técnicas

- **TypeScript** completo con tipos
- **React Hooks** (useState, useEffect, useCallback, useMemo, useRef)
- **Tailwind CSS** para estilos
- **React Markdown** para renderizado de markdown
- **Lucide React** para iconos
- **Accesibilidad** mejorada

## Mejoras Futuras Sugeridas

- [ ] Soporte para archivos adjuntos
- [ ] Editar mensajes enviados
- [ ] Respuestas con voz
- [ ] Notificaciones de escritura en tiempo real
- [ ] Soporte para múltiples conversaciones
- [ ] Integración con servicios de traducción
- [ ] Modo de pantalla completa
- [ ] Atajos de teclado personalizables


