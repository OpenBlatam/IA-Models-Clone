# Ejemplo de Integración - Bulk Chat API

## 📦 Hook Personalizado: `useBulkChat`

Ya está creado un hook personalizado en tu proyecto para facilitar la integración:

**Ubicación:** `lib/useBulkChat.ts`

### Uso Básico

```tsx
import { useBulkChat } from '@/lib/useBulkChat';

function MyChatComponent() {
  const {
    sessionId,
    messages,
    sendMessage,
    isConnected,
    isLoading,
    pause,
    resume,
    stop,
  } = useBulkChat({
    apiUrl: 'http://localhost:8006',
    autoConnect: true,
    autoContinue: true,
    enableWebSocket: true,
  });

  return (
    <div>
      {/* Tu UI aquí */}
    </div>
  );
}
```

### Opciones del Hook

```typescript
interface UseBulkChatOptions {
  apiUrl?: string;                    // URL de la API (default: http://localhost:8006)
  autoConnect?: boolean;              // Crear sesión automáticamente (default: false)
  initialMessage?: string;            // Mensaje inicial para la sesión
  autoContinue?: boolean;             // Continuar automáticamente (default: true)
  onMessage?: (message) => void;     // Callback cuando llega un mensaje
  onError?: (error) => void;         // Callback de errores
  onSessionCreated?: (session) => void; // Callback cuando se crea sesión
  enableWebSocket?: boolean;          // Usar WebSocket (default: true)
}
```

### Retorno del Hook

```typescript
interface UseBulkChatReturn {
  // Estado
  sessionId: string | null;
  session: BulkChatSession | null;
  messages: BulkChatMessage[];
  isConnected: boolean;
  isLoading: boolean;
  isPaused: boolean;
  error: Error | null;

  // Métodos REST
  createSession: (initialMessage?: string) => Promise<void>;
  sendMessage: (message: string) => Promise<void>;
  getMessages: (limit?: number) => Promise<BulkChatMessage[]>;
  pause: (reason?: string) => Promise<void>;
  resume: () => Promise<void>;
  stop: () => Promise<void>;
  refreshMessages: () => Promise<void>;

  // WebSocket
  connectWebSocket: () => void;
  disconnectWebSocket: () => void;

  // Utilidades
  clearError: () => void;
}
```

---

## 🎯 Ejemplo Completo

Ver el archivo `lib/bulkChatExample.tsx` para un ejemplo completo de componente.

---

## 🔧 Integración en ChatInterface.tsx

Para integrar en tu `ChatInterface.tsx` existente:

### 1. Importar el hook

```tsx
import { useBulkChat } from '@/lib/useBulkChat';
```

### 2. Agregar el hook al componente

```tsx
export default function ChatInterface() {
  // ... tus estados existentes ...
  
  // Agregar el hook de Bulk Chat
  const bulkChat = useBulkChat({
    apiUrl: process.env.NEXT_PUBLIC_BULK_CHAT_API_URL || 'http://localhost:8006',
    autoConnect: false,
    enableWebSocket: true,
    onMessage: (message) => {
      // Agregar mensaje al store existente
      addMessage({
        id: message.id,
        role: message.role,
        content: message.content,
        timestamp: new Date(message.timestamp),
      });
    },
  });

  // ... resto del componente ...
}
```

### 3. Usar los métodos

```tsx
// Enviar mensaje
const handleSendMessage = async () => {
  if (bulkChat.sessionId) {
    await bulkChat.sendMessage(input);
  } else {
    await bulkChat.createSession(input);
  }
  setInput('');
};

// Controlar sesión
const handlePause = () => bulkChat.pause();
const handleResume = () => bulkChat.resume();
const handleStop = () => bulkChat.stop();
```

---

## 🌐 Variables de Entorno

Agregar a tu `.env.local`:

```env
NEXT_PUBLIC_BULK_CHAT_API_URL=http://localhost:8006
```

---

## 📝 Características

✅ **REST API** - Fallback automático si WebSocket falla  
✅ **WebSocket** - Streaming en tiempo real  
✅ **Auto-reconexión** - Se reconecta automáticamente  
✅ **Polling** - Usa polling si WebSocket no está disponible  
✅ **TypeScript** - Completamente tipado  
✅ **Error Handling** - Manejo de errores robusto  
✅ **Toast Notifications** - Notificaciones integradas  

---

## 🚀 Iniciar el Servidor

```bash
cd bulk_chat
python -m bulk_chat.main --port 8006
```

---

## 📚 Más Documentación

- `FRONTEND_INTEGRATION.md` - Guía completa de integración
- `lib/useBulkChat.ts` - Código del hook
- `lib/bulkChatExample.tsx` - Ejemplo completo

---

¡Listo para usar! 🎉











