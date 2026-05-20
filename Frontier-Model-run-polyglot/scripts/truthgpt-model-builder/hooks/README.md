# Hooks Personalizados

## useChat

Hook para gestionar el estado y lógica del chat.

```tsx
import { useChat } from './hooks/useChat';

function ChatComponent() {
  const {
    messages,
    input,
    isLoading,
    sendMessage,
    clearMessages,
    retryLastMessage,
  } = useChat({
    onSendMessage: async (message) => {
      const response = await fetch('/api/chat', {
        method: 'POST',
        body: JSON.stringify({ message }),
      });
      const data = await response.json();
      return data.response;
    },
    autoSave: true,
    maxMessages: 1000,
  });

  return (
    <div>
      {/* UI del chat */}
    </div>
  );
}
```

## useConversations

Hook para gestionar múltiples conversaciones.

```tsx
import { useConversations } from './hooks/useConversations';

function ConversationsComponent() {
  const {
    conversations,
    activeConversationId,
    createConversation,
    deleteConversation,
    renameConversation,
  } = useConversations({
    autoSave: true,
  });

  return (
    <div>
      {/* UI de conversaciones */}
    </div>
  );
}
```

## Utilidades

### truthgpt-service.ts

Servicio mejorado con:
- ✅ Retry automático
- ✅ Cache
- ✅ Streaming
- ✅ Timeout configurable
- ✅ Manejo de errores

### chatUtils.ts

Utilidades para:
- ✅ Formatear timestamps
- ✅ Exportar conversaciones
- ✅ Buscar en mensajes
- ✅ Validar mensajes
- ✅ Detectar idioma
- ✅ Debounce/Throttle


