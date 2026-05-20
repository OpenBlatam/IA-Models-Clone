# Componentes de Chat Mejorados

## Componentes Disponibles

### 1. ChatInterface
Componente principal de chat con todas las funcionalidades avanzadas.

**Características:**
- ✅ Modo oscuro/claro
- ✅ Búsqueda en mensajes
- ✅ Exportar conversación
- ✅ Soporte Markdown
- ✅ Auto-scroll inteligente
- ✅ Manejo de errores

### 2. MessageBubble
Componente reutilizable para mensajes individuales.

**Características:**
- ✅ Editar mensajes
- ✅ Copiar mensajes
- ✅ Like/Dislike
- ✅ Eliminar mensajes
- ✅ Renderizado Markdown

### 3. ChatSidebar
Barra lateral para gestionar múltiples conversaciones.

**Características:**
- ✅ Lista de conversaciones
- ✅ Búsqueda de conversaciones
- ✅ Renombrar conversaciones
- ✅ Fijar conversaciones
- ✅ Crear carpetas
- ✅ Modo colapsable

### 4. ChatSettings
Panel de configuración completo.

**Características:**
- ✅ Configuración de tema
- ✅ Tamaño de fuente
- ✅ Opciones de mensajes
- ✅ Notificaciones
- ✅ Idioma
- ✅ Guardado automático

## Instalación

```bash
npm install react-markdown remark-gfm lucide-react
```

## Uso Básico

```tsx
import { ChatInterface } from './components/ChatInterface';
import { ChatSidebar } from './components/ChatSidebar';
import { ChatSettings } from './components/ChatSettings';

function App() {
  const [settings, setSettings] = useState(null);
  const [showSettings, setShowSettings] = useState(false);

  return (
    <div className="flex h-screen">
      <ChatSidebar
        conversations={conversations}
        onSelectConversation={handleSelect}
        onNewConversation={handleNew}
      />
      <ChatInterface
        onSendMessage={handleSend}
        enableMarkdown={settings?.enableMarkdown}
        theme={settings?.theme}
      />
      <ChatSettings
        isOpen={showSettings}
        onClose={() => setShowSettings(false)}
        onSave={setSettings}
      />
    </div>
  );
}
```

## Mejoras Futuras

- [ ] Soporte para archivos adjuntos
- [ ] Respuestas con voz
- [ ] Modo de pantalla completa
- [ ] Atajos de teclado
- [ ] Temas personalizados
- [ ] Integración con servicios externos


