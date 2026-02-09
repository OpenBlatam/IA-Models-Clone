# Mejoras del Frontend - Versión 7

## 📋 Resumen

Esta versión incluye hooks para APIs del navegador avanzadas (WebSocket, Speech, Battery, Network) y componentes de UI avanzados (NotificationCenter, CommandPalette).

## ✨ Nuevas Funcionalidades

### 1. Hooks de APIs Avanzadas

#### `useWebSocket`
Hook para conexiones WebSocket con reconexión automática.

```typescript
const { status, lastMessage, send, close, reconnect } = useWebSocket({
  url: 'ws://localhost:8006/ws',
  onOpen: () => console.log('Connected'),
  onMessage: (event) => console.log('Message:', event.data),
  onError: (event) => console.error('Error:', event),
  reconnect: true,
  reconnectInterval: 3000,
  reconnectAttempts: 5,
});

// Send message
send(JSON.stringify({ type: 'ping' }));

// Close connection
close();
```

**Características:**
- Reconexión automática
- Estados de conexión
- Manejo de errores
- Último mensaje recibido

#### `useSpeechRecognition`
Hook para reconocimiento de voz (speech-to-text).

```typescript
const {
  isListening,
  transcript,
  error,
  start,
  stop,
  abort,
  isSupported,
} = useSpeechRecognition({
  continuous: true,
  interimResults: true,
  lang: 'es-ES',
  onResult: (text, isFinal) => {
    console.log('Transcript:', text, isFinal);
  },
  onError: (error) => {
    console.error('Recognition error:', error);
  },
});

// Start listening
start();

// Stop listening
stop();
```

**Características:**
- Reconocimiento continuo
- Resultados intermedios
- Múltiples idiomas
- Manejo de errores

#### `useSpeechSynthesis`
Hook para síntesis de voz (text-to-speech).

```typescript
const {
  speak,
  cancel,
  pause,
  resume,
  isSpeaking,
  voices,
  isSupported,
} = useSpeechSynthesis({
  voice: voices.find(v => v.lang === 'es-ES'),
  pitch: 1.2,
  rate: 1.0,
  volume: 1.0,
  lang: 'es-ES',
});

// Speak text
speak('Hello, this is a test');

// Pause/Resume
pause();
resume();

// Cancel
cancel();
```

**Características:**
- Control de voz (pitch, rate, volume)
- Lista de voces disponibles
- Pausa y reanudación
- Cancelación

#### `useBattery`
Hook para obtener información de la batería.

```typescript
const { charging, chargingTime, dischargingTime, level, supported } = useBattery();

if (!supported) return <div>Battery API not supported</div>;

return (
  <div>
    <p>Level: {(level * 100).toFixed(0)}%</p>
    <p>Charging: {charging ? 'Yes' : 'No'}</p>
    {charging && chargingTime && (
      <p>Time to full: {chargingTime}s</p>
    )}
  </div>
);
```

**Características:**
- Nivel de batería
- Estado de carga
- Tiempo estimado
- Actualización automática

#### `useNetworkStatus`
Hook para obtener información detallada de la red.

```typescript
const { online, effectiveType, downlink, rtt, saveData } = useNetworkStatus();

return (
  <div>
    <p>Status: {online ? 'Online' : 'Offline'}</p>
    <p>Connection: {effectiveType}</p>
    <p>Speed: {downlink} Mbps</p>
    <p>Latency: {rtt}ms</p>
    {saveData && <p>Data saver mode enabled</p>}
  </div>
);
```

**Características:**
- Estado online/offline
- Tipo de conexión (2g, 3g, 4g)
- Velocidad de descarga
- Latencia (RTT)
- Modo ahorro de datos

### 2. Componentes Avanzados de UI

#### `NotificationCenter`
Centro de notificaciones completo con gestión de estado.

```tsx
<NotificationCenter
  notifications={notifications}
  onMarkAsRead={(id) => markAsRead(id)}
  onMarkAllAsRead={() => markAllAsRead()}
  onClear={() => clearNotifications()}
  maxNotifications={10}
/>
```

**Características:**
- Contador de no leídas
- Marcar como leída individual
- Marcar todas como leídas
- Limpiar notificaciones
- Timestamps relativos
- Acciones en notificaciones
- Categorización por tipo

#### `CommandPalette`
Paleta de comandos estilo VS Code con búsqueda.

```tsx
<CommandPalette
  commands={[
    {
      id: 'new-analysis',
      label: 'New Analysis',
      description: 'Start a new skin analysis',
      icon: <Plus />,
      keywords: ['new', 'analysis', 'start'],
      category: 'Actions',
      action: () => router.push('/'),
    },
    {
      id: 'settings',
      label: 'Settings',
      description: 'Open settings',
      icon: <Settings />,
      keywords: ['settings', 'preferences'],
      category: 'Navigation',
      action: () => router.push('/settings'),
    },
  ]}
  placeholder="Type a command..."
/>
```

**Características:**
- Búsqueda en tiempo real
- Navegación con teclado
- Categorización
- Atajo de teclado (Cmd/Ctrl + K)
- Iconos y descripciones
- Keywords para búsqueda

## 🎯 Ejemplos de Uso

### WebSocket para Actualizaciones en Tiempo Real

```tsx
import { useWebSocket } from '@/lib/hooks';

function RealTimeUpdates() {
  const { status, lastMessage, send } = useWebSocket({
    url: 'ws://localhost:8006/updates',
    onMessage: (event) => {
      const data = JSON.parse(event.data);
      // Update UI with real-time data
    },
  });

  return (
    <div>
      <p>Status: {status}</p>
      {lastMessage && (
        <div>Last update: {lastMessage.data}</div>
      )}
    </div>
  );
}
```

### Comandos de Voz

```tsx
import { useSpeechRecognition } from '@/lib/hooks';

function VoiceCommands() {
  const { isListening, transcript, start, stop } = useSpeechRecognition({
    onResult: (text) => {
      if (text.includes('analyze')) {
        startAnalysis();
      }
    },
  });

  return (
    <div>
      <button onClick={isListening ? stop : start}>
        {isListening ? 'Stop Listening' : 'Start Voice Command'}
      </button>
      {transcript && <p>You said: {transcript}</p>}
    </div>
  );
}
```

### Lectura de Texto

```tsx
import { useSpeechSynthesis } from '@/lib/hooks';

function TextReader({ text }: { text: string }) {
  const { speak, isSpeaking, cancel } = useSpeechSynthesis();

  return (
    <div>
      <button onClick={() => speak(text)} disabled={isSpeaking}>
        {isSpeaking ? 'Reading...' : 'Read Aloud'}
      </button>
      {isSpeaking && (
        <button onClick={cancel}>Stop</button>
      )}
    </div>
  );
}
```

### Optimización Basada en Batería

```tsx
import { useBattery } from '@/lib/hooks';

function AdaptiveUI() {
  const { level, charging } = useBattery();

  // Reduce animations when battery is low
  const shouldReduceAnimations = level < 0.2 && !charging;

  return (
    <div className={shouldReduceAnimations ? 'no-animations' : ''}>
      {/* UI content */}
    </div>
  );
}
```

### Optimización Basada en Red

```tsx
import { useNetworkStatus } from '@/lib/hooks';

function AdaptiveLoading() {
  const { effectiveType, saveData } = useNetworkStatus();

  // Load lower quality images on slow connections
  const imageQuality = effectiveType === '2g' || saveData ? 'low' : 'high';

  return (
    <img
      src={imageQuality === 'high' ? highResImage : lowResImage}
      alt="Analysis"
    />
  );
}
```

### Notification Center Completo

```tsx
import { NotificationCenter } from '@/lib/components';
import { useState } from 'react';

function App() {
  const [notifications, setNotifications] = useState([
    {
      id: '1',
      title: 'Analysis Complete',
      message: 'Your skin analysis is ready',
      type: 'success' as const,
      timestamp: new Date(),
      read: false,
      action: {
        label: 'View',
        onClick: () => router.push('/results'),
      },
    },
  ]);

  return (
    <Header>
      <NotificationCenter
        notifications={notifications}
        onMarkAsRead={(id) => {
          setNotifications(prev =>
            prev.map(n => n.id === id ? { ...n, read: true } : n)
          );
        }}
        onMarkAllAsRead={() => {
          setNotifications(prev => prev.map(n => ({ ...n, read: true })));
        }}
        onClear={() => setNotifications([])}
      />
    </Header>
  );
}
```

### Command Palette Global

```tsx
import { CommandPalette } from '@/lib/components';
import { useRouter } from 'next/navigation';

function App() {
  const router = useRouter();
  const [isOpen, setIsOpen] = useState(false);

  const commands = [
    {
      id: 'new-analysis',
      label: 'New Analysis',
      description: 'Start a new skin analysis',
      category: 'Actions',
      action: () => {
        router.push('/');
        setIsOpen(false);
      },
    },
    {
      id: 'dashboard',
      label: 'Go to Dashboard',
      description: 'View your dashboard',
      category: 'Navigation',
      action: () => {
        router.push('/dashboard');
        setIsOpen(false);
      },
    },
  ];

  return (
    <>
      <CommandPalette
        commands={commands}
        isOpen={isOpen}
        onClose={() => setIsOpen(false)}
      />
      {/* Rest of app */}
    </>
  );
}
```

## 📦 Archivos Creados

**Hooks:**
- `lib/hooks/useWebSocket.ts`
- `lib/hooks/useSpeechRecognition.ts`
- `lib/hooks/useSpeechSynthesis.ts`
- `lib/hooks/useBattery.ts`
- `lib/hooks/useNetworkStatus.ts`

**Componentes:**
- `lib/components/NotificationCenter.tsx`
- `lib/components/CommandPalette.tsx`

## 🎨 Características Destacadas

### WebSocket
- ✅ Reconexión automática
- ✅ Estados de conexión
- ✅ Manejo de errores
- ✅ Último mensaje

### Speech APIs
- ✅ Reconocimiento de voz
- ✅ Síntesis de voz
- ✅ Control completo
- ✅ Múltiples idiomas

### Browser APIs
- ✅ Información de batería
- ✅ Estado de red detallado
- ✅ Optimizaciones adaptativas

### UI Components
- ✅ Notification center completo
- ✅ Command palette avanzado
- ✅ Búsqueda y navegación

## 🚀 Beneficios

1. **Funcionalidades Avanzadas:**
   - WebSocket para tiempo real
   - Comandos de voz
   - Lectura de texto
   - Optimizaciones adaptativas

2. **Mejor UX:**
   - Notificaciones centralizadas
   - Comandos rápidos
   - Optimización automática

3. **Accesibilidad:**
   - Comandos de voz
   - Lectura de texto
   - Navegación por teclado

4. **Rendimiento:**
   - Optimización basada en batería
   - Optimización basada en red
   - Modo ahorro de datos

## 📚 Documentación

- Ver `lib/hooks/index.ts` para todos los hooks
- Ver `lib/components/index.ts` para todos los componentes

## 🔄 Resumen de Versiones

### Versión 2-6
- Hooks básicos y avanzados
- Componentes de UI
- Utilidades fundamentales

### Versión 7
- Hooks de APIs avanzadas (WebSocket, Speech, Battery, Network)
- Componentes avanzados (NotificationCenter, CommandPalette)
- Funcionalidades de accesibilidad y optimización

## 📊 Estadísticas Totales

- **Total de hooks:** 33
- **Total de componentes:** 21
- **Total de utilidades:** 9 módulos
- **Archivos creados:** 70+
- **Líneas de código:** 6000+



