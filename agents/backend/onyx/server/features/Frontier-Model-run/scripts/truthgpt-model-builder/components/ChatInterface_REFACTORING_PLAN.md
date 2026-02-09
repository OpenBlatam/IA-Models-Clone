# Plan de Refactorización - ChatInterface.tsx

## 🚨 Problema Identificado

**Archivo actual:** `ChatInterface.tsx`
- **Líneas:** 12,669
- **Hooks:** 1,127+ (useState, useEffect, useCallback, useMemo)
- **Estados:** 100+ estados diferentes
- **Responsabilidades:** Múltiples (UI, lógica de negocio, estado, efectos secundarios)

**Problemas:**
- ❌ Imposible de mantener
- ❌ Difícil de testear
- ❌ Performance degradada (re-renders innecesarios)
- ❌ Violación de Single Responsibility Principle
- ❌ Código duplicado
- ❌ Dependencias circulares potenciales

---

## 🎯 Objetivo de Refactorización

Dividir el componente en:
- ✅ Componentes más pequeños y enfocados
- ✅ Custom hooks para lógica reutilizable
- ✅ Context providers para estado compartido
- ✅ Utilidades separadas
- ✅ Mejor separación de responsabilidades

---

## 📐 Arquitectura Propuesta

### Estructura de Directorios

```
components/
├── ChatInterface/
│   ├── index.tsx                    # Componente principal (orquestador)
│   ├── ChatInterface.tsx            # UI principal simplificada
│   │
│   ├── hooks/                       # Custom hooks
│   │   ├── useChatState.ts          # Estado principal del chat
│   │   ├── useMessageManagement.ts   # Gestión de mensajes
│   │   ├── useSearch.ts             # Funcionalidad de búsqueda
│   │   ├── useFilters.ts            # Filtros y vistas
│   │   ├── useVoiceFeatures.ts      # Funcionalidad de voz
│   │   ├── useExportImport.ts       # Exportar/importar
│   │   ├── useCollaboration.ts      # Colaboración
│   │   ├── useAccessibility.ts      # Accesibilidad
│   │   ├── usePerformance.ts        # Optimización de performance
│   │   └── useNotifications.ts      # Notificaciones
│   │
│   ├── contexts/                    # Context providers
│   │   ├── ChatContext.tsx          # Estado global del chat
│   │   ├── SettingsContext.tsx      # Configuraciones
│   │   ├── ThemeContext.tsx         # Tema y personalización
│   │   └── AccessibilityContext.tsx # Accesibilidad
│   │
│   ├── components/                  # Sub-componentes
│   │   ├── MessageList/            # Lista de mensajes
│   │   │   ├── MessageList.tsx
│   │   │   ├── MessageItem.tsx
│   │   │   ├── MessageActions.tsx
│   │   │   └── MessageContent.tsx
│   │   │
│   │   ├── InputArea/              # Área de entrada
│   │   │   ├── InputArea.tsx
│   │   │   ├── TextInput.tsx
│   │   │   ├── VoiceInput.tsx
│   │   │   └── QuickActions.tsx
│   │   │
│   │   ├── Sidebar/                # Barra lateral
│   │   │   ├── Sidebar.tsx
│   │   │   ├── HistoryPanel.tsx
│   │   │   ├── SettingsPanel.tsx
│   │   │   └── BookmarksPanel.tsx
│   │   │
│   │   ├── Toolbar/                # Barra de herramientas
│   │   │   ├── Toolbar.tsx
│   │   │   ├── SearchBar.tsx
│   │   │   ├── FilterBar.tsx
│   │   │   └── ViewControls.tsx
│   │   │
│   │   └── Modals/                 # Modales y diálogos
│   │       ├── SettingsModal.tsx
│   │       ├── ExportModal.tsx
│   │       ├── ShareModal.tsx
│   │       └── CommandPalette.tsx
│   │
│   ├── utils/                      # Utilidades
│   │   ├── messageUtils.ts         # Utilidades de mensajes
│   │   ├── searchUtils.ts          # Utilidades de búsqueda
│   │   ├── exportUtils.ts          # Utilidades de exportación
│   │   ├── storageUtils.ts          # Utilidades de almacenamiento
│   │   └── validationUtils.ts      # Utilidades de validación
│   │
│   └── types/                      # Tipos TypeScript
│       ├── chat.types.ts
│       ├── message.types.ts
│       ├── settings.types.ts
│       └── state.types.ts
```

---

## 🔧 Fase 1: Extracción de Custom Hooks

### 1.1 useChatState.ts

**Responsabilidad:** Estado principal del chat

```typescript
// hooks/useChatState.ts
export function useChatState() {
  const [messages, setMessages] = useState<Message[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [input, setInput] = useState('')
  const [error, setError] = useState<string | null>(null)
  
  // Lógica de estado centralizada
  // ...
  
  return {
    messages,
    setMessages,
    isLoading,
    setIsLoading,
    input,
    setInput,
    error,
    setError,
    // ... más estado relacionado
  }
}
```

**Estados a extraer:**
- `messages`, `isLoading`, `input`
- `error`, `validation`, `previewSpec`
- Estados básicos del chat

### 1.2 useMessageManagement.ts

**Responsabilidad:** Gestión de mensajes (CRUD, favoritos, etc.)

```typescript
// hooks/useMessageManagement.ts
export function useMessageManagement() {
  const [favoriteMessages, setFavoriteMessages] = useState<Set<string>>(new Set())
  const [pinnedMessages, setPinnedMessages] = useState<Set<string>>(new Set())
  const [archivedMessages, setArchivedMessages] = useState<Set<string>>(new Set())
  const [selectedMessages, setSelectedMessages] = useState<Set<string>>(new Set())
  
  const toggleFavorite = useCallback((messageId: string) => {
    // Lógica
  }, [])
  
  // ... más funciones de gestión
  
  return {
    favoriteMessages,
    pinnedMessages,
    archivedMessages,
    selectedMessages,
    toggleFavorite,
    // ... más funciones
  }
}
```

**Estados a extraer:**
- `favoriteMessages`, `pinnedMessages`, `archivedMessages`
- `selectedMessages`, `messageTags`, `messageNotes`
- `messageReactions`, `messageBookmarks`

### 1.3 useSearch.ts

**Responsabilidad:** Funcionalidad de búsqueda

```typescript
// hooks/useSearch.ts
export function useSearch(messages: Message[]) {
  const [searchQuery, setSearchQuery] = useState('')
  const [currentSearchIndex, setCurrentSearchIndex] = useState(-1)
  const [filteredMessages, setFilteredMessages] = useState<Message[]>([])
  const [searchFilters, setSearchFilters] = useState<SearchFilters>({})
  
  // Lógica de búsqueda y filtrado
  // ...
  
  return {
    searchQuery,
    setSearchQuery,
    currentSearchIndex,
    setCurrentSearchIndex,
    filteredMessages,
    searchFilters,
    setSearchFilters,
    // ... más funciones de búsqueda
  }
}
```

**Estados a extraer:**
- `searchQuery`, `currentSearchIndex`, `filteredMessages`
- `searchFilters`, `advancedSearch`, `searchIndex`

### 1.4 useFilters.ts

**Responsabilidad:** Filtros y vistas

```typescript
// hooks/useFilters.ts
export function useFilters() {
  const [filterRole, setFilterRole] = useState<'all' | 'user' | 'assistant'>('all')
  const [viewMode, setViewMode] = useState<'normal' | 'compact' | 'comfortable'>('normal')
  const [groupingMode, setGroupingMode] = useState<'none' | 'time' | 'topic' | 'role'>('none')
  const [showFilters, setShowFilters] = useState(false)
  
  // Lógica de filtrado
  // ...
  
  return {
    filterRole,
    setFilterRole,
    viewMode,
    setViewMode,
    groupingMode,
    setGroupingMode,
    showFilters,
    setShowFilters,
    // ... más funciones de filtrado
  }
}
```

**Estados a extraer:**
- `filterRole`, `viewMode`, `groupingMode`
- `showFilters`, `collapsedMessages`, `compactMode`

### 1.5 useVoiceFeatures.ts

**Responsabilidad:** Funcionalidad de voz

```typescript
// hooks/useVoiceFeatures.ts
export function useVoiceFeatures() {
  const [voiceInputEnabled, setVoiceInputEnabled] = useState(false)
  const [voiceOutputEnabled, setVoiceOutputEnabled] = useState(false)
  const [isRecording, setIsRecording] = useState(false)
  const [dictationMode, setDictationMode] = useState(false)
  
  // Lógica de voz
  // ...
  
  return {
    voiceInputEnabled,
    setVoiceInputEnabled,
    voiceOutputEnabled,
    setVoiceOutputEnabled,
    isRecording,
    setIsRecording,
    dictationMode,
    setDictationMode,
    // ... más funciones de voz
  }
}
```

**Estados a extraer:**
- `voiceInputEnabled`, `voiceOutputEnabled`, `isRecording`
- `dictationMode`, `audioRecording`, `videoRecording`

### 1.6 useExportImport.ts

**Responsabilidad:** Exportar e importar

```typescript
// hooks/useExportImport.ts
export function useExportImport() {
  const [exportFormats, setExportFormats] = useState<Set<string>>(new Set(['json', 'txt', 'md']))
  const [showExportMenu, setShowExportMenu] = useState(false)
  const [importEnabled, setImportEnabled] = useState(true)
  
  const exportMessages = useCallback((format: string) => {
    // Lógica de exportación
  }, [])
  
  const importMessages = useCallback((file: File) => {
    // Lógica de importación
  }, [])
  
  return {
    exportFormats,
    setExportFormats,
    showExportMenu,
    setShowExportMenu,
    importEnabled,
    setImportEnabled,
    exportMessages,
    importMessages,
    // ... más funciones
  }
}
```

**Estados a extraer:**
- `exportFormats`, `showExportMenu`, `importEnabled`
- `exportTemplates`, `messageExportAdvanced`

### 1.7 useCollaboration.ts

**Responsabilidad:** Colaboración

```typescript
// hooks/useCollaboration.ts
export function useCollaboration() {
  const [collaborationMode, setCollaborationMode] = useState(false)
  const [collaborators, setCollaborators] = useState<string[]>([])
  const [messageSharing, setMessageSharing] = useState<Map<string, string[]>>(new Map())
  
  // Lógica de colaboración
  // ...
  
  return {
    collaborationMode,
    setCollaborationMode,
    collaborators,
    setCollaborators,
    messageSharing,
    setMessageSharing,
    // ... más funciones
  }
}
```

**Estados a extraer:**
- `collaborationMode`, `collaborators`, `messageSharing`
- `showShareMenu`, `shareTarget`

### 1.8 useAccessibility.ts

**Responsabilidad:** Accesibilidad

```typescript
// hooks/useAccessibility.ts
export function useAccessibility() {
  const [accessibilityMode, setAccessibilityMode] = useState(false)
  const [screenReader, setScreenReader] = useState(false)
  const [highContrast, setHighContrast] = useState(false)
  const [keyboardNavigation, setKeyboardNavigation] = useState(true)
  const [fontSize, setFontSize] = useState<'small' | 'medium' | 'large'>('medium')
  
  // Lógica de accesibilidad
  // ...
  
  return {
    accessibilityMode,
    setAccessibilityMode,
    screenReader,
    setScreenReader,
    highContrast,
    setHighContrast,
    keyboardNavigation,
    setKeyboardNavigation,
    fontSize,
    setFontSize,
    // ... más funciones
  }
}
```

**Estados a extraer:**
- `accessibilityMode`, `screenReader`, `highContrast`
- `keyboardNavigation`, `fontSize`, `accessibilityFeatures`

### 1.9 usePerformance.ts

**Responsabilidad:** Optimización de performance

```typescript
// hooks/usePerformance.ts
export function usePerformance() {
  const [showPerformance, setShowPerformance] = useState(false)
  const [performanceMetrics, setPerformanceMetrics] = useState<Map<string, number>>(new Map())
  const [messageCache, setMessageCache] = useState<Map<string, any>>(new Map())
  const [cacheEnabled, setCacheEnabled] = useState(true)
  
  // Lógica de performance
  // ...
  
  return {
    showPerformance,
    setShowPerformance,
    performanceMetrics,
    setPerformanceMetrics,
    messageCache,
    setMessageCache,
    cacheEnabled,
    setCacheEnabled,
    // ... más funciones
  }
}
```

**Estados a extraer:**
- `showPerformance`, `performanceMetrics`
- `messageCache`, `cacheEnabled`, `realTimeStats`

### 1.10 useNotifications.ts

**Responsabilidad:** Notificaciones

```typescript
// hooks/useNotifications.ts
export function useNotifications() {
  const [smartNotifications, setSmartNotifications] = useState(true)
  const [notificationRules, setNotificationRules] = useState<Map<string, any>>(new Map())
  const [pushNotifications, setPushNotifications] = useState(false)
  const [notificationPermission, setNotificationPermission] = useState<NotificationPermission>('default')
  
  // Lógica de notificaciones
  // ...
  
  return {
    smartNotifications,
    setSmartNotifications,
    notificationRules,
    setNotificationRules,
    pushNotifications,
    setPushNotifications,
    notificationPermission,
    setNotificationPermission,
    // ... más funciones
  }
}
```

**Estados a extraer:**
- `smartNotifications`, `notificationRules`
- `pushNotifications`, `notificationPermission`
- `messageNotifications`, `notificationSettings`

---

## 🏗️ Fase 2: Creación de Context Providers

### 2.1 ChatContext.tsx

**Responsabilidad:** Estado global del chat

```typescript
// contexts/ChatContext.tsx
interface ChatContextType {
  messages: Message[]
  setMessages: (messages: Message[]) => void
  isLoading: boolean
  setIsLoading: (loading: boolean) => void
  // ... más estado compartido
}

export const ChatContext = createContext<ChatContextType | null>(null)

export function ChatProvider({ children }: { children: React.ReactNode }) {
  const chatState = useChatState()
  
  return (
    <ChatContext.Provider value={chatState}>
      {children}
    </ChatContext.Provider>
  )
}

export function useChat() {
  const context = useContext(ChatContext)
  if (!context) {
    throw new Error('useChat must be used within ChatProvider')
  }
  return context
}
```

### 2.2 SettingsContext.tsx

**Responsabilidad:** Configuraciones

```typescript
// contexts/SettingsContext.tsx
interface SettingsContextType {
  theme: 'dark' | 'light' | 'auto'
  setTheme: (theme: 'dark' | 'light' | 'auto') => void
  autoSave: boolean
  setAutoSave: (enabled: boolean) => void
  // ... más configuraciones
}

export const SettingsContext = createContext<SettingsContextType | null>(null)

export function SettingsProvider({ children }: { children: React.ReactNode }) {
  // Lógica de settings
  // ...
  
  return (
    <SettingsContext.Provider value={settings}>
      {children}
    </SettingsContext.Provider>
  )
}
```

### 2.3 ThemeContext.tsx

**Responsabilidad:** Tema y personalización

```typescript
// contexts/ThemeContext.tsx
interface ThemeContextType {
  theme: 'dark' | 'light' | 'auto'
  customThemes: Map<string, any>
  activeTheme: string
  setActiveTheme: (theme: string) => void
  // ... más tema
}

export const ThemeContext = createContext<ThemeContextType | null>(null)
```

---

## 🧩 Fase 3: Componentes UI

### 3.1 MessageList Component

```typescript
// components/MessageList/MessageList.tsx
interface MessageListProps {
  messages: Message[]
  filters: FilterState
  viewMode: ViewMode
}

export function MessageList({ messages, filters, viewMode }: MessageListProps) {
  const { favoriteMessages, toggleFavorite } = useMessageManagement()
  const { searchQuery, filteredMessages } = useSearch(messages)
  
  return (
    <div className="message-list">
      {filteredMessages.map(message => (
        <MessageItem
          key={message.id}
          message={message}
          isFavorite={favoriteMessages.has(message.id)}
          onToggleFavorite={() => toggleFavorite(message.id)}
        />
      ))}
    </div>
  )
}
```

### 3.2 InputArea Component

```typescript
// components/InputArea/InputArea.tsx
export function InputArea() {
  const { input, setInput, sendMessage } = useChat()
  const { voiceInputEnabled, isRecording } = useVoiceFeatures()
  
  return (
    <div className="input-area">
      <TextInput
        value={input}
        onChange={setInput}
        onSend={sendMessage}
      />
      {voiceInputEnabled && <VoiceInput isRecording={isRecording} />}
      <QuickActions />
    </div>
  )
}
```

---

## 📋 Plan de Migración

### Paso 1: Preparación (1-2 días)
- [ ] Crear estructura de directorios
- [ ] Definir tipos TypeScript
- [ ] Crear utilidades base

### Paso 2: Extraer Hooks (3-5 días)
- [ ] `useChatState` - Estado principal
- [ ] `useMessageManagement` - Gestión de mensajes
- [ ] `useSearch` - Búsqueda
- [ ] `useFilters` - Filtros
- [ ] `useVoiceFeatures` - Voz
- [ ] `useExportImport` - Exportar/importar
- [ ] `useCollaboration` - Colaboración
- [ ] `useAccessibility` - Accesibilidad
- [ ] `usePerformance` - Performance
- [ ] `useNotifications` - Notificaciones

### Paso 3: Context Providers (2-3 días)
- [ ] `ChatContext` - Estado global
- [ ] `SettingsContext` - Configuraciones
- [ ] `ThemeContext` - Tema
- [ ] `AccessibilityContext` - Accesibilidad

### Paso 4: Componentes UI (5-7 días)
- [ ] `MessageList` - Lista de mensajes
- [ ] `InputArea` - Área de entrada
- [ ] `Sidebar` - Barra lateral
- [ ] `Toolbar` - Barra de herramientas
- [ ] `Modals` - Modales

### Paso 5: Refactorizar Componente Principal (2-3 días)
- [ ] Simplificar `ChatInterface.tsx`
- [ ] Usar hooks y contexts
- [ ] Integrar componentes nuevos

### Paso 6: Testing (2-3 días)
- [ ] Tests unitarios para hooks
- [ ] Tests de componentes
- [ ] Tests de integración

### Paso 7: Optimización (1-2 días)
- [ ] Performance optimization
- [ ] Code splitting
- [ ] Lazy loading

**Total estimado: 16-25 días**

---

## ✅ Checklist de Refactorización

### Por Hook
- [ ] Hook creado y documentado
- [ ] Estados extraídos del componente principal
- [ ] Lógica movida al hook
- [ ] Tests escritos
- [ ] Integrado en componente principal

### Por Context
- [ ] Context creado
- [ ] Provider implementado
- [ ] Hook de consumo creado
- [ ] Integrado en app

### Por Componente
- [ ] Componente creado
- [ ] Props tipadas
- [ ] Tests escritos
- [ ] Integrado en UI

### Global
- [ ] Componente principal < 500 líneas
- [ ] Todos los hooks funcionando
- [ ] Todos los tests pasando
- [ ] Performance mejorada
- [ ] Sin regresiones

---

## 📊 Métricas de Éxito

### Antes
- **Líneas:** 12,669
- **Hooks:** 1,127+
- **Estados:** 100+
- **Componentes:** 1 (monolítico)

### Después (Objetivo)
- **Líneas principales:** < 500
- **Hooks:** 10-15 custom hooks
- **Estados:** Organizados en contexts
- **Componentes:** 20+ componentes pequeños

### Mejoras Esperadas
- ✅ **Mantenibilidad:** +500%
- ✅ **Testabilidad:** +1000%
- ✅ **Performance:** +30% (menos re-renders)
- ✅ **Legibilidad:** +400%

---

## 🚨 Riesgos y Mitigaciones

### Riesgo 1: Breaking Changes
**Mitigación:** 
- Migración incremental
- Feature flags
- Tests exhaustivos

### Riesgo 2: Tiempo de Refactorización
**Mitigación:**
- Priorizar hooks más críticos
- Refactorizar por fases
- No bloquear features nuevas

### Riesgo 3: Bugs Introducidos
**Mitigación:**
- Tests antes de refactorizar
- Code review estricto
- Testing manual exhaustivo

---

**Versión:** 1.0  
**Fecha:** 2024  
**Prioridad:** 🔴 CRÍTICA




