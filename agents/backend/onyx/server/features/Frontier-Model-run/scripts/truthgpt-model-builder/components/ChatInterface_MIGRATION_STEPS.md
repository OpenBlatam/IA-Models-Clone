# Pasos de Migración Detallados - ChatInterface.tsx

## 🎯 Guía Paso a Paso para Refactorización

Esta guía te lleva paso a paso a través de la refactorización completa.

---

## 📋 Paso 1: Preparar el Entorno

### 1.1 Crear Estructura de Directorios

```bash
# Desde el directorio components/
mkdir -p ChatInterface/hooks
mkdir -p ChatInterface/contexts
mkdir -p ChatInterface/components/MessageList
mkdir -p ChatInterface/components/InputArea
mkdir -p ChatInterface/components/Sidebar
mkdir -p ChatInterface/components/Toolbar
mkdir -p ChatInterface/components/Modals
mkdir -p ChatInterface/utils
mkdir -p ChatInterface/types
mkdir -p ChatInterface/hooks/__tests__
```

### 1.2 Crear Archivos Base

```bash
# Crear archivos __init__ o index
touch ChatInterface/hooks/index.ts
touch ChatInterface/contexts/index.ts
touch ChatInterface/components/index.ts
touch ChatInterface/utils/index.ts
touch ChatInterface/types/index.ts
```

### 1.3 Definir Tipos Base

```typescript
// ChatInterface/types/message.types.ts
export interface Message {
  id: string
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp: number
  metadata?: Record<string, any>
}

// ChatInterface/types/chat.types.ts
export interface ChatState {
  messages: Message[]
  isLoading: boolean
  error: string | null
}

// ChatInterface/types/index.ts
export * from './message.types'
export * from './chat.types'
```

---

## 📋 Paso 2: Análisis y Limpieza

### 2.1 Ejecutar Análisis

```bash
# Ejecutar script de análisis
node scripts/analyze-unused-states.js

# Revisar reporte
cat ChatInterface_STATE_ANALYSIS_REPORT.txt
```

### 2.2 Identificar Estados a Eliminar

**Criterios para eliminar:**
- ✅ Estado nunca se lee
- ✅ Estado nunca se actualiza
- ✅ Estado duplicado
- ✅ Estado obsoleto

**Criterios para mantener:**
- ✅ Estado se usa activamente
- ✅ Estado es crítico para funcionalidad
- ✅ Estado tiene lógica compleja asociada

### 2.3 Eliminar Estados No Usados

**Proceso seguro:**
1. Hacer backup del archivo
2. Eliminar UN estado a la vez
3. Verificar que compila
4. Ejecutar tests
5. Si todo OK, commit
6. Repetir

**Ejemplo:**
```typescript
// ❌ ANTES: Estado no usado
const [workflowManager, setWorkflowManager] = useState(false)

// ✅ DESPUÉS: Eliminado
// (línea eliminada)
```

---

## 📋 Paso 3: Extraer Primer Hook (useChatState)

### 3.1 Generar Template

```bash
node scripts/extract-hook-template.js useChatState input isLoading messages error validation previewSpec
```

### 3.2 Completar el Hook

```typescript
// ChatInterface/hooks/useChatState.ts
import { useState, useCallback } from 'react'
import { Message } from '../types'

interface ChatState {
  input: string
  isLoading: boolean
  messages: Message[]
  error: string | null
  validation: any
  previewSpec: any
}

interface ChatActions {
  setInput: (input: string) => void
  setIsLoading: (loading: boolean) => void
  setMessages: (messages: Message[]) => void
  setError: (error: string | null) => void
  setValidation: (validation: any) => void
  setPreviewSpec: (spec: any) => void
  sendMessage: (content: string) => Promise<void>
  clearError: () => void
}

export function useChatState(): ChatState & ChatActions {
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [messages, setMessages] = useState<Message[]>([])
  const [error, setError] = useState<string | null>(null)
  const [validation, setValidation] = useState<any>(null)
  const [previewSpec, setPreviewSpec] = useState<any>(null)
  
  const sendMessage = useCallback(async (content: string) => {
    setIsLoading(true)
    setError(null)
    
    try {
      const newMessage: Message = {
        id: Date.now().toString(),
        role: 'user',
        content,
        timestamp: Date.now()
      }
      
      setMessages(prev => [...prev, newMessage])
      setInput('')
      
      // TODO: Lógica de envío real
      // await api.sendMessage(content)
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Error desconocido')
    } finally {
      setIsLoading(false)
    }
  }, [])
  
  const clearError = useCallback(() => {
    setError(null)
  }, [])
  
  return {
    // State
    input,
    isLoading,
    messages,
    error,
    validation,
    previewSpec,
    
    // Actions
    setInput,
    setIsLoading,
    setMessages,
    setError,
    setValidation,
    setPreviewSpec,
    sendMessage,
    clearError
  }
}
```

### 3.3 Escribir Tests

```typescript
// ChatInterface/hooks/__tests__/useChatState.test.ts
import { renderHook, act } from '@testing-library/react'
import { useChatState } from '../useChatState'

describe('useChatState', () => {
  it('should initialize with empty state', () => {
    const { result } = renderHook(() => useChatState())
    
    expect(result.current.input).toBe('')
    expect(result.current.messages).toEqual([])
    expect(result.current.isLoading).toBe(false)
    expect(result.current.error).toBeNull()
  })
  
  it('should update input', () => {
    const { result } = renderHook(() => useChatState())
    
    act(() => {
      result.current.setInput('test message')
    })
    
    expect(result.current.input).toBe('test message')
  })
  
  it('should send message', async () => {
    const { result } = renderHook(() => useChatState())
    
    await act(async () => {
      await result.current.sendMessage('Hello')
    })
    
    expect(result.current.messages).toHaveLength(1)
    expect(result.current.messages[0].content).toBe('Hello')
    expect(result.current.input).toBe('')
  })
})
```

### 3.4 Integrar en Componente Principal

```typescript
// ChatInterface.tsx
import { useChatState } from './ChatInterface/hooks/useChatState'

export default function ChatInterface() {
  // ❌ ANTES: Múltiples useState
  // const [input, setInput] = useState('')
  // const [isLoading, setIsLoading] = useState(false)
  // const [messages, setMessages] = useState<Message[]>([])
  // const [error, setError] = useState<string | null>(null)
  
  // ✅ DESPUÉS: Un hook
  const chatState = useChatState()
  
  // Usar chatState.input, chatState.setInput, etc.
  return (
    <div>
      <input 
        value={chatState.input}
        onChange={(e) => chatState.setInput(e.target.value)}
      />
      {chatState.isLoading && <div>Loading...</div>}
      {chatState.error && <div>Error: {chatState.error}</div>}
      {/* ... más UI */}
    </div>
  )
}
```

### 3.5 Verificar

- [ ] Componente compila
- [ ] Tests pasan
- [ ] Funcionalidad funciona igual
- [ ] No hay regresiones
- [ ] Commit: "Extract useChatState hook"

---

## 📋 Paso 4: Extraer Hook useSearch

### 4.1 Identificar Estados Relacionados

```bash
# Ejecutar análisis de dependencias
node scripts/find-state-dependencies.js

# Buscar estados de búsqueda
grep -n "search" ChatInterface.tsx | grep "useState"
```

### 4.2 Generar Template

```bash
node scripts/extract-hook-template.js useSearch searchQuery filteredMessages currentSearchIndex filterRole highlightSearch advancedSearch
```

### 4.3 Implementar Lógica

```typescript
// ChatInterface/hooks/useSearch.ts
import { useState, useMemo, useCallback } from 'react'
import { Message } from '../types'

export function useSearch(messages: Message[]) {
  const [searchQuery, setSearchQuery] = useState('')
  const [currentSearchIndex, setCurrentSearchIndex] = useState(-1)
  const [filterRole, setFilterRole] = useState<'all' | 'user' | 'assistant'>('all')
  const [highlightSearch, setHighlightSearch] = useState(true)
  const [advancedSearch, setAdvancedSearch] = useState(false)
  
  const filteredMessages = useMemo(() => {
    if (!searchQuery.trim()) {
      return messages
    }
    
    const query = searchQuery.toLowerCase()
    
    return messages.filter(message => {
      if (filterRole !== 'all' && message.role !== filterRole) {
        return false
      }
      
      return message.content.toLowerCase().includes(query)
    })
  }, [messages, searchQuery, filterRole])
  
  const nextMatch = useCallback(() => {
    if (filteredMessages.length === 0) return
    
    setCurrentSearchIndex(prev => 
      prev < filteredMessages.length - 1 ? prev + 1 : 0
    )
  }, [filteredMessages.length])
  
  const previousMatch = useCallback(() => {
    if (filteredMessages.length === 0) return
    
    setCurrentSearchIndex(prev => 
      prev > 0 ? prev - 1 : filteredMessages.length - 1
    )
  }, [filteredMessages.length])
  
  const clearSearch = useCallback(() => {
    setSearchQuery('')
    setCurrentSearchIndex(-1)
    setFilterRole('all')
  }, [])
  
  return {
    searchQuery,
    setSearchQuery,
    currentSearchIndex,
    setCurrentSearchIndex,
    filteredMessages,
    filterRole,
    setFilterRole,
    highlightSearch,
    setHighlightSearch,
    advancedSearch,
    setAdvancedSearch,
    nextMatch,
    previousMatch,
    clearSearch
  }
}
```

### 4.4 Integrar

```typescript
// ChatInterface.tsx
import { useSearch } from './ChatInterface/hooks/useSearch'

export default function ChatInterface() {
  const { messages } = useChatState()
  const search = useSearch(messages)
  
  // Usar search.filteredMessages en lugar de messages
  return (
    <div>
      <SearchBar
        query={search.searchQuery}
        onQueryChange={search.setSearchQuery}
        onNext={search.nextMatch}
        onPrevious={search.previousMatch}
      />
      <MessageList messages={search.filteredMessages} />
    </div>
  )
}
```

---

## 📋 Paso 5: Crear Componente MessageList

### 5.1 Crear Estructura

```bash
mkdir -p ChatInterface/components/MessageList
touch ChatInterface/components/MessageList/MessageList.tsx
touch ChatInterface/components/MessageList/MessageItem.tsx
touch ChatInterface/components/MessageList/index.ts
```

### 5.2 Implementar MessageItem

```typescript
// ChatInterface/components/MessageList/MessageItem.tsx
import { memo } from 'react'
import { Message } from '../../types'

interface MessageItemProps {
  message: Message
  isFavorite: boolean
  onToggleFavorite: () => void
}

export const MessageItem = memo(function MessageItem({
  message,
  isFavorite,
  onToggleFavorite
}: MessageItemProps) {
  return (
    <div className={`message message--${message.role}`}>
      <div className="message-content">{message.content}</div>
      <button onClick={onToggleFavorite}>
        {isFavorite ? '★' : '☆'}
      </button>
    </div>
  )
})
```

### 5.3 Implementar MessageList

```typescript
// ChatInterface/components/MessageList/MessageList.tsx
import { memo } from 'react'
import { Message } from '../../types'
import { MessageItem } from './MessageItem'
import { useMessageManagement } from '../../hooks/useMessageManagement'

interface MessageListProps {
  messages: Message[]
  highlightSearch?: boolean
  searchQuery?: string
}

export const MessageList = memo(function MessageList({
  messages,
  highlightSearch = false,
  searchQuery = ''
}: MessageListProps) {
  const { favoriteMessages, toggleFavorite } = useMessageManagement()
  
  return (
    <div className="message-list">
      {messages.map(message => (
        <MessageItem
          key={message.id}
          message={message}
          isFavorite={favoriteMessages.has(message.id)}
          onToggleFavorite={() => toggleFavorite(message.id)}
        />
      ))}
    </div>
  )
})
```

### 5.4 Integrar

```typescript
// ChatInterface.tsx
import { MessageList } from './ChatInterface/components/MessageList'

export default function ChatInterface() {
  const { messages } = useChatState()
  const search = useSearch(messages)
  
  return (
    <div>
      <MessageList 
        messages={search.filteredMessages}
        highlightSearch={search.highlightSearch}
        searchQuery={search.searchQuery}
      />
    </div>
  )
}
```

---

## 📋 Paso 6: Crear Context Provider

### 6.1 Crear ChatContext

```typescript
// ChatInterface/contexts/ChatContext.tsx
import { createContext, useContext, ReactNode } from 'react'
import { useChatState } from '../hooks/useChatState'

interface ChatContextType {
  input: string
  setInput: (input: string) => void
  isLoading: boolean
  messages: Message[]
  sendMessage: (content: string) => Promise<void>
  // ... más
}

const ChatContext = createContext<ChatContextType | null>(null)

export function ChatProvider({ children }: { children: ReactNode }) {
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

### 6.2 Integrar en App

```typescript
// app.tsx o _app.tsx
import { ChatProvider } from './components/ChatInterface/contexts/ChatContext'

export default function App() {
  return (
    <ChatProvider>
      <ChatInterface />
    </ChatProvider>
  )
}
```

### 6.3 Usar en Componentes

```typescript
// Cualquier componente hijo
import { useChat } from '../contexts/ChatContext'

function SomeComponent() {
  const { messages, sendMessage } = useChat()
  // Sin props drilling!
}
```

---

## 📋 Paso 7: Optimización Final

### 7.1 Code Splitting

```typescript
// ChatInterface.tsx
import { lazy, Suspense } from 'react'

const MessageList = lazy(() => import('./ChatInterface/components/MessageList'))
const ModelHistory = lazy(() => import('./ModelHistory'))
const ModelComparator = lazy(() => import('./ModelComparator'))

export default function ChatInterface() {
  return (
    <Suspense fallback={<div>Loading...</div>}>
      <MessageList />
      {showHistory && <ModelHistory />}
      {showComparator && <ModelComparator />}
    </Suspense>
  )
}
```

### 7.2 Virtual Scrolling

```typescript
// Para listas grandes
import { useVirtualizer } from '@tanstack/react-virtual'

function MessageList({ messages }: { messages: Message[] }) {
  const parentRef = useRef<HTMLDivElement>(null)
  
  const virtualizer = useVirtualizer({
    count: messages.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 100,
    overscan: 5,
  })
  
  return (
    <div ref={parentRef} style={{ height: 600, overflow: 'auto' }}>
      {virtualizer.getVirtualItems().map(virtualItem => (
        <MessageItem 
          key={virtualItem.key}
          message={messages[virtualItem.index]}
        />
      ))}
    </div>
  )
}
```

---

## ✅ Checklist de Verificación

### Después de Cada Paso

- [ ] Código compila sin errores
- [ ] Tests pasan
- [ ] Funcionalidad funciona igual
- [ ] No hay regresiones visuales
- [ ] Performance no degradada
- [ ] Commit realizado

### Al Final

- [ ] Componente principal < 500 líneas
- [ ] Todos los hooks extraídos
- [ ] Todos los contexts creados
- [ ] Todos los componentes extraídos
- [ ] Tests completos (> 80% cobertura)
- [ ] Performance optimizada
- [ ] Documentación actualizada

---

## 🚨 Troubleshooting

### Problema: "Hook is called conditionally"

**Solución:**
```typescript
// ❌ Incorrecto
if (condition) {
  const value = useState(0)
}

// ✅ Correcto
const value = useState(0)
if (condition) {
  // usar value
}
```

### Problema: "Cannot read property of undefined"

**Solución:**
```typescript
// ✅ Siempre verificar que context existe
const context = useContext(MyContext)
if (!context) {
  throw new Error('Must be used within provider')
}
```

### Problema: "Too many re-renders"

**Solución:**
```typescript
// ✅ Usar useCallback para funciones
const handleClick = useCallback(() => {
  // ...
}, [dependencies])

// ✅ Usar useMemo para valores calculados
const filtered = useMemo(() => {
  return items.filter(...)
}, [items, filter])
```

---

**Versión:** 1.0  
**Fecha:** 2024




