# Ejemplos de Refactorización - ChatInterface.tsx

## 💻 Ejemplos de Código para Refactorización

Este documento proporciona ejemplos concretos de cómo refactorizar secciones específicas del componente `ChatInterface.tsx`.

---

## 📦 Ejemplo 1: Extraer Hook de Estado Principal

### ❌ Antes (En ChatInterface.tsx)

```typescript
export default function ChatInterface() {
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [messages, setMessages] = useState<Message[]>([])
  const [error, setError] = useState<string | null>(null)
  const [validation, setValidation] = useState<any>(null)
  const [previewSpec, setPreviewSpec] = useState<any>(null)
  
  // ... 12,000+ líneas más
}
```

### ✅ Después (Hook Extraído)

```typescript
// hooks/useChatState.ts
import { useState, useCallback } from 'react'
import { Message } from '../types/message.types'

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
      // Lógica de envío de mensaje
      const newMessage: Message = {
        id: Date.now().toString(),
        role: 'user',
        content,
        timestamp: Date.now()
      }
      
      setMessages(prev => [...prev, newMessage])
      setInput('')
      
      // Simular respuesta
      // ... lógica de API
      
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

// Uso en ChatInterface.tsx
export default function ChatInterface() {
  const chatState = useChatState()
  
  // Componente mucho más simple
  return (
    <div>
      {/* UI simplificada */}
    </div>
  )
}
```

---

## 🔍 Ejemplo 2: Extraer Hook de Búsqueda

### ❌ Antes

```typescript
export default function ChatInterface() {
  const [searchQuery, setSearchQuery] = useState('')
  const [currentSearchIndex, setCurrentSearchIndex] = useState(-1)
  const [filteredMessages, setFilteredMessages] = useState<Message[]>([])
  const [filterRole, setFilterRole] = useState<'all' | 'user' | 'assistant'>('all')
  const [highlightSearch, setHighlightSearch] = useState(true)
  const [advancedSearch, setAdvancedSearch] = useState(false)
  const [searchFilters, setSearchFilters] = useState<SearchFilters>({})
  
  // 200+ líneas de lógica de búsqueda mezclada con UI
  useEffect(() => {
    if (!searchQuery || typeof searchQuery !== 'string') {
      setFilteredMessages(messages || [])
      setCurrentSearchIndex(-1)
      return
    }
    
    // ... lógica compleja de búsqueda
  }, [searchQuery, messages, filterRole])
  
  // ... más lógica
}
```

### ✅ Después (Hook Extraído)

```typescript
// hooks/useSearch.ts
import { useState, useEffect, useMemo, useCallback } from 'react'
import { Message } from '../types/message.types'

interface SearchFilters {
  dateRange?: { start: Date; end: Date }
  minWords?: number
  maxWords?: number
  hasCode?: boolean
  hasLinks?: boolean
}

interface SearchState {
  searchQuery: string
  currentSearchIndex: number
  filteredMessages: Message[]
  filterRole: 'all' | 'user' | 'assistant'
  highlightSearch: boolean
  advancedSearch: boolean
  searchFilters: SearchFilters
}

interface SearchActions {
  setSearchQuery: (query: string) => void
  setCurrentSearchIndex: (index: number) => void
  setFilterRole: (role: 'all' | 'user' | 'assistant') => void
  setHighlightSearch: (enabled: boolean) => void
  setAdvancedSearch: (enabled: boolean) => void
  setSearchFilters: (filters: SearchFilters) => void
  nextMatch: () => void
  previousMatch: () => void
  clearSearch: () => void
}

export function useSearch(messages: Message[]): SearchState & SearchActions {
  const [searchQuery, setSearchQuery] = useState('')
  const [currentSearchIndex, setCurrentSearchIndex] = useState(-1)
  const [filterRole, setFilterRole] = useState<'all' | 'user' | 'assistant'>('all')
  const [highlightSearch, setHighlightSearch] = useState(true)
  const [advancedSearch, setAdvancedSearch] = useState(false)
  const [searchFilters, setSearchFilters] = useState<SearchFilters>({})
  
  // Memoizar mensajes filtrados
  const filteredMessages = useMemo(() => {
    if (!searchQuery?.trim()) {
      return messages
    }
    
    const query = searchQuery.toLowerCase().trim()
    
    return messages.filter(message => {
      // Filtro por rol
      if (filterRole !== 'all' && message.role !== filterRole) {
        return false
      }
      
      // Búsqueda básica
      const content = message.content?.toLowerCase() || ''
      const matchesBasic = content.includes(query)
      
      if (!advancedSearch) {
        return matchesBasic
      }
      
      // Búsqueda avanzada
      const wordCount = message.content?.split(/\s+/).length || 0
      const hasCode = message.content?.includes('```') || false
      const hasLinks = /https?:\/\//.test(message.content || '')
      
      if (searchFilters.minWords && wordCount < searchFilters.minWords) return false
      if (searchFilters.maxWords && wordCount > searchFilters.maxWords) return false
      if (searchFilters.hasCode !== undefined && hasCode !== searchFilters.hasCode) return false
      if (searchFilters.hasLinks !== undefined && hasLinks !== searchFilters.hasLinks) return false
      
      return matchesBasic
    })
  }, [messages, searchQuery, filterRole, advancedSearch, searchFilters])
  
  // Actualizar índice cuando cambian los resultados
  useEffect(() => {
    if (filteredMessages.length === 0) {
      setCurrentSearchIndex(-1)
    } else if (currentSearchIndex >= filteredMessages.length) {
      setCurrentSearchIndex(0)
    }
  }, [filteredMessages.length, currentSearchIndex])
  
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
    setSearchFilters({})
  }, [])
  
  return {
    // State
    searchQuery,
    currentSearchIndex,
    filteredMessages,
    filterRole,
    highlightSearch,
    advancedSearch,
    searchFilters,
    
    // Actions
    setSearchQuery,
    setCurrentSearchIndex,
    setFilterRole,
    setHighlightSearch,
    setAdvancedSearch,
    setSearchFilters,
    nextMatch,
    previousMatch,
    clearSearch
  }
}

// Uso en ChatInterface.tsx
export default function ChatInterface() {
  const { messages } = useChatState()
  const search = useSearch(messages)
  
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

## 🎨 Ejemplo 3: Extraer Componente MessageList

### ❌ Antes (Todo en ChatInterface.tsx)

```typescript
export default function ChatInterface() {
  // ... 100+ estados
  
  return (
    <div className="chat-interface">
      {/* 500+ líneas de JSX mezclado */}
      <div className="messages-container">
        {filteredMessages.map((message, index) => (
          <div key={message.id} className="message">
            {/* Lógica compleja de renderizado */}
            {message.role === 'user' ? (
              <div className="user-message">
                {/* ... */}
              </div>
            ) : (
              <div className="assistant-message">
                {/* ... */}
              </div>
            )}
            {/* Botones, acciones, etc. */}
          </div>
        ))}
      </div>
    </div>
  )
}
```

### ✅ Después (Componente Separado)

```typescript
// components/MessageList/MessageList.tsx
import { memo } from 'react'
import { Message } from '../../types/message.types'
import { MessageItem } from './MessageItem'
import { useMessageManagement } from '../../hooks/useMessageManagement'
import { useSearch } from '../../hooks/useSearch'

interface MessageListProps {
  messages: Message[]
  viewMode: 'normal' | 'compact' | 'comfortable'
  groupingMode: 'none' | 'time' | 'topic' | 'role'
  showTimestamps: boolean
  highlightSearch: boolean
  searchQuery?: string
}

export const MessageList = memo(function MessageList({
  messages,
  viewMode,
  groupingMode,
  showTimestamps,
  highlightSearch,
  searchQuery
}: MessageListProps) {
  const { favoriteMessages, pinnedMessages, toggleFavorite } = useMessageManagement()
  const { filteredMessages } = useSearch(messages)
  
  // Agrupar mensajes si es necesario
  const groupedMessages = useMemo(() => {
    if (groupingMode === 'none') {
      return filteredMessages.map(msg => [msg])
    }
    
    // Lógica de agrupación
    // ...
    
    return filteredMessages
  }, [filteredMessages, groupingMode])
  
  return (
    <div className={`message-list message-list--${viewMode}`}>
      {groupedMessages.map((group, groupIndex) => (
        <div key={groupIndex} className="message-group">
          {group.map(message => (
            <MessageItem
              key={message.id}
              message={message}
              isFavorite={favoriteMessages.has(message.id)}
              isPinned={pinnedMessages.has(message.id)}
              showTimestamp={showTimestamps}
              highlightSearch={highlightSearch && !!searchQuery}
              searchQuery={searchQuery}
              onToggleFavorite={() => toggleFavorite(message.id)}
            />
          ))}
        </div>
      ))}
    </div>
  )
})

// components/MessageList/MessageItem.tsx
import { memo } from 'react'
import { Message } from '../../types/message.types'
import { MessageContent } from './MessageContent'
import { MessageActions } from './MessageActions'

interface MessageItemProps {
  message: Message
  isFavorite: boolean
  isPinned: boolean
  showTimestamp: boolean
  highlightSearch: boolean
  searchQuery?: string
  onToggleFavorite: () => void
}

export const MessageItem = memo(function MessageItem({
  message,
  isFavorite,
  isPinned,
  showTimestamp,
  highlightSearch,
  searchQuery,
  onToggleFavorite
}: MessageItemProps) {
  return (
    <div 
      className={`message-item message-item--${message.role} ${isPinned ? 'message-item--pinned' : ''}`}
      data-message-id={message.id}
    >
      <MessageContent
        message={message}
        highlightSearch={highlightSearch}
        searchQuery={searchQuery}
      />
      
      {showTimestamp && (
        <div className="message-timestamp">
          {new Date(message.timestamp).toLocaleTimeString()}
        </div>
      )}
      
      <MessageActions
        message={message}
        isFavorite={isFavorite}
        isPinned={isPinned}
        onToggleFavorite={onToggleFavorite}
      />
    </div>
  )
})

// Uso en ChatInterface.tsx
export default function ChatInterface() {
  const { messages } = useChatState()
  const { viewMode, groupingMode } = useFilters()
  const { highlightSearch, searchQuery } = useSearch(messages)
  
  return (
    <div className="chat-interface">
      <MessageList
        messages={messages}
        viewMode={viewMode}
        groupingMode={groupingMode}
        showTimestamps={true}
        highlightSearch={highlightSearch}
        searchQuery={searchQuery}
      />
    </div>
  )
}
```

---

## 🎤 Ejemplo 4: Extraer Hook de Voz

### ❌ Antes

```typescript
export default function ChatInterface() {
  const [voiceInputEnabled, setVoiceInputEnabled] = useState(false)
  const [voiceOutputEnabled, setVoiceOutputEnabled] = useState(false)
  const [isRecording, setIsRecording] = useState(false)
  const [dictationMode, setDictationMode] = useState(false)
  
  // 300+ líneas de lógica de voz mezclada
  const startRecording = () => {
    // Lógica compleja
  }
  
  const stopRecording = () => {
    // Lógica compleja
  }
  
  // ... más funciones
}
```

### ✅ Después (Hook Extraído)

```typescript
// hooks/useVoiceFeatures.ts
import { useState, useCallback, useRef, useEffect } from 'react'

interface VoiceState {
  voiceInputEnabled: boolean
  voiceOutputEnabled: boolean
  isRecording: boolean
  dictationMode: boolean
  transcript: string
  error: string | null
}

interface VoiceActions {
  setVoiceInputEnabled: (enabled: boolean) => void
  setVoiceOutputEnabled: (enabled: boolean) => void
  setDictationMode: (enabled: boolean) => void
  startRecording: () => Promise<void>
  stopRecording: () => void
  clearTranscript: () => void
}

export function useVoiceFeatures(): VoiceState & VoiceActions {
  const [voiceInputEnabled, setVoiceInputEnabled] = useState(false)
  const [voiceOutputEnabled, setVoiceOutputEnabled] = useState(false)
  const [isRecording, setIsRecording] = useState(false)
  const [dictationMode, setDictationMode] = useState(false)
  const [transcript, setTranscript] = useState('')
  const [error, setError] = useState<string | null>(null)
  
  const recognitionRef = useRef<SpeechRecognition | null>(null)
  
  useEffect(() => {
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
      setError('Speech recognition no está disponible en este navegador')
      return
    }
    
    const SpeechRecognition = (window as any).webkitSpeechRecognition || 
                             (window as any).SpeechRecognition
    
    recognitionRef.current = new SpeechRecognition()
    recognitionRef.current.continuous = dictationMode
    recognitionRef.current.interimResults = true
    recognitionRef.current.lang = 'es-ES'
    
    recognitionRef.current.onresult = (event: any) => {
      const transcript = Array.from(event.results)
        .map((result: any) => result[0].transcript)
        .join('')
      setTranscript(transcript)
    }
    
    recognitionRef.current.onerror = (event: any) => {
      setError(`Error de reconocimiento: ${event.error}`)
      setIsRecording(false)
    }
    
    recognitionRef.current.onend = () => {
      setIsRecording(false)
    }
    
    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.stop()
      }
    }
  }, [dictationMode])
  
  const startRecording = useCallback(async () => {
    if (!recognitionRef.current) {
      setError('Reconocimiento de voz no inicializado')
      return
    }
    
    try {
      setError(null)
      setTranscript('')
      recognitionRef.current.start()
      setIsRecording(true)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Error al iniciar grabación')
      setIsRecording(false)
    }
  }, [])
  
  const stopRecording = useCallback(() => {
    if (recognitionRef.current && isRecording) {
      recognitionRef.current.stop()
      setIsRecording(false)
    }
  }, [isRecording])
  
  const clearTranscript = useCallback(() => {
    setTranscript('')
  }, [])
  
  return {
    // State
    voiceInputEnabled,
    voiceOutputEnabled,
    isRecording,
    dictationMode,
    transcript,
    error,
    
    // Actions
    setVoiceInputEnabled,
    setVoiceOutputEnabled,
    setDictationMode,
    startRecording,
    stopRecording,
    clearTranscript
  }
}

// Uso en ChatInterface.tsx
export default function ChatInterface() {
  const { input, setInput } = useChatState()
  const voice = useVoiceFeatures()
  
  // Sincronizar transcript con input
  useEffect(() => {
    if (voice.transcript && voice.voiceInputEnabled) {
      setInput(voice.transcript)
    }
  }, [voice.transcript, voice.voiceInputEnabled, setInput])
  
  return (
    <div>
      <InputArea
        value={input}
        onChange={setInput}
        voiceEnabled={voice.voiceInputEnabled}
        isRecording={voice.isRecording}
        onStartRecording={voice.startRecording}
        onStopRecording={voice.stopRecording}
      />
    </div>
  )
}
```

---

## 🎯 Ejemplo 5: Crear Context Provider

### ❌ Antes (Props Drilling)

```typescript
// ChatInterface.tsx
export default function ChatInterface() {
  const [theme, setTheme] = useState<'dark' | 'light' | 'auto'>('dark')
  const [fontSize, setFontSize] = useState<'small' | 'medium' | 'large'>('medium')
  // ... más estados
  
  return (
    <div>
      <MessageList theme={theme} fontSize={fontSize} />
      <InputArea theme={theme} fontSize={fontSize} />
      <Sidebar theme={theme} fontSize={fontSize} />
      {/* Props drilling en todos los componentes */}
    </div>
  )
}
```

### ✅ Después (Context Provider)

```typescript
// contexts/ThemeContext.tsx
import { createContext, useContext, useState, useCallback, ReactNode } from 'react'

interface ThemeContextType {
  theme: 'dark' | 'light' | 'auto'
  fontSize: 'small' | 'medium' | 'large'
  customThemes: Map<string, any>
  activeTheme: string
  setTheme: (theme: 'dark' | 'light' | 'auto') => void
  setFontSize: (size: 'small' | 'medium' | 'large') => void
  setActiveTheme: (theme: string) => void
  addCustomTheme: (name: string, theme: any) => void
}

const ThemeContext = createContext<ThemeContextType | null>(null)

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setTheme] = useState<'dark' | 'light' | 'auto'>('dark')
  const [fontSize, setFontSize] = useState<'small' | 'medium' | 'large'>('medium')
  const [customThemes, setCustomThemes] = useState<Map<string, any>>(new Map())
  const [activeTheme, setActiveTheme] = useState<string>('default')
  
  const addCustomTheme = useCallback((name: string, themeData: any) => {
    setCustomThemes(prev => new Map(prev).set(name, themeData))
  }, [])
  
  const value: ThemeContextType = {
    theme,
    fontSize,
    customThemes,
    activeTheme,
    setTheme,
    setFontSize,
    setActiveTheme,
    addCustomTheme
  }
  
  return (
    <ThemeContext.Provider value={value}>
      {children}
    </ThemeContext.Provider>
  )
}

export function useTheme() {
  const context = useContext(ThemeContext)
  if (!context) {
    throw new Error('useTheme must be used within ThemeProvider')
  }
  return context
}

// Uso en ChatInterface.tsx
export default function ChatInterface() {
  return (
    <ThemeProvider>
      <div className="chat-interface">
        <MessageList /> {/* No necesita props de tema */}
        <InputArea /> {/* No necesita props de tema */}
        <Sidebar /> {/* No necesita props de tema */}
      </div>
    </ThemeProvider>
  )
}

// Uso en cualquier componente hijo
function MessageList() {
  const { theme, fontSize } = useTheme() // Acceso directo sin props
  
  return (
    <div className={`message-list theme-${theme} font-${fontSize}`}>
      {/* ... */}
    </div>
  )
}
```

---

## 📊 Ejemplo 6: Optimización con useMemo y useCallback

### ❌ Antes (Sin Optimización)

```typescript
export default function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([])
  const [searchQuery, setSearchQuery] = useState('')
  
  // Se recalcula en cada render
  const filteredMessages = messages.filter(msg => 
    msg.content.toLowerCase().includes(searchQuery.toLowerCase())
  )
  
  // Se recrea en cada render
  const handleSend = (content: string) => {
    setMessages(prev => [...prev, { id: Date.now().toString(), content }])
  }
  
  return (
    <div>
      {filteredMessages.map(msg => (
        <MessageItem key={msg.id} message={msg} onSend={handleSend} />
      ))}
    </div>
  )
}
```

### ✅ Después (Optimizado)

```typescript
export default function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([])
  const [searchQuery, setSearchQuery] = useState('')
  
  // Memoizado - solo se recalcula cuando cambian messages o searchQuery
  const filteredMessages = useMemo(() => {
    if (!searchQuery.trim()) return messages
    
    const query = searchQuery.toLowerCase()
    return messages.filter(msg => 
      msg.content.toLowerCase().includes(query)
    )
  }, [messages, searchQuery])
  
  // Memoizado - solo se recrea cuando cambia setMessages
  const handleSend = useCallback((content: string) => {
    setMessages(prev => [...prev, { 
      id: Date.now().toString(), 
      content 
    }])
  }, [])
  
  // Memoizado - solo se recrea cuando cambian filteredMessages o handleSend
  const messageItems = useMemo(() => 
    filteredMessages.map(msg => (
      <MessageItem 
        key={msg.id} 
        message={msg} 
        onSend={handleSend} 
      />
    )),
    [filteredMessages, handleSend]
  )
  
  return (
    <div>
      {messageItems}
    </div>
  )
}
```

---

## 🧪 Ejemplo 7: Testing de Hooks

```typescript
// hooks/__tests__/useChatState.test.ts
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
  
  it('should send message and update state', async () => {
    const { result } = renderHook(() => useChatState())
    
    await act(async () => {
      await result.current.sendMessage('Hello')
    })
    
    expect(result.current.messages).toHaveLength(1)
    expect(result.current.messages[0].content).toBe('Hello')
    expect(result.current.input).toBe('')
    expect(result.current.isLoading).toBe(false)
  })
  
  it('should handle errors', async () => {
    const { result } = renderHook(() => useChatState())
    
    // Mock error scenario
    // ...
    
    await act(async () => {
      await result.current.sendMessage('test')
    })
    
    expect(result.current.error).not.toBeNull()
  })
})
```

---

## 📝 Resumen de Mejoras

### Por Ejemplo

1. **useChatState Hook**
   - ✅ Estado centralizado
   - ✅ Lógica separada de UI
   - ✅ Fácil de testear

2. **useSearch Hook**
   - ✅ Búsqueda compleja aislada
   - ✅ Memoización para performance
   - ✅ Funciones reutilizables

3. **MessageList Component**
   - ✅ Componente enfocado
   - ✅ Props claras
   - ✅ Memoización con React.memo

4. **useVoiceFeatures Hook**
   - ✅ Lógica de voz encapsulada
   - ✅ Manejo de errores
   - ✅ Cleanup automático

5. **ThemeContext**
   - ✅ Sin props drilling
   - ✅ Estado global accesible
   - ✅ Fácil de extender

6. **Optimización**
   - ✅ useMemo para cálculos costosos
   - ✅ useCallback para funciones
   - ✅ Menos re-renders

7. **Testing**
   - ✅ Hooks testeables
   - ✅ Componentes testeables
   - ✅ Cobertura mejorada

---

**Versión:** 1.0  
**Fecha:** 2024




