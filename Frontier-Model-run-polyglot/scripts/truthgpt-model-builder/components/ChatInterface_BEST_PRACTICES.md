# Mejores Prácticas - ChatInterface.tsx

## 📋 Guía para Mantener el Código Limpio

Después de la refactorización, estas prácticas evitarán que el componente vuelva a crecer descontroladamente.

---

## 🎯 Principios Fundamentales

### 1. Single Responsibility Principle (SRP)

**✅ Correcto:**
```typescript
// Un hook = una responsabilidad
function useSearch(messages: Message[]) {
  // Solo lógica de búsqueda
}

function useMessageManagement() {
  // Solo gestión de mensajes
}
```

**❌ Incorrecto:**
```typescript
// Un hook hace demasiadas cosas
function useChatEverything() {
  // Búsqueda, gestión, voz, exportación, etc.
}
```

### 2. Component Size Limit

**Regla:** Un componente no debe exceder **300 líneas**

**✅ Correcto:**
```typescript
// Componente pequeño y enfocado
function MessageItem({ message }: { message: Message }) {
  return <div>{message.content}</div>
}
```

**❌ Incorrecto:**
```typescript
// Componente gigante
function ChatInterface() {
  // 12,000 líneas
}
```

### 3. Hook Size Limit

**Regla:** Un hook no debe exceder **200 líneas**

**✅ Correcto:**
```typescript
// Hook pequeño y enfocado
function useSearch(messages: Message[]) {
  // 50 líneas de lógica de búsqueda
}
```

**❌ Incorrecto:**
```typescript
// Hook gigante
function useChatState() {
  // 2,000 líneas de lógica
}
```

---

## 💻 Patrones de Código

### 1. Custom Hooks para Lógica Reutilizable

**✅ Correcto:**
```typescript
// hooks/useSearch.ts
export function useSearch(messages: Message[]) {
  const [query, setQuery] = useState('')
  const filtered = useMemo(() => 
    messages.filter(m => m.content.includes(query)),
    [messages, query]
  )
  return { query, setQuery, filtered }
}

// Uso
function ChatInterface() {
  const { query, setQuery, filtered } = useSearch(messages)
}
```

**❌ Incorrecto:**
```typescript
// Lógica mezclada en componente
function ChatInterface() {
  const [query, setQuery] = useState('')
  const [filtered, setFiltered] = useState([])
  
  useEffect(() => {
    // 100 líneas de lógica de búsqueda aquí
  }, [query])
}
```

### 2. Context para Estado Global

**✅ Correcto:**
```typescript
// contexts/ChatContext.tsx
export function ChatProvider({ children }) {
  const [messages, setMessages] = useState([])
  return (
    <ChatContext.Provider value={{ messages, setMessages }}>
      {children}
    </ChatContext.Provider>
  )
}

// Uso en cualquier componente
function MessageList() {
  const { messages } = useChat()
}
```

**❌ Incorrecto:**
```typescript
// Props drilling
function ChatInterface() {
  const [messages, setMessages] = useState([])
  return <MessageList messages={messages} setMessages={setMessages} />
}

function MessageList({ messages, setMessages }) {
  return <MessageItem messages={messages} setMessages={setMessages} />
}
```

### 3. Componentes Pequeños y Enfocados

**✅ Correcto:**
```typescript
// components/MessageList/MessageList.tsx (50 líneas)
function MessageList({ messages }: Props) {
  return messages.map(msg => <MessageItem key={msg.id} message={msg} />)
}

// components/MessageList/MessageItem.tsx (30 líneas)
function MessageItem({ message }: Props) {
  return <div>{message.content}</div>
}
```

**❌ Incorrecto:**
```typescript
// Todo en un componente
function ChatInterface() {
  return (
    <div>
      {messages.map(msg => (
        <div>
          {/* 500 líneas de JSX aquí */}
        </div>
      ))}
    </div>
  )
}
```

---

## 🔧 Convenciones de Código

### Naming Conventions

**Hooks:**
```typescript
// ✅ Correcto: Empiezan con "use"
function useSearch() { }
function useMessageManagement() { }

// ❌ Incorrecto
function searchMessages() { }
function manageMessages() { }
```

**Componentes:**
```typescript
// ✅ Correcto: PascalCase, descriptivo
function MessageList() { }
function InputArea() { }

// ❌ Incorrecto
function msgList() { }
function input() { }
```

**Utilidades:**
```typescript
// ✅ Correcto: camelCase, verbos
function formatMessage() { }
function searchMessages() { }

// ❌ Incorrecto
function MessageFormatter() { }
function MessageSearcher() { }
```

### File Organization

**✅ Estructura Correcta:**
```
components/
├── ChatInterface/
│   ├── index.tsx
│   ├── hooks/
│   │   ├── useSearch.ts
│   │   └── useMessageManagement.ts
│   ├── components/
│   │   ├── MessageList/
│   │   │   ├── index.tsx
│   │   │   └── MessageItem.tsx
│   │   └── InputArea/
│   │       └── index.tsx
│   └── utils/
│       └── messageUtils.ts
```

**❌ Estructura Incorrecta:**
```
components/
├── ChatInterface.tsx (12,000 líneas)
└── utils.ts (todo mezclado)
```

---

## ⚡ Performance Best Practices

### 1. Memoización Estratégica

**✅ Correcto:**
```typescript
// Memoizar cálculos costosos
const filteredMessages = useMemo(() => 
  messages.filter(m => m.content.includes(query)),
  [messages, query]
)

// Memoizar callbacks
const handleClick = useCallback(() => {
  // ...
}, [dependencies])

// Memoizar componentes
export const MessageItem = memo(function MessageItem({ message }) {
  return <div>{message.content}</div>
})
```

**❌ Incorrecto:**
```typescript
// Sin memoización
const filtered = messages.filter(m => m.content.includes(query)) // Se recalcula siempre

const handleClick = () => { } // Se recrea en cada render
```

### 2. Lazy Loading

**✅ Correcto:**
```typescript
const HeavyComponent = lazy(() => import('./HeavyComponent'))

function ChatInterface() {
  return (
    <Suspense fallback={<Loading />}>
      <HeavyComponent />
    </Suspense>
  )
}
```

**❌ Incorrecto:**
```typescript
import HeavyComponent from './HeavyComponent' // Carga siempre
```

### 3. Virtual Scrolling

**✅ Correcto:**
```typescript
// Para listas grandes
function MessageList({ messages }) {
  return (
    <VirtualList
      items={messages}
      renderItem={message => <MessageItem message={message} />}
    />
  )
}
```

**❌ Incorrecto:**
```typescript
// Renderiza todos los mensajes
function MessageList({ messages }) {
  return messages.map(msg => <MessageItem message={msg} />)
}
```

---

## 🧪 Testing Best Practices

### 1. Testear Hooks

```typescript
// hooks/__tests__/useSearch.test.ts
import { renderHook, act } from '@testing-library/react'
import { useSearch } from '../useSearch'

describe('useSearch', () => {
  it('should filter messages by query', () => {
    const messages = [
      { id: '1', content: 'Hello' },
      { id: '2', content: 'World' }
    ]
    
    const { result } = renderHook(() => useSearch(messages))
    
    act(() => {
      result.current.setQuery('Hello')
    })
    
    expect(result.current.filtered).toHaveLength(1)
    expect(result.current.filtered[0].content).toBe('Hello')
  })
})
```

### 2. Testear Componentes

```typescript
// components/__tests__/MessageList.test.tsx
import { render, screen } from '@testing-library/react'
import { MessageList } from '../MessageList'

describe('MessageList', () => {
  it('should render messages', () => {
    const messages = [
      { id: '1', content: 'Hello' },
      { id: '2', content: 'World' }
    ]
    
    render(<MessageList messages={messages} />)
    
    expect(screen.getByText('Hello')).toBeInTheDocument()
    expect(screen.getByText('World')).toBeInTheDocument()
  })
})
```

---

## 🚫 Anti-Patterns a Evitar

### 1. ❌ God Component

```typescript
// ❌ NO HACER: Componente que hace todo
function ChatInterface() {
  // 12,000 líneas
  // Maneja: UI, estado, lógica, efectos, etc.
}
```

### 2. ❌ Prop Drilling

```typescript
// ❌ NO HACER: Pasar props por muchos niveles
function A({ data }) {
  return <B data={data} />
}
function B({ data }) {
  return <C data={data} />
}
function C({ data }) {
  return <D data={data} />
}
```

### 3. ❌ Estado Duplicado

```typescript
// ❌ NO HACER: Mismo estado en múltiples lugares
function Component1() {
  const [messages, setMessages] = useState([])
}
function Component2() {
  const [messages, setMessages] = useState([]) // Duplicado!
}
```

### 4. ❌ Efectos con Muchas Dependencias

```typescript
// ❌ NO HACER: Efecto con 20 dependencias
useEffect(() => {
  // ...
}, [dep1, dep2, dep3, ..., dep20]) // Muy difícil de mantener
```

### 5. ❌ Lógica en JSX

```typescript
// ❌ NO HACER: Lógica compleja en JSX
return (
  <div>
    {messages
      .filter(m => m.role === 'user')
      .map(m => m.content)
      .join(', ')
      .toUpperCase()
      .split(' ')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ')}
  </div>
)
```

---

## ✅ Checklist de Code Review

### Antes de Merge

- [ ] Componente < 300 líneas
- [ ] Hook < 200 líneas
- [ ] No hay props drilling (> 2 niveles)
- [ ] Estados relacionados agrupados
- [ ] Funciones memoizadas donde corresponde
- [ ] Types TypeScript completos (no `any`)
- [ ] Tests escritos
- [ ] Sin código duplicado
- [ ] Sin lógica compleja en JSX
- [ ] Performance optimizada

---

## 📊 Métricas a Monitorear

### Code Quality

- **Complexity:** < 10 por función
- **Lines per file:** < 300
- **Test coverage:** > 80%
- **TypeScript strict:** Enabled

### Performance

- **Bundle size:** Monitorear crecimiento
- **Render time:** < 16ms (60fps)
- **Re-renders:** Minimizar innecesarios
- **Memory leaks:** 0

---

## 🔄 Proceso de Desarrollo

### 1. Antes de Agregar Código

- [ ] ¿Existe un hook/componente similar?
- [ ] ¿Puedo reutilizar código existente?
- [ ] ¿Necesito crear un nuevo hook/componente?
- [ ] ¿El tamaño será razonable?

### 2. Durante Desarrollo

- [ ] Separar lógica de UI
- [ ] Crear hooks para lógica reutilizable
- [ ] Memoizar donde sea necesario
- [ ] Escribir tests

### 3. Antes de Commit

- [ ] Code review
- [ ] Tests pasando
- [ ] Linter sin errores
- [ ] Performance verificada

---

**Versión:** 1.0  
**Fecha:** 2024  
**Objetivo:** Mantener código limpio y mantenible




