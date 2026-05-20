# Mejores Prácticas - ChatInterface

## 🎯 Principios de Diseño

### 1. Separación de Responsabilidades
- **Hooks** - Lógica de negocio y estado
- **Componentes** - Presentación y UI
- **Utils** - Funciones puras y reutilizables
- **Contexts** - Estado global compartido

### 2. Composición sobre Herencia
```typescript
// ✅ Bueno - Composición
function ChatInterface() {
  const chatState = useChatState()
  const messageManagement = useMessageManagement()
  return <MessageList messages={chatState.messages} />
}

// ❌ Malo - Herencia
class ChatInterface extends BaseComponent { }
```

### 3. Custom Hooks para Lógica Reutilizable
```typescript
// ✅ Bueno - Hook personalizado
function useMessageManagement() {
  const [favorites, setFavorites] = useState(new Set())
  const toggleFavorite = useCallback((id) => { /* ... */ }, [])
  return { favorites, toggleFavorite }
}

// ❌ Malo - Lógica en componente
function Component() {
  const [favorites, setFavorites] = useState(new Set())
  // 50+ líneas de lógica...
}
```

## 📝 Patrones de Código

### 1. Uso de Hooks

```typescript
// ✅ Bueno - Hooks al inicio
function ChatInterface() {
  const chatState = useChatState()
  const messageManagement = useMessageManagement()
  const search = useSearchAndFilters()
  
  // Lógica de orquestación
  const handleSend = useCallback(() => { /* ... */ }, [])
  
  return <div>...</div>
}

// ❌ Malo - Hooks mezclados con lógica
function ChatInterface() {
  const [state, setState] = useState()
  // 100 líneas de lógica...
  const chatState = useChatState() // Hook después de lógica
}
```

### 2. Memoización

```typescript
// ✅ Bueno - Memoizar componentes pesados
const MessageList = memo(function MessageList({ messages }) {
  return messages.map(msg => <MessageItem key={msg.id} message={msg} />)
})

// ✅ Bueno - Memoizar valores calculados
const filteredMessages = useMemo(() => {
  return messages.filter(msg => msg.role === filterRole)
}, [messages, filterRole])

// ❌ Malo - Sin memoización
function MessageList({ messages }) {
  return messages.map(msg => <MessageItem message={msg} />) // Re-render innecesario
}
```

### 3. Callbacks

```typescript
// ✅ Bueno - useCallback para callbacks
const handleClick = useCallback((id: string) => {
  toggleFavorite(id)
}, [toggleFavorite])

// ❌ Malo - Función inline sin memoizar
<button onClick={() => toggleFavorite(id)} /> // Nueva función en cada render
```

### 4. Manejo de Errores

```typescript
// ✅ Bueno - Manejo centralizado
try {
  const result = await chatActions.handleSendMessage(input)
} catch (error) {
  handleError(error, 'ChatInterface')
  toast.error(getUserFriendlyError(error))
}

// ❌ Malo - Errores sin manejar
const result = await chatActions.handleSendMessage(input) // Sin try/catch
```

## 🎨 Componentes

### 1. Props Interface

```typescript
// ✅ Bueno - Interface clara
interface MessageListProps {
  messages: Message[]
  viewMode?: 'normal' | 'compact' | 'comfortable'
  onMessageClick?: (id: string) => void
}

// ❌ Malo - Props sin tipo
function MessageList(props: any) { }
```

### 2. Componentes Pequeños

```typescript
// ✅ Bueno - Componente pequeño y enfocado
function MessageItem({ message }: { message: Message }) {
  return <div>{message.content}</div>
}

// ❌ Malo - Componente gigante
function MessageList() {
  // 500+ líneas de código...
}
```

### 3. Default Props

```typescript
// ✅ Bueno - Default props
function InputArea({
  maxLength = 10000,
  autoFocus = false,
}: InputAreaProps) { }

// ❌ Malo - Props requeridas cuando no es necesario
function InputArea({
  maxLength, // Requerido pero tiene valor por defecto
}: InputAreaProps) { }
```

## 🔧 Utilidades

### 1. Funciones Puras

```typescript
// ✅ Bueno - Función pura
function getWordCount(text: string): number {
  return text.trim().split(/\s+/).length
}

// ❌ Malo - Efectos secundarios
function getWordCount(text: string): number {
  console.log('Counting words') // Efecto secundario
  return text.trim().split(/\s+/).length
}
```

### 2. Validación de Input

```typescript
// ✅ Bueno - Validar input
function validateMessage(content: string): ValidationResult {
  if (!content || content.trim().length === 0) {
    return { valid: false, errors: ['Content is required'] }
  }
  return { valid: true, errors: [] }
}

// ❌ Malo - Sin validación
function processMessage(content: string) {
  return content.toUpperCase() // Puede fallar si content es null
}
```

### 3. Manejo de Edge Cases

```typescript
// ✅ Bueno - Manejar edge cases
function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 Bytes'
  if (bytes < 0) return 'Invalid size'
  // ... resto de lógica
}

// ❌ Malo - Sin manejo de edge cases
function formatFileSize(bytes: number): string {
  return `${bytes / 1024} KB` // Falla si bytes es 0 o negativo
}
```

## 🚀 Performance

### 1. Virtual Scrolling

```typescript
// ✅ Bueno - Virtual scrolling para listas grandes
const virtualization = useVirtualization(messages, {
  itemHeight: 80,
  containerHeight: 600,
})

// ❌ Malo - Renderizar todos los mensajes
{messages.map(msg => <MessageItem message={msg} />)} // Lento con 1000+ mensajes
```

### 2. Debounce/Throttle

```typescript
// ✅ Bueno - Debounce para búsqueda
const debouncedSearch = useMemo(
  () => debounce((query: string) => {
    performSearch(query)
  }, 300),
  []
)

// ❌ Malo - Búsqueda en cada keystroke
<input onChange={(e) => performSearch(e.target.value)} /> // Muy costoso
```

### 3. Lazy Loading

```typescript
// ✅ Bueno - Lazy load componentes pesados
const HeavyComponent = lazy(() => import('./HeavyComponent'))

// ❌ Malo - Importar todo al inicio
import HeavyComponent from './HeavyComponent' // Aumenta bundle inicial
```

## 🧪 Testing

### 1. Tests de Hooks

```typescript
// ✅ Bueno - Test hook aislado
describe('useChatState', () => {
  it('should initialize with default state', () => {
    const { result } = renderHook(() => useChatState())
    expect(result.current.state.input).toBe('')
  })
})
```

### 2. Tests de Componentes

```typescript
// ✅ Bueno - Test con providers
test('renders message list', () => {
  const { getByText } = renderWithProviders(
    <MessageList messages={mockMessages} />
  )
  expect(getByText('Hello')).toBeInTheDocument()
})
```

### 3. Mocks

```typescript
// ✅ Bueno - Mock servicios externos
beforeEach(() => {
  global.fetch = jest.fn().mockResolvedValue({
    ok: true,
    json: () => Promise.resolve({ data: [] }),
  })
})
```

## 📊 Métricas de Calidad

### Código Bueno
- ✅ Componentes < 200 líneas
- ✅ Hooks < 300 líneas
- ✅ Funciones < 50 líneas
- ✅ Complejidad ciclomática < 10
- ✅ Cobertura de tests > 80%

### Código a Mejorar
- ❌ Componentes > 500 líneas
- ❌ Hooks > 500 líneas
- ❌ Funciones > 100 líneas
- ❌ Complejidad ciclomática > 20
- ❌ Cobertura de tests < 50%

## 🎯 Checklist de Revisión

Antes de hacer commit:

- [ ] Componentes < 200 líneas
- [ ] Hooks < 300 líneas
- [ ] Props tipadas con TypeScript
- [ ] Errores manejados correctamente
- [ ] Memoización donde sea necesario
- [ ] Tests escritos para nueva funcionalidad
- [ ] Documentación actualizada
- [ ] Sin console.logs en producción
- [ ] Performance optimizado
- [ ] Accesibilidad considerada




