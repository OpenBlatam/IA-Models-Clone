# ChatInterface - Arquitectura Modular

## 📁 Estructura

```
ChatInterface/
├── types/              # Tipos TypeScript organizados
├── services/           # Servicios específicos (attachments, links, etc.)
├── repositories/       # Acceso a datos (Repository Pattern)
├── validators/         # Validación centralizada
├── strategies/         # Algoritmos intercambiables (Strategy Pattern)
├── events/             # Comunicación desacoplada (Event Bus)
├── builders/           # Construcción de objetos (Builder Pattern)
├── factories/          # Creación de estados iniciales
├── interfaces/         # Contratos y abstracciones
├── managers/           # Coordinación de servicios
├── decorators/         # Funcionalidades transversales
├── compositors/        # Combinación de servicios
├── hooks/              # Hooks React (orquestación)
├── utils/              # Utilidades compartidas
└── components/         # Componentes UI
```

## 🚀 Uso Rápido

### Hook Consolidado (Recomendado)

```typescript
import { useMessageState } from './ChatInterface/hooks'

function ChatInterface() {
  const { state, actions } = useMessageState()
  
  // Usar acciones
  actions.addLink('msg-1', 'https://example.com', 'Example', 'Description')
  actions.createPoll('msg-1', 'Question?', ['Option A', 'Option B'])
  actions.sortMessages('timestamp', 'desc')
}
```

### Servicios Específicos

```typescript
import { AttachmentService } from './ChatInterface/services/attachments/AttachmentService'
import { LinkService } from './ChatInterface/services/links/LinkService'

const attachments = AttachmentService.get(state.messageAttachments, 'msg-1')
const links = LinkService.get(state.messageLinks, 'msg-1')
```

### Builders

```typescript
import { MessageAttachmentBuilder } from './ChatInterface/builders'

const attachment = new MessageAttachmentBuilder()
  .withType('image')
  .withUrl('https://example.com/img.jpg')
  .withName('image.jpg')
  .build()
```

### Event Bus

```typescript
import { eventBus, MessageEvents } from './ChatInterface/events'

eventBus.on(MessageEvents.ATTACHMENT_ADDED, (data) => {
  console.log('Attachment added:', data)
})

eventBus.emit(MessageEvents.ATTACHMENT_ADDED, { messageId: 'msg-1' })
```

## 📚 Documentación Completa

- **Arquitectura**: Ver `ARCHITECTURE_IMPROVEMENTS.md`
- **Modularidad**: Ver `MAXIMUM_MODULARITY.md`
- **Limpieza**: Ver `CLEAN_CODE_SUMMARY.md`

## 🎯 Principios

- **Single Responsibility**: Cada módulo una responsabilidad
- **DRY**: Sin duplicación de código
- **SOLID**: Principios aplicados
- **Clean Code**: Código limpio y mantenible
