# Arquitectura del Módulo Agents

Este módulo implementa **Clean Architecture** (Arquitectura Limpia) con separación clara de responsabilidades y principios SOLID.

## 🏗️ Estructura de Capas

```
agents/
├── domain/                    # Capa de Dominio (Core Business Logic)
│   ├── entities/              # Entidades de negocio
│   │   └── Agent.ts           # Entidad Agent con lógica de negocio
│   └── repositories/           # Interfaces de repositorios
│       └── IAgentRepository.ts
│
├── application/                # Capa de Aplicación (Use Cases)
│   └── useCases/
│       ├── GetAgentsUseCase.ts
│       └── ToggleAgentUseCase.ts
│
├── infrastructure/             # Capa de Infraestructura (Implementaciones)
│   ├── repositories/
│   │   └── AgentRepository.ts # Implementación del repositorio
│   └── adapters/
│       └── AgentAdapter.ts    # Adaptador para transformación de datos
│
├── presentation/               # Capa de Presentación (UI)
│   ├── components/
│   │   └── AgentsSection.tsx  # Componente principal
│   ├── hooks/
│   │   ├── useAgentsContainer.ts
│   │   ├── useAgentActionsContainer.ts
│   │   ├── useFiltersContainer.ts
│   │   └── useAgentStatsContainer.ts
│   ├── context/
│   │   └── AgentsContext.tsx  # Context API para estado global
│   └── error/
│       └── ErrorBoundary.tsx  # Error Boundary
│
├── components/                # Componentes UI compartidos
├── config/                    # Configuración
├── hooks/                     # Hooks legacy (deprecados)
├── services/                  # Servicios legacy (deprecados)
└── utils/                     # Utilidades
```

## 📐 Principios de Arquitectura

### 1. **Clean Architecture**
- **Domain**: Lógica de negocio pura, sin dependencias externas
- **Application**: Casos de uso que orquestan la lógica de negocio
- **Infrastructure**: Implementaciones concretas (API, storage, etc.)
- **Presentation**: UI y hooks de React

### 2. **Dependency Inversion Principle (DIP)**
- Las capas internas no dependen de las externas
- Las dependencias apuntan hacia adentro
- Interfaces en domain, implementaciones en infrastructure

### 3. **Single Responsibility Principle (SRP)**
- Cada clase/componente tiene una única responsabilidad
- Use cases separados por funcionalidad
- Componentes UI atómicos

### 4. **Open/Closed Principle (OCP)**
- Abierto para extensión, cerrado para modificación
- Nuevos use cases sin modificar existentes
- Nuevos repositorios implementando interfaces

## 🔄 Flujo de Datos

```
Presentation Layer
    ↓ (usa)
Application Layer (Use Cases)
    ↓ (depende de)
Domain Layer (Interfaces)
    ↑ (implementado por)
Infrastructure Layer
```

### Ejemplo: Toggle Agent

1. **Presentation**: `useAgentActionsContainer` hook
2. **Application**: `ToggleAgentUseCase.execute()`
3. **Domain**: `AgentEntity.canBePaused()` (validación de negocio)
4. **Infrastructure**: `AgentRepository.toggleActive()` (llamada API)
5. **Infrastructure**: `AgentAdapter.toDomain()` (transformación)

## 🎯 Capas en Detalle

### Domain Layer (Dominio)

**Responsabilidades:**
- Definir entidades de negocio
- Contener lógica de negocio pura
- Definir interfaces de repositorios
- Validación con Zod schemas

**Características:**
- ✅ Sin dependencias externas
- ✅ Framework-agnostic
- ✅ Testeable sin mocks complejos
- ✅ Lógica de negocio inmutable

**Ejemplo:**
```typescript
// domain/entities/Agent.ts
export class AgentEntity {
  canBePaused(): boolean {
    return this.data.isActive;
  }
  
  getSuccessRate(): number {
    // Lógica de negocio pura
  }
}
```

### Application Layer (Aplicación)

**Responsabilidades:**
- Orquestar casos de uso
- Coordinar entre domain e infrastructure
- Validar reglas de negocio antes de ejecutar

**Características:**
- ✅ Depende solo de domain (interfaces)
- ✅ Contiene lógica de aplicación
- ✅ Un caso de uso = una funcionalidad

**Ejemplo:**
```typescript
// application/useCases/ToggleAgentUseCase.ts
export class ToggleAgentUseCase {
  async execute(agent: AgentEntity): Promise<AgentEntity> {
    if (agent.isActive && !agent.canBePaused()) {
      throw new Error("Agent cannot be paused");
    }
    return this.repository.toggleActive(agent.id, !agent.isActive);
  }
}
```

### Infrastructure Layer (Infraestructura)

**Responsabilidades:**
- Implementar interfaces de domain
- Comunicación con APIs externas
- Transformación de datos (DTOs ↔ Entities)
- Manejo de errores de red

**Características:**
- ✅ Implementa interfaces de domain
- ✅ Puede usar librerías externas (fetch, etc.)
- ✅ Adaptadores para transformación

**Ejemplo:**
```typescript
// infrastructure/repositories/AgentRepository.ts
export class AgentRepository implements IAgentRepository {
  async findAll(): Promise<AgentEntity[]> {
    const data = await fetch(API_ENDPOINTS.AGENTS);
    return this.adapter.toDomain(data);
  }
}
```

### Presentation Layer (Presentación)

**Responsabilidades:**
- Componentes React
- Hooks personalizados
- Context API para estado
- Error boundaries

**Características:**
- ✅ Depende de application layer
- ✅ Usa hooks para orquestar
- ✅ Componentes puros cuando es posible

**Ejemplo:**
```typescript
// presentation/hooks/useAgentsContainer.ts
export function useAgentsContainer() {
  const getAgentsUseCase = new GetAgentsUseCase(repository);
  // ...
}
```

## 🔌 Dependency Injection

Las dependencias se inyectan en los hooks containers:

```typescript
// Dependency Injection Container
const adapter = new AgentAdapter();
const repository = new AgentRepository(adapter);
const getAgentsUseCase = new GetAgentsUseCase(repository);
```

**Ventajas:**
- Fácil de testear (mock dependencies)
- Fácil de cambiar implementaciones
- Desacoplamiento

## 🧪 Testabilidad

Cada capa es testeable independientemente:

1. **Domain**: Tests unitarios puros
2. **Application**: Tests con mocks de repositorios
3. **Infrastructure**: Tests de integración
4. **Presentation**: Tests con React Testing Library

## 📦 Validación

Validación con **Zod** en la capa de dominio:

```typescript
export const AgentSchema = z.object({
  id: z.string().uuid(),
  name: z.string().min(1),
  // ...
});
```

## 🛡️ Manejo de Errores

- **Error Boundaries**: Para errores de React
- **Try/Catch**: En use cases y repositorios
- **Result Pattern**: Para errores funcionales (futuro)

## 🚀 Ventajas de esta Arquitectura

1. **Mantenibilidad**: Código organizado y claro
2. **Testabilidad**: Cada capa testeable aisladamente
3. **Escalabilidad**: Fácil agregar nuevas funcionalidades
4. **Flexibilidad**: Cambiar implementaciones sin afectar otras capas
5. **Type Safety**: TypeScript + Zod en toda la aplicación
6. **Separación de Concerns**: Cada capa tiene su responsabilidad

## 🔄 Migración desde Arquitectura Anterior

Los hooks y servicios antiguos están deprecados pero aún funcionan:
- `hooks/useAgents.ts` → `presentation/hooks/useAgentsContainer.ts`
- `services/agentsService.ts` → `infrastructure/repositories/AgentRepository.ts`

## 📚 Referencias

- [Clean Architecture by Robert C. Martin](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html)
- [SOLID Principles](https://en.wikipedia.org/wiki/SOLID)
- [Domain-Driven Design](https://martinfowler.com/bliki/DomainDrivenDesign.html)








