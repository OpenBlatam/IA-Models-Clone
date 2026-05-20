# Architecture Guide

## 🏗️ Clean Architecture + Feature-Based

This app follows **Clean Architecture** principles with **Feature-Based** organization.

## 📁 Architecture Layers

```
src/
├── core/                    # Core infrastructure
│   ├── config/             # Configuration
│   ├── errors/              # Error handling
│   └── repository/           # Base repository pattern
├── features/                # Feature modules (Clean Architecture)
│   ├── auth/
│   │   ├── data/           # Data layer (repositories)
│   │   ├── domain/         # Business logic (services)
│   │   └── presentation/    # UI layer (stores, hooks, components)
│   ├── jobs/
│   ├── gamification/
│   └── ...
├── components/              # Shared UI components
├── hooks/                   # Shared hooks
├── theme/                   # Theme system
└── utils/                   # Shared utilities
```

## 🎯 Layer Responsibilities

### Core Layer
**Purpose**: Infrastructure and cross-cutting concerns

- **config/**: Environment configuration
- **errors/**: Error classes and handling
- **repository/**: Base repository with interceptors

### Feature Layer (Clean Architecture)

Each feature has 3 layers:

#### 1. **Data Layer** (`data/`)
- **Repositories**: API communication
- **DTOs**: Data transfer objects
- **Mappers**: Transform API responses

```typescript
// Example: AuthRepository
class AuthRepository extends BaseRepository {
  async login(credentials: LoginCredentials): Promise<Session> {
    return this.post(ENDPOINTS.AUTH.LOGIN, credentials);
  }
}
```

#### 2. **Domain Layer** (`domain/`)
- **Services**: Business logic
- **Types**: Domain models
- **Use Cases**: Feature-specific operations

```typescript
// Example: AuthService
class AuthService {
  constructor(private repository: AuthRepository) {}
  
  async login(credentials: LoginCredentials): Promise<Session> {
    const session = await this.repository.login(credentials);
    // Business logic: store session, update state, etc.
    return session;
  }
}
```

#### 3. **Presentation Layer** (`presentation/`)
- **Stores**: State management (Zustand)
- **Hooks**: React Query integration
- **Components**: UI components

```typescript
// Example: useAuthStore
export const useAuthStore = create<AuthState>((set) => ({
  login: async (credentials) => {
    const service = new AuthService();
    const session = await service.login(credentials);
    set({ user: session.user, isAuthenticated: true });
  },
}));
```

## 🔄 Data Flow

```
User Action
    ↓
Presentation Layer (Component/Hook)
    ↓
Domain Layer (Service)
    ↓
Data Layer (Repository)
    ↓
API/Backend
    ↓
Response flows back up
```

## 📦 Feature Structure

```
feature-name/
├── data/
│   └── feature-repository.ts    # API communication
├── domain/
│   ├── feature-types.ts          # Domain types
│   └── feature-service.ts        # Business logic
└── presentation/
    ├── feature-store.ts          # State (if needed)
    ├── hooks/
    │   └── use-feature.ts        # React Query hooks
    └── components/
        └── feature-component.tsx # UI components
```

## 🎯 Benefits

### 1. **Separation of Concerns**
- Data layer: API only
- Domain layer: Business logic only
- Presentation layer: UI only

### 2. **Testability**
```typescript
// Test service independently
const mockRepository = { login: jest.fn() };
const service = new AuthService(mockRepository);
await service.login(credentials);
expect(mockRepository.login).toHaveBeenCalled();
```

### 3. **Dependency Inversion**
- Services depend on repository interfaces
- Easy to swap implementations
- Mock for testing

### 4. **Scalability**
- Add features without affecting others
- Clear boundaries
- Easy to understand

## 📝 Example: Complete Feature

### Auth Feature

```typescript
// data/auth-repository.ts
class AuthRepository extends BaseRepository {
  async login(credentials: LoginCredentials): Promise<Session> {
    return this.post(ENDPOINTS.AUTH.LOGIN, credentials);
  }
}

// domain/auth-service.ts
class AuthService {
  constructor(private repository: AuthRepository) {}
  
  async login(credentials: LoginCredentials): Promise<Session> {
    const session = await this.repository.login(credentials);
    // Store session, handle errors, etc.
    return session;
  }
}

// presentation/auth-store.ts
export const useAuthStore = create((set) => ({
  login: async (credentials) => {
    const service = new AuthService();
    await service.login(credentials);
  },
}));
```

## 🔧 Core Infrastructure

### BaseRepository
- Handles authentication
- Error mapping
- Request/response interceptors
- Automatic token management

### Error Handling
- Custom error classes
- Centralized error handling
- User-friendly messages
- Error tracking integration

### Environment Config
- Environment detection
- API URL configuration
- Feature flags
- Logging configuration

## 🎓 Best Practices

1. **One feature = One module**
   - All related code in one place
   - Clear boundaries

2. **Dependency flow**
   - Presentation → Domain → Data
   - Never reverse

3. **Services are pure**
   - No UI logic
   - No React hooks
   - Just business logic

4. **Repositories handle API**
   - Transform requests/responses
   - Handle errors
   - No business logic

5. **Types in domain**
   - Domain types in domain/
   - API types in data/
   - UI types in presentation/

## 📊 Comparison

| Aspect | Old Structure | New Architecture |
|--------|---------------|------------------|
| **Organization** | By type | By feature + layer |
| **Dependencies** | Circular | Unidirectional |
| **Testing** | Hard | Easy |
| **Scalability** | Limited | High |
| **Maintainability** | Medium | High |

## 🚀 Migration

### Step 1: Create Feature Structure
```
features/auth/
├── data/
├── domain/
└── presentation/
```

### Step 2: Move Code
- API calls → `data/repository.ts`
- Business logic → `domain/service.ts`
- State/UI → `presentation/`

### Step 3: Update Imports
```typescript
// Old
import { apiService } from '@/services/api';

// New
import { AuthService } from '@/features/auth';
```

## 📚 References

- [Clean Architecture](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html)
- [Feature-Based Architecture](https://khalilstemmler.com/articles/software-design-architecture/organizing-app-logic/)
- [Repository Pattern](https://martinfowler.com/eaaCatalog/repository.html)

---

**Architecture**: Clean Architecture + Feature-Based
**Version**: 3.0
**Last Updated**: 2024

