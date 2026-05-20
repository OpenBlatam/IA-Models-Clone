# Modular Structure Guide

## 📁 New Modular Architecture

The codebase has been reorganized into feature-based modules following best practices.

## 🏗️ Structure

```
src/
├── modules/                    # Feature modules
│   ├── auth/                   # Authentication module
│   │   ├── types.ts           # Type definitions
│   │   ├── constants.ts       # Constants
│   │   ├── validators.ts      # Validation schemas
│   │   ├── services/          # Business logic
│   │   │   └── auth-service.ts
│   │   └── index.ts           # Public exports
│   ├── jobs/                   # Jobs module
│   │   ├── types.ts
│   │   ├── constants.ts
│   │   ├── services/
│   │   │   └── job-service.ts
│   │   ├── components/
│   │   │   └── job-card.tsx
│   │   ├── hooks/
│   │   │   └── use-job-search.ts
│   │   └── index.ts
│   ├── gamification/           # Gamification module
│   │   ├── types.ts
│   │   ├── services/
│   │   │   └── gamification-service.ts
│   │   ├── utils/
│   │   │   └── level-calculator.ts
│   │   └── index.ts
│   ├── dashboard/              # Dashboard module
│   │   ├── types.ts
│   │   ├── constants.ts
│   │   ├── services/
│   │   │   └── dashboard-service.ts
│   │   └── index.ts
│   └── shared/                 # Shared utilities
│       ├── utils/
│       │   ├── error-handler.ts
│       │   ├── format-helpers.ts
│       │   └── validation-helpers.ts
│       └── index.ts
├── components/                  # Reusable UI components
├── hooks/                       # Shared hooks
├── services/                    # API services
├── store/                       # State management
├── theme/                       # Theme system
└── utils/                       # General utilities
```

## 📦 Module Structure

Each module follows this structure:

```
module-name/
├── types.ts              # TypeScript interfaces
├── constants.ts         # Module constants
├── services/            # Business logic services
│   └── module-service.ts
├── components/          # Module-specific components (optional)
│   └── module-component.tsx
├── hooks/               # Module-specific hooks (optional)
│   └── use-module.ts
├── utils/               # Module utilities (optional)
│   └── module-utils.ts
└── index.ts             # Public API exports
```

## 🎯 Benefits

### 1. **Separation of Concerns**
- Business logic separated from UI
- Services handle API calls
- Components focus on presentation

### 2. **Reusability**
- Services can be used across different components
- Utilities are shared and tested independently
- Types are centralized

### 3. **Maintainability**
- Easy to find related code
- Changes are localized to modules
- Clear dependencies

### 4. **Testability**
- Services can be tested independently
- Components can be tested with mocked services
- Utilities are pure functions

### 5. **Scalability**
- Easy to add new features
- Modules can be developed independently
- Clear boundaries between features

## 📝 Usage Examples

### Using Auth Module

```typescript
import { authService, loginSchema, type LoginFormData } from '@/modules/auth';

// In component
const form = useForm<LoginFormData>({
  initialValues: { email: '', password: '' },
  validationSchema: loginSchema,
  onSubmit: async (values) => {
    await authService.login(values);
  },
});
```

### Using Jobs Module

```typescript
import { jobService, useJobSearch, JobCard } from '@/modules/jobs';

// In component
const { data, isLoading } = useJobSearch({ keywords: 'developer', location: 'Madrid' });

// Render
<JobCard job={job} onSwipe={handleSwipe} onApply={handleApply} />
```

### Using Shared Utilities

```typescript
import { handleError, formatCurrency, validateEmail } from '@/modules/shared';

// Error handling
try {
  await someOperation();
} catch (error) {
  handleError(error, { showAlert: true });
}

// Formatting
const price = formatCurrency(1000); // "$1,000.00"

// Validation
if (validateEmail(email)) {
  // Valid email
}
```

## 🔄 Migration Guide

### Before (Non-modular)
```typescript
// In component file
const response = await apiService.login(email, password);
if (!response.data) {
  Alert.alert('Error', response.error);
}
```

### After (Modular)
```typescript
// Import from module
import { authService } from '@/modules/auth';
import { handleError } from '@/modules/shared';

// In component
try {
  await authService.login({ email, password });
} catch (error) {
  handleError(error);
}
```

## ✅ Best Practices

1. **Always export through index.ts**
   - Don't import from internal files
   - Use `@/modules/module-name` imports

2. **Keep services pure**
   - No UI logic in services
   - Services return data, components handle UI

3. **Type everything**
   - Export types from types.ts
   - Use interfaces, not types

4. **Constants in constants.ts**
   - Magic numbers/strings go here
   - Export as const for type safety

5. **One responsibility per module**
   - Auth module = authentication only
   - Jobs module = job-related features only

6. **Shared code in shared module**
   - Common utilities
   - Error handling
   - Format helpers

## 📊 Module Dependencies

```
modules/
├── auth (no dependencies)
├── jobs (depends on: auth for userId)
├── gamification (no dependencies)
├── dashboard (depends on: auth, jobs, gamification)
└── shared (no dependencies, used by all)
```

## 🚀 Adding New Modules

1. Create module directory: `src/modules/new-module/`
2. Add `types.ts` with interfaces
3. Add `constants.ts` if needed
4. Create `services/` for business logic
5. Add `components/` if module-specific UI
6. Add `hooks/` if module-specific hooks
7. Export everything in `index.ts`
8. Document in this file

## 📚 Module Documentation

Each module should have:
- Clear type definitions
- Documented services
- Usage examples in comments
- Error handling patterns

---

**Last Updated**: 2024
**Structure Version**: 2.0


