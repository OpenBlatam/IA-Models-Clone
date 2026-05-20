# Frontend Structure - Modular Architecture

## 📁 Directory Structure

```
frontend/
├── app/                    # Next.js App Router
│   ├── layout.tsx         # Root layout
│   ├── page.tsx           # Main dashboard page
│   └── globals.css        # Global styles
│
├── components/
│   ├── ui/                # Reusable UI components
│   │   ├── Button.tsx
│   │   ├── Input.tsx
│   │   ├── Textarea.tsx
│   │   ├── Select.tsx
│   │   ├── Checkbox.tsx
│   │   ├── LoadingSpinner.tsx
│   │   ├── ErrorMessage.tsx
│   │   ├── SuccessMessage.tsx
│   │   ├── EmptyState.tsx
│   │   ├── ProgressBar.tsx
│   │   ├── ConfirmDialog.tsx
│   │   ├── Tooltip.tsx
│   │   └── index.ts       # Barrel export
│   │
│   ├── features/          # Feature-specific components
│   │   ├── ProjectGeneratorForm.tsx
│   │   ├── ProjectQueue.tsx
│   │   ├── ProjectList.tsx
│   │   ├── Statistics.tsx
│   │   ├── StatusIndicator.tsx
│   │   └── index.ts       # Barrel export
│   │
│   └── layout/            # Layout components
│       ├── Header.tsx
│       ├── Navigation.tsx
│       └── index.ts       # Barrel export
│
├── hooks/
│   ├── api/               # API-related hooks
│   │   ├── useDashboardData.ts
│   │   ├── useProjectGenerator.ts
│   │   ├── useGeneratorControl.ts
│   │   └── index.ts       # Barrel export
│   │
│   └── useWebSocket.ts    # WebSocket hook
│
├── lib/
│   ├── api/               # API client
│   │   └── index.ts
│   │
│   ├── constants/         # Constants and config
│   │   └── index.ts
│   │
│   └── utils/             # Utility functions
│       └── index.ts
│
└── types/                 # TypeScript types
    └── index.ts
```

## 🎯 Module Organization

### UI Components (`components/ui/`)
Reusable, framework-agnostic UI components:
- **Button**: Configurable button with variants and sizes
- **Input**: Text input with validation and icons
- **Textarea**: Multi-line input with character count
- **Select**: Dropdown select with options
- **Checkbox**: Checkbox with label and helper text
- **LoadingSpinner**: Loading indicator
- **ErrorMessage**: Error display component
- **SuccessMessage**: Success display component
- **EmptyState**: Empty state display
- **ProgressBar**: Progress indicator
- **ConfirmDialog**: Modal confirmation dialog
- **Tooltip**: Contextual help tooltip

### Feature Components (`components/features/`)
Business logic components specific to features:
- **ProjectGeneratorForm**: Form for generating projects
- **ProjectQueue**: Queue management view
- **ProjectList**: Projects list with filters
- **Statistics**: Statistics dashboard
- **StatusIndicator**: Generator status display

### Layout Components (`components/layout/`)
Layout and navigation components:
- **Header**: Application header
- **Navigation**: Tab navigation

### Hooks (`hooks/`)
Custom React hooks organized by concern:

#### API Hooks (`hooks/api/`)
- **useDashboardData**: Fetches and manages dashboard data
- **useProjectGenerator**: Handles project generation
- **useGeneratorControl**: Controls generator start/stop

#### UI Hooks
- **useWebSocket**: WebSocket connection management

### Constants (`lib/constants/`)
Application-wide constants:
- **API_CONFIG**: API configuration
- **REFRESH_INTERVALS**: Auto-refresh intervals
- **TABS**: Tab identifiers
- **PROJECT_STATUS**: Project status values
- **FILTER_TYPES**: Filter type values

### Utilities (`lib/utils/`)
Helper functions:
- **cn**: Class name utility (clsx wrapper)
- **formatDate**: Date formatting
- **formatRelativeTime**: Relative time formatting
- **getErrorMessage**: Error message extraction

## 🔄 Import Patterns

### Using Barrel Exports
```typescript
// ✅ Good - Use barrel exports
import { Button, Input } from '@/components/ui'
import { ProjectGeneratorForm } from '@/components/features'
import { useDashboardData } from '@/hooks/api'

// ❌ Bad - Direct imports
import Button from '@/components/ui/Button'
```

### Using Constants
```typescript
// ✅ Good - Use constants
import { TABS, API_CONFIG } from '@/lib/constants'
const activeTab = TABS.GENERATE

// ❌ Bad - Magic strings
const activeTab = 'generate'
```

## 📦 Benefits of Modular Structure

1. **Separation of Concerns**: UI, features, and layout are clearly separated
2. **Reusability**: UI components can be used across features
3. **Maintainability**: Easy to locate and update specific functionality
4. **Testability**: Each module can be tested independently
5. **Scalability**: Easy to add new features without affecting existing code
6. **Type Safety**: Centralized types and constants
7. **Barrel Exports**: Cleaner imports with index files

## 🚀 Adding New Features

1. Create feature component in `components/features/`
2. Create API hook in `hooks/api/` if needed
3. Add constants to `lib/constants/` if needed
4. Export from appropriate `index.ts` file
5. Use in main page or create new route

