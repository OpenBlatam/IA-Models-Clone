# Quality Control AI Frontend - Modular Architecture

Frontend application for the Quality Control AI system built with Next.js, React, and TypeScript using a modular architecture.

## Architecture

The frontend is organized into **modules** by domain/feature:

```
frontend/
├── modules/              # Feature modules
│   ├── camera/          # Camera operations
│   │   ├── api.ts       # API client
│   │   ├── types.ts     # TypeScript types
│   │   ├── hooks/       # Custom hooks
│   │   └── components/  # React components
│   ├── inspection/      # Inspection operations
│   ├── alerts/          # Alert system
│   ├── detection/       # Detection settings
│   ├── reports/         # Report generation
│   ├── statistics/      # Statistics & charts
│   └── control/         # Control panel
├── components/          # Shared UI components
│   ├── ui/             # Reusable UI primitives
│   └── layout/         # Layout components
├── lib/                # Shared utilities
│   ├── api/            # API client setup
│   ├── hooks/          # Shared hooks
│   ├── store.ts        # Global state
│   └── utils.ts        # Utility functions
└── app/                # Next.js app directory
```

## Module Structure

Each module follows this structure:

```
module-name/
├── api.ts              # API client for this module
├── types.ts            # TypeScript types
├── hooks/              # Custom React hooks
│   └── useModuleName.ts
└── components/         # Module-specific components
    └── ComponentName.tsx
```

## Features

- **Modular Architecture**: Features organized by domain
- **Type Safety**: Full TypeScript coverage
- **Reusable Components**: Shared UI components
- **Custom Hooks**: Domain-specific hooks for logic
- **Error Handling**: Comprehensive error boundaries
- **Accessibility**: Full keyboard navigation and ARIA support
- **Responsive Design**: Mobile-first approach

## Getting Started

```bash
npm install
npm run dev
```

## Module Usage

### Camera Module

```typescript
import { useCamera, CameraView } from '@/modules/camera';

const MyComponent = () => {
  const { cameraInfo, initialize, captureFrame } = useCamera();
  // ...
};
```

### Inspection Module

```typescript
import { useInspection, InspectionResults } from '@/modules/inspection';

const MyComponent = () => {
  const { start, stop, inspectFrame } = useInspection();
  // ...
};
```

## Best Practices

1. **Keep modules independent**: Each module should be self-contained
2. **Use hooks for logic**: Extract business logic into custom hooks
3. **Share types**: Export types from module index
4. **Component composition**: Build complex UIs from simple components
5. **Error boundaries**: Wrap modules in error boundaries

## License

Blatam Academy
