# Frontend Architecture

## Overview

The frontend is built with Next.js 14 using the App Router, TypeScript, and TailwindCSS. It provides a complete interface for all backend API functionality.

## Architecture Principles

1. **Type Safety** - Full TypeScript coverage with types matching backend models
2. **Component Reusability** - DRY principle with reusable UI components
3. **Accessibility** - ARIA labels, keyboard navigation, screen reader support
4. **Responsive Design** - Mobile-first approach with TailwindCSS
5. **Performance** - React Query for efficient data fetching and caching

## Project Structure

```
frontend/
├── app/                          # Next.js App Router
│   ├── layout.tsx               # Root layout with providers
│   ├── page.tsx                 # Home page
│   ├── globals.css              # Global styles and Tailwind
│   ├── extract-profile/         # Profile extraction
│   ├── build-identity/         # Identity building
│   ├── generate-content/        # Content generation
│   ├── dashboard/               # Analytics dashboard
│   ├── identities/              # Identity management
│   │   └── [id]/               # Identity detail page
│   ├── templates/               # Template management
│   ├── tasks/                    # Task monitoring
│   └── alerts/                  # Alert management
│
├── components/                   # React components
│   ├── Layout/                  # Layout components
│   │   ├── Navbar.tsx          # Navigation bar
│   │   └── PageLayout.tsx      # Page wrapper
│   └── UI/                      # Reusable UI components
│       ├── Button.tsx
│       ├── Input.tsx
│       ├── Select.tsx
│       ├── Card.tsx
│       ├── LoadingSpinner.tsx
│       ├── Alert.tsx
│       └── Badge.tsx
│
├── lib/                          # Utilities and services
│   ├── api/
│   │   └── client.ts           # API client (all endpoints)
│   └── utils.ts                # Utility functions
│
└── types/                        # TypeScript definitions
    └── index.ts                 # All type definitions
```

## Data Flow

1. **User Interaction** → Component event handler
2. **API Call** → `apiClient` method
3. **React Query** → Caching and state management
4. **UI Update** → Component re-render with new data

## API Client

The `apiClient` (`lib/api/client.ts`) provides methods for all backend endpoints:

- Core: `extractProfile`, `buildIdentity`, `generateContent`
- Identity: `getIdentity`, `getGeneratedContent`
- Tasks: `createExtractProfileTask`, `getTask`, `getTasks`
- Analytics: `getMetrics`, `getDashboard`, `getAnalyticsStats`
- Templates: `getTemplates`, `createTemplate`, `deleteTemplate`
- Alerts: `getAlerts`, `acknowledgeAlert`, `resolveAlert`
- And many more...

## State Management

- **React Query** - Server state (API data, caching, refetching)
- **React State** - Local component state (form inputs, UI state)
- **No global state library** - React Query handles most needs

## Styling

- **TailwindCSS** - Utility-first CSS
- **Custom utilities** - Defined in `app/globals.css`
- **Component classes** - Reusable style patterns (`.btn`, `.card`, `.input`)

## Component Patterns

### Page Components
- Use `PageLayout` wrapper
- Fetch data with React Query
- Handle loading and error states
- Display data in Cards

### Form Components
- Use React Hook Form (when needed)
- Validate with Zod
- Submit via API client
- Show loading states during submission

### List Components
- Map over data arrays
- Use Cards for items
- Include loading and empty states
- Add pagination if needed

## Error Handling

- API client catches and formats errors
- Components display error messages
- React Query retries failed requests
- User-friendly error messages

## Accessibility Features

- ARIA labels on interactive elements
- Keyboard navigation support
- Focus management
- Screen reader compatibility
- Semantic HTML

## Performance Optimizations

- React Query caching reduces API calls
- Code splitting with Next.js App Router
- Image optimization (when needed)
- Lazy loading for heavy components

## Testing Strategy

- Unit tests for utilities
- Component tests for UI
- Integration tests for API calls
- E2E tests for critical flows

## Future Enhancements

- [ ] Add authentication/authorization
- [ ] Implement real-time updates with WebSockets
- [ ] Add more visualization charts
- [ ] Implement advanced search and filters
- [ ] Add export functionality
- [ ] Implement dark mode
- [ ] Add internationalization (i18n)



