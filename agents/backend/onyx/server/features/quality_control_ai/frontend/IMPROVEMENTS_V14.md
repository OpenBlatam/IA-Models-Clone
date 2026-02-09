# Improvements V14

This document outlines the fourteenth round of improvements made to enhance the frontend application.

## New Custom Hooks

### useAsyncState
- **Purpose**: Manage async state with loading and error handling
- **Returns**: `{ data, error, isLoading, execute, reset, setData, setError }`
- **Features**:
  - Automatic loading state
  - Error handling
  - Success/error callbacks
  - Reset functionality
  - Manual state setters

### useConditionalEffect
- **Purpose**: Run effect only when condition is true
- **Features**:
  - Conditional execution
  - Dependency array support
  - Clean API

## Enhanced Query Utilities

### Query Utilities (`lib/utils/query.ts`)
- **buildQueryString**: Build query string from object
- **parseQueryString**: Parse query string to object
- **getQueryParam**: Get single query parameter
- **getAllQueryParams**: Get all query parameters
- **setQueryParam**: Set query parameter in URL
- **removeQueryParam**: Remove query parameter from URL
- **updateQueryParams**: Update multiple query parameters
- **Features**:
  - SSR-safe
  - History API integration
  - Replace/push options

## New Format Utilities

### Format Utilities (`lib/utils/format.ts`)
- **formatBytes**: Format bytes to human-readable size
- **formatNumber**: Format number with Intl
- **formatCurrency**: Format as currency
- **formatPercentage**: Format as percentage
- **formatPhoneNumber**: Format phone number (XXX) XXX-XXXX
- **formatCreditCard**: Format credit card XXXX XXXX XXXX XXXX
- **formatSSN**: Format SSN XXX-XX-XXXX
- **formatInitials**: Extract initials from name
- **formatSlug**: Convert to URL-friendly slug
- **formatCamelCase**: Convert to readable camel case
- **formatSnakeCase/formatKebabCase/formatPascalCase**: Case conversions
- **formatTruncate/formatTruncateWords**: Truncate text

## Improvements Summary

### Custom Hooks
1. **useAsyncState**: Async state management
2. **useConditionalEffect**: Conditional effects

### Utility Functions
- Comprehensive query parameter handling
- Rich formatting utilities
- Data formatting

## Benefits

1. **Better Developer Experience**:
   - Async state management
   - Conditional effects
   - Query parameter utilities
   - Formatting utilities

2. **Code Quality**:
   - Type-safe operations
   - Reusable utilities
   - Better error handling

3. **Functionality**:
   - URL state management
   - Data formatting
   - Better async handling

## Usage Examples

### useAsyncState
```tsx
const { data, error, isLoading, execute, reset } = useAsyncState<User>({
  onSuccess: (user) => {
    console.log('User loaded:', user);
  },
  onError: (error) => {
    console.error('Error:', error);
  },
});

// Execute async function
await execute(() => fetchUser(userId));
```

### useConditionalEffect
```tsx
useConditionalEffect(
  () => {
    // Only runs when condition is true
    fetchData();
  },
  shouldFetch,
  [shouldFetch]
);
```

### Query Utilities
```tsx
import { getQueryParam, setQueryParam, updateQueryParams } from '@/lib/utils';

// Get query param
const page = getQueryParam('page', '1');

// Set query param
setQueryParam('page', '2', true); // replace history

// Update multiple
updateQueryParams({ page: '2', filter: 'active' }, true);
```

### Format Utilities
```tsx
import {
  formatBytes,
  formatCurrency,
  formatPhoneNumber,
  formatInitials,
  formatTruncate,
} from '@/lib/utils';

// Bytes
const size = formatBytes(1024); // "1 KB"

// Currency
const price = formatCurrency(1234.56); // "$1,234.56"

// Phone
const phone = formatPhoneNumber('1234567890'); // "(123) 456-7890"

// Initials
const initials = formatInitials('John Doe'); // "JD"

// Truncate
const short = formatTruncate('Long text', 10); // "Long te..."
```

These improvements add async state management, conditional effects, query parameter handling, and comprehensive formatting utilities that enhance both developer productivity and user experience.

