# Improvements V13

This document outlines the thirteenth round of improvements made to enhance the frontend application.

## New Custom Hooks

### useMap
- **Purpose**: Manage Map data structure
- **Returns**: `{ map, set, get, has, remove, clear, setAll, size, isEmpty, keys, values, entries }`
- **Features**:
  - Key-value pair management
  - Set/get/remove operations
  - Clear all entries
  - Set all entries at once
  - Size and isEmpty helpers
  - Access to keys, values, entries arrays

### useSet
- **Purpose**: Manage Set data structure
- **Returns**: `{ set, add, remove, has, clear, toggle, size, isEmpty, values }`
- **Features**:
  - Unique value collection
  - Add/remove/has operations
  - Toggle item presence
  - Clear all items
  - Size and isEmpty helpers
  - Access to values array

## New Utility Classes

### Logger (`lib/utils/logger.ts`)
- **Purpose**: Structured logging with levels
- **Features**:
  - Log levels: debug, info, warn, error
  - Configurable prefix
  - Timestamp formatting
  - Enable/disable logging
  - Group/table support
  - Production-safe (only warnings/errors)

### Cache (`lib/utils/cache.ts`)
- **Purpose**: In-memory cache with TTL
- **Features**:
  - Time-to-live (TTL) support
  - Maximum size limit
  - Automatic expiration
  - LRU eviction when max size reached
  - Type-safe operations

### EventEmitter (`lib/utils/eventEmitter.ts`)
- **Purpose**: Event-driven architecture
- **Features**:
  - Subscribe/unsubscribe to events
  - Once listeners
  - Emit events with arguments
  - Remove all listeners
  - Listener count
  - Event names listing

## Improvements Summary

### Custom Hooks
1. **useMap**: Map data structure management
2. **useSet**: Set data structure management

### Utility Classes
- Structured logging
- In-memory caching
- Event emission

## Benefits

1. **Better Developer Experience**:
   - Map/Set data structures
   - Structured logging
   - Event-driven patterns
   - Caching utilities

2. **Code Quality**:
   - Type-safe operations
   - Reusable utilities
   - Better debugging

3. **Functionality**:
   - Data structure management
   - Logging infrastructure
   - Event system
   - Caching support

## Usage Examples

### useMap
```tsx
const {
  map,
  set,
  get,
  has,
  remove,
  size,
  keys,
  values,
} = useMap<string, number>();

set('key1', 1);
set('key2', 2);
const value = get('key1'); // 1
const exists = has('key2'); // true
remove('key1');
```

### useSet
```tsx
const { set, add, remove, has, toggle, size } = useSet<string>();

add('item1');
add('item2');
const exists = has('item1'); // true
toggle('item1'); // Removes if exists, adds if not
```

### Logger
```tsx
import { createLogger, logger } from '@/lib/utils';

// Default logger
logger.debug('Debug message');
logger.info('Info message');
logger.warn('Warning message');
logger.error('Error message');

// Custom logger
const customLogger = createLogger({
  level: 'info',
  prefix: 'MyApp',
  enabled: true,
});

customLogger.group('User Actions');
customLogger.info('User clicked button');
customLogger.groupEnd();
```

### Cache
```tsx
import { createCache } from '@/lib/utils';

const cache = createCache<string, User>({
  ttl: 60000, // 1 minute
  maxSize: 100,
});

cache.set('user:123', userData);
const user = cache.get('user:123');
const exists = cache.has('user:123');
cache.delete('user:123');
cache.clear();
```

### EventEmitter
```tsx
import { createEventEmitter } from '@/lib/utils';

const emitter = createEventEmitter();

// Subscribe
const unsubscribe = emitter.on('user:login', (user) => {
  console.log('User logged in:', user);
});

// Once listener
emitter.once('app:ready', () => {
  console.log('App is ready');
});

// Emit
emitter.emit('user:login', { id: 1, name: 'John' });

// Unsubscribe
unsubscribe();

// Remove all listeners
emitter.removeAllListeners('user:login');
```

These improvements add data structure management (Map/Set), structured logging, caching, and event emission that enhance both developer productivity and application architecture.

