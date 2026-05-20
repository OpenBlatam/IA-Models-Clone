# Utils Directory

This directory contains all utility functions organized by category.

## Organization

### Core Utilities
- `cn.ts` - Class name merging utility
- `logger.ts` - Logging system
- `cache.ts` - In-memory cache
- `storage.ts` - Enhanced storage with encryption
- `constants.ts` - Application constants

### Data Manipulation
- `array.ts` - Array operations (chunk, unique, groupBy, etc.)
- `object.ts` - Object operations (deepClone, omit, pick, etc.)
- `transform.ts` - Data transformations (mapToObject, zip, transpose)
- `filter.ts` - Filtering utilities
- `sort.ts` - Sorting utilities
- `search.ts` - Search utilities

### Type & Validation
- `type.ts` - Type guards and checks
- `safe.ts` - Safe operations (parse, get, call)
- `validation.ts` - Basic validation schemas
- `validation-advanced.ts` - Advanced validation (credit card, IBAN, etc.)

### Formatting
- `format.ts` - Number, date, currency formatting
- `string.ts` - String utilities
- `date.ts` - Date utilities
- `color.ts` - Color utilities

### Math & Random
- `math.ts` - Math utilities
- `random.ts` - Random generation utilities

### Network & API
- `network.ts` - Network detection
- `promise.ts` - Promise utilities
- `retry.ts` - Retry logic

### Performance
- `performance.ts` - Performance monitoring
- `benchmark.ts` - Benchmarking utilities

### File & Storage
- `file.ts` - File operations
- `localStorage.ts` - LocalStorage wrapper

### DOM & Events
- `dom.ts` - DOM manipulation
- `event.ts` - Event utilities

### Encoding & Compression
- `encoding.ts` - Encoding/decoding (Base64, URL, HTML)
- `compression.ts` - Compression utilities
- `hash.ts` - Hashing utilities

### UI Utilities
- `clipboard.ts` - Clipboard operations
- `print.ts` - Print utilities
- `animations.ts` - Animation easing functions
- `accessibility.ts` - Accessibility utilities

### Device & Platform
- `device.ts` - Device detection
- `network.ts` - Network status

### Domain Specific
- `robot.ts` - Robot-specific utilities
- `i18n.ts` - Internationalization
- `analytics.ts` - Analytics utilities
- `keyboard.ts` - Keyboard shortcuts
- `toast.ts` - Toast notifications

### Testing
- `testing.ts` - Testing utilities

## Usage

Import from the main index:

```typescript
import { chunk, groupBy, formatNumber } from '@/lib/utils';
```

Or import directly for better tree-shaking:

```typescript
import { chunk } from '@/lib/utils/array';
import { formatNumber } from '@/lib/utils/format';
```



