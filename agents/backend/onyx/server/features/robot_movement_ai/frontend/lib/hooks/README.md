# Hooks Directory

This directory contains all custom React hooks organized by category.

## Organization

### State Management
- `useToggle` - Toggle boolean state
- `useCounter` - Counter with increment/decrement
- `usePrevious` - Get previous value
- `useQueue` - Queue data structure
- `useStack` - Stack data structure
- `useMap` - Map data structure
- `useSet` - Set data structure
- `useReducerWithMiddleware` - Reducer with middleware
- `useReducerWithDevTools` - Reducer with dev tools

### Storage
- `useLocalStorage` - LocalStorage hook with encryption
- `useSessionStorage` - SessionStorage hook

### Async Operations
- `useAsync` - Handle async operations
- `useRetry` - Retry with backoff
- `useCache` - Cache with TTL

### Lifecycle
- `useMount` - On mount effect
- `useUnmount` - On unmount effect
- `useUpdateEffect` - Effect only on updates
- `useDeepCompareEffect` - Effect with deep comparison
- `useIsomorphicLayoutEffect` - SSR-safe layout effect
- `useIsFirstRender` - Check if first render

### Timing
- `useTimeout` - Timeout hook
- `useInterval` - Interval hook
- `useDebounce` - Debounce values
- `useThrottle` - Throttle values

### UI Interactions
- `useClickOutside` - Detect clicks outside
- `useHover` - Detect hover state
- `useFocus` - Detect focus state
- `useIntersectionObserver` - Intersection observer
- `useMediaQuery` - Media query hook
- `useWindowSize` - Window dimensions

### Network
- `useOnline` - Online/offline status
- `useNetwork` - Network information

### Device
- `useDevice` - Device detection

### Utilities
- `useCopyToClipboard` - Copy to clipboard
- `useURLParams` - URL parameters
- `useErrorHandler` - Error handling
- `useEventCallback` - Stable callback
- `useMemoizedCallback` - Memoized callback
- `usePerformance` - Performance monitoring
- `useLogger` - Logging hook
- `useAnalytics` - Analytics tracking
- `useI18n` - Internationalization

### Debug
- `useWhyDidYouUpdate` - Debug prop changes

## Usage

Import from the main index:

```typescript
import { useToggle, useDebounce, useLocalStorage } from '@/lib/hooks';
```

Or import directly for better tree-shaking:

```typescript
import { useToggle } from '@/lib/hooks/useToggle';
import { useDebounce } from '@/lib/hooks/useDebounce';
```



