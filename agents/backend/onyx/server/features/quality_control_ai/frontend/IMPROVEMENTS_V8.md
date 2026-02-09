# Improvements V8

This document outlines the eighth round of improvements made to enhance the frontend application.

## New Custom Hooks

### useAnimationFrame
- **Purpose**: Execute callback on each animation frame
- **Parameters**: 
  - `callback`: Function receiving deltaTime
  - `isActive`: Boolean to enable/disable
- **Features**:
  - Smooth animations
  - Delta time calculation
  - Automatic cleanup
  - Performance optimized

### useSize
- **Purpose**: Track element size using ResizeObserver
- **Returns**: `{ width, height }`
- **Features**:
  - Real-time size tracking
  - ResizeObserver-based
  - Automatic cleanup
  - Type-safe ref

### useRect
- **Purpose**: Track element's bounding rectangle
- **Returns**: Complete rect object with position and size
- **Features**:
  - Tracks position and size
  - Updates on scroll and resize
  - ResizeObserver integration
  - Complete bounding box info

### useToggleState
- **Purpose**: Enhanced toggle state with multiple controls
- **Returns**: `[state, { toggle, setTrue, setFalse, set }]`
- **Features**:
  - Multiple control methods
  - More flexible than basic toggle
  - Type-safe

## New URL Utilities

### URL Utilities (`lib/utils/url.ts`)
- **getQueryParam**: Get single query parameter
- **getAllQueryParams**: Get all query parameters as object
- **setQueryParam**: Set query parameter
- **removeQueryParam**: Remove query parameter
- **buildQueryString**: Build query string from object
- **parseQueryString**: Parse query string to object
- **updateUrl**: Update multiple query params in URL
- **getHash/setHash**: Get/set URL hash
- **isValidUrl**: Validate URL format
- **getDomain**: Extract domain from URL
- **getPathname**: Get pathname from URL

## New Cookie Utilities

### Cookie Utilities (`lib/utils/cookie.ts`)
- **setCookie**: Set cookie with options
- **getCookie**: Get cookie value
- **getAllCookies**: Get all cookies as object
- **removeCookie**: Remove cookie
- **hasCookie**: Check if cookie exists
- **Features**:
  - Full cookie options support (expires, path, domain, secure, sameSite)
  - Automatic encoding/decoding
  - Type-safe operations

## Improvements Summary

### Custom Hooks
1. **useAnimationFrame**: Smooth animations with delta time
2. **useSize**: Element size tracking
3. **useRect**: Complete bounding rectangle tracking
4. **useToggleState**: Enhanced toggle with multiple controls

### Utility Functions
- Comprehensive URL manipulation
- Cookie management
- Query parameter handling

## Benefits

1. **Better User Experience**:
   - Smooth animations
   - Responsive layouts with size tracking
   - Better URL management

2. **Developer Experience**:
   - Simple URL utilities
   - Cookie management
   - Element tracking hooks

3. **Code Quality**:
   - Type-safe operations
   - Reusable utilities
   - Performance optimized

4. **Functionality**:
   - Animation support
   - Size/position tracking
   - URL state management
   - Cookie handling

## Usage Examples

### useAnimationFrame
```tsx
const [position, setPosition] = useState(0);

useAnimationFrame((deltaTime) => {
  setPosition((prev) => prev + deltaTime * 0.01);
}, true);

return <div style={{ transform: `translateX(${position}px)` }}>Animated</div>;
```

### useSize
```tsx
const containerRef = useRef<HTMLDivElement>(null);
const { width, height } = useSize(containerRef);

return (
  <div ref={containerRef}>
    Size: {width}x{height}
  </div>
);
```

### useRect
```tsx
const elementRef = useRef<HTMLDivElement>(null);
const rect = useRect(elementRef);

if (rect) {
  console.log('Position:', rect.top, rect.left);
  console.log('Size:', rect.width, rect.height);
}
```

### useToggleState
```tsx
const [isOpen, { toggle, setTrue, setFalse }] = useToggleState(false);

return (
  <>
    <button onClick={toggle}>Toggle</button>
    <button onClick={setTrue}>Open</button>
    <button onClick={setFalse}>Close</button>
  </>
);
```

### URL Utilities
```tsx
import { getQueryParam, updateUrl, setHash } from '@/lib/utils';

// Get query param
const page = getQueryParam('page'); // "1"

// Update URL
updateUrl({ page: '2', filter: 'active' });

// Set hash
setHash('section-1');
```

### Cookie Utilities
```tsx
import { setCookie, getCookie, removeCookie } from '@/lib/utils';

// Set cookie
setCookie('theme', 'dark', {
  expires: 30, // days
  path: '/',
  secure: true,
  sameSite: 'strict',
});

// Get cookie
const theme = getCookie('theme');

// Remove cookie
removeCookie('theme');
```

These improvements add animation support, element tracking, URL management, and cookie handling that enhance both user experience and developer productivity.

