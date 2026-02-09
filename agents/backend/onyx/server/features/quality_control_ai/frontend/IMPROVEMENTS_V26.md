# Improvements V26

This document outlines the twenty-sixth round of improvements made to enhance the frontend application.

## New Custom Hooks

### usePrint
- **Purpose**: Trigger browser print dialog
- **Parameters**: None
- **Returns**: Print function
- **Features**:
  - Simple print trigger
  - Browser native print dialog
  - Useful for printing pages/content

### useShare
- **Purpose**: Use Web Share API to share content
- **Parameters**: Share data (title, text, url, files)
- **Returns**: Share function, support status, error
- **Features**:
  - Web Share API integration
  - Share text, URLs, files
  - Support detection
  - Error handling
  - Useful for social sharing

### useWakeLock
- **Purpose**: Prevent screen from sleeping
- **Parameters**: None
- **Returns**: Request/release functions, status, error
- **Features**:
  - Wake Lock API integration
  - Screen wake lock
  - Auto-release on unmount
  - Support detection
  - Error handling
  - Useful for media players, presentations

## New Utility Functions

### Device Utilities (`lib/utils/device.ts`)
- **isMobile**: Check if device is mobile
- **isTablet**: Check if device is tablet
- **isDesktop**: Check if device is desktop
- **isIOS**: Check if device is iOS
- **isAndroid**: Check if device is Android
- **isWindows**: Check if OS is Windows
- **isMac**: Check if OS is Mac
- **isLinux**: Check if OS is Linux
- **getDeviceType**: Get device type (mobile/tablet/desktop)
- **getOS**: Get operating system
- **isTouchDevice**: Check if device has touch support
- **isStandalone**: Check if app is in standalone mode (PWA)
- **Features**:
  - Device detection
  - OS detection
  - Touch detection
  - PWA detection
  - SSR-safe

## New UI Components

### ShareButton
- **Purpose**: Button component for sharing content
- **Features**:
  - Uses Web Share API
  - Share title, text, URL, files
  - Auto-hides if not supported
  - Customizable styling
  - Callbacks for share/error
  - Accessible

### PrintButton
- **Purpose**: Button component for printing
- **Features**:
  - Triggers browser print
  - Customizable styling
  - Callback on print
  - Accessible
  - Icon and text support

## Improvements Summary

### Custom Hooks
1. **usePrint**: Print functionality
2. **useShare**: Web Share API
3. **useWakeLock**: Screen wake lock

### Utility Functions
- Comprehensive device detection
- OS detection
- Touch detection
- PWA detection

### UI Components
- ShareButton for sharing
- PrintButton for printing

## Benefits

1. **Better User Experience**:
   - Native sharing
   - Print functionality
   - Screen wake lock
   - Device-aware features

2. **Developer Experience**:
   - Reusable hooks
   - Device utilities
   - Pre-built components
   - Simple APIs

3. **Code Quality**:
   - Type-safe operations
   - Reusable utilities
   - Accessible components
   - Consistent patterns

4. **Functionality**:
   - Web APIs integration
   - Device detection
   - Sharing capabilities
   - Print capabilities

## Usage Examples

### usePrint
```tsx
import { usePrint } from '@/lib/hooks';

const MyComponent = () => {
  const { print } = usePrint();

  return <button onClick={print}>Print</button>;
};
```

### useShare
```tsx
import { useShare } from '@/lib/hooks';

const MyComponent = () => {
  const { share, isSupported, error } = useShare();

  const handleShare = async () => {
    await share({
      title: 'My Article',
      text: 'Check out this article',
      url: 'https://example.com/article',
    });
  };

  if (!isSupported) return null;

  return <button onClick={handleShare}>Share</button>;
};
```

### useWakeLock
```tsx
import { useWakeLock } from '@/lib/hooks';

const MyComponent = () => {
  const { request, release, isActive, isSupported } = useWakeLock();

  useEffect(() => {
    if (isSupported) {
      request();
    }
    return () => release();
  }, []);

  return <div>Screen will stay awake</div>;
};
```

### Device Utilities
```tsx
import {
  isMobile,
  isTablet,
  isDesktop,
  getDeviceType,
  getOS,
  isTouchDevice,
} from '@/lib/utils';

// Device type
if (isMobile()) {
  // Mobile-specific code
}

// OS
const os = getOS(); // 'ios' | 'android' | 'windows' | 'mac' | 'linux'

// Touch
if (isTouchDevice()) {
  // Touch-specific code
}
```

### ShareButton
```tsx
import { ShareButton } from '@/components/ui';

<ShareButton
  title="My Article"
  text="Check out this article"
  url="https://example.com/article"
  onShare={() => console.log('Shared')}
  onError={(error) => console.error(error)}
  showText={true}
/>
```

### PrintButton
```tsx
import { PrintButton } from '@/components/ui';

<PrintButton
  onPrint={() => console.log('Printing')}
  showText={true}
  variant="ghost"
/>
```

These improvements add Web API integrations (Share, Print, Wake Lock), comprehensive device detection utilities, and convenient UI components that enhance both user experience and developer productivity.

