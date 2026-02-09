# Improvements V23

This document outlines the twenty-third round of improvements made to enhance the frontend application.

## New Custom Hooks

### useImageLoader
- **Purpose**: Load images with loading and error states
- **Parameters**: Source, onLoad, onError callbacks
- **Returns**: Loading state, error state, and loaded image source
- **Features**:
  - Automatic image loading
  - Loading state tracking
  - Error handling
  - Callbacks for load/error events
  - Useful for image components

### useScript
- **Purpose**: Dynamically load external scripts
- **Parameters**: Script source, async, defer, callbacks
- **Returns**: Loading state (loading, ready, error)
- **Features**:
  - Dynamic script loading
  - Prevents duplicate loading
  - Async/defer support
  - Load/error callbacks
  - State tracking
  - Useful for third-party scripts

### useTitle
- **Purpose**: Set document title with cleanup
- **Parameters**: Title string
- **Returns**: void
- **Features**:
  - Sets document title
  - Automatic cleanup on unmount
  - Restores previous title
  - Useful for page titles

## Enhanced Color Utilities

### Color Utilities (`lib/utils/color.ts`)
- **hexToRgb**: Convert hex to RGB
- **rgbToHex**: Convert RGB to hex
- **hexToHsl**: Convert hex to HSL
- **rgbToHsl**: Convert RGB to HSL
- **hslToRgb**: Convert HSL to RGB
- **lighten**: Lighten color by percentage
- **darken**: Darken color by percentage
- **getContrastColor**: Get contrasting color (black/white)
- **isValidHex**: Validate hex color
- **mix**: Mix two colors
- **alpha**: Add alpha channel to hex color
- **Features**:
  - Color format conversion
  - Color manipulation
  - Contrast calculation
  - Color validation
  - Alpha channel support

## New UI Components

### CustomImage
- **Purpose**: Enhanced image component with loading states
- **Features**:
  - Uses useImageLoader hook
  - Loading skeleton
  - Error fallback
  - Next.js Image integration
  - Customizable fallback
  - Object fit options
  - Accessible

### ColorPicker
- **Purpose**: Color picker component
- **Features**:
  - Native color input
  - Hex input field
  - Real-time updates
  - Validation
  - Label support
  - Customizable styling
  - Accessible

## Improvements Summary

### Custom Hooks
1. **useImageLoader**: Image loading with states
2. **useScript**: Dynamic script loading
3. **useTitle**: Document title management

### Utility Functions
- Enhanced color utilities
- Color format conversion
- Color manipulation

### UI Components
- CustomImage for enhanced image display
- ColorPicker for color selection

## Benefits

1. **Better User Experience**:
   - Image loading states
   - Error handling
   - Color selection
   - Smooth interactions

2. **Developer Experience**:
   - Reusable image loader hook
   - Script loading hook
   - Title management hook
   - Color utilities
   - Pre-built components

3. **Code Quality**:
   - Type-safe operations
   - Reusable utilities
   - Accessible components
   - Consistent patterns

4. **Functionality**:
   - Image loading
   - Script management
   - Title management
   - Color manipulation
   - Color selection

## Usage Examples

### useImageLoader
```tsx
import { useImageLoader } from '@/lib/hooks';

const MyComponent = () => {
  const { isLoading, hasError, imageSrc } = useImageLoader({
    src: 'https://example.com/image.jpg',
    onLoad: () => console.log('Loaded'),
    onError: (error) => console.error(error),
  });

  if (isLoading) return <div>Loading...</div>;
  if (hasError) return <div>Error</div>;
  return <img src={imageSrc} alt="Image" />;
};
```

### useScript
```tsx
import { useScript } from '@/lib/hooks';

const MyComponent = () => {
  const status = useScript({
    src: 'https://example.com/script.js',
    async: true,
    onLoad: () => console.log('Script loaded'),
    onError: (error) => console.error(error),
  });

  return <div>Status: {status}</div>;
};
```

### useTitle
```tsx
import { useTitle } from '@/lib/hooks';

const MyPage = () => {
  useTitle('My Page Title');
  return <div>Content</div>;
};
```

### Color Utilities
```tsx
import { lighten, darken, mix, alpha, getContrastColor } from '@/lib/utils/color';

// Lighten
const lighter = lighten('#ff0000', 20); // Lighter red

// Darken
const darker = darken('#ff0000', 20); // Darker red

// Mix
const mixed = mix('#ff0000', '#0000ff', 50); // Purple

// Alpha
const transparent = alpha('#ff0000', 0.5); // rgba(255, 0, 0, 0.5)

// Contrast
const contrast = getContrastColor('#ff0000'); // '#000000' or '#FFFFFF'
```

### CustomImage
```tsx
import { CustomImage } from '@/components/ui';

<CustomImage
  src="https://example.com/image.jpg"
  alt="Description"
  width={400}
  height={300}
  objectFit="cover"
  onLoad={() => console.log('Loaded')}
  onError={(error) => console.error(error)}
  fallback={<div>Failed to load</div>}
/>
```

### ColorPicker
```tsx
import { ColorPicker } from '@/components/ui';

<ColorPicker
  value={color}
  onChange={setColor}
  label="Select color"
  showInput={true}
/>
```

These improvements add image loading capabilities, script management, title management, enhanced color utilities, and convenient UI components that enhance both user experience and developer productivity.

