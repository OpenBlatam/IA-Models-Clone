# Improvements V25

This document outlines the twenty-fifth round of improvements made to enhance the frontend application.

## New Custom Hooks

### useHotkeys
- **Purpose**: Register keyboard shortcuts/hotkeys
- **Parameters**: Array of hotkey configurations
- **Returns**: void
- **Features**:
  - Multiple hotkeys support
  - Modifier keys (Ctrl, Shift, Alt, Meta)
  - Key matching
  - Event prevention
  - Automatic cleanup
  - Useful for keyboard shortcuts

### useBeforeUnload
- **Purpose**: Warn user before leaving page
- **Parameters**: Options with enabled flag and message
- **Returns**: void
- **Features**:
  - Before unload warning
  - Custom message
  - Enable/disable toggle
  - Automatic cleanup
  - Useful for unsaved changes

### useVisibilityChange
- **Purpose**: Track page visibility (tab focus)
- **Parameters**: None
- **Returns**: Boolean indicating if page is visible
- **Features**:
  - Visibility API integration
  - Real-time tracking
  - SSR-safe
  - Automatic cleanup
  - Useful for pausing/resuming activities

## New Utility Functions

### File Utilities (`lib/utils/file.ts`)
- **getFileExtension**: Get file extension
- **getFileName**: Get filename from path
- **getFileSize**: Get file size
- **formatFileSize**: Format bytes to human-readable
- **isValidImageFile**: Check if file is image
- **isValidVideoFile**: Check if file is video
- **isValidAudioFile**: Check if file is audio
- **isValidPdfFile**: Check if file is PDF
- **isValidTextFile**: Check if file is text
- **readFileAsText**: Read file as text
- **readFileAsDataURL**: Read file as data URL
- **readFileAsArrayBuffer**: Read file as array buffer
- **downloadFile**: Download blob as file
- **createFileFromText**: Create file from text
- **Features**:
  - File type validation
  - File reading
  - File creation
  - File download
  - Type-safe operations

## New UI Components

### FileUpload
- **Purpose**: File upload component with drag & drop
- **Features**:
  - Drag & drop support
  - File type validation
  - Max size validation
  - Multiple file support
  - File preview
  - Remove files
  - Customizable styling
  - Accessible
  - Uses react-dropzone

### KeyboardShortcuts
- **Purpose**: Component for registering keyboard shortcuts
- **Features**:
  - Multiple shortcuts
  - Modifier keys
  - Uses useHotkeys hook
  - Invisible component
  - Useful for global shortcuts

## Improvements Summary

### Custom Hooks
1. **useHotkeys**: Keyboard shortcuts
2. **useBeforeUnload**: Page unload warning
3. **useVisibilityChange**: Page visibility tracking

### Utility Functions
- Comprehensive file utilities
- File type validation
- File reading/writing

### UI Components
- FileUpload for file uploads
- KeyboardShortcuts for shortcuts

## Benefits

1. **Better User Experience**:
   - Keyboard shortcuts
   - File upload with drag & drop
   - Page visibility awareness
   - Unsaved changes warning

2. **Developer Experience**:
   - Reusable hooks
   - File utilities
   - Pre-built components
   - Simple APIs

3. **Code Quality**:
   - Type-safe operations
   - Reusable utilities
   - Accessible components
   - Consistent patterns

4. **Functionality**:
   - Keyboard shortcuts
   - File handling
   - Page visibility
   - User warnings

## Usage Examples

### useHotkeys
```tsx
import { useHotkeys } from '@/lib/hooks';

const MyComponent = () => {
  useHotkeys([
    {
      key: 's',
      ctrl: true,
      callback: () => console.log('Save'),
    },
    {
      key: 'Escape',
      callback: () => console.log('Close'),
    },
  ]);

  return <div>Content</div>;
};
```

### useBeforeUnload
```tsx
import { useBeforeUnload } from '@/lib/hooks';

const MyComponent = () => {
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);

  useBeforeUnload({
    enabled: hasUnsavedChanges,
    message: 'You have unsaved changes. Are you sure you want to leave?',
  });

  return <div>Content</div>;
};
```

### useVisibilityChange
```tsx
import { useVisibilityChange } from '@/lib/hooks';

const MyComponent = () => {
  const isVisible = useVisibilityChange();

  useEffect(() => {
    if (!isVisible) {
      // Pause activity
    } else {
      // Resume activity
    }
  }, [isVisible]);

  return <div>Content</div>;
};
```

### File Utilities
```tsx
import {
  formatFileSize,
  isValidImageFile,
  readFileAsText,
  downloadFile,
} from '@/lib/utils';

// Format size
const size = formatFileSize(1024 * 1024); // "1 MB"

// Validate
const isValid = isValidImageFile(file);

// Read file
const text = await readFileAsText(file);

// Download
downloadFile(blob, 'file.pdf');
```

### FileUpload
```tsx
import { FileUpload } from '@/components/ui';

<FileUpload
  onUpload={(files) => console.log(files)}
  accept="image/*"
  multiple={true}
  maxSize={5 * 1024 * 1024} // 5MB
  showPreview={true}
  label="Upload images"
/>
```

### KeyboardShortcuts
```tsx
import { KeyboardShortcuts } from '@/components/ui';

<KeyboardShortcuts
  shortcuts={[
    {
      keys: ['Ctrl', 'S'],
      description: 'Save',
      action: handleSave,
      ctrl: true,
    },
    {
      keys: ['Escape'],
      description: 'Close',
      action: handleClose,
    },
  ]}
/>
```

These improvements add keyboard shortcuts, file upload capabilities, page visibility tracking, and comprehensive file utilities that enhance both user experience and developer productivity.

