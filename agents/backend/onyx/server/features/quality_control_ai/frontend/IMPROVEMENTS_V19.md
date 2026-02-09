# Improvements V19

This document outlines the nineteenth round of improvements made to enhance the frontend application.

## New Custom Hooks

### useClickOutside
- **Purpose**: Detect clicks outside an element
- **Parameters**: Handler function
- **Returns**: Ref to attach to element
- **Features**:
  - Mouse and touch event support
  - Automatic cleanup
  - Type-safe refs
  - Useful for modals, dropdowns, popovers

### useDoubleClick
- **Purpose**: Distinguish between single and double clicks
- **Parameters**: Options with callbacks and delay
- **Returns**: Click handler function
- **Features**:
  - Configurable delay
  - Separate callbacks for single/double click
  - Automatic timeout handling
  - Useful for interactive elements

## New Utility Functions

### Diff Utilities (`lib/utils/diff.ts`)
- **diff**: Shallow diff between two objects
- **deepDiff**: Deep diff between nested objects
- **hasChanges**: Check if objects have differences
- **hasDeepChanges**: Check if nested objects have differences
- **Features**:
  - Object comparison
  - Change detection
  - Nested object support
  - Type-safe operations

### Merge Utilities (`lib/utils/merge.ts`)
- **merge**: Shallow merge objects
- **deepMerge**: Deep merge nested objects
- **mergeArrays**: Merge multiple arrays
- **mergeUnique**: Merge arrays with unique values
- **mergeBy**: Merge arrays by key function
- **Features**:
  - Object merging
  - Array merging
  - Deep merging support
  - Unique value handling
  - Key-based merging

## New UI Components

### DropdownMenu
- **Purpose**: Dropdown menu component
- **Features**:
  - Custom trigger element
  - Menu items with icons
  - Divider support
  - Disabled items
  - Click outside to close
  - Left/right alignment
  - Keyboard accessible
  - ARIA attributes

### Accordion
- **Purpose**: Accordion component for collapsible content
- **Features**:
  - Multiple items
  - Allow multiple open items
  - Default open items
  - Smooth animations
  - Keyboard accessible
  - ARIA attributes
  - Customizable styling

### Collapsible
- **Purpose**: Single collapsible section
- **Features**:
  - Default open state
  - Smooth animations
  - Customizable header/content
  - Keyboard accessible
  - ARIA attributes
  - Simple API

## Improvements Summary

### Custom Hooks
1. **useClickOutside**: Click outside detection
2. **useDoubleClick**: Single/double click distinction

### Utility Functions
- Object diff utilities
- Object merge utilities
- Array merge utilities

### UI Components
- DropdownMenu for menus
- Accordion for collapsible sections
- Collapsible for single sections

## Benefits

1. **Better User Experience**:
   - Dropdown menus
   - Collapsible content
   - Click outside to close
   - Single/double click handling

2. **Developer Experience**:
   - Reusable hooks
   - Utility functions
   - Pre-built components
   - Type-safe operations

3. **Code Quality**:
   - Consistent patterns
   - Accessible components
   - Reusable utilities
   - Type safety

4. **Functionality**:
   - Object comparison
   - Object merging
   - Interactive components
   - Event handling

## Usage Examples

### useClickOutside
```tsx
import { useClickOutside } from '@/lib/hooks';

const MyComponent = () => {
  const ref = useClickOutside<HTMLDivElement>(() => {
    console.log('Clicked outside!');
  });

  return <div ref={ref}>Content</div>;
};
```

### useDoubleClick
```tsx
import { useDoubleClick } from '@/lib/hooks';

const MyComponent = () => {
  const handleClick = useDoubleClick({
    onSingleClick: () => console.log('Single click'),
    onDoubleClick: () => console.log('Double click'),
    delay: 300,
  });

  return <button onClick={handleClick}>Click me</button>;
};
```

### Diff Utilities
```tsx
import { diff, deepDiff, hasChanges } from '@/lib/utils';

const oldObj = { name: 'John', age: 30 };
const newObj = { name: 'John', age: 31 };

const changes = diff(oldObj, newObj); // { age: 31 }
const hasDiff = hasChanges(oldObj, newObj); // true
```

### Merge Utilities
```tsx
import { merge, deepMerge, mergeUnique } from '@/lib/utils';

const obj1 = { a: 1, b: 2 };
const obj2 = { b: 3, c: 4 };
const merged = merge(obj1, obj2); // { a: 1, b: 3, c: 4 }

const arr1 = [1, 2, 3];
const arr2 = [3, 4, 5];
const unique = mergeUnique(arr1, arr2); // [1, 2, 3, 4, 5]
```

### DropdownMenu
```tsx
import { DropdownMenu } from '@/components/ui';
import { Settings, Edit, Trash } from 'lucide-react';

<DropdownMenu
  trigger={<Button>Actions</Button>}
  items={[
    { label: 'Edit', onClick: () => {}, icon: <Edit /> },
    { label: 'Delete', onClick: () => {}, icon: <Trash />, divider: true },
    { label: 'Settings', onClick: () => {}, icon: <Settings /> },
  ]}
  align="right"
/>
```

### Accordion
```tsx
import { Accordion } from '@/components/ui';

<Accordion
  items={[
    { title: 'Section 1', content: <p>Content 1</p>, defaultOpen: true },
    { title: 'Section 2', content: <p>Content 2</p> },
  ]}
  allowMultiple={false}
/>
```

### Collapsible
```tsx
import { Collapsible } from '@/components/ui';

<Collapsible title="Click to expand" defaultOpen={false}>
  <p>Collapsible content here</p>
</Collapsible>
```

These improvements add click outside detection, double click handling, object diff/merge utilities, and interactive UI components that enhance both user experience and developer productivity.

