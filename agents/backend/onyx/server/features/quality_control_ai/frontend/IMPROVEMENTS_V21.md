# Improvements V21

This document outlines the twenty-first round of improvements made to enhance the frontend application.

## New Custom Hooks

### useSwipe
- **Purpose**: Detect swipe gestures (left, right, up, down)
- **Parameters**: Options with callbacks, threshold, and velocity threshold
- **Returns**: Swipe props and swiping state
- **Features**:
  - Four-direction swipe detection
  - Threshold for minimum swipe distance
  - Velocity threshold for swipe speed
  - Touch and mouse support
  - Useful for carousels, navigation, actions

### useWindowScroll
- **Purpose**: Track window scroll position
- **Returns**: Current scroll position (x, y)
- **Features**:
  - Real-time scroll tracking
  - SSR-safe
  - Passive event listeners
  - Automatic cleanup

### useScrollToTop
- **Purpose**: Scroll to top of page
- **Returns**: Scroll function
- **Features**:
  - Smooth scrolling
  - Simple API

### useScrollToBottom
- **Purpose**: Scroll to bottom of page
- **Returns**: Scroll function
- **Features**:
  - Smooth scrolling
  - Simple API

### useScrollTo
- **Purpose**: Scroll to specific position
- **Returns**: Scroll function with options
- **Features**:
  - Flexible scrolling options
  - Smooth scrolling support

## New Utility Functions

### Group Utilities (`lib/utils/group.ts`)
- **groupBy**: Group array by key function
- **groupByMultiple**: Group by multiple keys
- **partition**: Split array into two groups
- **chunk**: Split array into chunks
- **split**: Split array at index
- **splitBy**: Split array by predicate
- **groupByCount**: Group into fixed-size chunks
- **Features**:
  - Multiple grouping strategies
  - Flexible partitioning
  - Type-safe operations

### Aggregate Utilities (`lib/utils/aggregate.ts`)
- **sum**: Sum values by key function
- **average**: Calculate average
- **min**: Find minimum value
- **max**: Find maximum value
- **count**: Count items (with optional predicate)
- **countBy**: Count items by key
- **aggregate**: Generic aggregation
- **aggregateBy**: Aggregate by key
- **median**: Calculate median
- **mode**: Find most common value
- **Features**:
  - Statistical operations
  - Flexible aggregation
  - Type-safe operations

## New UI Components

### ScrollToTop
- **Purpose**: Floating button to scroll to top
- **Features**:
  - Auto-show/hide based on scroll position
  - Configurable threshold
  - Smooth scrolling
  - Fixed positioning
  - Accessible
  - Customizable styling

### DataTable
- **Purpose**: Feature-rich data table component
- **Features**:
  - Sortable columns
  - Custom cell rendering
  - Loading state with skeletons
  - Empty state message
  - Row click handling
  - Custom row styling
  - Responsive design
  - Accessible
  - Type-safe with generics

## Improvements Summary

### Custom Hooks
1. **useSwipe**: Swipe gesture detection
2. **useWindowScroll**: Window scroll tracking
3. **useScrollToTop**: Scroll to top
4. **useScrollToBottom**: Scroll to bottom
5. **useScrollTo**: Flexible scrolling

### Utility Functions
- Comprehensive grouping utilities
- Statistical aggregation utilities

### UI Components
- ScrollToTop for navigation
- DataTable for data display

## Benefits

1. **Better User Experience**:
   - Swipe gestures
   - Scroll navigation
   - Data tables
   - Smooth interactions

2. **Developer Experience**:
   - Reusable gesture hooks
   - Scroll utilities
   - Grouping utilities
   - Aggregation utilities
   - Pre-built data table

3. **Code Quality**:
   - Type-safe operations
   - Reusable utilities
   - Accessible components
   - Consistent patterns

4. **Functionality**:
   - Gesture handling
   - Scroll management
   - Data grouping
   - Statistical operations
   - Data display

## Usage Examples

### useSwipe
```tsx
import { useSwipe } from '@/lib/hooks';

const MyComponent = () => {
  const { swipeProps, isSwiping } = useSwipe({
    onSwipeLeft: () => console.log('Swiped left'),
    onSwipeRight: () => console.log('Swiped right'),
    threshold: 50,
    velocityThreshold: 0.3,
  });

  return (
    <div {...swipeProps} className={isSwiping ? 'opacity-50' : ''}>
      Swipe me
    </div>
  );
};
```

### useWindowScroll
```tsx
import { useWindowScroll } from '@/lib/hooks';

const MyComponent = () => {
  const { x, y } = useWindowScroll();
  return <div>Scroll position: {x}, {y}</div>;
};
```

### Scroll Utilities
```tsx
import { useScrollToTop, useScrollToBottom, useScrollTo } from '@/lib/hooks';

const MyComponent = () => {
  const scrollToTop = useScrollToTop();
  const scrollToBottom = useScrollToBottom();
  const scrollTo = useScrollTo();

  return (
    <div>
      <button onClick={scrollToTop}>Top</button>
      <button onClick={scrollToBottom}>Bottom</button>
      <button onClick={() => scrollTo({ top: 500, behavior: 'smooth' })}>
        Scroll to 500px
      </button>
    </div>
  );
};
```

### Group Utilities
```tsx
import { groupBy, partition, chunk } from '@/lib/utils';

// Group by property
const grouped = groupBy(users, (user) => user.role);

// Partition
const [active, inactive] = partition(users, (user) => user.active);

// Chunk
const chunks = chunk(items, 10);
```

### Aggregate Utilities
```tsx
import { sum, average, count, median } from '@/lib/utils';

// Sum
const total = sum(orders, (order) => order.total);

// Average
const avg = average(scores, (score) => score.value);

// Count
const count = count(users, (user) => user.active);

// Median
const median = median(prices, (price) => price.value);
```

### ScrollToTop
```tsx
import { ScrollToTop } from '@/components/ui';

<ScrollToTop threshold={400} />
```

### DataTable
```tsx
import { DataTable } from '@/components/ui';

<DataTable
  data={users}
  columns={[
    { key: 'name', header: 'Name', sortable: true },
    { key: 'email', header: 'Email', sortable: true },
    {
      key: 'status',
      header: 'Status',
      render: (value) => <Badge>{value}</Badge>,
    },
  ]}
  onSort={(key, direction) => handleSort(key, direction)}
  sortKey={sortKey}
  sortDirection={sortDirection}
  onRowClick={(row) => handleRowClick(row)}
  loading={isLoading}
/>
```

These improvements add swipe gesture detection, scroll management utilities, comprehensive grouping and aggregation functions, and powerful data table component that enhance both user experience and developer productivity.

