# Improvements V12

This document outlines the twelfth round of improvements made to enhance the frontend application.

## New Custom Hooks

### useQueue
- **Purpose**: Manage queue data structure
- **Returns**: `{ queue, enqueue, dequeue, peek, clear, size, isEmpty }`
- **Features**:
  - FIFO (First In, First Out) operations
  - Enqueue items to back
  - Dequeue items from front
  - Peek at front item
  - Clear queue
  - Size and isEmpty helpers

### useStack
- **Purpose**: Manage stack data structure
- **Returns**: `{ stack, push, pop, peek, clear, size, isEmpty }`
- **Features**:
  - LIFO (Last In, First Out) operations
  - Push items to top
  - Pop items from top
  - Peek at top item
  - Clear stack
  - Size and isEmpty helpers

## New Math Utilities

### Math Utilities (`lib/utils/math.ts`)
- **clamp**: Clamp value between min and max
- **lerp**: Linear interpolation
- **normalize/denormalize**: Normalize values to 0-1 range
- **roundTo/floorTo/ceilTo**: Round to specific decimals
- **distance**: Calculate distance between two points
- **angle**: Calculate angle between two points
- **toRadians/toDegrees**: Convert between radians and degrees
- **sum/average/median**: Statistical functions
- **min/max**: Array min/max
- **range**: Generate number range
- **factorial**: Calculate factorial
- **gcd/lcm**: Greatest common divisor / Least common multiple
- **isPrime**: Check if number is prime

## New Time Utilities

### Time Utilities (`lib/utils/time.ts`)
- **formatDuration/formatDurationShort**: Format milliseconds to duration
- **parseDuration**: Parse duration string to milliseconds
- **addDays/addHours/addMinutes/addSeconds**: Add time to date
- **startOfDay/endOfDay**: Get start/end of day
- **startOfWeek/endOfWeek**: Get start/end of week
- **startOfMonth/endOfMonth**: Get start/end of month
- **startOfYear/endOfYear**: Get start/end of year
- **isSameDay/isSameWeek/isSameMonth/isSameYear**: Compare dates
- **differenceInDays/Hours/Minutes/Seconds**: Calculate time differences

## Improvements Summary

### Custom Hooks
1. **useQueue**: Queue data structure management
2. **useStack**: Stack data structure management

### Utility Functions
- Comprehensive math operations
- Date/time manipulation
- Duration formatting

## Benefits

1. **Better Developer Experience**:
   - Queue/Stack data structures
   - Math utilities for calculations
   - Time utilities for date handling

2. **Code Quality**:
   - Type-safe operations
   - Reusable utilities
   - Well-tested functions

3. **Functionality**:
   - Data structure management
   - Mathematical calculations
   - Date/time operations

## Usage Examples

### useQueue
```tsx
const { queue, enqueue, dequeue, peek, size, isEmpty } = useQueue<string>();

enqueue('task1');
enqueue('task2');
const first = peek(); // 'task1'
const processed = dequeue(); // 'task1'
```

### useStack
```tsx
const { stack, push, pop, peek, size, isEmpty } = useStack<number>();

push(1);
push(2);
const top = peek(); // 2
const popped = pop(); // 2
```

### Math Utilities
```tsx
import { clamp, lerp, distance, average, gcd, isPrime } from '@/lib/utils';

// Clamp
const clamped = clamp(value, 0, 100);

// Linear interpolation
const interpolated = lerp(0, 100, 0.5); // 50

// Distance
const dist = distance(0, 0, 3, 4); // 5

// Average
const avg = average([1, 2, 3, 4, 5]); // 3

// GCD
const commonDivisor = gcd(48, 18); // 6

// Prime check
const prime = isPrime(17); // true
```

### Time Utilities
```tsx
import {
  formatDuration,
  addDays,
  startOfWeek,
  isSameDay,
  differenceInDays,
} from '@/lib/utils';

// Format duration
const duration = formatDuration(3661000); // "1h 1m 1s"

// Add time
const tomorrow = addDays(new Date(), 1);

// Start of week
const weekStart = startOfWeek(new Date(), 0); // Sunday

// Compare dates
const same = isSameDay(date1, date2);

// Difference
const daysDiff = differenceInDays(date1, date2);
```

These improvements add data structure management, comprehensive math utilities, and time manipulation that enhance both developer productivity and application functionality.

