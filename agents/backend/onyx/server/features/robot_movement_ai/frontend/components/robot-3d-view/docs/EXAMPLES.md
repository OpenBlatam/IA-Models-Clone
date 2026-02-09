# Robot 3D View - Examples

## 📚 Usage Examples

### Basic Usage

```tsx
import Robot3DView from '@/components/robot-3d-view';

export default function RobotPage() {
  return (
    <div className="container mx-auto p-4">
      <h1>Robot 3D Visualization</h1>
      <Robot3DView />
    </div>
  );
}
```

### Fullscreen Mode

```tsx
import Robot3DView from '@/components/robot-3d-view';

export default function FullscreenPage() {
  return (
    <Robot3DView 
      fullscreen={true}
      className="custom-class"
    />
  );
}
```

### With Type-Safe Props

```tsx
import Robot3DView from '@/components/robot-3d-view';
import { robot3DViewPropsSchema } from '@/components/robot-3d-view/schemas/validation-schemas';
import type { Robot3DViewProps } from '@/components/robot-3d-view/schemas/validation-schemas';

export default function TypedPage() {
  // Validate props at runtime
  const props: Robot3DViewProps = robot3DViewPropsSchema.parse({
    fullscreen: false,
    className: 'my-custom-class',
  });

  return <Robot3DView {...props} />;
}
```

### With Error Handling

```tsx
import Robot3DView from '@/components/robot-3d-view';
import { ErrorBoundary } from '@/components/robot-3d-view/utils/error-boundary';

export default function ErrorHandledPage() {
  return (
    <ErrorBoundary
      fallback={
        <div className="p-4 bg-red-100 text-red-800">
          Error loading 3D view. Please refresh the page.
        </div>
      }
      onError={(error) => {
        console.error('3D View Error:', error);
        // Send to error tracking service
      }}
    >
      <Robot3DView />
    </ErrorBoundary>
  );
}
```

### With Performance Monitoring

```tsx
'use client';

import { usePerformanceMonitor } from '@/components/robot-3d-view/hooks/use-performance-monitor';
import Robot3DView from '@/components/robot-3d-view';

export default function MonitoredPage() {
  const { metrics, isLowPerformance } = usePerformanceMonitor({
    enabled: true,
    onMetricsUpdate: (m) => {
      if (m.fps < 30) {
        console.warn('Low FPS detected:', m.fps);
      }
    },
  });

  return (
    <div>
      {isLowPerformance && (
        <div className="bg-yellow-100 p-2 mb-4">
          Performance warning: Low FPS ({metrics.fps})
        </div>
      )}
      <Robot3DView />
    </div>
  );
}
```

### Custom Position Validation

```tsx
'use client';

import { positionTo3D, createError } from '@/components/robot-3d-view/lib';
import { useRobotStore } from '@/lib/store/robotStore';

export default function CustomValidationPage() {
  const { currentPosition } = useRobotStore();

  const handlePosition = () => {
    try {
      const pos = positionTo3D(currentPosition, [0, 0, 0], true);
      console.log('Valid position:', pos);
    } catch (error) {
      if (createError.position('').constructor === error.constructor) {
        console.error('Position validation error:', error);
      }
    }
  };

  return (
    <div>
      <button onClick={handlePosition}>Validate Position</button>
      <Robot3DView />
    </div>
  );
}
```

### Keyboard Shortcuts

The component supports keyboard shortcuts:

- `S` - Toggle stats
- `G` - Toggle gizmo
- `H` - Toggle grid
- `O` - Toggle objects
- `R` - Toggle auto-rotate
- `P` - Screenshot
- `1` - Front camera
- `2` - Top camera
- `3` - Side camera
- `4` - Isometric camera

### Accessibility

The component is fully accessible:

- ARIA labels on all interactive elements
- Keyboard navigation support
- Screen reader compatible
- Focus management

```tsx
// All controls have proper ARIA labels
<button
  aria-label="Toggle statistics"
  onClick={toggleStats}
>
  Stats
</button>
```

### Responsive Design

The component is responsive and mobile-first:

```tsx
// Automatically adjusts height based on screen size
// Mobile: 400px
// Tablet: 500px
// Desktop: 600px
<Robot3DView fullscreen={false} />
```

### Testing

Example test for the component:

```tsx
import { render, screen } from '@testing-library/react';
import Robot3DView from '@/components/robot-3d-view';

describe('Robot3DView', () => {
  it('should render loading state', () => {
    render(<Robot3DView />);
    expect(screen.getByLabelText(/cargando/i)).toBeInTheDocument();
  });
});
```



