# Robot 3D View - API Documentation

## Overview

Complete API documentation for the Robot 3D View component.

## Core Components

### Robot3DView

Main component for rendering the 3D robot view.

```tsx
import { Robot3DView } from './components/robot-3d-view';

<Robot3DView
  currentPos={[0, 0, 0]}
  targetPos={[10, 10, 10]}
  status="idle"
/>
```

#### Props

- `currentPos: Position3D` - Current robot position
- `targetPos: Position3D | null` - Target position
- `status: RobotStatus` - Robot status
- `fullscreen?: boolean` - Enable fullscreen mode

## Hooks

### useRobot3DView

Main hook for 3D view state and actions.

```tsx
const {
  currentPos,
  targetPos,
  trajectory,
  config,
  toggleStats,
  toggleGizmo,
  // ... more
} = useRobot3DView(fullscreen);
```

### useEvents

Subscribe to events.

```tsx
useEvents(Events.CONFIG_CHANGED, (config) => {
  console.log('Config changed:', config);
}, []);
```

### useEmit

Emit events.

```tsx
const emit = useEmit();
emit(Events.CONFIG_CHANGED, newConfig);
```

### useI18n

Internationalization hook.

```tsx
const { t, setLanguage, currentLanguage } = useI18n();
<button>{t('common.save')}</button>
```

## Utilities

### EventManager

Event system for pub/sub pattern.

```tsx
import { eventManager, Events } from './utils/event-system';

// Subscribe
const subscription = eventManager.on(Events.CONFIG_CHANGED, (config) => {
  console.log(config);
});

// Emit
await eventManager.emit(Events.CONFIG_CHANGED, config);

// Unsubscribe
subscription.unsubscribe();
```

### MetricsManager

Metrics and statistics tracking.

```tsx
import { metricsManager } from './utils/metrics';

// Record metric
metricsManager.record('render.fps', 60);

// Get statistics
const average = metricsManager.getAverage('render.fps', 60000);
const min = metricsManager.getMin('render.fps', 60000);
const max = metricsManager.getMax('render.fps', 60000);
```

### I18nManager

Internationalization manager.

```tsx
import { i18nManager } from './utils/i18n';

// Translate
const text = i18nManager.t('common.save');

// Set language
i18nManager.setLanguage('es');
```

### Memoization

Advanced memoization utilities.

```tsx
import { memoize, memoizeAsync } from './utils/memoization';

// Memoize function
const memoized = memoize((x: number) => x * 2);
memoized(5); // Computed
memoized(5); // Cached

// Memoize async
const memoizedAsync = memoizeAsync(async (x: number) => x * 2);
await memoizedAsync(5);
```

## Event Types

```tsx
Events.CONFIG_CHANGED
Events.POSITION_CHANGED
Events.TARGET_CHANGED
Events.TRAJECTORY_UPDATED
Events.UI_OPENED
Events.UI_CLOSED
Events.CLICK
Events.HOVER
Events.KEYBOARD_SHORTCUT
Events.PERFORMANCE_WARNING
Events.ERROR
Events.WARNING
Events.INITIALIZED
Events.DESTROYED
```

## Keyboard Shortcuts

- `S` - Toggle stats
- `G` - Toggle gizmo
- `H` - Toggle grid
- `O` - Toggle objects
- `R` - Toggle auto-rotate
- `K` - Toggle stars
- `W` - Toggle waypoints
- `P` - Screenshot
- `1-4` - Camera presets
- `0` - Reset camera
- `F` - Toggle fullscreen
- `Ctrl+Z` - Undo
- `Ctrl+Shift+Z` - Redo
- `Ctrl+K` - Command palette
- `Ctrl+L` - Log viewer
- `Ctrl+M` - Metrics panel

## Configuration

### SceneConfig

```tsx
interface SceneConfig {
  showStats: boolean;
  showGizmo: boolean;
  showStars: boolean;
  showWaypoints: boolean;
  showGrid: boolean;
  showObjects: boolean;
  autoRotate: boolean;
  cameraPreset: 'front' | 'top' | 'side' | 'iso' | null;
  gridSize: number;
}
```

## Examples

See `docs/EXAMPLES.md` for more detailed examples.



