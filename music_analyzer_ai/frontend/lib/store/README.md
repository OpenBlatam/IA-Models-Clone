# Store Documentation

## Overview

The store uses Zustand for global state management with the following features:

- ✅ Immer for immutable updates
- ✅ Persistence for user preferences
- ✅ DevTools integration
- ✅ Selectors for optimized re-renders
- ✅ Type-safe state management

## Usage

### Basic Store Access

```typescript
import { useMusicStore } from '@/lib/store';

// Get entire state (not recommended for performance)
const state = useMusicStore();
```

### Using Selectors

```typescript
import {
  useCurrentTrack,
  usePlaybackState,
  usePlaybackActions,
} from '@/lib/store';

// Only re-renders when currentTrack changes
const track = useCurrentTrack();

// Only re-renders when playback state changes
const { isPlaying, currentTime } = usePlaybackState();

// Actions don't cause re-renders
const { setIsPlaying, moveToNext } = usePlaybackActions();
```

### Store Actions

```typescript
import { useMusicStore } from '@/lib/store';

function MyComponent() {
  const setCurrentTrack = useMusicStore((state) => state.setCurrentTrack);
  const addToQueue = useMusicStore((state) => state.addToQueue);
  
  const handleSelect = (track: Track) => {
    setCurrentTrack(track);
    addToQueue(track);
  };
}
```

## State Structure

### Playback State

```typescript
playback: {
  isPlaying: boolean;
  currentTime: number;
  duration: number;
  playbackSpeed: number;
  volume: number;
  isMuted: boolean;
  isShuffled: boolean;
  repeatMode: 'off' | 'one' | 'all';
}
```

### View Preferences

```typescript
viewPreferences: {
  viewMode: 'grid' | 'list' | 'compact';
  sortBy: string;
  sortOrder: 'asc' | 'desc';
  itemsPerPage: number;
}
```

## Persistence

The following state is persisted to localStorage:

- View preferences
- Playback settings (speed, volume, mute, repeat, shuffle)
- Recent searches
- Filters

## Selectors

All selectors are optimized to prevent unnecessary re-renders:

- `useCurrentTrack()` - Current track only
- `usePlaybackState()` - Playback state only
- `usePlaylistQueue()` - Queue only
- `usePlaybackActions()` - Actions only (no re-renders)
- `usePlaybackInfo()` - Computed playback info

