# Guía de Gestión de Estado

## 📋 Overview

El sistema de gestión de estado utiliza **Zustand** para estado global y **React Query** para estado del servidor, siguiendo las mejores prácticas de Next.js 14.

## 🏗️ Arquitectura

### Estado Global (Zustand)

- **Música**: Playback, queue, preferencias, filtros
- **Persistencia**: Preferencias de usuario en localStorage
- **Optimización**: Selectores para prevenir re-renders innecesarios

### Estado del Servidor (React Query)

- **Caché**: Datos de API con invalidación inteligente
- **Sincronización**: Actualización automática y refetch
- **Optimistic Updates**: Actualizaciones optimistas para mejor UX

## 🎯 Zustand Store

### Uso Básico

```typescript
import { useMusicStore } from '@/lib/store';

// Acceder a todo el estado (no recomendado para performance)
const state = useMusicStore();

// Usar selectores optimizados (recomendado)
const currentTrack = useCurrentTrack();
const playback = usePlaybackState();
```

### Selectores Optimizados

Los selectores previenen re-renders innecesarios:

```typescript
import {
  useCurrentTrack,
  usePlaybackState,
  usePlaylistQueue,
  usePlaybackActions,
  usePlaybackInfo,
} from '@/lib/store';

function PlayerComponent() {
  // Solo se re-renderiza cuando currentTrack cambia
  const track = useCurrentTrack();
  
  // Solo se re-renderiza cuando playback cambia
  const { isPlaying, currentTime } = usePlaybackState();
  
  // Solo acciones, no causa re-renders
  const { setIsPlaying, moveToNext } = usePlaybackActions();
  
  // Estado computado
  const { progress, hasNext, hasPrevious } = usePlaybackInfo();
  
  return (
    <div>
      <button onClick={() => setIsPlaying(!isPlaying)}>
        {isPlaying ? 'Pause' : 'Play'}
      </button>
    </div>
  );
}
```

### Acciones del Store

```typescript
import { useMusicStore } from '@/lib/store';

function MyComponent() {
  const setCurrentTrack = useMusicStore((state) => state.setCurrentTrack);
  const addToQueue = useMusicStore((state) => state.addToQueue);
  const toggleShuffle = useMusicStore((state) => state.toggleShuffle);
  
  const handleTrackSelect = (track: Track) => {
    setCurrentTrack(track);
    addToQueue(track);
  };
}
```

### Persistencia

El store persiste automáticamente:

```typescript
// Se persiste automáticamente en localStorage
const { viewMode, playbackSpeed, recentSearches } = useMusicStore(
  (state) => ({
    viewMode: state.viewPreferences.viewMode,
    playbackSpeed: state.playback.playbackSpeed,
    recentSearches: state.recentSearches,
  })
);
```

## 🔄 React Query

### Uso Básico

```typescript
import { useQuery } from '@tanstack/react-query';
import { searchTracks } from '@/lib/api';
import { QUERY_KEYS } from '@/lib/constants';

function SearchComponent() {
  const { data, isLoading, error } = useQuery({
    queryKey: QUERY_KEYS.MUSIC.SEARCH(query),
    queryFn: () => searchTracks(query, 10),
    enabled: query.length > 0,
  });
}
```

### Hooks Mejorados

```typescript
import { useQueryWithErrorHandling } from '@/lib/hooks';

// Manejo automático de errores con toast
const { data, isLoading } = useQueryWithErrorHandling(
  QUERY_KEYS.MUSIC.SEARCH(query),
  () => searchTracks(query, 10),
  {
    showErrorToast: true,
    errorMessage: 'Error al buscar canciones',
  }
);
```

### Mutations con Manejo Automático

```typescript
import { useMutationWithHandling } from '@/lib/hooks';
import { addToFavorites } from '@/lib/api';

const addFavorite = useMutationWithHandling(
  (trackId: string) => addToFavorites('user123', trackId, 'Track', ['Artist']),
  {
    showSuccessToast: true,
    successMessage: 'Agregado a favoritos',
    showErrorToast: true,
    invalidateQueries: [QUERY_KEYS.MUSIC.FAVORITES()],
  }
);

// Usar
addFavorite.mutate('track-id');
```

### Optimistic Updates

```typescript
import { useOptimisticMutation } from '@/lib/hooks';

const toggleFavorite = useOptimisticMutation(
  (trackId: string) => toggleFavoriteTrack(trackId),
  {
    queryKey: QUERY_KEYS.MUSIC.FAVORITES(),
    optimisticUpdate: (trackId) => {
      // Actualización optimista
      return { ...currentData, [trackId]: !currentData[trackId] };
    },
    showSuccessToast: true,
  }
);
```

## 🎣 Hooks Personalizados

### useMusicState

Hook que integra Zustand y React Query:

```typescript
import { useMusicState } from '@/app/music/hooks/use-music-state';

function MusicPage() {
  const {
    activeTab,
    selectedTrack,
    analysisData,
    isLoadingAnalysis,
    handleTrackSelect,
  } = useMusicState();
  
  // El estado está sincronizado entre componentes
}
```

## 📊 Estructura del Estado

### Music Store

```typescript
interface MusicState {
  // Current track
  currentTrack: Track | null;
  currentTrackIndex: number;
  
  // Playlist queue
  playlistQueue: Track[];
  playlistHistory: Track[];
  
  // Playback state
  playback: {
    isPlaying: boolean;
    currentTime: number;
    duration: number;
    playbackSpeed: number;
    volume: number;
    isMuted: boolean;
    isShuffled: boolean;
    repeatMode: 'off' | 'one' | 'all';
  };
  
  // View preferences
  viewPreferences: {
    viewMode: 'grid' | 'list' | 'compact';
    sortBy: string;
    sortOrder: 'asc' | 'desc';
    itemsPerPage: number;
  };
  
  // Recent searches
  recentSearches: string[];
  
  // Filters
  filters: FilterState;
  
  // Selection
  selectedTracks: Set<string>;
  isSelectMode: boolean;
}
```

## ⚡ Optimizaciones

### 1. Selectores Específicos

```typescript
// ❌ Mal: Re-renderiza en cualquier cambio
const state = useMusicStore();

// ✅ Bien: Solo re-renderiza cuando currentTrack cambia
const track = useCurrentTrack();
```

### 2. useShallow para Múltiples Valores

```typescript
// ✅ Usar shallow para objetos/arrays
const { isPlaying, currentTime } = useMusicStore(
  (state) => ({
    isPlaying: state.playback.isPlaying,
    currentTime: state.playback.currentTime,
  }),
  shallow
);
```

### 3. Separar Acciones de Estado

```typescript
// ✅ Acciones no causan re-renders
const { setIsPlaying, moveToNext } = usePlaybackActions();
```

### 4. React Query Caching

```typescript
// ✅ Configurar staleTime y gcTime apropiadamente
useQuery({
  queryKey: QUERY_KEYS.MUSIC.ANALYSIS(trackId),
  queryFn: () => analyzeTrack({ trackId }),
  staleTime: 5 * 60 * 1000, // 5 minutos
  gcTime: 10 * 60 * 1000, // 10 minutos
});
```

## 🔄 Sincronización

### Entre Componentes

El estado global se sincroniza automáticamente:

```typescript
// Componente A
function ComponentA() {
  const setCurrentTrack = useMusicStore((state) => state.setCurrentTrack);
  setCurrentTrack(track);
}

// Componente B (se actualiza automáticamente)
function ComponentB() {
  const track = useCurrentTrack(); // Recibe el track actualizado
}
```

### Con React Query

```typescript
// Invalidar queries después de mutations
const queryClient = useQueryClient();

const addFavorite = useMutation({
  mutationFn: addToFavorites,
  onSuccess: () => {
    queryClient.invalidateQueries({ queryKey: QUERY_KEYS.MUSIC.FAVORITES() });
  },
});
```

## 🐛 Debugging

### Redux DevTools

Zustand está configurado con Redux DevTools:

1. Instala la extensión Redux DevTools
2. Abre las DevTools
3. Selecciona "MusicStore" en el dropdown
4. Verás todas las acciones y cambios de estado

### React Query DevTools

```typescript
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';

// En providers.tsx
<QueryClientProvider client={queryClient}>
  {children}
  <ReactQueryDevtools initialIsOpen={false} />
</QueryClientProvider>
```

## ✅ Mejores Prácticas

1. **Usa selectores específicos**: Previene re-renders innecesarios
2. **Separa estado local y global**: Usa useState para UI local, Zustand para global
3. **Usa React Query para datos del servidor**: Caché automático y sincronización
4. **Persiste solo lo necesario**: No persistas estado temporal
5. **Usa computed values**: Deriva estado en lugar de almacenarlo
6. **Optimistic updates**: Mejora la UX con actualizaciones optimistas

## 🔗 Recursos

- [Zustand Documentation](https://github.com/pmndrs/zustand)
- [React Query Documentation](https://tanstack.com/query/latest)
- [Zustand with Immer](https://github.com/pmndrs/zustand#using-immer)
- [React Query Best Practices](https://tanstack.com/query/latest/docs/react/guides/best-practices)

