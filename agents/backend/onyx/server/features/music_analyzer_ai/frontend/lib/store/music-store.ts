/**
 * Zustand store for music-related global state.
 * Manages global music state that needs to be shared across components.
 * Optimized with selectors and proper state structure.
 */

import { create } from 'zustand';
import { devtools, persist, subscribeWithSelector } from 'zustand/middleware';
import { type Track } from '@/lib/api/types';
import { STORAGE_KEYS } from '@/lib/constants';

/**
 * Playback state interface.
 */
interface PlaybackState {
  isPlaying: boolean;
  currentTime: number;
  duration: number;
  playbackSpeed: number;
  volume: number;
  isMuted: boolean;
  isShuffled: boolean;
  repeatMode: 'off' | 'one' | 'all';
}

/**
 * View preferences interface.
 */
interface ViewPreferences {
  viewMode: 'grid' | 'list' | 'compact';
  sortBy: string;
  sortOrder: 'asc' | 'desc';
  itemsPerPage: number;
}

/**
 * Filter state interface.
 */
interface FilterState {
  genre: string[];
  year: { min: number; max: number } | null;
  bpm: { min: number; max: number } | null;
  energy: { min: number; max: number } | null;
  custom: Record<string, unknown>;
}

/**
 * Music state interface.
 */
interface MusicState {
  // Current track
  currentTrack: Track | null;
  currentTrackIndex: number;

  // Playlist queue
  playlistQueue: Track[];
  playlistHistory: Track[];

  // Playback state
  playback: PlaybackState;

  // View preferences
  viewPreferences: ViewPreferences;

  // Recent searches
  recentSearches: string[];
  maxRecentSearches: number;

  // Filters
  filters: FilterState;

  // UI state
  selectedTracks: Set<string>;
  isSelectMode: boolean;

  // Actions - Current track
  setCurrentTrack: (track: Track | null) => void;
  setCurrentTrackIndex: (index: number) => void;

  // Actions - Playlist queue
  setPlaylistQueue: (queue: Track[]) => void;
  addToQueue: (track: Track, position?: 'start' | 'end') => void;
  addMultipleToQueue: (tracks: Track[], position?: 'start' | 'end') => void;
  removeFromQueue: (trackId: string) => void;
  clearQueue: () => void;
  reorderQueue: (fromIndex: number, toIndex: number) => void;
  moveToNext: () => void;
  moveToPrevious: () => void;
  moveToTrack: (trackId: string) => void;

  // Actions - Playback
  setIsPlaying: (playing: boolean) => void;
  setCurrentTime: (time: number) => void;
  setDuration: (duration: number) => void;
  setPlaybackSpeed: (speed: number) => void;
  setVolume: (volume: number) => void;
  toggleMute: () => void;
  toggleShuffle: () => void;
  setRepeatMode: (mode: 'off' | 'one' | 'all') => void;
  resetPlayback: () => void;

  // Actions - View preferences
  setViewMode: (mode: 'grid' | 'list' | 'compact') => void;
  setSortBy: (field: string) => void;
  setSortOrder: (order: 'asc' | 'desc') => void;
  setItemsPerPage: (count: number) => void;
  resetViewPreferences: () => void;

  // Actions - Recent searches
  addRecentSearch: (query: string) => void;
  removeRecentSearch: (query: string) => void;
  clearRecentSearches: () => void;

  // Actions - Filters
  setFilters: (filters: Partial<FilterState>) => void;
  setFilterValue: (key: keyof FilterState, value: unknown) => void;
  clearFilters: () => void;
  resetFilters: () => void;

  // Actions - Selection
  toggleTrackSelection: (trackId: string) => void;
  selectAllTracks: (trackIds: string[]) => void;
  clearSelection: () => void;
  setSelectMode: (enabled: boolean) => void;

  // Actions - History
  addToHistory: (track: Track) => void;
  clearHistory: () => void;

  // Computed getters (via selectors)
  hasNextTrack: () => boolean;
  hasPreviousTrack: () => boolean;
  getQueueLength: () => number;
  isTrackInQueue: (trackId: string) => boolean;
}

/**
 * Initial playback state.
 */
const initialPlaybackState: PlaybackState = {
  isPlaying: false,
  currentTime: 0,
  duration: 0,
  playbackSpeed: 1,
  volume: 1,
  isMuted: false,
  isShuffled: false,
  repeatMode: 'off',
};

/**
 * Initial view preferences.
 */
const initialViewPreferences: ViewPreferences = {
  viewMode: 'list',
  sortBy: 'name',
  sortOrder: 'asc',
  itemsPerPage: 20,
};

/**
 * Initial filter state.
 */
const initialFilterState: FilterState = {
  genre: [],
  year: null,
  bpm: null,
  energy: null,
  custom: {},
};

/**
 * Music store using Zustand with persistence, devtools, and subscribeWithSelector.
 * Uses functional updates for immutable state management.
 */
export const useMusicStore = create<MusicState>()(
  devtools(
    subscribeWithSelector(
      persist(
        (set, get) => ({
          // Initial state
          currentTrack: null,
          currentTrackIndex: -1,
          playlistQueue: [],
          playlistHistory: [],
          playback: initialPlaybackState,
          viewPreferences: initialViewPreferences,
          recentSearches: [],
          maxRecentSearches: 10,
          filters: initialFilterState,
          selectedTracks: new Set<string>(),
          isSelectMode: false,

          // Current track actions
          setCurrentTrack: (track) =>
            set((state) => {
              if (track) {
                const index = state.playlistQueue.findIndex(
                  (t) => t.id === track.id
                );
                return {
                  ...state,
                  currentTrack: track,
                  currentTrackIndex: index >= 0 ? index : -1,
                };
              }
              return {
                ...state,
                currentTrack: null,
                currentTrackIndex: -1,
              };
            }),

          setCurrentTrackIndex: (index) =>
            set((state) => {
              if (
                index >= 0 &&
                index < state.playlistQueue.length
              ) {
                return {
                  ...state,
                  currentTrackIndex: index,
                  currentTrack: state.playlistQueue[index],
                };
              }
              return state;
            }),

          // Playlist queue actions
          setPlaylistQueue: (queue) =>
            set((state) => {
              if (state.currentTrackIndex >= queue.length) {
                return {
                  ...state,
                  playlistQueue: queue,
                  currentTrackIndex: -1,
                  currentTrack: null,
                };
              }
              return {
                ...state,
                playlistQueue: queue,
              };
            }),

          addToQueue: (track, position = 'end') =>
            set((state) => {
              if (state.playlistQueue.some((t) => t.id === track.id)) {
                return state;
              }
              if (position === 'start') {
                return {
                  ...state,
                  playlistQueue: [track, ...state.playlistQueue],
                  currentTrackIndex:
                    state.currentTrackIndex >= 0
                      ? state.currentTrackIndex + 1
                      : state.currentTrackIndex,
                };
              }
              return {
                ...state,
                playlistQueue: [...state.playlistQueue, track],
              };
            }),

          addMultipleToQueue: (tracks, position = 'end') =>
            set((state) => {
              const newTracks = tracks.filter(
                (t) => !state.playlistQueue.some((existing) => existing.id === t.id)
              );
              if (newTracks.length === 0) {
                return state;
              }
              if (position === 'start') {
                return {
                  ...state,
                  playlistQueue: [...newTracks, ...state.playlistQueue],
                  currentTrackIndex:
                    state.currentTrackIndex >= 0
                      ? state.currentTrackIndex + newTracks.length
                      : state.currentTrackIndex,
                };
              }
              return {
                ...state,
                playlistQueue: [...state.playlistQueue, ...newTracks],
              };
            }),

          removeFromQueue: (trackId) =>
            set((state) => {
              const index = state.playlistQueue.findIndex(
                (t) => t.id === trackId
              );
              if (index < 0) {
                return state;
              }
              const newQueue = state.playlistQueue.filter((t) => t.id !== trackId);
              if (state.currentTrackIndex === index) {
                // If removed track was current, move to next or previous
                if (newQueue.length > 0) {
                  const newIndex = Math.min(index, newQueue.length - 1);
                  return {
                    ...state,
                    playlistQueue: newQueue,
                    currentTrackIndex: newIndex,
                    currentTrack: newQueue[newIndex],
                  };
                }
                return {
                  ...state,
                  playlistQueue: newQueue,
                  currentTrackIndex: -1,
                  currentTrack: null,
                };
              }
              return {
                ...state,
                playlistQueue: newQueue,
                currentTrackIndex:
                  state.currentTrackIndex > index
                    ? state.currentTrackIndex - 1
                    : state.currentTrackIndex,
              };
            }),

          clearQueue: () =>
            set((state) => ({
              ...state,
              playlistQueue: [],
              currentTrack: null,
              currentTrackIndex: -1,
            })),

          reorderQueue: (fromIndex, toIndex) =>
            set((state) => {
              const newQueue = [...state.playlistQueue];
              const [removed] = newQueue.splice(fromIndex, 1);
              newQueue.splice(toIndex, 0, removed);
              
              // Update current track index
              let newIndex = state.currentTrackIndex;
              if (state.currentTrackIndex === fromIndex) {
                newIndex = toIndex;
              } else if (
                state.currentTrackIndex > fromIndex &&
                state.currentTrackIndex <= toIndex
              ) {
                newIndex = state.currentTrackIndex - 1;
              } else if (
                state.currentTrackIndex < fromIndex &&
                state.currentTrackIndex >= toIndex
              ) {
                newIndex = state.currentTrackIndex + 1;
              }
              
              return {
                ...state,
                playlistQueue: newQueue,
                currentTrackIndex: newIndex,
                currentTrack: newIndex >= 0 ? newQueue[newIndex] : state.currentTrack,
              };
            }),

          moveToNext: () =>
            set((state) => {
              if (state.playlistQueue.length === 0) return state;

              if (state.playback.repeatMode === 'one') {
                // Stay on current track
                return state;
              }

              let nextIndex = state.currentTrackIndex + 1;

              if (nextIndex >= state.playlistQueue.length) {
                if (state.playback.repeatMode === 'all') {
                  nextIndex = 0;
                } else {
                  // Stop playback
                  return {
                    ...state,
                    playback: {
                      ...state.playback,
                      isPlaying: false,
                    },
                  };
                }
              }

              return {
                ...state,
                currentTrackIndex: nextIndex,
                currentTrack: state.playlistQueue[nextIndex],
              };
            }),

          moveToPrevious: () =>
            set((state) => {
              if (state.playlistQueue.length === 0) return state;

              let prevIndex = state.currentTrackIndex - 1;

              if (prevIndex < 0) {
                if (state.playback.repeatMode === 'all') {
                  prevIndex = state.playlistQueue.length - 1;
                } else {
                  // Go to beginning
                  prevIndex = 0;
                }
              }

              return {
                ...state,
                currentTrackIndex: prevIndex,
                currentTrack: state.playlistQueue[prevIndex],
              };
            }),

          moveToTrack: (trackId) =>
            set((state) => {
              const index = state.playlistQueue.findIndex(
                (t) => t.id === trackId
              );
              if (index >= 0) {
                return {
                  ...state,
                  currentTrackIndex: index,
                  currentTrack: state.playlistQueue[index],
                };
              }
              return state;
            }),

          // Playback actions
          setIsPlaying: (playing) =>
            set((state) => ({
              ...state,
              playback: {
                ...state.playback,
                isPlaying: playing,
              },
            })),

          setCurrentTime: (time) =>
            set((state) => ({
              ...state,
              playback: {
                ...state.playback,
                currentTime: Math.max(
                  0,
                  Math.min(time, state.playback.duration)
                ),
              },
            })),

          setDuration: (duration) =>
            set((state) => ({
              ...state,
              playback: {
                ...state.playback,
                duration: Math.max(0, duration),
              },
            })),

          setPlaybackSpeed: (speed) =>
            set((state) => ({
              ...state,
              playback: {
                ...state.playback,
                playbackSpeed: Math.max(0.25, Math.min(2, speed)),
              },
            })),

          setVolume: (volume) =>
            set((state) => ({
              ...state,
              playback: {
                ...state.playback,
                volume: Math.max(0, Math.min(1, volume)),
                isMuted: volume > 0 ? false : state.playback.isMuted,
              },
            })),

          toggleMute: () =>
            set((state) => ({
              ...state,
              playback: {
                ...state.playback,
                isMuted: !state.playback.isMuted,
              },
            })),

          toggleShuffle: () =>
            set((state) => ({
              ...state,
              playback: {
                ...state.playback,
                isShuffled: !state.playback.isShuffled,
              },
            })),

          setRepeatMode: (mode) =>
            set((state) => ({
              ...state,
              playback: {
                ...state.playback,
                repeatMode: mode,
              },
            })),

          resetPlayback: () =>
            set((state) => ({
              ...state,
              playback: { ...initialPlaybackState },
            })),

          // View preferences actions
          setViewMode: (mode) =>
            set((state) => ({
              ...state,
              viewPreferences: {
                ...state.viewPreferences,
                viewMode: mode,
              },
            })),

          setSortBy: (field) =>
            set((state) => ({
              ...state,
              viewPreferences: {
                ...state.viewPreferences,
                sortBy: field,
              },
            })),

          setSortOrder: (order) =>
            set((state) => ({
              ...state,
              viewPreferences: {
                ...state.viewPreferences,
                sortOrder: order,
              },
            })),

          setItemsPerPage: (count) =>
            set((state) => ({
              ...state,
              viewPreferences: {
                ...state.viewPreferences,
                itemsPerPage: Math.max(1, Math.min(100, count)),
              },
            })),

          resetViewPreferences: () =>
            set((state) => ({
              ...state,
              viewPreferences: { ...initialViewPreferences },
            })),

          // Recent searches actions
          addRecentSearch: (query) =>
            set((state) => {
              const trimmed = query.trim();
              if (!trimmed) return state;

              // Remove if already exists
              const filtered = state.recentSearches.filter((q) => q !== trimmed);
              // Add to beginning and limit
              return {
                ...state,
                recentSearches: [trimmed, ...filtered].slice(
                  0,
                  state.maxRecentSearches
                ),
              };
            }),

          removeRecentSearch: (query) =>
            set((state) => ({
              ...state,
              recentSearches: state.recentSearches.filter((q) => q !== query),
            })),

          clearRecentSearches: () =>
            set((state) => ({
              ...state,
              recentSearches: [],
            })),

          // Filters actions
          setFilters: (newFilters) =>
            set((state) => ({
              ...state,
              filters: { ...state.filters, ...newFilters },
            })),

          setFilterValue: (key, value) =>
            set((state) => ({
              ...state,
              filters: {
                ...state.filters,
                [key]: value,
              },
            })),

          clearFilters: () =>
            set((state) => ({
              ...state,
              filters: { ...initialFilterState },
            })),

          resetFilters: () =>
            set((state) => ({
              ...state,
              filters: { ...initialFilterState },
            })),

          // Selection actions
          toggleTrackSelection: (trackId) =>
            set((state) => {
              const newSelected = new Set(state.selectedTracks);
              if (newSelected.has(trackId)) {
                newSelected.delete(trackId);
              } else {
                newSelected.add(trackId);
              }
              return {
                ...state,
                selectedTracks: newSelected,
              };
            }),

          selectAllTracks: (trackIds) =>
            set((state) => ({
              ...state,
              selectedTracks: new Set(trackIds),
            })),

          clearSelection: () =>
            set((state) => ({
              ...state,
              selectedTracks: new Set<string>(),
            })),

          setSelectMode: (enabled) =>
            set((state) => ({
              ...state,
              isSelectMode: enabled,
              selectedTracks: enabled ? state.selectedTracks : new Set<string>(),
            })),

          // History actions
          addToHistory: (track) =>
            set((state) => {
              // Remove if already exists
              const filtered = state.playlistHistory.filter(
                (t) => t.id !== track.id
              );
              // Add to beginning and limit to 50
              return {
                ...state,
                playlistHistory: [track, ...filtered].slice(0, 50),
              };
            }),

          clearHistory: () =>
            set((state) => ({
              ...state,
              playlistHistory: [],
            })),

          // Computed getters
          hasNextTrack: () => {
            const state = get();
            if (state.playback.repeatMode === 'all') {
              return state.playlistQueue.length > 0;
            }
            return (
              state.currentTrackIndex >= 0 &&
              state.currentTrackIndex < state.playlistQueue.length - 1
            );
          },

          hasPreviousTrack: () => {
            const state = get();
            if (state.playback.repeatMode === 'all') {
              return state.playlistQueue.length > 0;
            }
            return state.currentTrackIndex > 0;
          },

          getQueueLength: () => {
            return get().playlistQueue.length;
          },

          isTrackInQueue: (trackId) => {
            return get().playlistQueue.some((t) => t.id === trackId);
          },
        })),
        {
          name: STORAGE_KEYS.USER_PREFERENCES,
          partialize: (state) => ({
            viewPreferences: state.viewPreferences,
            playback: {
              playbackSpeed: state.playback.playbackSpeed,
              volume: state.playback.volume,
              isMuted: state.playback.isMuted,
              repeatMode: state.playback.repeatMode,
              isShuffled: state.playback.isShuffled,
            },
            recentSearches: state.recentSearches,
            filters: state.filters,
          }),
        }
      )
    ),
    { name: 'MusicStore' }
  )
);
