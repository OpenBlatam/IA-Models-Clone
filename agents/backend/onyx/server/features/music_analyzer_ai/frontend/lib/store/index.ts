/**
 * Store exports.
 * Centralized export point for all Zustand stores and selectors.
 */

export { useMusicStore } from './music-store';
export type { MusicState } from './music-store';

// Selectors for optimized re-renders
export {
  useCurrentTrack,
  usePlaybackState,
  usePlaylistQueue,
  useViewPreferences,
  useFilters,
  useRecentSearches,
  useSelectedTracks,
  usePlaybackActions,
  useQueueActions,
  useSelectionActions,
  usePlaybackInfo,
  useCurrentTrackInfo,
} from './selectors';
