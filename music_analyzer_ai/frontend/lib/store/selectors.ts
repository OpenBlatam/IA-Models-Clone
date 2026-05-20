/**
 * Zustand store selectors for optimized re-renders.
 * Using selectors prevents unnecessary re-renders by subscribing to specific state slices.
 */

import { useMusicStore } from './music-store';
import { shallow } from 'zustand/shallow';

/**
 * Selector for current track only.
 * Component will only re-render when currentTrack changes.
 */
export function useCurrentTrack() {
  return useMusicStore((state) => state.currentTrack);
}

/**
 * Selector for playback state only.
 */
export function usePlaybackState() {
  return useMusicStore((state) => state.playback);
}

/**
 * Selector for playlist queue only.
 */
export function usePlaylistQueue() {
  return useMusicStore((state) => state.playlistQueue);
}

/**
 * Selector for view preferences only.
 */
export function useViewPreferences() {
  return useMusicStore((state) => state.viewPreferences);
}

/**
 * Selector for filters only.
 */
export function useFilters() {
  return useMusicStore((state) => state.filters);
}

/**
 * Selector for recent searches only.
 */
export function useRecentSearches() {
  return useMusicStore((state) => state.recentSearches);
}

/**
 * Selector for selected tracks only.
 */
export function useSelectedTracks() {
  return useMusicStore((state) => state.selectedTracks);
}

/**
 * Selector for playback controls (actions only, no state).
 * Useful when you only need actions and don't want to re-render on state changes.
 */
export function usePlaybackActions() {
  return useMusicStore(
    (state) => ({
      setIsPlaying: state.setIsPlaying,
      setCurrentTime: state.setCurrentTime,
      setDuration: state.setDuration,
      setPlaybackSpeed: state.setPlaybackSpeed,
      setVolume: state.setVolume,
      toggleMute: state.toggleMute,
      toggleShuffle: state.toggleShuffle,
      setRepeatMode: state.setRepeatMode,
      resetPlayback: state.resetPlayback,
      moveToNext: state.moveToNext,
      moveToPrevious: state.moveToPrevious,
      moveToTrack: state.moveToTrack,
    }),
    shallow
  );
}

/**
 * Selector for queue actions only.
 */
export function useQueueActions() {
  return useMusicStore(
    (state) => ({
      setPlaylistQueue: state.setPlaylistQueue,
      addToQueue: state.addToQueue,
      addMultipleToQueue: state.addMultipleToQueue,
      removeFromQueue: state.removeFromQueue,
      clearQueue: state.clearQueue,
      reorderQueue: state.reorderQueue,
    }),
    shallow
  );
}

/**
 * Selector for selection actions only.
 */
export function useSelectionActions() {
  return useMusicStore(
    (state) => ({
      toggleTrackSelection: state.toggleTrackSelection,
      selectAllTracks: state.selectAllTracks,
      clearSelection: state.clearSelection,
      setSelectMode: state.setSelectMode,
    }),
    shallow
  );
}

/**
 * Selector for computed playback state.
 * Returns derived state that doesn't cause re-renders unless dependencies change.
 */
export function usePlaybackInfo() {
  return useMusicStore(
    (state) => ({
      isPlaying: state.playback.isPlaying,
      currentTime: state.playback.currentTime,
      duration: state.playback.duration,
      progress: state.playback.duration > 0
        ? state.playback.currentTime / state.playback.duration
        : 0,
      hasNext: state.hasNextTrack(),
      hasPrevious: state.hasPreviousTrack(),
      queueLength: state.getQueueLength(),
    }),
    shallow
  );
}

/**
 * Selector for current track with queue info.
 */
export function useCurrentTrackInfo() {
  return useMusicStore(
    (state) => ({
      track: state.currentTrack,
      index: state.currentTrackIndex,
      queueLength: state.playlistQueue.length,
      isInQueue: state.currentTrack
        ? state.isTrackInQueue(state.currentTrack.id)
        : false,
      hasNext: state.hasNextTrack(),
      hasPrevious: state.hasPreviousTrack(),
    }),
    shallow
  );
}

