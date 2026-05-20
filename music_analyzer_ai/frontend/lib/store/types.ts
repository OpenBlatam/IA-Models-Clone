/**
 * Type definitions for Zustand stores.
 * Shared types for store interfaces.
 */

import { type Track } from '@/lib/api/types';

/**
 * Playback state interface.
 */
export interface PlaybackState {
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
export interface ViewPreferences {
  viewMode: 'grid' | 'list' | 'compact';
  sortBy: string;
  sortOrder: 'asc' | 'desc';
  itemsPerPage: number;
}

/**
 * Filter state interface.
 */
export interface FilterState {
  genre: string[];
  year: { min: number; max: number } | null;
  bpm: { min: number; max: number } | null;
  energy: { min: number; max: number } | null;
  custom: Record<string, unknown>;
}

/**
 * Playlist queue item with metadata.
 */
export interface QueueItem extends Track {
  addedAt: number;
  playCount: number;
}

