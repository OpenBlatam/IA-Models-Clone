import { act, renderHook } from '@testing-library/react';
import { useMusicStore } from '@/lib/store/music-store';
import { type Track } from '@/lib/api/types';

// Mock localStorage
const localStorageMock = (() => {
  let store: Record<string, string> = {};

  return {
    getItem: (key: string) => store[key] || null,
    setItem: (key: string, value: string) => {
      store[key] = value.toString();
    },
    removeItem: (key: string) => {
      delete store[key];
    },
    clear: () => {
      store = {};
    },
  };
})();

Object.defineProperty(window, 'localStorage', {
  value: localStorageMock,
});

const mockTrack1: Track = {
  id: 'track-1',
  name: 'Track 1',
  artists: ['Artist 1'],
  album: 'Album 1',
  duration_ms: 200000,
  popularity: 80,
  images: [],
};

const mockTrack2: Track = {
  id: 'track-2',
  name: 'Track 2',
  artists: ['Artist 2'],
  album: 'Album 2',
  duration_ms: 180000,
  popularity: 75,
  images: [],
};

const mockTrack3: Track = {
  id: 'track-3',
  name: 'Track 3',
  artists: ['Artist 3'],
  album: 'Album 3',
  duration_ms: 220000,
  popularity: 85,
  images: [],
};

describe('Music Store', () => {
  beforeEach(() => {
    // Reset store before each test
    act(() => {
      useMusicStore.getState().clearQueue();
      useMusicStore.getState().clearRecentSearches();
      useMusicStore.getState().clearFilters();
      useMusicStore.getState().clearSelection();
      useMusicStore.getState().clearHistory();
      useMusicStore.getState().setCurrentTrack(null);
      useMusicStore.getState().resetPlayback();
      useMusicStore.getState().resetViewPreferences();
    });
    localStorageMock.clear();
  });

  describe('Current Track', () => {
    it('should set current track', () => {
      const { result } = renderHook(() => useMusicStore());

      act(() => {
        result.current.setCurrentTrack(mockTrack1);
      });

      expect(result.current.currentTrack).toEqual(mockTrack1);
    });

    it('should clear current track when set to null', () => {
      const { result } = renderHook(() => useMusicStore());

      act(() => {
        result.current.setCurrentTrack(mockTrack1);
        result.current.setCurrentTrack(null);
      });

      expect(result.current.currentTrack).toBeNull();
      expect(result.current.currentTrackIndex).toBe(-1);
    });

    it('should update current track index when track is in queue', () => {
      const { result } = renderHook(() => useMusicStore());

      act(() => {
        result.current.setPlaylistQueue([mockTrack1, mockTrack2, mockTrack3]);
        result.current.setCurrentTrack(mockTrack2);
      });

      expect(result.current.currentTrackIndex).toBe(1);
    });
  });

  describe('Playlist Queue', () => {
    it('should set playlist queue', () => {
      const { result } = renderHook(() => useMusicStore());

      act(() => {
        result.current.setPlaylistQueue([mockTrack1, mockTrack2]);
      });

      expect(result.current.playlistQueue).toHaveLength(2);
      expect(result.current.getQueueLength()).toBe(2);
    });

    it('should add track to queue at end', () => {
      const { result } = renderHook(() => useMusicStore());

      act(() => {
        result.current.addToQueue(mockTrack1);
        result.current.addToQueue(mockTrack2);
      });

      expect(result.current.playlistQueue).toHaveLength(2);
      expect(result.current.playlistQueue[1]).toEqual(mockTrack2);
    });

    it('should add track to queue at start', () => {
      const { result } = renderHook(() => useMusicStore());

      act(() => {
        result.current.addToQueue(mockTrack1);
        result.current.addToQueue(mockTrack2, 'start');
      });

      expect(result.current.playlistQueue).toHaveLength(2);
      expect(result.current.playlistQueue[0]).toEqual(mockTrack2);
    });

    it('should not add duplicate tracks', () => {
      const { result } = renderHook(() => useMusicStore());

      act(() => {
        result.current.addToQueue(mockTrack1);
        result.current.addToQueue(mockTrack1);
      });

      expect(result.current.playlistQueue).toHaveLength(1);
    });

    it('should add multiple tracks to queue', () => {
      const { result } = renderHook(() => useMusicStore());

      act(() => {
        result.current.addMultipleToQueue([mockTrack1, mockTrack2, mockTrack3]);
      });

      expect(result.current.playlistQueue).toHaveLength(3);
    });

    it('should remove track from queue', () => {
      const { result } = renderHook(() => useMusicStore());

      act(() => {
        result.current.setPlaylistQueue([mockTrack1, mockTrack2, mockTrack3]);
        result.current.removeFromQueue('track-2');
      });

      expect(result.current.playlistQueue).toHaveLength(2);
      expect(result.current.isTrackInQueue('track-2')).toBe(false);
    });

    it('should clear queue', () => {
      const { result } = renderHook(() => useMusicStore());

      act(() => {
        result.current.setPlaylistQueue([mockTrack1, mockTrack2]);
        result.current.clearQueue();
      });

      expect(result.current.playlistQueue).toHaveLength(0);
      expect(result.current.currentTrack).toBeNull();
    });

    it('should reorder queue', () => {
      const { result } = renderHook(() => useMusicStore());

      act(() => {
        result.current.setPlaylistQueue([mockTrack1, mockTrack2, mockTrack3]);
        result.current.reorderQueue(0, 2);
      });

      expect(result.current.playlistQueue[0]).toEqual(mockTrack2);
      expect(result.current.playlistQueue[2]).toEqual(mockTrack1);
    });
  });

  describe('Navigation', () => {
    it('should move to next track', () => {
      const { result } = renderHook(() => useMusicStore());

      act(() => {
        result.current.setPlaylistQueue([mockTrack1, mockTrack2, mockTrack3]);
        result.current.setCurrentTrackIndex(0);
        result.current.moveToNext();
      });

      expect(result.current.currentTrackIndex).toBe(1);
      expect(result.current.currentTrack).toEqual(mockTrack2);
    });

    it('should move to previous track', () => {
      const { result } = renderHook(() => useMusicStore());

      act(() => {
        result.current.setPlaylistQueue([mockTrack1, mockTrack2, mockTrack3]);
        result.current.setCurrentTrackIndex(1);
        result.current.moveToPrevious();
      });

      expect(result.current.currentTrackIndex).toBe(0);
      expect(result.current.currentTrack).toEqual(mockTrack1);
    });

    it('should move to specific track', () => {
      const { result } = renderHook(() => useMusicStore());

      act(() => {
        result.current.setPlaylistQueue([mockTrack1, mockTrack2, mockTrack3]);
        result.current.moveToTrack('track-3');
      });

      expect(result.current.currentTrackIndex).toBe(2);
      expect(result.current.currentTrack).toEqual(mockTrack3);
    });

    it('should handle repeat all mode', () => {
      const { result } = renderHook(() => useMusicStore());

      act(() => {
        result.current.setPlaylistQueue([mockTrack1, mockTrack2]);
        result.current.setCurrentTrackIndex(1);
        result.current.setRepeatMode('all');
        result.current.moveToNext();
      });

      expect(result.current.currentTrackIndex).toBe(0);
    });

    it('should handle repeat one mode', () => {
      const { result } = renderHook(() => useMusicStore());

      act(() => {
        result.current.setPlaylistQueue([mockTrack1, mockTrack2]);
        result.current.setCurrentTrackIndex(0);
        result.current.setRepeatMode('one');
        result.current.moveToNext();
      });

      // Should stay on same track
      expect(result.current.currentTrackIndex).toBe(0);
    });
  });

  describe('Playback State', () => {
    it('should set playing state', () => {
      const { result } = renderHook(() => useMusicStore());

      act(() => {
        result.current.setIsPlaying(true);
      });

      expect(result.current.playback.isPlaying).toBe(true);
    });

    it('should set current time', () => {
      const { result } = renderHook(() => useMusicStore());

      act(() => {
        result.current.setDuration(200);
        result.current.setCurrentTime(100);
      });

      expect(result.current.playback.currentTime).toBe(100);
    });

    it('should clamp current time to duration', () => {
      const { result } = renderHook(() => useMusicStore());

      act(() => {
        result.current.setDuration(200);
        result.current.setCurrentTime(300);
      });

      expect(result.current.playback.currentTime).toBe(200);
    });

    it('should set volume', () => {
      const { result } = renderHook(() => useMusicStore());

      act(() => {
        result.current.setVolume(0.5);
      });

      expect(result.current.playback.volume).toBe(0.5);
    });

    it('should clamp volume between 0 and 1', () => {
      const { result } = renderHook(() => useMusicStore());

      act(() => {
        result.current.setVolume(1.5);
      });

      expect(result.current.playback.volume).toBe(1);

      act(() => {
        result.current.setVolume(-0.5);
      });

      expect(result.current.playback.volume).toBe(0);
    });

    it('should toggle mute', () => {
      const { result } = renderHook(() => useMusicStore());

      act(() => {
        result.current.toggleMute();
      });

      expect(result.current.playback.isMuted).toBe(true);

      act(() => {
        result.current.toggleMute();
      });

      expect(result.current.playback.isMuted).toBe(false);
    });

    it('should toggle shuffle', () => {
      const { result } = renderHook(() => useMusicStore());

      act(() => {
        result.current.toggleShuffle();
      });

      expect(result.current.playback.isShuffled).toBe(true);
    });

    it('should set playback speed', () => {
      const { result } = renderHook(() => useMusicStore());

      act(() => {
        result.current.setPlaybackSpeed(1.5);
      });

      expect(result.current.playback.playbackSpeed).toBe(1.5);
    });

    it('should clamp playback speed', () => {
      const { result } = renderHook(() => useMusicStore());

      act(() => {
        result.current.setPlaybackSpeed(3);
      });

      expect(result.current.playback.playbackSpeed).toBe(2);

      act(() => {
        result.current.setPlaybackSpeed(0.1);
      });

      expect(result.current.playback.playbackSpeed).toBe(0.25);
    });

    it('should reset playback', () => {
      const { result } = renderHook(() => useMusicStore());

      act(() => {
        result.current.setIsPlaying(true);
        result.current.setVolume(0.5);
        result.current.setPlaybackSpeed(1.5);
        result.current.resetPlayback();
      });

      expect(result.current.playback.isPlaying).toBe(false);
      expect(result.current.playback.volume).toBe(1);
      expect(result.current.playback.playbackSpeed).toBe(1);
    });
  });

  describe('View Preferences', () => {
    it('should set view mode', () => {
      const { result } = renderHook(() => useMusicStore());

      act(() => {
        result.current.setViewMode('grid');
      });

      expect(result.current.viewPreferences.viewMode).toBe('grid');
    });

    it('should set sort by', () => {
      const { result } = renderHook(() => useMusicStore());

      act(() => {
        result.current.setSortBy('popularity');
      });

      expect(result.current.viewPreferences.sortBy).toBe('popularity');
    });

    it('should set sort order', () => {
      const { result } = renderHook(() => useMusicStore());

      act(() => {
        result.current.setSortOrder('desc');
      });

      expect(result.current.viewPreferences.sortOrder).toBe('desc');
    });

    it('should set items per page', () => {
      const { result } = renderHook(() => useMusicStore());

      act(() => {
        result.current.setItemsPerPage(50);
      });

      expect(result.current.viewPreferences.itemsPerPage).toBe(50);
    });

    it('should clamp items per page', () => {
      const { result } = renderHook(() => useMusicStore());

      act(() => {
        result.current.setItemsPerPage(200);
      });

      expect(result.current.viewPreferences.itemsPerPage).toBe(100);

      act(() => {
        result.current.setItemsPerPage(0);
      });

      expect(result.current.viewPreferences.itemsPerPage).toBe(1);
    });

    it('should reset view preferences', () => {
      const { result } = renderHook(() => useMusicStore());

      act(() => {
        result.current.setViewMode('grid');
        result.current.setSortBy('popularity');
        result.current.resetViewPreferences();
      });

      expect(result.current.viewPreferences.viewMode).toBe('list');
      expect(result.current.viewPreferences.sortBy).toBe('name');
    });
  });

  describe('Recent Searches', () => {
    it('should add recent search', () => {
      const { result } = renderHook(() => useMusicStore());

      act(() => {
        result.current.addRecentSearch('test query');
      });

      expect(result.current.recentSearches).toContain('test query');
    });

    it('should limit recent searches', () => {
      const { result } = renderHook(() => useMusicStore());

      act(() => {
        for (let i = 0; i < 15; i++) {
          result.current.addRecentSearch(`query ${i}`);
        }
      });

      expect(result.current.recentSearches.length).toBeLessThanOrEqual(10);
    });

    it('should remove duplicate searches and move to top', () => {
      const { result } = renderHook(() => useMusicStore());

      act(() => {
        result.current.addRecentSearch('query 1');
        result.current.addRecentSearch('query 2');
        result.current.addRecentSearch('query 1');
      });

      expect(result.current.recentSearches[0]).toBe('query 1');
      expect(result.current.recentSearches).toHaveLength(2);
    });

    it('should remove recent search', () => {
      const { result } = renderHook(() => useMusicStore());

      act(() => {
        result.current.addRecentSearch('query 1');
        result.current.addRecentSearch('query 2');
        result.current.removeRecentSearch('query 1');
      });

      expect(result.current.recentSearches).not.toContain('query 1');
    });

    it('should clear recent searches', () => {
      const { result } = renderHook(() => useMusicStore());

      act(() => {
        result.current.addRecentSearch('query 1');
        result.current.clearRecentSearches();
      });

      expect(result.current.recentSearches).toHaveLength(0);
    });
  });

  describe('Filters', () => {
    it('should set filters', () => {
      const { result } = renderHook(() => useMusicStore());

      act(() => {
        result.current.setFilters({
          genre: ['rock', 'pop'],
        });
      });

      expect(result.current.filters.genre).toEqual(['rock', 'pop']);
    });

    it('should set filter value', () => {
      const { result } = renderHook(() => useMusicStore());

      act(() => {
        result.current.setFilterValue('genre', ['rock']);
      });

      expect(result.current.filters.genre).toEqual(['rock']);
    });

    it('should clear filters', () => {
      const { result } = renderHook(() => useMusicStore());

      act(() => {
        result.current.setFilters({
          genre: ['rock'],
          year: { min: 2000, max: 2020 },
        });
        result.current.clearFilters();
      });

      expect(result.current.filters.genre).toEqual([]);
      expect(result.current.filters.year).toBeNull();
    });
  });

  describe('Selection', () => {
    it('should toggle track selection', () => {
      const { result } = renderHook(() => useMusicStore());

      act(() => {
        result.current.toggleTrackSelection('track-1');
      });

      expect(result.current.selectedTracks.has('track-1')).toBe(true);

      act(() => {
        result.current.toggleTrackSelection('track-1');
      });

      expect(result.current.selectedTracks.has('track-1')).toBe(false);
    });

    it('should select all tracks', () => {
      const { result } = renderHook(() => useMusicStore());

      act(() => {
        result.current.selectAllTracks(['track-1', 'track-2', 'track-3']);
      });

      expect(result.current.selectedTracks.size).toBe(3);
    });

    it('should clear selection', () => {
      const { result } = renderHook(() => useMusicStore());

      act(() => {
        result.current.toggleTrackSelection('track-1');
        result.current.clearSelection();
      });

      expect(result.current.selectedTracks.size).toBe(0);
    });

    it('should set select mode', () => {
      const { result } = renderHook(() => useMusicStore());

      act(() => {
        result.current.setSelectMode(true);
      });

      expect(result.current.isSelectMode).toBe(true);
    });

    it('should clear selection when exiting select mode', () => {
      const { result } = renderHook(() => useMusicStore());

      act(() => {
        result.current.toggleTrackSelection('track-1');
        result.current.setSelectMode(false);
      });

      expect(result.current.selectedTracks.size).toBe(0);
    });
  });

  describe('History', () => {
    it('should add track to history', () => {
      const { result } = renderHook(() => useMusicStore());

      act(() => {
        result.current.addToHistory(mockTrack1);
      });

      expect(result.current.playlistHistory).toContain(mockTrack1);
    });

    it('should limit history to 50 tracks', () => {
      const { result } = renderHook(() => useMusicStore());

      act(() => {
        for (let i = 0; i < 60; i++) {
          result.current.addToHistory({
            ...mockTrack1,
            id: `track-${i}`,
          });
        }
      });

      expect(result.current.playlistHistory.length).toBeLessThanOrEqual(50);
    });

    it('should clear history', () => {
      const { result } = renderHook(() => useMusicStore());

      act(() => {
        result.current.addToHistory(mockTrack1);
        result.current.clearHistory();
      });

      expect(result.current.playlistHistory).toHaveLength(0);
    });
  });

  describe('Computed Getters', () => {
    it('should check if has next track', () => {
      const { result } = renderHook(() => useMusicStore());

      act(() => {
        result.current.setPlaylistQueue([mockTrack1, mockTrack2, mockTrack3]);
        result.current.setCurrentTrackIndex(0);
      });

      expect(result.current.hasNextTrack()).toBe(true);
    });

    it('should check if has previous track', () => {
      const { result } = renderHook(() => useMusicStore());

      act(() => {
        result.current.setPlaylistQueue([mockTrack1, mockTrack2, mockTrack3]);
        result.current.setCurrentTrackIndex(1);
      });

      expect(result.current.hasPreviousTrack()).toBe(true);
    });

    it('should check if track is in queue', () => {
      const { result } = renderHook(() => useMusicStore());

      act(() => {
        result.current.setPlaylistQueue([mockTrack1, mockTrack2]);
      });

      expect(result.current.isTrackInQueue('track-1')).toBe(true);
      expect(result.current.isTrackInQueue('track-3')).toBe(false);
    });
  });
});

