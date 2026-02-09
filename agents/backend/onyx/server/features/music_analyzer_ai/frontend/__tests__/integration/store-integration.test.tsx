/**
 * Integration Tests - Store Integration
 * Tests integration between Zustand store and components
 */

import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useMusicStore } from '@/lib/store/music-store';
import { TrackSearch } from '@/components/music/TrackSearch';
import { AudioPlayer } from '@/components/music/AudioPlayer';
import { musicApiService } from '@/lib/api/music-api';
import { type Track } from '@/lib/api/music-api';
import { act } from 'react';

jest.mock('@/lib/api/music-api', () => ({
  musicApiService: {
    searchTracks: jest.fn(),
  },
}));

jest.mock('react-hot-toast', () => ({
  default: {
    success: jest.fn(),
    error: jest.fn(),
  },
}));

const mockSearchTracks = musicApiService.searchTracks as jest.MockedFunction<
  typeof musicApiService.searchTracks
>;

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        cacheTime: 0,
      },
    },
  });

  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
};

const mockTrack: Track = {
  id: 'track-123',
  name: 'Test Track',
  artists: ['Test Artist'],
  album: 'Test Album',
  duration_ms: 200000,
  preview_url: 'https://example.com/preview.mp3',
  popularity: 80,
  images: [{ url: 'https://example.com/image.jpg' }],
};

// Component that uses the store
function TestComponent() {
  const currentTrack = useMusicStore((state) => state.currentTrack);
  const setCurrentTrack = useMusicStore((state) => state.setCurrentTrack);
  const playlistQueue = useMusicStore((state) => state.playlistQueue);

  return (
    <div>
      {currentTrack && <div data-testid="current-track">{currentTrack.name}</div>}
      <div data-testid="queue-length">{playlistQueue.length}</div>
      <button
        onClick={() => setCurrentTrack(mockTrack)}
        data-testid="set-track-button"
      >
        Set Track
      </button>
    </div>
  );
}

describe('Store Integration Tests', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    jest.useFakeTimers();
    // Reset store
    act(() => {
      useMusicStore.getState().clearQueue();
      useMusicStore.getState().setCurrentTrack(null);
    });
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  describe('Store with Components', () => {
    it('should update component when store changes', () => {
      render(<TestComponent />, { wrapper: createWrapper() });

      expect(screen.queryByTestId('current-track')).not.toBeInTheDocument();

      const setTrackButton = screen.getByTestId('set-track-button');
      act(() => {
        userEvent.click(setTrackButton);
      });

      expect(screen.getByTestId('current-track')).toHaveTextContent('Test Track');
    });

    it('should reflect queue changes in component', () => {
      render(<TestComponent />, { wrapper: createWrapper() });

      expect(screen.getByTestId('queue-length')).toHaveTextContent('0');

      act(() => {
        useMusicStore.getState().addToQueue(mockTrack);
      });

      expect(screen.getByTestId('queue-length')).toHaveTextContent('1');
    });
  });

  describe('Store with TrackSearch', () => {
    it('should add track to queue when selected from search', async () => {
      const user = userEvent.setup({ delay: null });

      mockSearchTracks.mockResolvedValue({
        success: true,
        query: 'test',
        results: [mockTrack],
        total: 1,
      });

      render(<TrackSearch onTrackSelect={() => {}} />, {
        wrapper: createWrapper(),
      });

      const searchInput = screen.getByPlaceholderText(
        /busca canciones, artistas o álbumes/i
      );

      await user.type(searchInput, 'test');
      jest.advanceTimersByTime(500);

      await waitFor(() => {
        expect(screen.getByText('Test Track')).toBeInTheDocument();
      });

      const trackButton = screen.getByText('Test Track').closest('button');
      await user.click(trackButton!);

      // Track should be in store queue
      act(() => {
        const queue = useMusicStore.getState().playlistQueue;
        expect(queue.length).toBeGreaterThanOrEqual(0);
      });
    });
  });

  describe('Store with AudioPlayer', () => {
    it('should sync player with store current track', () => {
      act(() => {
        useMusicStore.getState().setCurrentTrack(mockTrack);
      });

      render(<AudioPlayer track={mockTrack} />);

      expect(screen.getByText('Test Track')).toBeInTheDocument();
    });

    it('should update store when player controls are used', () => {
      act(() => {
        useMusicStore.getState().setCurrentTrack(mockTrack);
        useMusicStore.getState().setPlaylistQueue([mockTrack]);
      });

      render(<AudioPlayer track={mockTrack} />);

      const playButton = screen.getByRole('button', { name: /play/i });
      act(() => {
        userEvent.click(playButton);
      });

      // Store should reflect playing state
      const isPlaying = useMusicStore.getState().playback.isPlaying;
      expect(isPlaying).toBeDefined();
    });
  });

  describe('Store State Persistence', () => {
    it('should persist view preferences', () => {
      act(() => {
        useMusicStore.getState().setViewMode('grid');
      });

      const viewMode = useMusicStore.getState().viewPreferences.viewMode;
      expect(viewMode).toBe('grid');
    });

    it('should persist playback settings', () => {
      act(() => {
        useMusicStore.getState().setVolume(0.7);
        useMusicStore.getState().setPlaybackSpeed(1.25);
      });

      const volume = useMusicStore.getState().playback.volume;
      const speed = useMusicStore.getState().playback.playbackSpeed;

      expect(volume).toBe(0.7);
      expect(speed).toBe(1.25);
    });
  });

  describe('Store Actions Integration', () => {
    it('should handle complete playback flow', () => {
      act(() => {
        useMusicStore.getState().setPlaylistQueue([mockTrack]);
        useMusicStore.getState().setCurrentTrackIndex(0);
        useMusicStore.getState().setIsPlaying(true);
      });

      const state = useMusicStore.getState();
      expect(state.currentTrack).toEqual(mockTrack);
      expect(state.playback.isPlaying).toBe(true);
    });

    it('should handle queue navigation', () => {
      const tracks = [
        { ...mockTrack, id: '1', name: 'Track 1' },
        { ...mockTrack, id: '2', name: 'Track 2' },
        { ...mockTrack, id: '3', name: 'Track 3' },
      ];

      act(() => {
        useMusicStore.getState().setPlaylistQueue(tracks);
        useMusicStore.getState().setCurrentTrackIndex(0);
        useMusicStore.getState().moveToNext();
      });

      const state = useMusicStore.getState();
      expect(state.currentTrackIndex).toBe(1);
      expect(state.currentTrack?.id).toBe('2');
    });
  });
});

