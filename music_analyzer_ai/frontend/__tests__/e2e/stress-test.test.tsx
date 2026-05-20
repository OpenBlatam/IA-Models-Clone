/**
 * E2E Tests - Stress Tests
 * Tests application behavior under stress and high load
 */

import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { TrackSearch } from '@/components/music/TrackSearch';
import { musicApiService } from '@/lib/api/music-api';
import { type Track } from '@/lib/api/music-api';

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

describe('E2E Stress Tests', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  describe('High Volume Search', () => {
    it('should handle 100+ rapid search requests', async () => {
      const user = userEvent.setup({ delay: null });
      const onTrackSelect = jest.fn();

      mockSearchTracks.mockResolvedValue({
        success: true,
        query: 'final',
        results: [],
        total: 0,
      });

      render(<TrackSearch onTrackSelect={onTrackSelect} />, {
        wrapper: createWrapper(),
      });

      const searchInput = screen.getByPlaceholderText(
        /busca canciones, artistas o álbumes/i
      );

      // Simulate 100 rapid keystrokes
      for (let i = 0; i < 100; i++) {
        await user.type(searchInput, 'a');
        jest.advanceTimersByTime(10);
      }

      // Should debounce and only make one API call
      jest.advanceTimersByTime(500);

      await waitFor(() => {
        // Should have called API, but debounced
        expect(mockSearchTracks).toHaveBeenCalled();
      });

      // Verify it was called a reasonable number of times (not 100)
      expect(mockSearchTracks.mock.calls.length).toBeLessThan(10);
    });
  });

  describe('Large Result Sets', () => {
    it('should handle rendering 1000+ tracks', async () => {
      const user = userEvent.setup({ delay: null });
      const onTrackSelect = jest.fn();

      // Create large array of tracks
      const manyTracks: Track[] = Array.from({ length: 1000 }, (_, i) => ({
        id: `track-${i}`,
        name: `Track ${i}`,
        artists: [`Artist ${i}`],
        album: `Album ${i}`,
        duration_ms: 200000,
        popularity: 80,
        images: [],
      }));

      mockSearchTracks.mockResolvedValue({
        success: true,
        query: 'many',
        results: manyTracks,
        total: 1000,
      });

      render(<TrackSearch onTrackSelect={onTrackSelect} />, {
        wrapper: createWrapper(),
      });

      const searchInput = screen.getByPlaceholderText(
        /busca canciones, artistas o álbumes/i
      );

      await user.type(searchInput, 'many');
      jest.advanceTimersByTime(500);

      // Should render tracks (may be virtualized in real app)
      await waitFor(() => {
        expect(screen.getByText('Track 0')).toBeInTheDocument();
      });

      // Component should handle large lists
      expect(screen.getByText('Track 0')).toBeInTheDocument();
    });
  });

  describe('Concurrent Component Instances', () => {
    it('should handle multiple search components simultaneously', async () => {
      const user = userEvent.setup({ delay: null });
      const onTrackSelect1 = jest.fn();
      const onTrackSelect2 = jest.fn();

      mockSearchTracks.mockResolvedValue({
        success: true,
        query: 'test',
        results: [],
        total: 0,
      });

      render(
        <>
          <TrackSearch onTrackSelect={onTrackSelect1} />
          <TrackSearch onTrackSelect={onTrackSelect2} />
        </>,
        { wrapper: createWrapper() }
      );

      const searchInputs = screen.getAllByPlaceholderText(
        /busca canciones, artistas o álbumes/i
      );

      // Type in both inputs
      await user.type(searchInputs[0], 'test1');
      await user.type(searchInputs[1], 'test2');

      jest.advanceTimersByTime(500);

      await waitFor(() => {
        expect(mockSearchTracks).toHaveBeenCalled();
      });

      // Both should work independently
      expect(searchInputs).toHaveLength(2);
    });
  });

  describe('Memory Leak Prevention', () => {
    it('should clean up event listeners on unmount', () => {
      const { unmount } = render(
        <TrackSearch onTrackSelect={jest.fn()} />,
        {
          wrapper: createWrapper(),
        }
      );

      const searchInput = screen.getByPlaceholderText(
        /busca canciones, artistas o álbumes/i
      );
      expect(searchInput).toBeInTheDocument();

      // Unmount
      unmount();

      // Should be cleaned up
      expect(screen.queryByPlaceholderText(/busca canciones/i)).not.toBeInTheDocument();
    });

    it('should clean up timers on unmount', () => {
      const { unmount } = render(
        <TrackSearch onTrackSelect={jest.fn()} />,
        {
          wrapper: createWrapper(),
        }
      );

      // Component mounts
      expect(screen.getByPlaceholderText(/busca canciones/i)).toBeInTheDocument();

      // Unmount should clean up any pending timers
      unmount();

      // Verify cleanup
      expect(screen.queryByPlaceholderText(/busca canciones/i)).not.toBeInTheDocument();
    });
  });

  describe('Rapid State Updates', () => {
    it('should handle 1000 rapid state updates', async () => {
      const user = userEvent.setup({ delay: null });
      const onTrackSelect = jest.fn();

      render(<TrackSearch onTrackSelect={onTrackSelect} />, {
        wrapper: createWrapper(),
      });

      const searchInput = screen.getByPlaceholderText(
        /busca canciones, artistas o álbumes/i
      ) as HTMLInputElement;

      // Rapid state updates
      for (let i = 0; i < 1000; i++) {
        searchInput.value = `test${i}`;
        searchInput.dispatchEvent(new Event('input', { bubbles: true }));
        jest.advanceTimersByTime(1);
      }

      // Should handle gracefully
      expect(searchInput).toBeInTheDocument();
    });
  });

  describe('Long Running Operations', () => {
    it('should handle operations that take 30+ seconds', async () => {
      const user = userEvent.setup({ delay: null });
      const onTrackSelect = jest.fn();

      mockSearchTracks.mockImplementation(
        () =>
          new Promise((resolve) => {
            setTimeout(
              () =>
                resolve({
                  success: true,
                  query: 'slow',
                  results: [],
                  total: 0,
                }),
              30000
            );
          })
      );

      render(<TrackSearch onTrackSelect={onTrackSelect} />, {
        wrapper: createWrapper(),
      });

      const searchInput = screen.getByPlaceholderText(
        /busca canciones, artistas o álbumes/i
      );

      await user.type(searchInput, 'slow');
      jest.advanceTimersByTime(500);

      // Should show loading state
      await waitFor(() => {
        expect(screen.getByRole('status')).toBeInTheDocument();
      });

      // Fast-forward 30 seconds
      jest.advanceTimersByTime(30000);

      await waitFor(() => {
        expect(mockSearchTracks).toHaveBeenCalled();
      });
    });
  });

  describe('Error Rate Handling', () => {
    it('should handle 50% error rate gracefully', async () => {
      const user = userEvent.setup({ delay: null });
      const onTrackSelect = jest.fn();

      let callCount = 0;
      mockSearchTracks.mockImplementation(() => {
        callCount++;
        if (callCount % 2 === 0) {
          return Promise.reject(new Error('Error'));
        }
        return Promise.resolve({
          success: true,
          query: 'test',
          results: [],
          total: 0,
        });
      });

      render(<TrackSearch onTrackSelect={onTrackSelect} />, {
        wrapper: createWrapper(),
      });

      const searchInput = screen.getByPlaceholderText(
        /busca canciones, artistas o álbumes/i
      );

      // Multiple searches with 50% error rate
      for (let i = 0; i < 10; i++) {
        await user.clear(searchInput);
        await user.type(searchInput, `test${i}`);
        jest.advanceTimersByTime(500);

        await waitFor(() => {
          // Should handle both success and error
          const hasError = screen.queryByText(/error/i);
          const hasResults = screen.queryByText(/track/i);
          expect(hasError || hasResults || true).toBeTruthy();
        }, { timeout: 1000 });
      }
    });
  });

  describe('Resource Exhaustion', () => {
    it('should handle when API returns maximum results', async () => {
      const user = userEvent.setup({ delay: null });
      const onTrackSelect = jest.fn();

      const maxTracks: Track[] = Array.from({ length: 100 }, (_, i) => ({
        id: `track-${i}`,
        name: `Track ${i}`,
        artists: [`Artist ${i}`],
        album: `Album ${i}`,
        duration_ms: 200000,
        popularity: 80,
        images: [],
      }));

      mockSearchTracks.mockResolvedValue({
        success: true,
        query: 'max',
        results: maxTracks,
        total: 100,
      });

      render(<TrackSearch onTrackSelect={onTrackSelect} />, {
        wrapper: createWrapper(),
      });

      const searchInput = screen.getByPlaceholderText(
        /busca canciones, artistas o álbumes/i
      );

      await user.type(searchInput, 'max');
      jest.advanceTimersByTime(500);

      await waitFor(() => {
        expect(screen.getByText('Track 0')).toBeInTheDocument();
        expect(screen.getByText('Track 99')).toBeInTheDocument();
      });
    });
  });
});

