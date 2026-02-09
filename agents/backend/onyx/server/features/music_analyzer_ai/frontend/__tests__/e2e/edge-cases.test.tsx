/**
 * E2E Tests - Edge Cases
 * Tests edge cases, boundary conditions, and error scenarios
 */

import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { TrackSearch } from '@/components/music/TrackSearch';
import { AudioPlayer } from '@/components/music/AudioPlayer';
import { ProgressIndicator } from '@/components/music/ProgressIndicator';
import { ErrorBoundary } from '@/components/error-boundary';
import { musicApiService } from '@/lib/api/music-api';
import { type Track } from '@/lib/api/music-api';

jest.mock('@/lib/api/music-api', () => ({
  musicApiService: {
    searchTracks: jest.fn(),
    analyzeTrack: jest.fn(),
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

describe('E2E Edge Cases', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  describe('Network Edge Cases', () => {
    it('should handle slow network responses', async () => {
      const user = userEvent.setup({ delay: null });
      const onTrackSelect = jest.fn();

      mockSearchTracks.mockImplementation(
        () =>
          new Promise((resolve) => {
            setTimeout(
              () =>
                resolve({
                  success: true,
                  query: 'test',
                  results: [],
                  total: 0,
                }),
              2000
            );
          })
      );

      render(<TrackSearch onTrackSelect={onTrackSelect} />, {
        wrapper: createWrapper(),
      });

      const searchInput = screen.getByPlaceholderText(
        /busca canciones, artistas o álbumes/i
      );

      await user.type(searchInput, 'test');
      jest.advanceTimersByTime(500);

      // Should show loading state
      await waitFor(() => {
        expect(screen.getByRole('status')).toBeInTheDocument();
      });

      // Fast-forward time
      jest.advanceTimersByTime(2000);

      await waitFor(() => {
        expect(mockSearchTracks).toHaveBeenCalled();
      });
    });

    it('should handle request timeout', async () => {
      const user = userEvent.setup({ delay: null });
      const onTrackSelect = jest.fn();

      mockSearchTracks.mockRejectedValue(
        new Error('Request timeout after 10000ms')
      );

      render(<TrackSearch onTrackSelect={onTrackSelect} />, {
        wrapper: createWrapper(),
      });

      const searchInput = screen.getByPlaceholderText(
        /busca canciones, artistas o álbumes/i
      );

      await user.type(searchInput, 'test');
      jest.advanceTimersByTime(500);

      await waitFor(() => {
        expect(
          screen.getByText(/error al buscar canciones/i)
        ).toBeInTheDocument();
      });
    });

    it('should handle CORS errors', async () => {
      const user = userEvent.setup({ delay: null });
      const onTrackSelect = jest.fn();

      const corsError = new Error('CORS policy');
      corsError.name = 'CORS Error';
      mockSearchTracks.mockRejectedValue(corsError);

      render(<TrackSearch onTrackSelect={onTrackSelect} />, {
        wrapper: createWrapper(),
      });

      const searchInput = screen.getByPlaceholderText(
        /busca canciones, artistas o álbumes/i
      );

      await user.type(searchInput, 'test');
      jest.advanceTimersByTime(500);

      await waitFor(() => {
        expect(
          screen.getByText(/error al buscar canciones/i)
        ).toBeInTheDocument();
      });
    });
  });

  describe('Data Edge Cases', () => {
    it('should handle track with missing preview URL', () => {
      const trackWithoutPreview: Track = {
        id: 'track-123',
        name: 'Test Track',
        artists: ['Test Artist'],
        album: 'Test Album',
        duration_ms: 200000,
        preview_url: null,
        popularity: 80,
        images: [],
      };

      render(<AudioPlayer track={trackWithoutPreview} />);

      expect(
        screen.getByText(/preview no disponible/i)
      ).toBeInTheDocument();
    });

    it('should handle track with missing images', () => {
      const trackWithoutImages: Track = {
        id: 'track-123',
        name: 'Test Track',
        artists: ['Test Artist'],
        album: 'Test Album',
        duration_ms: 200000,
        preview_url: 'https://example.com/preview.mp3',
        popularity: 80,
        images: [],
      };

      render(<AudioPlayer track={trackWithoutImages} />);

      expect(screen.getByText('Test Track')).toBeInTheDocument();
      expect(screen.getByText('Test Artist')).toBeInTheDocument();
    });

    it('should handle track with very long names', () => {
      const longName = 'A'.repeat(200);
      const trackWithLongName: Track = {
        id: 'track-123',
        name: longName,
        artists: ['Test Artist'],
        album: 'Test Album',
        duration_ms: 200000,
        preview_url: 'https://example.com/preview.mp3',
        popularity: 80,
        images: [],
      };

      render(<AudioPlayer track={trackWithLongName} />);

      expect(screen.getByText(longName)).toBeInTheDocument();
    });

    it('should handle track with multiple artists', () => {
      const trackWithMultipleArtists: Track = {
        id: 'track-123',
        name: 'Test Track',
        artists: ['Artist 1', 'Artist 2', 'Artist 3', 'Artist 4'],
        album: 'Test Album',
        duration_ms: 200000,
        preview_url: 'https://example.com/preview.mp3',
        popularity: 80,
        images: [],
      };

      render(<AudioPlayer track={trackWithMultipleArtists} />);

      expect(screen.getByText('Test Track')).toBeInTheDocument();
      // Artists should be displayed (joined)
      const artistText = screen.getByText(/artist 1/i);
      expect(artistText).toBeInTheDocument();
    });
  });

  describe('UI Edge Cases', () => {
    it('should handle empty progress steps', () => {
      render(<ProgressIndicator steps={[]} />);

      expect(screen.getByText(/progreso del análisis/i)).toBeInTheDocument();
    });

    it('should handle all steps completed', () => {
      const allCompletedSteps = [
        { id: '1', label: 'Step 1', status: 'completed' as const },
        { id: '2', label: 'Step 2', status: 'completed' as const },
        { id: '3', label: 'Step 3', status: 'completed' as const },
      ];

      render(<ProgressIndicator steps={allCompletedSteps} />);

      allCompletedSteps.forEach((step) => {
        expect(screen.getByText(step.label)).toBeInTheDocument();
      });
    });

    it('should handle all steps with errors', () => {
      const allErrorSteps = [
        { id: '1', label: 'Step 1', status: 'error' as const },
        { id: '2', label: 'Step 2', status: 'error' as const },
      ];

      render(<ProgressIndicator steps={allErrorSteps} />);

      allErrorSteps.forEach((step) => {
        const stepElement = screen.getByText(step.label).closest('div');
        expect(stepElement?.querySelector('.text-red-400')).toBeInTheDocument();
      });
    });
  });

  describe('Error Boundary Edge Cases', () => {
    it('should catch and display component errors', () => {
      const ThrowError = () => {
        throw new Error('Test error');
      };

      render(
        <ErrorBoundary>
          <ThrowError />
        </ErrorBoundary>
      );

      expect(screen.getByText(/something went wrong/i)).toBeInTheDocument();
    });

    it('should allow error recovery', async () => {
      const user = userEvent.setup();
      const ThrowError = ({ shouldThrow }: { shouldThrow: boolean }) => {
        if (shouldThrow) {
          throw new Error('Test error');
        }
        return <div>No error</div>;
      };

      const { rerender } = render(
        <ErrorBoundary>
          <ThrowError shouldThrow={true} />
        </ErrorBoundary>
      );

      expect(screen.getByText(/something went wrong/i)).toBeInTheDocument();

      const resetButton = screen.getByText(/try again/i);
      await user.click(resetButton);

      // Re-render without error
      rerender(
        <ErrorBoundary>
          <ThrowError shouldThrow={false} />
        </ErrorBoundary>
      );

      expect(screen.getByText('No error')).toBeInTheDocument();
    });
  });

  describe('Input Edge Cases', () => {
    it('should handle empty search input', async () => {
      const user = userEvent.setup({ delay: null });
      const onTrackSelect = jest.fn();

      render(<TrackSearch onTrackSelect={onTrackSelect} />, {
        wrapper: createWrapper(),
      });

      const searchInput = screen.getByPlaceholderText(
        /busca canciones, artistas o álbumes/i
      ) as HTMLInputElement;

      // Type and then clear
      await user.type(searchInput, 'test');
      await user.clear(searchInput);

      jest.advanceTimersByTime(500);

      // Should not call API with empty query
      expect(mockSearchTracks).not.toHaveBeenCalled();
    });

    it('should handle only whitespace in search', async () => {
      const user = userEvent.setup({ delay: null });
      const onTrackSelect = jest.fn();

      render(<TrackSearch onTrackSelect={onTrackSelect} />, {
        wrapper: createWrapper(),
      });

      const searchInput = screen.getByPlaceholderText(
        /busca canciones, artistas o álbumes/i
      );

      await user.type(searchInput, '   ');
      jest.advanceTimersByTime(500);

      // Should not call API with only whitespace
      expect(mockSearchTracks).not.toHaveBeenCalled();
    });

    it('should handle unicode characters in search', async () => {
      const user = userEvent.setup({ delay: null });
      const onTrackSelect = jest.fn();

      mockSearchTracks.mockResolvedValue({
        success: true,
        query: 'café',
        results: [],
        total: 0,
      });

      render(<TrackSearch onTrackSelect={onTrackSelect} />, {
        wrapper: createWrapper(),
      });

      const searchInput = screen.getByPlaceholderText(
        /busca canciones, artistas o álbumes/i
      );

      await user.type(searchInput, 'café');
      jest.advanceTimersByTime(500);

      await waitFor(() => {
        expect(mockSearchTracks).toHaveBeenCalled();
      });
    });
  });

  describe('State Edge Cases', () => {
    it('should handle rapid state changes', async () => {
      const user = userEvent.setup({ delay: null });
      const onTrackSelect = jest.fn();

      mockSearchTracks.mockResolvedValue({
        success: true,
        query: 'final',
        results: [mockTrack],
        total: 1,
      });

      render(<TrackSearch onTrackSelect={onTrackSelect} />, {
        wrapper: createWrapper(),
      });

      const searchInput = screen.getByPlaceholderText(
        /busca canciones, artistas o álbumes/i
      );

      // Rapid state changes
      for (let i = 0; i < 10; i++) {
        await user.type(searchInput, 'a');
        jest.advanceTimersByTime(10);
        await user.clear(searchInput);
        jest.advanceTimersByTime(10);
      }

      // Should handle gracefully
      expect(searchInput).toBeInTheDocument();
    });

    it('should handle component unmount during async operation', async () => {
      const user = userEvent.setup({ delay: null });
      const onTrackSelect = jest.fn();

      mockSearchTracks.mockImplementation(
        () =>
          new Promise((resolve) => {
            setTimeout(
              () =>
                resolve({
                  success: true,
                  query: 'test',
                  results: [],
                  total: 0,
                }),
              1000
            );
          })
      );

      const { unmount } = render(<TrackSearch onTrackSelect={onTrackSelect} />, {
        wrapper: createWrapper(),
      });

      const searchInput = screen.getByPlaceholderText(
        /busca canciones, artistas o álbumes/i
      );

      await user.type(searchInput, 'test');
      jest.advanceTimersByTime(500);

      // Unmount before response
      unmount();

      // Should not crash
      jest.advanceTimersByTime(1000);
    });
  });

  describe('Browser Edge Cases', () => {
    it('should handle localStorage quota exceeded', () => {
      const originalSetItem = Storage.prototype.setItem;
      Storage.prototype.setItem = jest.fn(() => {
        throw new DOMException('QuotaExceededError');
      });

      const onSelect = jest.fn();

      // Should not crash
      render(<SearchSuggestions onSelect={onSelect} />);

      expect(screen.getByText('Tendencias')).toBeInTheDocument();

      // Restore
      Storage.prototype.setItem = originalSetItem;
    });

    it('should handle localStorage being disabled', () => {
      const originalGetItem = Storage.prototype.getItem;
      Storage.prototype.getItem = jest.fn(() => {
        throw new Error('localStorage is disabled');
      });

      const onSelect = jest.fn();

      // Should not crash
      render(<SearchSuggestions onSelect={onSelect} />);

      expect(screen.getByText('Tendencias')).toBeInTheDocument();

      // Restore
      Storage.prototype.getItem = originalGetItem;
    });
  });

  describe('API Response Edge Cases', () => {
    it('should handle malformed API response', async () => {
      const user = userEvent.setup({ delay: null });
      const onTrackSelect = jest.fn();

      // Mock malformed response
      mockSearchTracks.mockResolvedValue({
        success: true,
        query: 'test',
        results: null as any,
        total: 0,
      });

      render(<TrackSearch onTrackSelect={onTrackSelect} />, {
        wrapper: createWrapper(),
      });

      const searchInput = screen.getByPlaceholderText(
        /busca canciones, artistas o álbumes/i
      );

      await user.type(searchInput, 'test');
      jest.advanceTimersByTime(500);

      // Should handle gracefully
      await waitFor(() => {
        expect(mockSearchTracks).toHaveBeenCalled();
      });
    });

    it('should handle API response with unexpected structure', async () => {
      const user = userEvent.setup({ delay: null });
      const onTrackSelect = jest.fn();

      mockSearchTracks.mockResolvedValue({
        success: true,
        query: 'test',
        results: [
          {
            id: 'track-1',
            name: 'Track 1',
            artists: 'Single Artist String' as any, // Unexpected type
            album: 'Album',
            duration_ms: 200000,
            popularity: 80,
            images: [],
          },
        ],
        total: 1,
      });

      render(<TrackSearch onTrackSelect={onTrackSelect} />, {
        wrapper: createWrapper(),
      });

      const searchInput = screen.getByPlaceholderText(
        /busca canciones, artistas o álbumes/i
      );

      await user.type(searchInput, 'test');
      jest.advanceTimersByTime(500);

      // Should handle gracefully
      await waitFor(() => {
        expect(mockSearchTracks).toHaveBeenCalled();
      });
    });
  });
});

