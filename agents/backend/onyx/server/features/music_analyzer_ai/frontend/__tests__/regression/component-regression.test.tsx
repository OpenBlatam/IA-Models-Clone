/**
 * Regression Tests - Components
 * Tests to prevent regression of component functionality
 */

import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Navigation } from '@/components/Navigation';
import { TrackSearch } from '@/components/music/TrackSearch';
import { AudioPlayer } from '@/components/music/AudioPlayer';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
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

describe('Component Regression Tests', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  describe('Navigation Regression', () => {
    it('should maintain navigation structure', () => {
      render(<Navigation />);

      // Should always have these links
      expect(screen.getByText('Inicio')).toBeInTheDocument();
      expect(screen.getByText('Music AI')).toBeInTheDocument();
      expect(screen.getByText('Robot AI')).toBeInTheDocument();
    });

    it('should maintain brand name', () => {
      render(<Navigation />);
      expect(screen.getByText('Blatam Academy')).toBeInTheDocument();
    });
  });

  describe('TrackSearch Regression', () => {
    it('should maintain search functionality', async () => {
      const user = userEvent.setup({ delay: null });
      const onTrackSelect = jest.fn();

      mockSearchTracks.mockResolvedValue({
        success: true,
        query: 'test',
        results: [mockTrack],
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

      // Should still search and display results
      await screen.findByText('Test Track');
      expect(screen.getByText('Test Track')).toBeInTheDocument();
    });

    it('should maintain debounce behavior', async () => {
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

      // Rapid typing
      await user.type(searchInput, 't');
      jest.advanceTimersByTime(100);
      await user.type(searchInput, 'e');
      jest.advanceTimersByTime(100);
      await user.type(searchInput, 's');
      jest.advanceTimersByTime(100);
      await user.type(searchInput, 't');

      // Should not call immediately
      expect(mockSearchTracks).not.toHaveBeenCalled();

      // After debounce
      jest.advanceTimersByTime(500);

      // Should only call once
      expect(mockSearchTracks).toHaveBeenCalledTimes(1);
    });
  });

  describe('AudioPlayer Regression', () => {
    it('should maintain player controls', () => {
      render(<AudioPlayer track={mockTrack} />);

      // Should always have these elements
      expect(screen.getByText('Test Track')).toBeInTheDocument();
      expect(screen.getByText('Test Artist')).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /play/i })).toBeInTheDocument();
    });

    it('should maintain preview unavailable message', () => {
      const trackWithoutPreview: Track = {
        ...mockTrack,
        preview_url: null,
      };

      render(<AudioPlayer track={trackWithoutPreview} />);

      expect(
        screen.getByText(/preview no disponible/i)
      ).toBeInTheDocument();
    });
  });
});

