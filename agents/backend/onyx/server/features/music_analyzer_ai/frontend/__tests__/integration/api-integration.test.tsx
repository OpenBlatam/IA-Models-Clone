import { render, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { TrackSearch } from '@/components/music/TrackSearch';
import { ApiStatus } from '@/components/api-status';
import { musicApiService } from '@/lib/api/music-api';
import { checkApiHealth } from '@/lib/api/client';

// Mock API services
jest.mock('@/lib/api/music-api', () => ({
  musicApiService: {
    searchTracks: jest.fn(),
  },
}));

jest.mock('@/lib/api/client', () => ({
  checkApiHealth: jest.fn(),
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
const mockCheckApiHealth = checkApiHealth as jest.MockedFunction<
  typeof checkApiHealth
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

describe('API Integration Tests', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  describe('TrackSearch with API', () => {
    it('should integrate with API service for searching', async () => {
      const mockTracks = [
        {
          id: '1',
          name: 'Test Track',
          artists: ['Artist'],
          album: 'Album',
          duration_ms: 200000,
          preview_url: 'https://example.com/preview.mp3',
          popularity: 50,
          images: [{ url: 'https://example.com/image.jpg' }],
        },
      ];

      mockSearchTracks.mockResolvedValue({
        success: true,
        query: 'test',
        results: mockTracks,
        total: 1,
      });

      const onTrackSelect = jest.fn();

      render(<TrackSearch onTrackSelect={onTrackSelect} />, {
        wrapper: createWrapper(),
      });

      const input = screen.getByPlaceholderText(
        /busca canciones, artistas o álbumes/i
      );

      // Simulate user typing
      input.focus();
      Object.defineProperty(input, 'value', {
        writable: true,
        value: 'test',
      });
      input.dispatchEvent(new Event('input', { bubbles: true }));

      // Fast-forward debounce
      jest.advanceTimersByTime(500);

      await waitFor(() => {
        expect(mockSearchTracks).toHaveBeenCalledWith('test', 10);
      });
    });

    it('should handle API errors gracefully', async () => {
      mockSearchTracks.mockRejectedValue(new Error('API Error'));

      const onTrackSelect = jest.fn();

      render(<TrackSearch onTrackSelect={onTrackSelect} />, {
        wrapper: createWrapper(),
      });

      const input = screen.getByPlaceholderText(
        /busca canciones, artistas o álbumes/i
      );

      input.focus();
      Object.defineProperty(input, 'value', {
        writable: true,
        value: 'test',
      });
      input.dispatchEvent(new Event('input', { bubbles: true }));

      jest.advanceTimersByTime(500);

      await waitFor(() => {
        expect(
          screen.getByText(/error al buscar canciones/i)
        ).toBeInTheDocument();
      });
    });
  });

  describe('ApiStatus with Health Check', () => {
    it('should integrate with health check API', async () => {
      mockCheckApiHealth.mockResolvedValue({
        status: 'healthy',
        message: 'API is reachable',
        timestamp: Date.now(),
      });

      render(<ApiStatus />, {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(screen.getByText('Connected')).toBeInTheDocument();
      });

      expect(mockCheckApiHealth).toHaveBeenCalled();
    });

    it('should show disconnected when API is unhealthy', async () => {
      mockCheckApiHealth.mockResolvedValue({
        status: 'unhealthy',
        message: 'Connection failed',
        timestamp: Date.now(),
      });

      render(<ApiStatus />, {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(screen.getByText('Disconnected')).toBeInTheDocument();
      });
    });
  });

  describe('End-to-End Flow', () => {
    it('should complete full search and select flow', async () => {
      const mockTracks = [
        {
          id: '1',
          name: 'Selected Track',
          artists: ['Artist'],
          album: 'Album',
          duration_ms: 200000,
          preview_url: 'https://example.com/preview.mp3',
          popularity: 50,
          images: [{ url: 'https://example.com/image.jpg' }],
        },
      ];

      mockSearchTracks.mockResolvedValue({
        success: true,
        query: 'selected',
        results: mockTracks,
        total: 1,
      });

      const onTrackSelect = jest.fn();

      render(<TrackSearch onTrackSelect={onTrackSelect} />, {
        wrapper: createWrapper(),
      });

      const input = screen.getByPlaceholderText(
        /busca canciones, artistas o álbumes/i
      ) as HTMLInputElement;

      // Search
      input.focus();
      Object.defineProperty(input, 'value', {
        writable: true,
        value: 'selected',
      });
      input.dispatchEvent(new Event('input', { bubbles: true }));

      jest.advanceTimersByTime(500);

      // Wait for results
      await waitFor(() => {
        expect(screen.getByText('Selected Track')).toBeInTheDocument();
      });

      // Select track
      const trackButton = screen.getByText('Selected Track').closest('button');
      trackButton?.click();

      // Verify selection
      expect(onTrackSelect).toHaveBeenCalledWith(mockTracks[0]);
      expect(input.value).toBe('');
    });
  });
});

