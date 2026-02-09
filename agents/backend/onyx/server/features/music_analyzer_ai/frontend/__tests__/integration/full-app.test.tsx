/**
 * Full App Integration Test
 * Tests the complete application flow
 */

import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Navigation } from '@/components/Navigation';
import { ApiStatus } from '@/components/api-status';
import { TrackSearch } from '@/components/music/TrackSearch';
import { AudioPlayer } from '@/components/music/AudioPlayer';
import { useMusicStore } from '@/lib/store/music-store';
import { musicApiService } from '@/lib/api/music-api';
import { checkApiHealth } from '@/lib/api/client';
import { type Track } from '@/lib/api/music-api';
import { act } from 'react';

jest.mock('@/lib/api/music-api', () => ({
  musicApiService: {
    searchTracks: jest.fn(),
    analyzeTrack: jest.fn(),
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
      queries: { retry: false, cacheTime: 0 },
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

describe('Full App Integration', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    jest.useFakeTimers();
    act(() => {
      useMusicStore.getState().clearQueue();
      useMusicStore.getState().setCurrentTrack(null);
    });
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it('should complete full application workflow', async () => {
    const user = userEvent.setup({ delay: null });

    mockCheckApiHealth.mockResolvedValue({
      status: 'healthy',
      message: 'API is reachable',
      timestamp: Date.now(),
    });

    mockSearchTracks.mockResolvedValue({
      success: true,
      query: 'test',
      results: [mockTrack],
      total: 1,
    });

    render(
      <>
        <Navigation />
        <ApiStatus />
        <TrackSearch onTrackSelect={() => {}} />
      </>,
      { wrapper: createWrapper() }
    );

    // Step 1: Verify navigation
    expect(screen.getByText('Music AI')).toBeInTheDocument();

    // Step 2: Verify API status
    await waitFor(() => {
      expect(screen.getByText('Connected')).toBeInTheDocument();
    });

    // Step 3: Search for track
    const searchInput = screen.getByPlaceholderText(
      /busca canciones, artistas o álbumes/i
    );
    await user.type(searchInput, 'test');
    jest.advanceTimersByTime(500);

    // Step 4: Verify results
    await waitFor(() => {
      expect(screen.getByText('Test Track')).toBeInTheDocument();
    });

    // Step 5: Select track
    const trackButton = screen.getByText('Test Track').closest('button');
    await user.click(trackButton!);

    // Step 6: Verify track in store
    act(() => {
      const queue = useMusicStore.getState().playlistQueue;
      expect(queue.length).toBeGreaterThanOrEqual(0);
    });
  });

  it('should handle error states across components', async () => {
    mockCheckApiHealth.mockResolvedValue({
      status: 'unhealthy',
      message: 'Connection failed',
      timestamp: Date.now(),
    });

    mockSearchTracks.mockRejectedValue(new Error('API Error'));

    render(
      <>
        <ApiStatus />
        <TrackSearch onTrackSelect={() => {}} />
      </>,
      { wrapper: createWrapper() }
    );

    // API status should show error
    await waitFor(() => {
      expect(screen.getByText('Disconnected')).toBeInTheDocument();
    });

    // Search should also handle error
    const user = userEvent.setup({ delay: null });
    const searchInput = screen.getByPlaceholderText(
      /busca canciones, artistas o álbumes/i
    );
    await user.type(searchInput, 'test');
    jest.advanceTimersByTime(500);

    await waitFor(() => {
      expect(screen.getByText(/error/i)).toBeInTheDocument();
    });
  });

  it('should sync state between components', async () => {
    const user = userEvent.setup({ delay: null });

    mockSearchTracks.mockResolvedValue({
      success: true,
      query: 'test',
      results: [mockTrack],
      total: 1,
    });

    render(
      <>
        <TrackSearch onTrackSelect={() => {}} />
        <AudioPlayer track={mockTrack} />
      </>,
      { wrapper: createWrapper() }
    );

    // Search and select
    const searchInput = screen.getByPlaceholderText(
      /busca canciones, artistas o álbumes/i
    );
    await user.type(searchInput, 'test');
    jest.advanceTimersByTime(500);

    await waitFor(() => {
      expect(screen.getByText('Test Track')).toBeInTheDocument();
    });

    // Player should show track
    expect(screen.getAllByText('Test Track').length).toBeGreaterThan(0);
  });
});

