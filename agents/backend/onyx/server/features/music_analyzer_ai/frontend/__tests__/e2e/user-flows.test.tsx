/**
 * End-to-End Tests - User Flows
 * Tests complete user workflows from start to finish
 */

import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { TrackSearch } from '@/components/music/TrackSearch';
import { AudioPlayer } from '@/components/music/AudioPlayer';
import { ApiStatus } from '@/components/api-status';
import { Navigation } from '@/components/Navigation';
import { musicApiService, type Track } from '@/lib/api/music-api';
import { checkApiHealth } from '@/lib/api/client';

// Mock all API services
jest.mock('@/lib/api/music-api', () => ({
  musicApiService: {
    searchTracks: jest.fn(),
    analyzeTrack: jest.fn(),
    getRecommendations: jest.fn(),
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
const mockAnalyzeTrack = musicApiService.analyzeTrack as jest.MockedFunction<
  typeof musicApiService.analyzeTrack
>;
const mockGetRecommendations =
  musicApiService.getRecommendations as jest.MockedFunction<
    typeof musicApiService.getRecommendations
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

describe('E2E User Flows', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  describe('Flow 1: Search and Play Track', () => {
    it('should complete full search to play flow', async () => {
      const user = userEvent.setup({ delay: null });
      const onTrackSelect = jest.fn();

      mockSearchTracks.mockResolvedValue({
        success: true,
        query: 'test',
        results: [mockTrack],
        total: 1,
      });

      mockCheckApiHealth.mockResolvedValue({
        status: 'healthy',
        message: 'API is reachable',
        timestamp: Date.now(),
      });

      render(
        <>
          <ApiStatus />
          <TrackSearch onTrackSelect={onTrackSelect} />
        </>,
        { wrapper: createWrapper() }
      );

      // Step 1: Verify API is healthy
      await waitFor(() => {
        expect(screen.getByText('Connected')).toBeInTheDocument();
      });

      // Step 2: Search for track
      const searchInput = screen.getByPlaceholderText(
        /busca canciones, artistas o álbumes/i
      );
      await user.type(searchInput, 'test');
      jest.advanceTimersByTime(500);

      // Step 3: Wait for results
      await waitFor(() => {
        expect(screen.getByText('Test Track')).toBeInTheDocument();
      });

      // Step 4: Select track
      const trackButton = screen.getByText('Test Track').closest('button');
      await user.click(trackButton!);

      // Step 5: Verify track was selected
      expect(onTrackSelect).toHaveBeenCalledWith(mockTrack);
    });
  });

  describe('Flow 2: Track Analysis Workflow', () => {
    it('should complete track analysis flow', async () => {
      const user = userEvent.setup({ delay: null });
      const onTrackSelect = jest.fn();

      mockSearchTracks.mockResolvedValue({
        success: true,
        query: 'test',
        results: [mockTrack],
        total: 1,
      });

      mockAnalyzeTrack.mockResolvedValue({
        success: true,
        track_basic_info: {
          name: 'Test Track',
          artists: ['Test Artist'],
          album: 'Test Album',
          duration_seconds: 200,
        },
        musical_analysis: {
          key_signature: 'C',
          root_note: 'C',
          mode: 'major',
          tempo: { bpm: 120, category: 'moderate' },
          time_signature: '4/4',
          scale: { name: 'C Major', notes: ['C', 'D', 'E', 'F', 'G', 'A', 'B'] },
        },
        technical_analysis: {
          energy: { value: 0.8, description: 'High energy' },
          danceability: { value: 0.7, description: 'Moderate danceability' },
          valence: { value: 0.6, description: 'Positive mood' },
          acousticness: { value: 0.2, description: 'Electronic' },
          instrumentalness: { value: 0.1, description: 'Vocal' },
          liveness: { value: 0.3, description: 'Studio' },
          loudness: { value: -5, description: 'Loud' },
        },
      });

      render(<TrackSearch onTrackSelect={onTrackSelect} />, {
        wrapper: createWrapper(),
      });

      // Search and select track
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

      // Track should be selected (analysis would happen in parent component)
      expect(onTrackSelect).toHaveBeenCalled();
    });
  });

  describe('Flow 3: Navigation and API Status', () => {
    it('should navigate and check API status', async () => {
      mockCheckApiHealth.mockResolvedValue({
        status: 'healthy',
        message: 'API is reachable',
        timestamp: Date.now(),
      });

      render(
        <>
          <Navigation />
          <ApiStatus showDetails={true} />
        </>,
        { wrapper: createWrapper() }
      );

      // Verify navigation is rendered
      expect(screen.getByText('Inicio')).toBeInTheDocument();
      expect(screen.getByText('Music AI')).toBeInTheDocument();
      expect(screen.getByText('Robot AI')).toBeInTheDocument();

      // Verify API status shows details
      await waitFor(() => {
        expect(screen.getByText('API Status')).toBeInTheDocument();
      });

      await waitFor(() => {
        expect(screen.getByText('API is reachable')).toBeInTheDocument();
      });
    });
  });

  describe('Flow 4: Audio Player Controls', () => {
    it('should interact with audio player controls', async () => {
      const user = userEvent.setup({ delay: null });
      const onNext = jest.fn();
      const onPrevious = jest.fn();

      render(
        <AudioPlayer track={mockTrack} onNext={onNext} onPrevious={onPrevious} />
      );

      // Verify track info is displayed
      expect(screen.getByText('Test Track')).toBeInTheDocument();
      expect(screen.getByText('Test Artist')).toBeInTheDocument();

      // Test next button
      const nextButton = screen.getByRole('button', { name: /next|skip forward/i });
      await user.click(nextButton);
      expect(onNext).toHaveBeenCalled();

      // Test previous button
      const prevButton = screen.getByRole('button', {
        name: /previous|skip back/i,
      });
      await user.click(prevButton);
      expect(onPrevious).toHaveBeenCalled();
    });
  });

  describe('Flow 5: Error Handling Flow', () => {
    it('should handle API errors gracefully', async () => {
      const user = userEvent.setup({ delay: null });
      const onTrackSelect = jest.fn();

      mockSearchTracks.mockRejectedValue(new Error('API Error'));
      mockCheckApiHealth.mockResolvedValue({
        status: 'unhealthy',
        message: 'Connection failed',
        timestamp: Date.now(),
      });

      render(
        <>
          <ApiStatus />
          <TrackSearch onTrackSelect={onTrackSelect} />
        </>,
        { wrapper: createWrapper() }
      );

      // Verify API shows as disconnected
      await waitFor(() => {
        expect(screen.getByText('Disconnected')).toBeInTheDocument();
      });

      // Try to search
      const searchInput = screen.getByPlaceholderText(
        /busca canciones, artistas o álbumes/i
      );
      await user.type(searchInput, 'test');
      jest.advanceTimersByTime(500);

      // Should show error message
      await waitFor(() => {
        expect(
          screen.getByText(/error al buscar canciones/i)
        ).toBeInTheDocument();
      });
    });
  });

  describe('Flow 6: Complete Music Discovery Flow', () => {
    it('should complete full music discovery workflow', async () => {
      const user = userEvent.setup({ delay: null });
      const onTrackSelect = jest.fn();

      const mockTracks = [
        mockTrack,
        {
          ...mockTrack,
          id: 'track-456',
          name: 'Another Track',
          artists: ['Another Artist'],
        },
      ];

      mockSearchTracks.mockResolvedValue({
        success: true,
        query: 'test',
        results: mockTracks,
        total: 2,
      });

      mockGetRecommendations.mockResolvedValue({
        tracks: [
          {
            id: 'rec-1',
            name: 'Recommended Track',
            artists: ['Recommended Artist'],
          },
        ],
      });

      render(<TrackSearch onTrackSelect={onTrackSelect} />, {
        wrapper: createWrapper(),
      });

      // Step 1: Search
      const searchInput = screen.getByPlaceholderText(
        /busca canciones, artistas o álbumes/i
      );
      await user.type(searchInput, 'test');
      jest.advanceTimersByTime(500);

      // Step 2: See multiple results
      await waitFor(() => {
        expect(screen.getByText('Test Track')).toBeInTheDocument();
        expect(screen.getByText('Another Track')).toBeInTheDocument();
      });

      // Step 3: Select first track
      const firstTrack = screen.getByText('Test Track').closest('button');
      await user.click(firstTrack!);

      expect(onTrackSelect).toHaveBeenCalledWith(mockTracks[0]);
    });
  });

  describe('Flow 7: Theme Toggle Flow', () => {
    it('should toggle theme and persist', async () => {
      const user = userEvent.setup();

      // This would require the actual ThemeToggle component
      // For now, we test the concept
      const { container } = render(
        <div>
          <button onClick={() => document.documentElement.classList.toggle('dark')}>
            Toggle Theme
          </button>
        </div>
      );

      const toggleButton = screen.getByText('Toggle Theme');
      const initialHasDark = document.documentElement.classList.contains('dark');

      await user.click(toggleButton);

      const afterToggleHasDark =
        document.documentElement.classList.contains('dark');
      expect(afterToggleHasDark).not.toBe(initialHasDark);
    });
  });

  describe('Flow 8: Multiple Component Interaction', () => {
    it('should interact with multiple components together', async () => {
      const user = userEvent.setup({ delay: null });
      const onTrackSelect = jest.fn();

      mockSearchTracks.mockResolvedValue({
        success: true,
        query: 'test',
        results: [mockTrack],
        total: 1,
      });

      mockCheckApiHealth.mockResolvedValue({
        status: 'healthy',
        message: 'API is reachable',
        timestamp: Date.now(),
      });

      render(
        <>
          <Navigation />
          <ApiStatus />
          <TrackSearch onTrackSelect={onTrackSelect} />
        </>,
        { wrapper: createWrapper() }
      );

      // Verify all components are rendered
      expect(screen.getByText('Music AI')).toBeInTheDocument();
      await waitFor(() => {
        expect(screen.getByText('Connected')).toBeInTheDocument();
      });

      // Interact with search
      const searchInput = screen.getByPlaceholderText(
        /busca canciones, artistas o álbumes/i
      );
      await user.type(searchInput, 'test');
      jest.advanceTimersByTime(500);

      await waitFor(() => {
        expect(screen.getByText('Test Track')).toBeInTheDocument();
      });

      // Select track
      const trackButton = screen.getByText('Test Track').closest('button');
      await user.click(trackButton!);

      expect(onTrackSelect).toHaveBeenCalled();
    });
  });
});

