/**
 * E2E Tests - Cross-Component Integration
 * Tests interactions between multiple components
 */

import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { TrackSearch } from '@/components/music/TrackSearch';
import { AudioPlayer } from '@/components/music/AudioPlayer';
import { ApiStatus } from '@/components/api-status';
import { Navigation } from '@/components/Navigation';
import { SearchSuggestions } from '@/components/music/SearchSuggestions';
import { SortOptions } from '@/components/music/SortOptions';
import { ProgressIndicator } from '@/components/music/ProgressIndicator';
import { musicApiService } from '@/lib/api/music-api';
import { checkApiHealth } from '@/lib/api/client';
import { type Track } from '@/lib/api/music-api';

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

describe('E2E Cross-Component Integration', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    jest.useFakeTimers();
    localStorage.clear();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  describe('Full Application Flow', () => {
    it('should complete full application workflow', async () => {
      const user = userEvent.setup({ delay: null });
      const onTrackSelect = jest.fn();
      const onSelectSuggestion = jest.fn();
      const onSortChange = jest.fn();

      mockCheckApiHealth.mockResolvedValue({
        status: 'healthy',
        message: 'API is reachable',
        timestamp: Date.now(),
      });

      mockSearchTracks.mockResolvedValue({
        success: true,
        query: 'bohemian',
        results: [mockTrack],
        total: 1,
      });

      render(
        <>
          <Navigation />
          <ApiStatus />
          <SearchSuggestions onSelect={onSelectSuggestion} />
          <SortOptions onSortChange={onSortChange} />
          <TrackSearch onTrackSelect={onTrackSelect} />
        </>,
        { wrapper: createWrapper() }
      );

      // Step 1: Verify all components are rendered
      expect(screen.getByText('Music AI')).toBeInTheDocument();
      await waitFor(() => {
        expect(screen.getByText('Connected')).toBeInTheDocument();
      });
      expect(screen.getByText('Tendencias')).toBeInTheDocument();
      expect(screen.getByText('Ordenar')).toBeInTheDocument();

      // Step 2: User selects from suggestions
      const suggestionButton = screen.getByText('Bohemian Rhapsody');
      await user.click(suggestionButton);

      expect(onSelectSuggestion).toHaveBeenCalled();

      // Step 3: Search is performed
      jest.advanceTimersByTime(500);

      await waitFor(() => {
        expect(mockSearchTracks).toHaveBeenCalled();
      });

      // Step 4: Results appear
      await waitFor(() => {
        expect(screen.getByText('Test Track')).toBeInTheDocument();
      });

      // Step 5: User selects track
      const trackButton = screen.getByText('Test Track').closest('button');
      await user.click(trackButton!);

      expect(onTrackSelect).toHaveBeenCalledWith(mockTrack);
    });
  });

  describe('Component State Synchronization', () => {
    it('should synchronize state between search and player', async () => {
      const user = userEvent.setup({ delay: null });
      let selectedTrack: Track | null = null;

      const onTrackSelect = (track: Track) => {
        selectedTrack = track;
      };

      mockSearchTracks.mockResolvedValue({
        success: true,
        query: 'test',
        results: [mockTrack],
        total: 1,
      });

      const { rerender } = render(
        <>
          <TrackSearch onTrackSelect={onTrackSelect} />
          {selectedTrack && <AudioPlayer track={selectedTrack} />}
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

      const trackButton = screen.getByText('Test Track').closest('button');
      await user.click(trackButton!);

      // Re-render with selected track
      rerender(
        <>
          <TrackSearch onTrackSelect={onTrackSelect} />
          {selectedTrack && <AudioPlayer track={selectedTrack} />}
        </>
      );

      // Player should show selected track
      expect(screen.getByText('Test Track')).toBeInTheDocument();
      expect(screen.getByText('Test Artist')).toBeInTheDocument();
    });
  });

  describe('Error Propagation Across Components', () => {
    it('should propagate errors from API to UI components', async () => {
      const user = userEvent.setup({ delay: null });
      const onTrackSelect = jest.fn();

      mockCheckApiHealth.mockResolvedValue({
        status: 'unhealthy',
        message: 'Connection failed',
        timestamp: Date.now(),
      });

      mockSearchTracks.mockRejectedValue(new Error('API Error'));

      render(
        <>
          <ApiStatus />
          <TrackSearch onTrackSelect={onTrackSelect} />
        </>,
        { wrapper: createWrapper() }
      );

      // API status shows error
      await waitFor(() => {
        expect(screen.getByText('Disconnected')).toBeInTheDocument();
      });

      // Search also shows error
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

  describe('Multi-Component User Interaction', () => {
    it('should handle user interacting with multiple components simultaneously', async () => {
      const user = userEvent.setup({ delay: null });
      const onTrackSelect = jest.fn();
      const onSortChange = jest.fn();

      mockSearchTracks.mockResolvedValue({
        success: true,
        query: 'test',
        results: [mockTrack],
        total: 1,
      });

      render(
        <>
          <SortOptions onSortChange={onSortChange} />
          <TrackSearch onTrackSelect={onTrackSelect} />
        </>,
        { wrapper: createWrapper() }
      );

      // Interact with sort
      const sortButton = screen.getByText('Ordenar');
      await user.click(sortButton);

      const nameOption = screen.getByText('Nombre');
      await user.click(nameOption);

      expect(onSortChange).toHaveBeenCalled();

      // Interact with search
      const searchInput = screen.getByPlaceholderText(
        /busca canciones, artistas o álbumes/i
      );
      await user.type(searchInput, 'test');
      jest.advanceTimersByTime(500);

      await waitFor(() => {
        expect(screen.getByText('Test Track')).toBeInTheDocument();
      });

      // Both should work independently
      expect(onSortChange).toHaveBeenCalled();
      expect(mockSearchTracks).toHaveBeenCalled();
    });
  });

  describe('Progress and Search Integration', () => {
    it('should show progress during search and analysis', async () => {
      const user = userEvent.setup({ delay: null });
      const onTrackSelect = jest.fn();

      const steps = [
        { id: '1', label: 'Buscando...', status: 'loading' as const },
        { id: '2', label: 'Analizando...', status: 'pending' as const },
      ];

      mockSearchTracks.mockResolvedValue({
        success: true,
        query: 'test',
        results: [mockTrack],
        total: 1,
      });

      render(
        <>
          <ProgressIndicator steps={steps} />
          <TrackSearch onTrackSelect={onTrackSelect} />
        </>,
        { wrapper: createWrapper() }
      );

      // Progress indicator shows loading
      expect(screen.getByText('Buscando...')).toBeInTheDocument();

      // User searches
      const searchInput = screen.getByPlaceholderText(
        /busca canciones, artistas o álbumes/i
      );
      await user.type(searchInput, 'test');
      jest.advanceTimersByTime(500);

      // Both components work together
      await waitFor(() => {
        expect(screen.getByText('Test Track')).toBeInTheDocument();
      });
    });
  });

  describe('Navigation and Content Integration', () => {
    it('should maintain state when navigating', () => {
      render(<Navigation />);

      const homeLink = screen.getByText('Inicio').closest('a');
      const musicLink = screen.getByText('Music AI').closest('a');
      const robotLink = screen.getByText('Robot AI').closest('a');

      // All links should be present
      expect(homeLink).toBeInTheDocument();
      expect(musicLink).toBeInTheDocument();
      expect(robotLink).toBeInTheDocument();

      // Links should have correct hrefs
      expect(homeLink).toHaveAttribute('href', '/');
      expect(musicLink).toHaveAttribute('href', '/music');
      expect(robotLink).toHaveAttribute('href', '/robot');
    });
  });

  describe('Component Lifecycle Integration', () => {
    it('should handle component mounting and unmounting in sequence', () => {
      const { unmount: unmount1 } = render(
        <TrackSearch onTrackSelect={jest.fn()} />,
        { wrapper: createWrapper() }
      );

      expect(screen.getByPlaceholderText(/busca canciones/i)).toBeInTheDocument();

      unmount1();

      expect(screen.queryByPlaceholderText(/busca canciones/i)).not.toBeInTheDocument();

      // Mount again
      const { unmount: unmount2 } = render(
        <TrackSearch onTrackSelect={jest.fn()} />,
        { wrapper: createWrapper() }
      );

      expect(screen.getByPlaceholderText(/busca canciones/i)).toBeInTheDocument();

      unmount2();
    });
  });
});

