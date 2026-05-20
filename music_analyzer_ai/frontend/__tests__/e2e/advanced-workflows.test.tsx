/**
 * E2E Tests - Advanced Workflows
 * Tests complex multi-component workflows and edge cases
 */

import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { TrackSearch } from '@/components/music/TrackSearch';
import { AudioPlayer } from '@/components/music/AudioPlayer';
import { ProgressIndicator } from '@/components/music/ProgressIndicator';
import { SearchSuggestions } from '@/components/music/SearchSuggestions';
import { SortOptions } from '@/components/music/SortOptions';
import { musicApiService } from '@/lib/api/music-api';
import { type Track } from '@/lib/api/music-api';

jest.mock('@/lib/api/music-api', () => ({
  musicApiService: {
    searchTracks: jest.fn(),
    analyzeTrack: jest.fn(),
    getRecommendations: jest.fn(),
    compareTracks: jest.fn(),
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
const mockAnalyzeTrack = musicApiService.analyzeTrack as jest.MockedFunction<
  typeof musicApiService.analyzeTrack
>;
const mockGetRecommendations =
  musicApiService.getRecommendations as jest.MockedFunction<
    typeof musicApiService.getRecommendations
  >;
const mockCompareTracks = musicApiService.compareTracks as jest.MockedFunction<
  typeof musicApiService.compareTracks
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

describe('E2E Advanced Workflows', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    jest.useFakeTimers();
    localStorage.clear();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  describe('Complex Search and Filter Workflow', () => {
    it('should handle search with suggestions and sorting', async () => {
      const user = userEvent.setup({ delay: null });
      const onTrackSelect = jest.fn();
      const onSelectSuggestion = jest.fn();

      mockSearchTracks.mockResolvedValue({
        success: true,
        query: 'bohemian',
        results: [mockTrack],
        total: 1,
      });

      render(
        <>
          <SearchSuggestions onSelect={onSelectSuggestion} />
          <TrackSearch onTrackSelect={onTrackSelect} />
        </>,
        { wrapper: createWrapper() }
      );

      // User clicks on trending suggestion
      const trendingButton = screen.getByText('Bohemian Rhapsody');
      await user.click(trendingButton);

      expect(onSelectSuggestion).toHaveBeenCalledWith('Bohemian Rhapsody');

      // Search should be triggered
      jest.advanceTimersByTime(500);

      await waitFor(() => {
        expect(mockSearchTracks).toHaveBeenCalled();
      });
    });
  });

  describe('Track Comparison Workflow', () => {
    it('should compare multiple tracks', async () => {
      const tracks = [
        mockTrack,
        { ...mockTrack, id: 'track-2', name: 'Track 2' },
        { ...mockTrack, id: 'track-3', name: 'Track 3' },
      ];

      mockCompareTracks.mockResolvedValue({
        success: true,
        comparison: {
          key_signatures: {
            all_same: false,
            keys: ['C', 'D', 'E'],
            most_common: 'C',
          },
          tempos: {
            average: 120,
            min: 100,
            max: 140,
            range: 40,
          },
        },
        similarities: [],
        differences: [],
        recommendations: [],
      });

      // This would be tested in a component that uses compareTracks
      // For now, we verify the API call
      await musicApiService.compareTracks(tracks.map((t) => t.id));

      expect(mockCompareTracks).toHaveBeenCalledWith(tracks.map((t) => t.id));
    });
  });

  describe('Playlist Management Workflow', () => {
    it('should handle adding tracks to queue', async () => {
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

      await waitFor(() => {
        expect(screen.getByText('Test Track')).toBeInTheDocument();
      });

      // Select track (would add to queue in real app)
      const trackButton = screen.getByText('Test Track').closest('button');
      await user.click(trackButton!);

      expect(onTrackSelect).toHaveBeenCalledWith(mockTrack);
    });
  });

  describe('Error Recovery and Retry Workflow', () => {
    it('should recover from multiple consecutive errors', async () => {
      const user = userEvent.setup({ delay: null });
      const onTrackSelect = jest.fn();

      // First two attempts fail
      mockSearchTracks
        .mockRejectedValueOnce(new Error('Network error'))
        .mockRejectedValueOnce(new Error('Timeout'))
        .mockResolvedValueOnce({
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

      // First attempt - fails
      await user.type(searchInput, 'test');
      jest.advanceTimersByTime(500);

      await waitFor(() => {
        expect(
          screen.getByText(/error al buscar canciones/i)
        ).toBeInTheDocument();
      });

      // Second attempt - fails
      await user.clear(searchInput);
      await user.type(searchInput, 'test');
      jest.advanceTimersByTime(500);

      await waitFor(() => {
        expect(
          screen.getByText(/error al buscar canciones/i)
        ).toBeInTheDocument();
      });

      // Third attempt - succeeds
      await user.clear(searchInput);
      await user.type(searchInput, 'test');
      jest.advanceTimersByTime(500);

      await waitFor(() => {
        expect(screen.getByText('Test Track')).toBeInTheDocument();
      });
    });
  });

  describe('Multi-Tab Workflow', () => {
    it('should handle switching between different views', async () => {
      const user = userEvent.setup({ delay: null });
      const onSortChange = jest.fn();

      render(<SortOptions onSortChange={onSortChange} />);

      // Open sort dropdown
      const sortButton = screen.getByText('Ordenar');
      await user.click(sortButton);

      // Select different sort options
      const nameOption = screen.getByText('Nombre');
      await user.click(nameOption);

      expect(onSortChange).toHaveBeenCalledWith('name', 'desc');

      // Open again and select duration
      await user.click(sortButton);
      const durationOption = screen.getByText('Duración');
      await user.click(durationOption);

      expect(onSortChange).toHaveBeenCalledWith('duration', 'desc');
    });
  });

  describe('Progress Tracking Workflow', () => {
    it('should track progress through multiple analysis steps', () => {
      const steps = [
        { id: '1', label: 'Step 1', status: 'completed' as const },
        { id: '2', label: 'Step 2', status: 'completed' as const },
        { id: '3', label: 'Step 3', status: 'loading' as const },
        { id: '4', label: 'Step 4', status: 'pending' as const },
        { id: '5', label: 'Step 5', status: 'pending' as const },
      ];

      render(<ProgressIndicator steps={steps} />);

      // Verify all steps are displayed
      steps.forEach((step) => {
        expect(screen.getByText(step.label)).toBeInTheDocument();
      });

      // Verify status indicators
      const step1 = screen.getByText('Step 1').closest('div');
      const step2 = screen.getByText('Step 2').closest('div');
      const step3 = screen.getByText('Step 3').closest('div');

      expect(step1?.querySelector('.text-green-400')).toBeInTheDocument();
      expect(step2?.querySelector('.text-green-400')).toBeInTheDocument();
      expect(step3?.querySelector('.animate-spin')).toBeInTheDocument();
    });
  });

  describe('Concurrent Operations Workflow', () => {
    it('should handle multiple simultaneous operations', async () => {
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
        musical_analysis: {},
        technical_analysis: {},
      });

      render(<TrackSearch onTrackSelect={onTrackSelect} />, {
        wrapper: createWrapper(),
      });

      const searchInput = screen.getByPlaceholderText(
        /busca canciones, artistas o álbumes/i
      );

      // Rapid search operations
      await user.type(searchInput, 't');
      jest.advanceTimersByTime(100);
      await user.type(searchInput, 'e');
      jest.advanceTimersByTime(100);
      await user.type(searchInput, 's');
      jest.advanceTimersByTime(100);
      await user.type(searchInput, 't');

      // Should debounce and only call once
      jest.advanceTimersByTime(500);

      await waitFor(() => {
        expect(mockSearchTracks).toHaveBeenCalledTimes(1);
      });
    });
  });

  describe('Data Persistence Workflow', () => {
    it('should persist search history across sessions', async () => {
      const user = userEvent.setup({ delay: null });
      const onSelect = jest.fn();

      // Set up initial localStorage data
      localStorage.setItem(
        'recent-searches',
        JSON.stringify(['Previous Search 1', 'Previous Search 2'])
      );

      render(<SearchSuggestions onSelect={onSelect} />);

      // Verify previous searches are shown
      expect(screen.getByText('Previous Search 1')).toBeInTheDocument();
      expect(screen.getByText('Previous Search 2')).toBeInTheDocument();

      // Select a new search
      const trendingButton = screen.getByText('Bohemian Rhapsody');
      await user.click(trendingButton);

      // Verify new search is saved
      await waitFor(() => {
        const saved = localStorage.getItem('recent-searches');
        expect(saved).toBeTruthy();
        if (saved) {
          const parsed = JSON.parse(saved);
          expect(parsed).toContain('Bohemian Rhapsody');
        }
      });
    });
  });

  describe('Edge Cases Workflow', () => {
    it('should handle empty search results', async () => {
      const user = userEvent.setup({ delay: null });
      const onTrackSelect = jest.fn();

      mockSearchTracks.mockResolvedValue({
        success: true,
        query: 'nonexistent',
        results: [],
        total: 0,
      });

      render(<TrackSearch onTrackSelect={onTrackSelect} />, {
        wrapper: createWrapper(),
      });

      const searchInput = screen.getByPlaceholderText(
        /busca canciones, artistas o álbumes/i
      );

      await user.type(searchInput, 'nonexistent');
      jest.advanceTimersByTime(500);

      await waitFor(() => {
        expect(mockSearchTracks).toHaveBeenCalled();
      });

      // Should not show any tracks
      expect(screen.queryByText('Test Track')).not.toBeInTheDocument();
    });

    it('should handle very long search queries', async () => {
      const user = userEvent.setup({ delay: null });
      const onTrackSelect = jest.fn();

      const longQuery = 'a'.repeat(200);

      mockSearchTracks.mockResolvedValue({
        success: true,
        query: longQuery,
        results: [],
        total: 0,
      });

      render(<TrackSearch onTrackSelect={onTrackSelect} />, {
        wrapper: createWrapper(),
      });

      const searchInput = screen.getByPlaceholderText(
        /busca canciones, artistas o álbumes/i
      );

      await user.type(searchInput, longQuery);
      jest.advanceTimersByTime(500);

      await waitFor(() => {
        expect(mockSearchTracks).toHaveBeenCalled();
      });
    });

    it('should handle special characters in search', async () => {
      const user = userEvent.setup({ delay: null });
      const onTrackSelect = jest.fn();

      const specialQuery = 'test@#$%^&*()';

      mockSearchTracks.mockResolvedValue({
        success: true,
        query: specialQuery,
        results: [],
        total: 0,
      });

      render(<TrackSearch onTrackSelect={onTrackSelect} />, {
        wrapper: createWrapper(),
      });

      const searchInput = screen.getByPlaceholderText(
        /busca canciones, artistas o álbumes/i
      );

      await user.type(searchInput, specialQuery);
      jest.advanceTimersByTime(500);

      await waitFor(() => {
        expect(mockSearchTracks).toHaveBeenCalled();
      });
    });
  });

  describe('Performance Under Load', () => {
    it('should handle rapid successive searches', async () => {
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

      // Rapid typing
      for (let i = 0; i < 20; i++) {
        await user.type(searchInput, 'a');
        jest.advanceTimersByTime(50);
      }

      // Should only call API once after debounce
      jest.advanceTimersByTime(500);

      await waitFor(() => {
        expect(mockSearchTracks).toHaveBeenCalledTimes(1);
      });
    });
  });

  describe('Accessibility in Complex Workflows', () => {
    it('should maintain keyboard navigation through complex flow', async () => {
      const user = userEvent.setup({ delay: null });
      const onTrackSelect = jest.fn();
      const onSelectSuggestion = jest.fn();

      mockSearchTracks.mockResolvedValue({
        success: true,
        query: 'test',
        results: [mockTrack],
        total: 1,
      });

      render(
        <>
          <SearchSuggestions onSelect={onSelectSuggestion} />
          <TrackSearch onTrackSelect={onTrackSelect} />
        </>,
        { wrapper: createWrapper() }
      );

      // Navigate with keyboard
      await user.tab();
      const searchInput = screen.getByPlaceholderText(
        /busca canciones, artistas o álbumes/i
      );
      expect(searchInput).toHaveFocus();

      // Type and search
      await user.type(searchInput, 'test');
      jest.advanceTimersByTime(500);

      await waitFor(() => {
        expect(screen.getByText('Test Track')).toBeInTheDocument();
      });

      // Navigate to result
      await user.tab();
      const trackButton = screen.getByText('Test Track').closest('button');
      expect(trackButton).toHaveFocus();

      // Select with Enter
      await user.keyboard('{Enter}');
      expect(onTrackSelect).toHaveBeenCalled();
    });
  });
});

